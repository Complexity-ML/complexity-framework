"""Four-checkpoint retention, atomic replacement and verified history purges."""
import hashlib, json, shutil, time
from pathlib import Path, PurePosixPath
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete, hf_hub_download
from filelock import FileLock
from data import ROOT, sha
from retention import NAME, describe, select_keep
from checkpoint_json import prepare_upload

runtime=ROOT/'runtime'; runtime.mkdir(exist_ok=True)
recipe=json.loads((ROOT/'recipe.json').read_text())
policy=json.loads((ROOT/'retention.json').read_text())
repo=recipe['model_repo']; token=(runtime/'hf-token').read_text().strip()
api=HfApi(endpoint='https://huggingface.co',token=token)

def event(name, **kw): print(json.dumps(dict(event=name,**kw)),flush=True)

def atomic_json(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2)); tmp.replace(path)

def remote_json(name, revision):
    path=Path(hf_hub_download(repo,name,revision=revision,token=token))
    return json.loads(path.read_text()),sha(path)

def inventory():
    info=api.model_info(repo)
    revision=info.sha
    files=api.list_repo_files(repo,revision=revision)
    names=sorted({f.split('/')[0] for f in files if NAME.fullmatch(f.split('/')[0])})
    catalog={}
    for name in names:
        assert name+'/complete.json' in files, f'Incomplete remote checkpoint: {name}'
        marker,digest=remote_json(name+'/complete.json',revision)
        catalog[name]=dict(describe(name,marker),marker=marker,manifest_sha256=digest)
    return revision,files,catalog

def verify(name, item, revision):
    prefix=name+'/' if name else ''
    entries=dict(item['marker']['files'])
    for file in entries:
        p=PurePosixPath(file); assert not p.is_absolute() and '..' not in p.parts
    names=[prefix+file for file in entries]
    for start in range(0,len(names),100):
        infos=api.get_paths_info(repo,names[start:start+100],revision=revision)
        assert len(infos)==len(names[start:start+100])
        for info in infos:
            relative=info.path[len(prefix):]; expected=entries[relative]
            assert info.size==expected['bytes'],info.path
            if info.lfs:
                assert info.lfs.sha256==expected['sha256'],info.path
            else:
                downloaded=Path(hf_hub_download(repo,info.path,revision=revision,token=token))
                assert sha(downloaded)==expected['sha256'],info.path
    _,digest=remote_json(prefix+'complete.json',revision)
    assert digest==item['manifest_sha256']

def latest_payload(catalog):
    name=max(catalog,key=lambda n:catalog[n]['step']); item=catalog[name]
    return json.dumps(dict(format=2,checkpoint=name,step=item['step'],tokens_seen=item['step']*131072,
        manifest_sha256=item['manifest_sha256'],revision_policy='same snapshot as latest.json'),indent=2).encode()

def verify_all(catalog, revision):
    for name,item in catalog.items(): verify(name,item,revision)
    if catalog:
        latest,_=remote_json('latest.json',revision)
        assert latest['format']==2 and latest['checkpoint']==max(catalog,key=lambda n:catalog[n]['step'])
        assert latest['manifest_sha256']==catalog[latest['checkpoint']]['manifest_sha256']


def squash_if_pending():
    if not policy.get('purge_history_after_eviction', False): return
    pending=runtime/'purge-pending.json'
    if not pending.exists(): return
    revision,files,catalog=inventory()
    assert set(catalog)==select_keep(catalog,policy)
    verify_all(catalog,revision)
    refs=api.list_repo_refs(repo)
    assert {b.name for b in refs.branches}=={'main'} and not refs.tags, 'Unexpected branches/tags: automatic purge paused'
    api.super_squash_history(repo,branch='main',commit_message='Retain 3 regular checkpoints and latest interruption; purge obsolete history')
    after,afterfiles,aftercatalog=inventory()
    assert set(files)==set(afterfiles)
    assert {n:x['manifest_sha256'] for n,x in catalog.items()}=={n:x['manifest_sha256'] for n,x in aftercatalog.items()}
    verify_all(aftercatalog,after)
    assert len(api.list_repo_commits(repo,revision='main'))==1
    atomic_json(runtime/'last-purge.json',dict(revision=after,kept=sorted(aftercatalog),time=time.time()))
    for name,item in aftercatalog.items():
        atomic_json(runtime/(name+'.uploaded.json'),dict(revision=after,verified=True,step=item['step']))
    pending.unlink()
    event('history_purge_verified',revision=after,kept=sorted(aftercatalog),checkpoints=len(aftercatalog))


def reconcile():
    revision,files,catalog=inventory()
    keep=select_keep(catalog,policy); stale=set(catalog)-keep
    retained={n:catalog[n] for n in keep}
    for name in keep: verify(name,catalog[name],revision)
    oldlatest=None
    if 'latest.json' in files: oldlatest,_=remote_json('latest.json',revision)
    desired=json.loads(latest_payload(retained)) if retained else None
    ops=[CommitOperationDelete(path_in_repo=f) for f in files if f.split('/')[0] in stale]
    if desired is not None and oldlatest!=desired:
        ops.append(CommitOperationAdd(path_in_repo='latest.json',path_or_fileobj=latest_payload(retained)))
    if stale: atomic_json(runtime/'purge-pending.json',dict(reason='retention',removed=sorted(stale)))
    if ops:
        commit=api.create_commit(repo,operations=ops,parent_commit=revision,commit_message='Apply 3 regular plus 1 interruption checkpoint retention')
        verify_all(retained,commit.oid)
    squash_if_pending()
    # Only remove local snapshots when the retained remote set is verified.
    if retained:
        for path in (ROOT/'run').glob('step_*'):
            if not (path/'complete.json').exists(): continue
            marker=json.loads((path/'complete.json').read_text())
            candidate=dict(catalog); candidate[path.name]=dict(describe(path.name,marker))
            if path.name not in select_keep(candidate,policy):
                shutil.rmtree(path)
                staged=runtime/'upload-json'/path.name
                if staged.exists():shutil.rmtree(staged)


def iteration():
    reconcile()
    complete=sorted(p for p in (ROOT/'run').glob('step_*') if (p/'complete.json').exists())
    for path in complete:
        if not path.exists(): continue
        receipt=runtime/(path.name+'.uploaded.json')
        if receipt.exists(): continue
        path=prepare_upload(path,runtime/'upload-json')
        revision,files,catalog=inventory()
        marker=json.loads((path/'complete.json').read_text())
        item=dict(describe(path.name,marker),marker=marker,manifest_sha256=sha(path/'complete.json'))
        for name,meta in marker['files'].items():
            relative=PurePosixPath(name); assert not relative.is_absolute() and '..' not in relative.parts
            assert (path/name).stat().st_size==meta['bytes'] and sha(path/name)==meta['sha256'],name
        proposed=dict(catalog); proposed[path.name]=item
        keep=select_keep(proposed,policy)
        if path.name not in keep: continue
        # Verify the surviving existing snapshots before one atomic add/delete commit.
        for name in keep-{path.name}: verify(name,catalog[name],revision)
        stale=set(catalog)-keep
        retained={n:proposed[n] for n in keep}
        ops=[CommitOperationAdd(path_in_repo=path.name+'/'+str(p.relative_to(path)),path_or_fileobj=str(p)) for p in path.rglob('*') if p.is_file()]
        ops += [CommitOperationDelete(path_in_repo=f) for f in files if f.split('/')[0] in stale]
        ops.append(CommitOperationAdd(path_in_repo='latest.json',path_or_fileobj=latest_payload(retained)))
        if stale: atomic_json(runtime/'purge-pending.json',dict(reason='replacement',removed=sorted(stale)))
        commit=api.create_commit(repo,operations=ops,parent_commit=revision,commit_message=f'Complete checkpoint {marker["step"]}; retain at most 3 regular plus 1 interruption')
        verify_all(retained,commit.oid)
        atomic_json(receipt,dict(revision=commit.oid,verified=True,step=marker['step']))
        event('upload_verified',checkpoint=path.name,revision=commit.oid,kind=item['kind'],kept=sorted(keep))
        reconcile()
    final=ROOT/'final'
    if (final/'complete.json').exists() and not (runtime/'final.uploaded.json').exists():
        marker=json.loads((final/'complete.json').read_text())
        for name,meta in marker['files'].items(): assert sha(final/name)==meta['sha256']
        commit=api.create_commit(repo,operations=[CommitOperationAdd(path_in_repo=p.name,path_or_fileobj=str(p)) for p in final.iterdir() if p.is_file()],commit_message='Publish completed 70B-token model at repository root')
        verify('',dict(marker=marker,manifest_sha256=sha(final/'complete.json')),commit.oid)
        atomic_json(runtime/'final.uploaded.json',dict(revision=commit.oid,verified=True))
        event('final_upload_verified',revision=commit.oid)

if __name__=='__main__':
    with FileLock(str(runtime/'upload.lock'),timeout=0):
        delay=15
        while True:
            try:
                iteration(); delay=15
                if (runtime/'pause-uploader').exists(): break
            except Exception as error:
                event('upload_error',type=type(error).__name__,message=str(error)); delay=min(300,delay*2)
            time.sleep(delay)
