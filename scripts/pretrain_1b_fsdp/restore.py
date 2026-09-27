"""Restore from one current Hub snapshot, safely across history purges."""
import json,shutil,time,argparse
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download,snapshot_download
from huggingface_hub.errors import HfHubHTTPError
from data import ROOT,sha
from retention import NAME

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--revision', default=None, help='Pinned HF snapshot; default: current latest')
args=parser.parse_args()
recipe=json.loads((ROOT/'recipe.json').read_text())
token=(ROOT/'runtime/hf-token').read_text().strip() if (ROOT/'runtime/hf-token').exists() else None
api=HfApi(token=token,endpoint='https://huggingface.co')
repo=recipe['model_repo']; staging=ROOT/'runtime/restore-staging'
for attempt in range(3):
    try:
        revision=args.revision or api.model_info(repo).sha
        latest=json.loads(Path(hf_hub_download(repo,'latest.json',revision=revision,token=token)).read_text())
        name=latest['checkpoint']; assert NAME.fullmatch(name)
        manifest_file=Path(hf_hub_download(repo,name+'/complete.json',revision=revision,token=token))
        assert latest['format']==2 and sha(manifest_file)==latest['manifest_sha256']
        manifest=json.loads(manifest_file.read_text())
        state=json.loads(Path(hf_hub_download(repo,name+'/state.json',revision=revision,token=token)).read_text())
        if sha(ROOT/'recipe.json') != state['recipe_sha256']:
            raise ValueError('Local recipe differs from the HF checkpoint; refusing resume')
        # Recover the exact original schedule rather than regenerating it.
        assets=['recipe.json','model.json','data-plan.json','schedule.npy','source-update-offsets.npy']
        available=api.list_repo_files(repo,revision=revision)
        assets += [f.removeprefix('training/') for f in available if f.startswith('training/tokenizer/')]
        for asset in assets:
            target=ROOT/asset
            downloaded=Path(hf_hub_download(repo,'training/'+asset,revision=revision,token=token))
            if target.exists() and sha(target)!=sha(downloaded):
                raise ValueError(f'Local asset differs from HF: {asset}')
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists(): shutil.copyfile(downloaded,target)
        plan=json.loads((ROOT/'data-plan.json').read_text())
        for asset,expected in [('model.json',recipe['model_sha256']),('data-plan.json',state['data_plan_sha256']),
                               ('schedule.npy',plan['schedule_sha256']),('source-update-offsets.npy',plan['offsets_sha256'])]:
            if sha(ROOT/asset)!=expected: raise ValueError(f'Invalid resume asset: {asset}')
        snapshot_download(repo,revision=revision,token=token,allow_patterns=[name+'/*'],local_dir=staging)
        path=staging/name
        assert sha(path/'complete.json')==latest['manifest_sha256']
        for file,expected in manifest['files'].items():
            p=Path(file); assert not p.is_absolute() and '..' not in p.parts
            assert sha(path/file)==expected['sha256'] and (path/file).stat().st_size==expected['bytes'],file
        output=ROOT/'run'/name; output.parent.mkdir(exist_ok=True)
        if output.exists(): raise FileExistsError(f'Refusing to overwrite {output}; verify the existing snapshot first')
        path.rename(output)
        receipt=ROOT/'runtime'/(name+'.uploaded.json')
        receipt.write_text(json.dumps(dict(revision=revision,verified=True,step=manifest['step'])))
        print('Verified resume checkpoint:',name,'step:',manifest['step'],'revision:',revision)
        break
    except HfHubHTTPError as error:
        if error.response.status_code not in (404,409) or attempt==2: raise
        time.sleep(2)
