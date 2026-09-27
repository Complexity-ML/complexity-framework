"""Use the framework uploader and its upload-before-local-pruning policy."""
import argparse
import json
import time
from pathlib import Path
from huggingface_hub import HfApi, get_token, snapshot_download
from filelock import FileLock
from scripts.sync_checkpoints_to_hf import sync_once
from .data import ROOT, sha


def restore(api, repo):
    revision = api.model_info(repo).sha
    files = api.list_repo_files(repo, revision=revision)
    names = {f.split('/')[0] for f in files if f.endswith('/complete.json')
             and f.split('/')[0].rsplit('_', 1)[-1].isdigit()}
    if not names: raise FileNotFoundError('No complete remote checkpoint')
    name = max(names, key=lambda n: int(n.rsplit('_', 1)[1]))
    staging = ROOT/'restore-staging'
    snapshot_download(repo, revision=revision, allow_patterns=[name+'/*'], local_dir=staging)
    path = staging/name
    marker = json.loads((path/'complete.json').read_text())
    if marker['contract']['recipe_sha256'] != sha(ROOT/'recipe.json'):
        raise ValueError('Remote checkpoint belongs to a different recipe')
    for filename, info in marker['files'].items():
        if Path(filename).name != filename or sha(path/filename) != info['sha256']:
            raise ValueError(f'Checkpoint checksum failure: {filename}')
    destination = ROOT/'run'/name
    destination.parent.mkdir(exist_ok=True)
    if destination.exists(): raise FileExistsError(destination)
    path.rename(destination)
    print(f'Restored {destination} from pinned revision {revision}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--restore', action='store_true', help='Download and verify the latest complete checkpoint')
    args = parser.parse_args()
    recipe = json.loads((ROOT/'recipe.json').read_text())
    token = get_token()
    if not token:
        raise RuntimeError('Run hf auth login on this server first')
    api = HfApi(token=token)
    if api.whoami()['name'] != 'Pacific-i64':
        raise ValueError('Expected the personal account Pacific-i64')
    if args.restore:
        restore(api, recipe['model_repo'])
        return
    api.create_repo(recipe['model_repo'], private=True, exist_ok=True)
    # Reproduction assets and code, never runtime credentials, logs or token shards.
    for filename in ['recipe.json', 'model.json', 'data-plan.json', 'schedule.npy', 'source-update-offsets.npy']:
        api.upload_file(repo_id=recipe['model_repo'], path_or_fileobj=ROOT/filename,
                        path_in_repo='training/'+filename)
    api.upload_folder(repo_id=recipe['model_repo'], folder_path=Path(__file__).parent,
                      path_in_repo='training/scripts', allow_patterns=['*.py', '*.sh', '*.json', '*.md'])
    api.upload_folder(repo_id=recipe['model_repo'], folder_path=ROOT/'tokenizer',
                      path_in_repo='tokenizer', allow_patterns=['*.json', '*.jinja'])
    (ROOT/'run').mkdir(exist_ok=True)
    with FileLock(str(ROOT/'sync.lock'), timeout=0):
        while True:
            try:
                sync_once(ROOT/'run', recipe['model_repo'], token, private=True, keep_local=3)
            except Exception:
                if args.once: raise
                import logging
                logging.exception('Upload failed; complete local checkpoints retained for retry')
            if args.once: return
            time.sleep(30)


if __name__ == '__main__': main()
