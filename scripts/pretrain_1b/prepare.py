"""Adapt the original 70B schedule to 100B, preserving source weights and 16K packing."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download, HfApi
from .data import ROOT, sha

REPO = 'AETHORIA-AI/TR-HASH-Pretraining-125B-Agentic-32K'
REV = 'fc738b3a10c5c093e3b34b48bcf1cb7066184706'
TOKENIZER = 'AETHORIA-AI/TR-HASH-Tokenizer-32K-Agentic'
TOK_REV = '2fcbc2c5359ded0244ca14531f1b3806eebac55e'
TOK_SHA = 'd2053a9e99c484f7c4fce57a919fff644b0bd13348cab518230f32869c7609f9'


def build_schedule(mixture, manifests, *, target=100_000_000_000, unit=131072, context=16384):
    updates = target // unit  # Do not exceed the requested budget.
    weights = np.array([s['weight'] for s in mixture['sources']], dtype=np.float64)
    if abs(weights.sum()-1) > 1e-8:
        raise ValueError('Invalid corpus weights')
    counts = np.floor(weights*updates).astype(np.int64)
    for i in np.argsort(-(weights*updates-counts), kind='stable')[:updates-int(counts.sum())]:
        counts[i] += 1
    schedule = np.repeat(np.arange(len(counts), dtype=np.uint8), counts)
    np.random.default_rng(42).shuffle(schedule)
    seen = np.zeros(len(counts), dtype=np.int64)
    offsets = np.empty(updates, dtype=np.int64)
    for step, source in enumerate(schedule):
        offsets[step] = seen[source]
        seen[source] += 1
    sources = []
    for i, (source, manifest) in enumerate(zip(mixture['sources'], manifests)):
        if manifest['dtype'] != 'uint16':
            raise ValueError('Expected uint16 corpus')
        for shard in manifest['shards']:
            if shard['tokens'] != shard['rows']*2048+1 or shard['bytes'] != shard['tokens']*2:
                raise ValueError('Unexpected packed shard layout')
        # Same reader as the original: never bridge independent shard boundaries.
        available = sum((s['tokens']-1)//context*context for s in manifest['shards'])
        if unit+int(counts[i])*unit > available:
            raise ValueError(f"Insufficient non-replayed data for {source['name']}")
        sources.append(dict(name=source['name'], weight=float(weights[i]), updates=int(counts[i]),
                            train_start_token=unit, manifest=manifest))
    return schedule, offsets, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--create-repo', action='store_true', help='Create the private personal model repository')
    args = parser.parse_args()
    if (ROOT/'recipe.json').exists():
        raise FileExistsError(f'{ROOT}/recipe.json exists; reuse it, never regenerate a running recipe')
    ROOT.mkdir(parents=True, exist_ok=True)
    def fetch(name):
        return json.loads(Path(hf_hub_download(REPO, name, repo_type='dataset', revision=REV)).read_text())
    mixture = fetch('mixture_manifest.json')
    if mixture['tokenizer_sha256'] != TOK_SHA:
        raise ValueError('Tokenizer/corpus mismatch')
    manifests = [fetch(s['manifest']) for s in mixture['sources']]
    remote = set(HfApi().list_repo_files(REPO, repo_type='dataset', revision=REV))
    for source, manifest in zip(mixture['sources'], manifests):
        for shard in manifest['shards']:
            if f"corpora/{source['name']}/{shard['file']}" not in remote:
                raise ValueError('Remote corpus is incomplete')
    schedule, offsets, sources = build_schedule(mixture, manifests)
    np.save(ROOT/'schedule.npy', schedule)
    np.save(ROOT/'source-update-offsets.npy', offsets)
    shutil.copyfile(Path(__file__).with_name('model.json'), ROOT/'model.json')
    snapshot_download(TOKENIZER, revision=TOK_REV, local_dir=ROOT/'tokenizer',
                      allow_patterns=['tokenizer.json','tokenizer_config.json','chat_template.jinja','agentic_tokenizer_manifest.json'])
    if sha(ROOT/'tokenizer/tokenizer.json') != TOK_SHA:
        raise ValueError('Tokenizer checksum mismatch')
    plan = dict(dataset=REPO, revision=REV, tokenizer_contract=dict(repo_id=TOKENIZER, revision=TOK_REV, tokenizer_sha256=TOK_SHA),
                sources=sources, updates=len(schedule), tokens_per_update=131072,
                target_tokens_requested=100_000_000_000, scheduled_tokens=len(schedule)*131072,
                intentional_replay_tokens=0, global_document_deduplication=False,
                schedule_sha256=sha(ROOT/'schedule.npy'), offsets_sha256=sha(ROOT/'source-update-offsets.npy'),
                heldout_tokens_per_source=131072)
    (ROOT/'data-plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    recipe = dict(model_repo='Pacific-i64/TR-HASH-1B-100B', stage='pretraining', world_size=2,
                  microbatch=2, gradient_accumulation=2, context=16384, seed=42,
                  max_updates=len(schedule), lr=3e-4, min_lr_ratio=0.1, warmup_updates=2000,
                  weight_decay=0.1, betas=[0.9,0.95], grad_clip=1.0,
                  checkpoint_tokens=5_000_000_000, eval_every=1000, cache_gib=64, keep_checkpoints=3,
                  model_sha256=sha(ROOT/'model.json'), data_plan_sha256=sha(ROOT/'data-plan.json'),
                  precision='FP32 replicated weights and AdamW state; BF16 autocast',
                  distributed_mode='ddp', attention='full causal SDPA Flash Attention')
    (ROOT/'recipe.json').write_text(json.dumps(recipe, indent=2)+'\n')
    if args.create_repo:
        api = HfApi()
        if api.whoami()['name'] != 'Pacific-i64':
            raise ValueError('Expected personal HF account Pacific-i64')
        api.create_repo(recipe['model_repo'], private=True, exist_ok=True)
    print(json.dumps(dict(updates=len(schedule), tokens=plan['scheduled_tokens'], sources=len(sources)), indent=2))


if __name__ == '__main__':
    main()
