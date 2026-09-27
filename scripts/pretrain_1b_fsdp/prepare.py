import hashlib,json,math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests

ROOT=Path(__file__).parent
REPO='AETHORIA-AI/TR-HASH-Pretraining-125B-Agentic-32K'
REV='fc738b3a10c5c093e3b34b48bcf1cb7066184706'
BASE=f'https://huggingface.co/datasets/{REPO}/resolve/{REV}'
def fetch(path):
 r=requests.get(BASE+'/'+path,timeout=90);r.raise_for_status();return r.json()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
mixture=fetch('mixture_manifest.json');cfg=fetch('_metadata/config.json')
source_meta=list(ThreadPoolExecutor(4).map(lambda s:fetch(s['manifest']),mixture['sources']))
updates=70_000_000_000//131072
weights=np.array([s['weight'] for s in mixture['sources']],dtype=np.float64)
assert abs(weights.sum()-1)<1e-8
counts=np.floor(weights*updates).astype(np.int64)
for i in np.argsort(-(weights*updates-counts),kind='stable')[:updates-int(counts.sum())]:counts[i]+=1
schedule=np.repeat(np.arange(len(counts),dtype=np.uint8),counts)
np.random.default_rng(42).shuffle(schedule)
seen=np.zeros(len(counts),dtype=np.int64); offsets=np.empty(updates,dtype=np.int64)
for step,source in enumerate(schedule):offsets[step]=seen[source];seen[source]+=1
np.save(ROOT/'schedule.npy',schedule);np.save(ROOT/'source-update-offsets.npy',offsets)
sources=[]
for i,(source,manifest) in enumerate(zip(mixture['sources'],source_meta)):
 assert manifest['dtype']=='uint16'
 for shard in manifest['shards']:assert shard['tokens']==shard['rows']*2048+1 and shard['bytes']==shard['tokens']*2
 available=sum(s['tokens']-1 for s in manifest['shards'])
 assert 131072+int(counts[i])*131072+1 <= available
 sources.append({'name':source['name'],'weight':float(weights[i]),'updates':int(counts[i]),'train_start_token':131072,'manifest':manifest})
contract=cfg['tokenizer_contract'];tokdir=ROOT/'tokenizer';tokdir.mkdir(exist_ok=True)
tokbase=f"https://huggingface.co/{contract['repo_id']}/resolve/{contract['revision']}"
for name,key in [('tokenizer.json','tokenizer_sha256'),('agentic_tokenizer_manifest.json','manifest_sha256')]:
 r=requests.get(tokbase+'/'+name,timeout=90);r.raise_for_status();assert hashlib.sha256(r.content).hexdigest()==contract[key];(tokdir/name).write_bytes(r.content)
for name in ['tokenizer_config.json','chat_template.jinja','chat_template.json','special_tokens_map.json']:
 r=requests.get(tokbase+'/'+name,timeout=60)
 if r.status_code==200:(tokdir/name).write_bytes(r.content)
plan={'dataset':REPO,'revision':REV,'tokenizer_contract':contract,'sources':sources,'updates':updates,'tokens_per_update':131072,'target_tokens_requested':70_000_000_000,'scheduled_tokens':updates*131072,'intentional_replay_tokens':0,'global_document_deduplication':False,'schedule_sha256':sha(ROOT/'schedule.npy'),'offsets_sha256':sha(ROOT/'source-update-offsets.npy'),'heldout_tokens_per_source':131072}
(ROOT/'data-plan.json').write_text(json.dumps(plan,indent=2))
recipe={'model_repo':'Pacific-i64/TR-HASH-MoE-1B-70B-Agentic-Pretraining','world_size':4,'microbatch':2,'context':16384,'seed':42,'max_updates':updates,'lr':3e-4,'min_lr_ratio':0.1,'warmup_updates':2000,'weight_decay':0.1,'betas':[0.9,0.95],'grad_clip':1.0,'save_every':1000,'eval_every':1000,'first_checkpoint':10,'cache_gib':64,'keep_checkpoints':3,'model_sha256':sha(ROOT/'model.json'),'data_plan_sha256':sha(ROOT/'data-plan.json'),'precision':'FP32 sharded master weights and AdamW state; BF16 compute','attention':'full causal SDPA Flash Attention','kernels':'TR-HASH fused_cuda and Liger'}
(ROOT/'recipe.json').write_text(json.dumps(recipe,indent=2))
print(json.dumps({'scheduled_tokens':plan['scheduled_tokens'],'updates':updates,'sources':len(sources)}))
