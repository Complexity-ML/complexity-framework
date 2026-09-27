"""Resumable FSDP2 pretraining. Only complete checkpoints may be uploaded."""
import argparse, datetime, gc, json, math, os, random, shutil, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, get_model_state_dict, StateDictOptions
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.nn.attention import sdpa_kernel, SDPBackend
from complexity.config import ModelConfig
from model_impl import TrainingModel, has_liger_fused_linear_ce
from data import Corpus, ROOT, sha
from checkpoint_json import checkpoint_reader


def log(event, **kw):
    if dist.get_rank()==0: print(json.dumps(dict(event=event, **kw)), flush=True)

def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(), python=random.getstate(), numpy=np.random.get_state())

def restore_rng(state):
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state(state['cuda'])
    random.setstate(state['python']); np.random.set_state(state['numpy'])

def verify_checkpoint(path):
    marker=json.loads((path/'complete.json').read_text())
    for name, meta in marker['files'].items():
        file=path/name
        assert file.stat().st_size==meta['bytes'] and sha(file)==meta['sha256'], name
    return marker

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'run')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--verify-resume', action='store_true')
    parser.add_argument('--require-resume', action='store_true')
    args=parser.parse_args()
    if args.require_resume and not any(args.output.glob('step_*/complete.json')):
        raise RuntimeError('No restored checkpoint: refusing to start from zero')
    rank=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=45))
    recipe=json.loads((ROOT/'recipe.json').read_text()); world=dist.get_world_size()
    assert world==recipe['world_size']==4 and has_liger_fused_linear_ce()
    torch.manual_seed(recipe['seed']); torch.cuda.manual_seed_all(recipe['seed'])
    np.random.seed(recipe['seed']); random.seed(recipe['seed'])
    corpus=Corpus(rank); args.output.mkdir(parents=True, exist_ok=True)
    runtime=ROOT/'runtime'; runtime.mkdir(exist_ok=True)
    with torch.device(f'cuda:{rank}'):
        model=TrainingModel(ModelConfig(**json.loads((ROOT/'model.json').read_text())))
    policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for block in model.model.layers: fully_shard(block, mp_policy=policy, reshard_after_forward=True)
    fully_shard(model, mp_policy=policy, reshard_after_forward=True)
    optimizer=torch.optim.AdamW(model.parameters(), lr=recipe['lr'], betas=tuple(recipe['betas']), weight_decay=recipe['weight_decay'], foreach=False)
    options=StateDictOptions(cpu_offload=True)
    fingerprint=sha(ROOT/'recipe.json')

    def load(path):
        # Every rank verifies its RNG; rank zero verifies all saved files before loading.
        if rank==0: verify_checkpoint(path)
        dist.barrier()
        meta=json.loads((path/'state.json').read_text())
        assert meta['recipe_sha256']==fingerprint and meta['world_size']==world
        ms, osd=get_state_dict(model, optimizer, options=options)
        state={'model':ms, 'optimizer':osd}
        dcp.load(state, storage_reader=checkpoint_reader(path/'distributed'))
        set_state_dict(model, optimizer, model_state_dict=state['model'], optim_state_dict=state['optimizer'], options=options)
        del ms, osd, state; gc.collect()
        restore_rng(torch.load(path/f'rng_rank{rank}.pt', map_location='cpu', weights_only=False))
        log('resumed', step=meta['step'], next_update=meta['next_update'], checkpoint=str(path))
        return meta['step']

    def save(step, kind="regular"):
        final=args.output/f'step_{step:07d}'
        if final.exists(): return final
        pending=args.output/f'.incomplete_step_{step:07d}'
        if rank==0:
            if pending.exists(): shutil.rmtree(pending)
            pending.mkdir()
        dist.barrier()
        optimizer.zero_grad(set_to_none=True)
        state_rng=rng_state()
        ms, osd=get_state_dict(model, optimizer, options=options)
        dcp.save({'model':ms,'optimizer':osd}, checkpoint_id=str(pending/'distributed'))
        del ms, osd; gc.collect()
        torch.save(state_rng, pending/f'rng_rank{rank}.pt')
        dist.barrier()
        if rank==0:
            meta={'step':step,'next_update':step,'tokens_seen':step*corpus.plan['tokens_per_update'],
                  'world_size':world,'recipe_sha256':fingerprint,'data_plan_sha256':recipe['data_plan_sha256'],
                  'scheduler':'deterministic cosine by next update; see recipe.json',
                  'source_updates_consumed':np.bincount(corpus.schedule[:step], minlength=len(corpus.plan['sources'])).tolist()}
            (pending/'state.json').write_text(json.dumps(meta, indent=2))
            shutil.copy(ROOT/'model.json',pending/'config.json')
            files={str(p.relative_to(pending)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in pending.rglob('*') if p.is_file()}
            (pending/'complete.json').write_text(json.dumps({'step':step,'kind':kind,'regular':step==recipe['first_checkpoint'] or step%recipe['save_every']==0 or step==recipe['max_updates'], 'files':files},indent=2))
            pending.rename(final)
        dist.barrier(); restore_rng(state_rng)
        log('checkpoint_complete', step=step, path=str(final))
        return final

    checkpoints=sorted(p for p in args.output.glob('step_*') if (p/'complete.json').is_file())
    step=load(checkpoints[-1]) if checkpoints else 0
    limit=min(args.limit or recipe['max_updates'], recipe['max_updates'])
    log('start', next_update=step, target_updates=recipe['max_updates'], target_tokens=corpus.plan['scheduled_tokens'], world_size=world, context=recipe['context'], tokens_per_update=corpus.plan['tokens_per_update'], experts=8, top_k=2)
    model.train()
    writer=None
    if rank==0:
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(runtime/'tensorboard'))
    initial_step=step
    while step<limit:
        if rank==0:
            if step==initial_step or step%32==0: corpus.prefetch(step)
            # Do not fill disk with checkpoints if uploads cannot keep up.
            while len([p for p in args.output.glob('step_*') if (p/'complete.json').exists() and not (runtime/(p.name+'.uploaded.json')).exists()])>=3 and not args.limit:
                log('waiting_for_upload', step=step); time.sleep(30)
        dist.barrier(); started=time.monotonic()
        batch=torch.from_numpy(corpus.batch(step,rank)).to(f'cuda:{rank}')
        x,y=batch[:,:-1].contiguous(),batch[:,1:].contiguous()
        update=step+1
        if update<=recipe['warmup_updates']: scale=update/recipe['warmup_updates']
        else:
            progress=(update-recipe['warmup_updates'])/(recipe['max_updates']-recipe['warmup_updates'])
            scale=recipe['min_lr_ratio']+(1-recipe['min_lr_ratio'])*.5*(1+math.cos(math.pi*progress))
        lr=recipe['lr']*scale
        for group in optimizer.param_groups: group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION): loss=model(x,y); loss.backward()
        norm2=torch.zeros((),device=f'cuda:{rank}')
        for param in model.parameters():
            if param.grad is not None:
                grad=param.grad.to_local() if hasattr(param.grad,'to_local') else param.grad
                norm2+=grad.float().square().sum()
        dist.all_reduce(norm2); norm=norm2.sqrt()
        metrics=torch.stack([loss.detach(),norm]); dist.all_reduce(metrics,op=dist.ReduceOp.SUM)
        if not torch.isfinite(metrics).all(): raise RuntimeError('Nonfinite loss/gradient; stopping without checkpoint overwrite')
        coefficient=(recipe['grad_clip']/(norm+1e-6)).clamp(max=1)
        for param in model.parameters():
            if param.grad is not None:
                (param.grad.to_local() if hasattr(param.grad,'to_local') else param.grad).mul_(coefficient)
        optimizer.step(); torch.cuda.synchronize(); step=update
        seconds=time.monotonic()-started
        seconds_tensor=torch.tensor(seconds,device=f'cuda:{rank}'); dist.all_reduce(seconds_tensor,op=dist.ReduceOp.MAX)
        row=dict(step=step, tokens_seen=step*corpus.plan['tokens_per_update'], loss=float(metrics[0]/world), grad_norm=float(norm), lr=lr, tokens_per_second=corpus.plan['tokens_per_update']/float(seconds_tensor), peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, source=corpus.plan['sources'][int(corpus.schedule[step-1])]['name'])
        log('optimizer_update',**row)
        if rank==0:
            with open(runtime/'metrics.jsonl','a') as f: f.write(json.dumps(row)+'\n')
            writer.add_scalar('train/loss',row['loss'],step); writer.add_scalar('train/tokens_per_second',row['tokens_per_second'],step); writer.add_scalar('train/lr',lr,step)
        request=torch.tensor(int((runtime/'request-checkpoint').exists())+2*int((runtime/'request-stop').exists()),device=f'cuda:{rank}')
        dist.all_reduce(request,op=dist.ReduceOp.MAX)
        should_save=step==recipe['first_checkpoint'] or step%recipe['save_every']==0 or step==limit or int(request)>0
        if should_save:
            path=save(step, kind="interrupt" if int(request)>0 else "regular")
            if args.verify_resume:
                # Test restoring both model and Adam after a deliberately destructive mutation.
                before=[p.detach().to_local().clone() for p in model.parameters()]
                saved_rng=rng_state()
                with torch.no_grad():
                    for p in model.parameters(): p.to_local().zero_()
                optimizer.state.clear()
                assert load(path)==step
                matches=all(torch.equal(old,p.detach().to_local()) for old,p in zip(before, model.parameters()))
                assert matches and len(optimizer.state)>0
                assert all(float(s['step'])==step for s in optimizer.state.values())
                assert torch.equal(saved_rng['torch'],torch.get_rng_state()) and torch.equal(saved_rng['cuda'],torch.cuda.get_rng_state())
                del before; gc.collect()
                log('resume_roundtrip_pass',step=step,model_exact=True,optimizer_steps_restored=True,rng_exact=True)
            if rank==0: (runtime/'request-checkpoint').unlink(missing_ok=True)
        if int(request)>=2: log('graceful_stop',step=step); break
        if step%recipe['eval_every']==0:
            model.eval(); value=0.
            with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                for sid,source in enumerate(corpus.plan['sources']):
                    b=torch.from_numpy(corpus.batch(0,rank,validation_source=sid)).to(f'cuda:{rank}')
                    value+=model(b[:,:-1].contiguous(),b[:,1:].contiguous()).detach()*source['weight']
            dist.all_reduce(value); log('validation',step=step,loss=float(value/world)); model.train()
    if step==recipe['max_updates']:
        optimizer.zero_grad(set_to_none=True)
        full=get_model_state_dict(model, options=StateDictOptions(full_state_dict=True,cpu_offload=True))
        if rank==0:
            from safetensors.torch import save_file
            final=ROOT/'final'; final.mkdir(exist_ok=True)
            weights={k.removeprefix('model.'):v.contiguous().clone() for k,v in full.items()}
            save_file(weights,str(final/'model.safetensors'))
            shutil.copy(ROOT/'model.json',final/'config.json')
            for p in (ROOT/'tokenizer').iterdir(): shutil.copy(p,final/p.name)
            (final/'complete.json').write_text(json.dumps({'step':step,'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in final.iterdir() if p.name!='complete.json'}},indent=2))
        del full; dist.barrier()
    if writer: writer.close()
    log('run_exit',step=step,target_reached=step==recipe['max_updates'])
    dist.destroy_process_group()

if __name__=='__main__': main()
