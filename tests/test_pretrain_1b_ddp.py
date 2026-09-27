import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from complexity.config import ModelConfig
from complexity.models import ComplexityModel
from complexity.utils.checkpointing import CheckpointManager, TrainingState, peek_latest_checkpoint_step
from scripts.pretrain_1b.data import Corpus
from scripts.pretrain_1b.prepare import build_schedule
from scripts.pretrain_1b.train import OriginalCorpusDataset
from scripts.sync_checkpoints_to_hf import is_complete_checkpoint


def test_original_model_count_and_architecture():
    path = Path(__file__).parents[1]/'scripts/pretrain_1b/model.json'
    config = ModelConfig(**json.loads(path.read_text()))
    with torch.device('meta'):
        model = ComplexityModel(config)
    assert config.num_experts == 8 and config.top_k == 2
    assert config.max_position_embeddings == 16384
    assert sum(p.numel() for p in model.parameters()) == 1_011_823_104


def test_schedule_is_bounded_non_replayed_and_deterministic():
    mixture = {'sources': [{'name': 'a', 'weight': .6}, {'name': 'b', 'weight': .4}]}
    manifest = {'dtype': 'uint16', 'shards': [dict(rows=40_000_000, tokens=81_920_000_001, bytes=163_840_000_002)]}
    first = build_schedule(mixture, [manifest, manifest])
    second = build_schedule(mixture, [manifest, manifest])
    schedule, offsets, sources = first
    assert len(schedule)*131072 == 99_999_940_608
    assert np.array_equal(schedule, second[0])
    assert np.array_equal(offsets, second[1])
    for sid, source in enumerate(sources):
        assert np.array_equal(offsets[schedule == sid], np.arange(source['updates']))
    assert abs(sources[0]['updates']/len(schedule)-.6) < 1/len(schedule)
    boundaries = [s for s in range(1, len(schedule))
                  if s*131072//5_000_000_000 > (s-1)*131072//5_000_000_000]
    assert len(boundaries) == 19  # Plus the final checkpoint, just below 100B.
    assert boundaries[0] == 38147


def test_rank_and_accumulation_offsets_and_resume():
    corpus = Corpus.__new__(Corpus)
    corpus.recipe = dict(world_size=2, microbatch=2, gradient_accumulation=2)
    corpus.length = 4
    corpus.plan = dict(updates=4, tokens_per_update=32, sources=[dict(train_start_token=32)])
    corpus.schedule = np.zeros(4, dtype=np.uint8)
    corpus.offsets = np.arange(4)
    corpus.sequence = lambda sid, offset: np.arange(offset, offset+5)
    all_inputs = []
    for rank in range(2):
        for micro in range(2):
            all_inputs.extend(corpus.batch(0, rank, microstep=micro)[:, :-1].reshape(-1))
    assert sorted(all_inputs) == list(range(32, 64))
    for rank in range(2):
        uninterrupted = list(OriginalCorpusDataset(corpus, rank, 0))
        resumed = list(OriginalCorpusDataset(corpus, rank, 2))
        assert len(resumed) == 8
        for actual, expected in zip(resumed, uninterrupted[8:]):
            assert torch.equal(actual['input_ids'], expected['input_ids'])
            assert torch.equal(actual['labels'], expected['labels'])


def setup_manager(path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=.9)
    manager = CheckpointManager(str(path), model, optimizer, scheduler)
    manager.resume_contract = {'recipe': 'test'}
    return model, optimizer, scheduler, manager


def update(model, optimizer, scheduler):
    x = torch.randn(5, 3)
    model(x).square().mean().backward()
    optimizer.step(); scheduler.step(); optimizer.zero_grad()


def test_exact_resume_matches_next_optimizer_update(tmp_path):
    torch.manual_seed(7)
    model, optimizer, scheduler, manager = setup_manager(tmp_path)
    update(model, optimizer, scheduler)
    checkpoint = Path(manager.save(1, TrainingState(step=1, total_tokens=32)))
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_scheduler = copy.deepcopy(scheduler.state_dict())
    update(model, optimizer, scheduler)
    expected = copy.deepcopy(model.state_dict())
    state = manager.load(checkpoint)
    assert state.step == 1 and state.total_tokens == 32
    assert scheduler.state_dict() == saved_scheduler
    for key, value in saved_optimizer['state'].items():
        for name, tensor in value.items():
            assert torch.equal(optimizer.state_dict()['state'][key][name], tensor)
    update(model, optimizer, scheduler)
    for name, value in expected.items():
        assert torch.equal(model.state_dict()[name], value)


def test_partial_corrupt_and_changed_contract_are_rejected(tmp_path):
    model, optimizer, scheduler, manager = setup_manager(tmp_path)
    update(model, optimizer, scheduler)
    checkpoint = Path(manager.save(1, TrainingState(step=1)))
    pending = tmp_path/'.pending-step_999.writing'
    pending.mkdir(); (pending/'checkpoint.pt').write_bytes(b'partial')
    assert not is_complete_checkpoint(pending)
    assert peek_latest_checkpoint_step(tmp_path) == 1
    manager.resume_contract = {'recipe': 'changed'}
    with pytest.raises(ValueError, match='contract'):
        manager.load(checkpoint)
    manager.resume_contract = {'recipe': 'test'}
    (checkpoint/'optimizer_rank0.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        manager.load(checkpoint)


def test_retention_counts_across_tags(tmp_path):
    model, optimizer, scheduler, manager = setup_manager(tmp_path)
    for step, tag in enumerate(['step', 'interrupted', 'step', 'final'], 1):
        update(model, optimizer, scheduler)
        manager.save(step, TrainingState(step=step), tag=tag)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['final_4', 'interrupted_2', 'step_3']


def _ddp_checkpoint_worker(rank, root, rendezvous):
    import datetime
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank,
                            world_size=2, timeout=datetime.timedelta(seconds=40))
    try:
        torch.manual_seed(100+rank)
        model = DistributedDataParallel(torch.nn.Linear(3, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=.9)
        manager = CheckpointManager(root, model, optimizer, scheduler)
        manager.resume_contract = {'recipe': 'two-ranks'}
        update(model, optimizer, scheduler)
        path = Path(manager.save(1, TrainingState(step=1)))
        update(model, optimizer, scheduler)
        expected = copy.deepcopy(model.state_dict())
        manager.load(path)
        update(model, optimizer, scheduler)
        assert all(torch.equal(value, expected[name]) for name, value in model.state_dict().items())
        # A missing RNG file must fail on BOTH ranks, not leave one waiting in load.
        if rank == 0: (path/'rng_rank1.pt').unlink()
        dist.barrier()
        try:
            manager.load(path)
        except RuntimeError as exc:
            assert 'Checkpoint failed on a rank' in str(exc)
        else: raise AssertionError('Missing rank state was accepted')
    finally:
        dist.destroy_process_group()


def test_two_process_ddp_resume(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_ddp_checkpoint_worker, args=(str(tmp_path/'ckpt'), str(tmp_path/'rendezvous')),
             nprocs=2, join=True)
