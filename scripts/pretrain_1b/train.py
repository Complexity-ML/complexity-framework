"""Original 1B/16K architecture and corpus, using the framework's DDP runner."""
import json
import random
import hashlib
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import IterableDataset
from complexity.config import ModelConfig
from complexity.core.losses import fused_linear_causal_lm_loss
from complexity.training.runner import TrainRunner
from complexity.utils.checkpointing import peek_latest_checkpoint_step
from .data import Corpus, ROOT, sha


class OriginalCorpusDataset(IterableDataset):
    """Keep the original random-access 16K packing and update-level source mix."""
    def __init__(self, corpus, rank, start):
        self.corpus, self.rank, self.start = corpus, rank, start

    def __iter__(self):
        for step in range(self.start, self.corpus.plan['updates']):
            for micro in range(self.corpus.recipe['gradient_accumulation']):
                for row in self.corpus.batch(step, self.rank, microstep=micro):
                    yield {'input_ids': torch.from_numpy(row[:-1].copy()),
                           'labels': torch.from_numpy(row[1:].copy())}


class Pretraining1BRunner(TrainRunner):
    def __init__(self):
        self.recipe = json.loads((ROOT/'recipe.json').read_text())
        recipe = self.recipe
        super().__init__(
            make_config=lambda: ModelConfig(**json.loads((ROOT/'model.json').read_text())),
            run_name='TR-HASH-1B-100B', checkpoint_dir=str(ROOT/'run'),
            default_lr=recipe['lr'], default_batch_size=recipe['microbatch'],
            default_seq_len=recipe['context'], default_target_tokens=100_000_000_000,
            default_gradient_accumulation=recipe['gradient_accumulation'],
            default_gradient_checkpointing=True, default_distributed_mode='ddp',
            default_save_steps=recipe['max_updates']+1, label_smoothing=0.0,
        )

    def add_args(self, parser):
        parser.add_argument('--stop-after', type=int, default=None,
                            help='Stop cleanly at this absolute update, keeping the full LR schedule')
        parser.set_defaults(tokenizer=str(ROOT/'tokenizer'), num_workers=0,
                            warmup_steps=self.recipe['warmup_updates'], lr_scheduler='cosine',
                            max_steps=self.recipe['max_updates'], resume='auto', require_cuda=True)

    def run(self):
        random.seed(self.recipe['seed']); np.random.seed(self.recipe['seed'])
        torch.manual_seed(self.recipe['seed'])
        super().run()

    def build_dataset(self, tokenizer, args, rank, world_size):
        recipe = self.recipe
        expected = dict(batch_size=recipe['microbatch'], gradient_accumulation=recipe['gradient_accumulation'],
                        seq_len=recipe['context'], num_workers=0, distributed_mode='ddp',
                        precision='bf16', lr=recipe['lr'], weight_decay=recipe['weight_decay'],
                        warmup_steps=recipe['warmup_updates'], warmup_tokens=None,
                        lr_scheduler='cosine', save_steps=recipe['max_updates']+1,
                        token_packs=1, gradient_checkpointing=True, max_steps=recipe['max_updates'],
                        optimizer='adamw', use_custom_kernels='auto',
                        checkpoint_dir=str(ROOT/'run'))
        for name, value in expected.items():
            if getattr(args, name) != value:
                raise ValueError(f'{name} differs from the prepared recipe; use a fresh run directory')
        if world_size != recipe['world_size'] or args.init_checkpoint:
            raise ValueError('This entry point is fresh pretraining on exactly two DDP ranks')
        if args.top_k not in (None, 2):
            raise ValueError('The original architecture uses top-2 routing')
        corpus = Corpus(rank)
        contract = corpus.plan['tokenizer_contract']
        if len(tokenizer) != 32000 or sha(Path(args.tokenizer)/'tokenizer.json') != contract['tokenizer_sha256']:
            raise ValueError('Tokenizer differs from the prepared corpus')
        if args.resume == 'auto':
            step = peek_latest_checkpoint_step(args.checkpoint_dir) or 0
        elif args.resume:
            step = json.loads((Path(args.resume)/'training_state.json').read_text())['step']
        else:
            step = 0
        return OriginalCorpusDataset(corpus, rank, step)

    def build_compute_loss(self, trainer, model):
        def compute_loss(wrapped, batch):
            x, labels = batch['input_ids'].to(trainer.device), batch['labels'].to(trainer.device)
            hidden = wrapped(x, return_logits=False)['last_hidden_state']
            # The reader already shifts labels, including the last position.
            loss, _ = fused_linear_causal_lm_loss(hidden, model.get_output_embeddings().weight,
                                                labels, use_liger=True, sync_metrics=False)
            return loss
        return compute_loss

    def extra_callbacks(self, trainer, args, is_main):
        unit = self.recipe['world_size']*args.batch_size*args.gradient_accumulation*args.seq_len
        interval = self.recipe['checkpoint_tokens']
        sources = sorted(Path('complexity').rglob('*.py')) + sorted(Path(__file__).parent.glob('*.py'))
        code_hash = hashlib.sha256(''.join(sha(path) for path in sources).encode()).hexdigest()
        trainer.checkpoint_manager.resume_contract = dict(recipe_sha256=sha(ROOT/'recipe.json'),
                                                          code_sha256=code_hash)
        # The existing uploader owns pruning, only after a successful remote upload.
        trainer.checkpoint_manager.defer_rotation = True
        if args.stop_after is not None and args.stop_after <= (peek_latest_checkpoint_step(args.checkpoint_dir) or 0):
            raise ValueError('stop-after must be later than the resumed update')
        def checkpoint_boundary(trainer, step, loss):
            trainer.state.total_tokens = step*unit
            if step < self.recipe['max_updates'] and step*unit//interval > (step-1)*unit//interval:
                trainer._save_checkpoint()
            if args.stop_after is not None and step >= args.stop_after:
                trainer.stop_requested = True
        return [checkpoint_boundary]


if __name__ == '__main__':
    Pretraining1BRunner().run()
