# Optimizer-update boundaries

`Trainer` synchronizes gradients at the end of each accumulation group.
A finite epoch's final group also produces an update, even when it contains
fewer microbatches than `gradient_accumulation_steps`. Its gradients and logged
loss are averaged over the microbatches actually processed. Each microbatch
has equal weight, as in complete groups. The scheduler and callbacks advance
once per optimizer update; gradients never carry into the next epoch.

This works with unsized iterable loaders. The trainer keeps at most one batch
of lookahead within an update to identify the final microbatch before DDP's
backward synchronization. It does not consume a batch from the next update
before checkpoint callbacks run. An empty or exhausted non-reiterable loader
raises an error instead of looping without progress. DDP ranks must have
matching microbatch counts and accumulation boundaries.

Changing the treatment of an incomplete group changes the number of updates
per epoch: it is now `ceil(number_of_batches / accumulation_steps)`. Pipelines
that require an exact token budget should supply complete groups, as the
prepared 1B corpus does; a short final group consumes fewer tokens than a full
update. No existing recipe or corpus schedule is changed by this correction.

An unchanged WSD schedule can resume during its decay phase. Changing its
bounds after decay has begun still raises an error, because that requires
reconstructing the learning-rate trajectory. Both the milestones and the
cosine decay length are compared before deciding whether resynchronization
is necessary.

## Compatibility with existing runs

These edits take effect only when the updated code is deployed and loaded.
Do not replace code underneath an active run: validate the change in a
separate process against a copied checkpoint before restarting that run.
Neither model tensors nor optimizer/checkpoint formats are changed here.

The strict `scripts/pretrain_1b` DDP recipe fingerprints Python source code.
Its existing checkpoints will reject resume under a changed source hash;
the update does not bypass that guard or migrate those contracts. The FSDP
recipe checks its unchanged recipe hash instead. Passing the CPU/Gloo
regressions does not qualify a GPU checkpoint restart.

The `Training reliability` workflow checks CPU training and two-process Gloo
coordination on Python 3.11 and 3.12. This includes partial accumulation,
checkpoint cursor boundaries, WSD continuation and upload backpressure.
It does not validate GPU kernels or a four-GPU FSDP training run.
