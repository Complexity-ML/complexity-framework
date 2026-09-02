# Portable Supervisor jobs

Complexity Framework generates Supervisor configuration from validated Python
objects. Long-running runs should use this interface instead of committing
instance-specific `.conf` files or copying ad-hoc waiter scripts to a server.

## Manifest format

A version-1 TOML manifest contains one or more `[[jobs]]` tables. Commands are
argv arrays and are never passed through a shell by the framework. Relative
`directory` and `log_path` values are resolved from `--root`.

Built-in variables are:

- `{root}`: the runtime project root;
- `{python}`: the interpreter running the Complexity CLI;
- `{python_dir}`: that interpreter's binary directory.

Additional defaults belong in `[variables]`. Values that must be supplied by
the operator belong in `required_variables` and are passed with `--set
NAME=VALUE`. Secret values are rejected from Supervisor environments. Pass a
protected credential file path to the program instead.

Validate and inspect a manifest without changing Supervisor:

```bash
complexity jobs render-file configs/jobs/tr_hash_agentic_100m_pipeline.toml \
  --root /workspace/complexity-framework \
  --set artifact_root=/workspace/artifacts \
  --set hf_home=/workspace/.hf_home \
  --set tokenizer=/workspace/tokenizer-agentic-32k \
  --set hf_token_file=/workspace/.hf_home/token
```

Install the whole job group with one Supervisor update:

```bash
sudo /venv/main/bin/complexity jobs submit-file \
  configs/jobs/tr_hash_agentic_100m_pipeline.toml \
  --root /workspace/complexity-framework \
  --set artifact_root=/workspace/artifacts \
  --set hf_home=/workspace/.hf_home \
  --set tokenizer=/workspace/tokenizer-agentic-32k \
  --set hf_token_file=/workspace/.hf_home/token
```

The included Agentic 100M pipeline defines pretraining, checkpoint sync,
refinement, and refinement sync. The refinement process uses `jobs run-after`:
it starts only after Supervisor reports a clean pretraining exit and both final
model artifacts are non-empty. If pretraining exits without them or enters
`FATAL`, the refinement fails visibly instead of training from an incomplete
checkpoint.

## Operations

```bash
complexity jobs list
complexity jobs status tr_hash_100m_pretraining
complexity jobs logs --follow tr_hash_100m_pretraining
complexity jobs restart tr_hash_100m_pretraining
```

Generated files are private (`0600`), logs are written to real files with
bounded rotation, and commands use process-group termination so distributed
workers do not survive a stop or restart.
