# fix-my-elo
FixMyElo - ML Projects 2025-2026

## Team2 Checkpoint Utility

Team2 includes a helper script at `Team2/save_checkpoint.py` for creating and validating `.pth` files for `SLPolicyValueNetwork`.

Use the configured Python interpreter (example shown below):

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python Team2/save_checkpoint.py <command> [options]
```

### Create a new checkpoint

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python Team2/save_checkpoint.py create --out Team2/model_files/sl_policy_value_bootstrap.pth
```

### Validate one checkpoint

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python Team2/save_checkpoint.py validate --path Team2/model_files/sl_policy_value_bootstrap.pth
```

### Validate all checkpoints under Team2

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python Team2/save_checkpoint.py validate-all --root Team2
```

### Inspect checkpoint metadata and tensor stats

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python Team2/save_checkpoint.py inspect --path Team2/model_files/sl_policy_value_bootstrap.pth
```

### Notes

- A created checkpoint is architecture-compatible but randomly initialized (not trained).
- For website inference, point `TEAM2_MODEL_PATH` in `fme-app/.env.local` to your chosen compatible checkpoint.
