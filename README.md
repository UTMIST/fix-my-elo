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

## Deploying Without Committing `.pth`

For deployment, keep the model file out of git and host it on a free public file service such as Hugging Face Hub or a GitHub Release asset.

The Docker image will download the checkpoint at startup when `TEAM2_MODEL_URL` is set.

### Required environment variables

Set these in Render (or your host's env settings):

- `TEAM2_PYTHON=/usr/bin/python3`
- `TEAM2_MODEL_URL=<public direct download URL to your .pth file>`
- `TEAM2_MODEL_PATH=/app/Team2/model_files/model.pth`

### How it works

1. The container starts.
2. `docker-entrypoint.sh` checks whether `TEAM2_MODEL_URL` is set.
3. If the checkpoint is missing at `TEAM2_MODEL_PATH`, it downloads the file.
4. The Next.js API route calls Team2 using that local file.

### Good free options for the model file

- Hugging Face Hub public file
- GitHub Release asset

Use a direct download URL, not a webpage URL.
