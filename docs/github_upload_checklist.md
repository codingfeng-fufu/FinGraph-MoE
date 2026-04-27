# GitHub Upload Checklist

## Current Situation

`/home/u2023312337/MoE_GNN` is inside a larger parent git repository:

```text
/home/u2023312337
```

Do not upload from the parent repository. Create a dedicated git repository inside `MoE_GNN` or copy this directory to a clean location before publishing.

## Keep In Git

- `README.md`
- `pyproject.toml`
- `.gitignore`
- `.env.example`
- `apps/demo/`
- `src/`
- `scripts/`
- `tests/`
- `docs/*.md`
- `data/demo/`

## Ignore Or Keep Local Only

- `.env`
- `.claude/`
- `.playwright-mcp/`
- `.pytest_cache/`
- `__pycache__/`
- `data/realdata_inputs/`
- `data/moe_router_realdata_v1/*.faiss`
- `data/moe_router_realdata_v1/*.joblib`
- `runs/`
- `archive/`
- `tmp/`
- `catboost_info/`
- `*.pt`
- `*.joblib`
- `*.faiss`
- generated `.pptx`
- generated screenshots
- local reference PDFs
- legacy synthetic GNN modules
- old Predict v2/v3/v4/v7/v8/v9/v10 experiment branches
- old MoE pipeline scripts
- presentation-generation JavaScript files

## Why These Files Are Ignored

- `data/realdata_inputs/` contains large/private benchmark data.
- `runs/` contains generated checkpoints and benchmark outputs.
- `archive/` contains old experiment artifacts.
- `tmp/` contains rendered slides, node modules, and temporary screenshots.
- `.env` contains API keys.
- PDF references and generated presentations are not required for a source-code release.
- Legacy GNN and prediction experiment branches make the repository harder to explain. They stay local but are excluded from the GitHub showcase.

## Recommended Publish Flow

Run these commands from `/home/u2023312337/MoE_GNN`:

```bash
git init
git status --short
git add .
git status --short
git commit -m "Initial FinGraph-MoE prototype"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

Before committing, check that these paths do not appear in `git status --short`:

```text
.env
data/realdata_inputs/
runs/
archive/
tmp/
catboost_info/
```

If any of them appears, stop and update `.gitignore` before committing.

## Optional Cleanup After Confirmation

The current cleanup is non-destructive: local generated files are ignored, not deleted.

If you want a physically clean directory later, remove or move these local-only paths after backing up anything important:

```text
archive/
tmp/
runs/
catboost_info/
.pytest_cache/
.playwright-mcp/
```
