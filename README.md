# AuraCast

Automated, AI-driven Instagram curation system. Curates candidate images by scoring them with on-prem vision models (CLIP/SigLIP + Qwen2-VL) on an A100 GPU, surfacing the top picks for human review through a Streamlit interface.

📐 **Read [CLAUDE.md](./CLAUDE.md) first** — module layout, data flow, conventions.

## Module layout

| Module | Responsibility |
|---|---|
| `auracast/auth/`   | Google OAuth2 + API connection management |
| `auracast/ingest/` | Async image acquisition — local dirs (dev) + Google APIs (prod) |
| `auracast/engine/` | Vision model orchestration — CLIP/SigLIP + Qwen2-VL, FP16 on A100 |
| `auracast/app/`    | Streamlit interface for preview / approve |
| `auracast/schema/` | Pydantic data models (single source of truth) |

## Quick start (dev box, CPU/MPS)

```bash
pip install -e ".[dev]"
pytest                                              # 12+ CPU tests
python -m auracast.scripts.mock_pipeline_demo data/mock_images
streamlit run auracast/app/streamlit_app.py
```

## Quick start (cluster, A100)

```bash
sbatch scripts/slurm/verify_a100.sh                 # CLIP load + embed test on A100
# then check the log for "[verify] OK"
```

## Status

Phase 0 — scaffold. Pydantic schema + mock pipeline + CLIP A100 verification.
See [CLAUDE.md §Phase plan](./CLAUDE.md#phase-plan).
