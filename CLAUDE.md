# AuraCast

**An automated, AI-driven Instagram curation system.**

Curates a stream of candidate images (pulled from Google Drive in
production, local directories in development) by scoring them with on-prem
vision models, then surfacing the top picks for human review and
publication.

## Architectural principles

- **Modular boundaries.** Each subsystem has one responsibility and a typed
  interface (Pydantic) at the boundary. No subsystem reaches across.
- **Local-first dev loop.** Every code path is exercised against local image
  directories before touching any external API. The Google integration is a
  drop-in adapter behind the same `IngestSource` interface as the local one.
- **GPU is a single chokepoint.** Vision models live in `engine/` and nowhere
  else. Everything upstream produces `ImageRecord`s; everything downstream
  consumes `Embedding`s + `AestheticScore`s. The engine module is the only
  place that imports `torch`.
- **GPU-aware by default.** FP16 mixed precision when CUDA is available;
  falls back to FP32 on Apple MPS or CPU (slow but functional for tests).
- **Production-grade, not research-grade.** This is a system that should be
  reliable for an end user, not a sprint of experiments. Tests, type hints,
  structured logging, retries on the network boundary.

## Module layout

```
auracast/
├── auth/        Google OAuth2 + API connection. Token cache, refresh,
│                scope management. Returns authenticated httpx clients.
├── ingest/      Async image acquisition. Backends behind one interface:
│                LocalDirectoryIngest (dev) and GoogleDriveIngest (prod).
│                Produces ImageRecord rows; does NOT score or embed.
├── engine/      Vision model orchestration. CLIP / LAION-Aesthetic /
│                Qwen2-VL. FP16 on CUDA, FP32 on MPS/CPU. The only module
│                that imports torch / transformers.
├── app/         Streamlit interface. Multi-project curation UI: project
│                picker, custom-prompt scoring, approve/reject, finalize.
├── schema/      Pydantic models. Single source of truth for the on-disk
│                manifest format and the in-memory record types.
└── scripts/     Entry-point runners (auth_setup, pipeline, mock demo).
```

## Data flow

```
  [LocalDir | Google Drive]                     (ingest/)
            │
            ▼
      ImageRecord                               (schema/)
            │
            ▼
  CLIP / LAION / Qwen2-VL scoring              (engine/)
            │
            ▼
   AestheticScore + Embedding                   (schema/)
            │
            ▼
   Streamlit preview / approve                  (app/)
            │
            ▼
   Drive Trash (Finalize) — published manually
```

## Technology choices

- **Python 3.10+**. Type hints + Pydantic v2 throughout.
- **Pydantic v2** for all data records (`auracast/schema/models.py`). No
  ad-hoc dicts across module boundaries.
- **PyTorch + Transformers** for model loading. Default scorer:
  `openai/clip-vit-base-patch32` (lightweight, ~150 MB). Premium scorers:
  LAION Aesthetic Predictor (CLIP-L/14 + trained MLP head), Qwen2-VL-7B.
- **httpx (async)** for the network boundary. `aiofiles` for disk I/O on
  the ingest path.
- **Pillow** for decode + light pre-processing only. Heavy transforms
  belong in the model's own processor.
- **Streamlit** for the UI. Single-file app in `auracast/app/streamlit_app.py`.

## Hardware target

Local dev box (Mac M-series with MPS, or any CPU): all CLIP/LAION scoring
runs fine, ~1s/image. Qwen2-VL is slow on MPS (~5s/image) but functional
for small batches.

CUDA host (any NVIDIA card, ideally ≥24 GB VRAM for Qwen2-VL): FP16
mixed precision selected automatically by `pick_device_and_dtype()`.
Batch size scales with VRAM headroom — start at 32 for CLIP-B/32.

## Conventions

- All public functions are typed.
- All cross-module data uses Pydantic models from `schema/`.
- All disk paths use `pathlib.Path`, never raw strings.
- All times are `datetime.datetime` with explicit `tz=timezone.utc`.
- Logging via `logging` with a per-module `logger = logging.getLogger(__name__)`.
- Tests in `tests/test_<module>.py`, CPU-runnable.

## Phase status (current)

- **Phase 0**: scaffold, Pydantic schema, mock pipeline. ✅
- **Phase 1**: local end-to-end with persistence + dedupe. ✅
- **Phase 1.5**: Qwen2-VL captioning. ✅
- **Phase 2**: Google OAuth + Drive ingest. ✅
- **Phase 2b**: multi-project UI + folder picker + Finalize. ✅
- **Phase 2c**: custom prompts, scorer model dropdown, normalization. ✅
- **Phase 3 — Instagram publishing**: not started.

## Out of scope (for now)

- Actually publishing to Instagram (no Graph API integration yet).
- Multi-user / multi-tenant.
- Persistent database (Postgres etc.) — manifest is a JSONL file on disk;
  migration when it's actually a bottleneck.
