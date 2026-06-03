# AuraCast

**An automated, AI-driven Instagram curation system.**

Curates a stream of candidate images (eventually pulled live from Google services,
locally during development) by scoring them with on-prem vision models, then
surfacing the top picks for human review and publication.

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
- **A100-aware by default.** FP16 mixed precision, batched inference,
  `device_map="auto"` for multi-GPU when present. Falls back to CPU/MPS on
  the dev box for unit tests only.
- **Production-grade, not research-grade.** This is a system that should be
  reliable for an end user, not a sprint of experiments. Tests, type hints,
  structured logging, retries on the network boundary.

## Module layout

```
auracast/
├── auth/        Google OAuth2 + API connection. Token cache, refresh,
│                scope management. Returns authenticated httpx clients.
├── ingest/      Async image acquisition. Two backends behind one interface:
│                LocalDirectoryIngest (dev) and GoogleAPIIngest (prod).
│                Produces ImageRecord rows; does NOT score or embed.
├── engine/      Vision model orchestration. CLIP/SigLIP for aesthetic
│                scoring + embeddings. Qwen2-VL for captions / structured
│                attribute extraction. FP16 mixed-precision A100 path.
│                The only module that imports torch / transformers.
├── app/         Streamlit interface. Loads scored records, displays
│                synthesized post previews, captures human approve/reject.
├── schema/      Pydantic models. Single source of truth for the on-disk
│                manifest format and the in-memory record types.
└── scripts/     Verification + one-off runners.
                 - verify_clip_a100.py: smoke test on the GPU.
                 - mock_pipeline_demo.py: end-to-end local-dir demo.
```

## Data flow

```
  [LocalDir | Google API]                       (ingest/)
            │
            ▼
      ImageRecord                               (schema/)
            │
            ▼
  CLIP/SigLIP scoring + embedding              (engine/)
            │
            ▼
   AestheticScore + Embedding                   (schema/)
            │
            ▼
   Streamlit preview / approve                  (app/)
            │
            ▼
      Publish (out of scope for v0)
```

## Technology choices

- **Python 3.10+**. Type hints + Pydantic v2 throughout.
- **Pydantic v2** for all data records (`auracast/schema/models.py`). No
  ad-hoc dicts across module boundaries.
- **PyTorch + Transformers** for model loading. Default scorer:
  `openai/clip-vit-base-patch32` (lightweight, ~150 MB). Production:
  `google/siglip-so400m-patch14-384` once latency budget allows.
- **httpx (async)** for the network boundary. `aiofiles` for disk I/O on
  the ingest path so a single worker can saturate the link.
- **Pillow** for decode + light pre-processing only. Heavy transforms
  belong in the model's own processor.
- **Streamlit** for the UI. Single-file app in `auracast/app/streamlit_app.py`.

## Hardware target

Primary: a single NVIDIA A100 80 GB on the MIT Beery vision cluster
(see `~/.claude/projects/...auracast/memory/reference_beery_cluster.md`).
FP16 mixed precision. Batch size scales with VRAM headroom — start at
32 for CLIP-B/32, autotune up.

Dev box: Mac M4 (CPU/MPS) — only used for unit tests, never for actual
scoring runs. The GPU code paths are CPU-tested via dimensions + dtypes
but model-loading tests are gated by `torch.cuda.is_available()`.

## Conventions

- All public functions are typed.
- All cross-module data uses Pydantic models from `schema/`.
- All disk paths use `pathlib.Path`, never raw strings.
- All times are `datetime.datetime` with explicit `tz=timezone.utc`.
- Logging via `logging` with a per-module `logger = logging.getLogger(__name__)`.
- Tests in `tests/test_<module>.py`, CPU-runnable.

## Phase plan

- **Phase 0 — scaffold (this commit)**: directories, Pydantic schema, mock
  ingest pipeline, CLIP A100 verification script, Streamlit stub.
- **Phase 1 — local end-to-end**: ingest local dir → score + embed → write
  manifest → Streamlit preview. No external APIs.
- **Phase 2 — Google integration**: OAuth2 flow, Photos / Drive ingest
  adapter behind the existing `IngestSource` interface.
- **Phase 3 — quality + UX**: human feedback loop captured back into the
  manifest; per-user aesthetic preference learning; post-composition logic.

## Out of scope (for now)

- Actually publishing to Instagram (no Graph API integration yet).
- Multi-user / multi-tenant.
- Persistent database (Postgres etc.) — manifest is a JSONL file on disk
  through Phase 1; migration when it's actually a bottleneck.
