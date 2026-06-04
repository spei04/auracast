# AuraCast

**AI-assisted curation for your Instagram-bound photos.**

AuraCast pulls candidate photos from a Google Drive folder, scores each one against an aesthetic you describe in plain English, and gives you a clean web interface to approve or reject. When you're done, a single click moves the rejects to Drive's Trash — your folder ends up holding exactly the curated set you want.

Built for the workflow of "I have 200 photos from this shoot / weekend / trip, and I need to whittle them down to the 5-10 best for a feed."

## What you can do with it

- **Organize candidates into projects.** Each project = one Drive folder + its own scoring criteria. Switch between projects via a sidebar dropdown ("Spring Trip", "Studio Portraits", "Daily Stories" — whatever scheme you like).
- **Describe your aesthetic in words.** Type "warm authentic smiles with golden-hour light" or "moody cinematic with deep shadows" into a text box. The chosen model scores every image against your description.
- **Pick the right scoring model for the job:**
  - **CLIP — custom prompt** *(default, fast)* — flexible, handles anything you can describe in a sentence. Best general-purpose pick.
  - **LAION Aesthetic** *(no prompt needed)* — trained on ~600k human aesthetic ratings. Strong for landscapes and "is this objectively beautiful" judgments.
  - **Qwen2-VL — smart prompt** *(slowest, smartest)* — a vision-language model that reads your prompt with full nuance. Best for portraits, facial expression, eye contact, mood.
- **Approve, reject, or undo** — three buttons per image card. Sorted top-down by score so the best candidates surface first.
- **Filter and re-rank** — sidebar sliders show only the top X% of a project, or only approved/rejected/pending. Score is normalized within the project so "0.85" always means "near the top of this batch".
- **Sync incrementally** — add new photos to your Drive folder, click 🔄 Sync now; only the *new* ones get scored (content-hash deduplication, so re-syncs are cheap).
- **Finalize** — one button moves all rejected files to Drive Trash (recoverable for 30 days). What's left in your Drive folder is your shortlist.

## Quick start

### 1. Install

```bash
git clone https://github.com/spei04/auracast.git
cd auracast
pip install -e ".[dev]"
```

Python 3.10+ required. Works on Mac (M-series MPS), Linux, and Windows. NVIDIA GPU optional — only Qwen2-VL really benefits from one.

### 2. Set up Google access (one-time)

1. **Create a Google Cloud project** at https://console.cloud.google.com/projectcreate.
2. **Enable the Drive API**: https://console.cloud.google.com/apis/api/drive.googleapis.com — click *Enable*.
3. **Configure the OAuth consent screen**: APIs & Services → OAuth consent screen. Choose "External" user type. Add yourself under **Test users**.
4. **Create credentials**: APIs & Services → Credentials → *Create credentials* → *OAuth client ID* → *Desktop app*. Download the resulting JSON.
5. **Place credentials and run the auth helper:**
   ```bash
   mkdir -p ~/.config/auracast
   mv ~/Downloads/client_secret_*.json ~/.config/auracast/client_secrets.json
   python -m auracast.scripts.auth_setup
   ```
   A browser opens. Click through the "this app is unverified" warning (it's your app — that's expected for personal use). Grant the Drive scopes. A `token.json` file appears alongside `client_secrets.json` and you're done.

### 3. Launch the app

```bash
streamlit run auracast/app/streamlit_app.py
```

Opens at http://localhost:8501.

## Typical workflow

1. **Create a project** from the sidebar (➕ New project).
   - Give it a name like "Beach Trip 2026".
   - Click **🔄 Refresh** to load your Drive folders, then pick one from the dropdown. (Or paste a folder URL / ID directly.)
2. **Set your aesthetic goal** (top of the project view):
   - Pick a scoring model.
   - If it's a prompt-based model, describe what you want — e.g.
     `warm candid laughter, motion blur in the hands, soft natural light`.
3. **🔄 Sync now** (sidebar) — pulls images from Drive, runs them through the scorer, populates the grid.
4. **Review the grid** — images appear sorted by score, best first. Click **Approve** / **Reject** / **↺** under each.
5. When you're satisfied, scroll to the bottom and click **🗑 Finalize project**. Rejected files are moved to Drive Trash. Your Drive folder is now your shortlist.

Each project stays independent — its own folder, scoring criteria, scored manifest, and approval history.

## Useful tips

- **Change your mind?** Add `rejected` to the sidebar "Review status" filter, find the image, and click **↺** to reset it to pending — or **Approve** to flip it directly. Works *until* you click Finalize.
- **Rejected file in Trash by accident?** Go to https://drive.google.com → Trash → Restore. Files stay recoverable for 30 days.
- **Want to try a different aesthetic?** Just edit the prompt, click **🎯 Score all N image(s)**. All images are re-ranked against the new criterion in seconds.
- **Adding more photos later?** Drop them in the same Drive folder, click **🔄 Sync now**. The pipeline only scores *new* photos (content-hash dedupe).

## For developers

- Architecture, module layout, and conventions: see [CLAUDE.md](./CLAUDE.md).
- Tests: `pytest` (104 tests, all CPU-runnable).
- Cluster slurm scripts: `scripts/slurm/verify_a100.sh`, `scripts/slurm/verify_qwen2vl.sh`. Configured for the MIT Beery vision cluster but easy to adapt.

## Limitations / things to know

- **Google Photos** is **not** supported. Google restricted the Photos Library API for new third-party apps in 2024-2025; we use Drive instead. Drop your photos into a Drive folder.
- **No automatic Instagram publishing yet.** Approved photos sit in your curated Drive folder; the actual posting is manual (or a future feature — see CLAUDE.md Phase 3).
- **Qwen2-VL is slow on Mac.** ~5 seconds per image on M-series MPS. Fine for ~20 images, painful for hundreds. Use CLIP or LAION for fast iteration, switch to Qwen2-VL for a final sort if it matters.
- **Single user.** Each install runs locally with one user's credentials. No multi-tenant or hosted version.

## Status

Actively under development. Current capability:

- Phase 0: scaffold ✅
- Phase 1: local + Drive end-to-end with persistence ✅
- Phase 1.5: Qwen2-VL captioning ✅
- Phase 2: Google OAuth + Drive ingest ✅
- Phase 2b: multi-project UI + folder picker + Finalize ✅
- Phase 2c: custom prompts + scorer model dropdown + score normalization ✅
- Phase 3: Instagram publishing — *not started*
