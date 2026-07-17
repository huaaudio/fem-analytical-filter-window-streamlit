# METAVISION acoustic panel listening demo

A public Streamlit demo for hearing how a bare A4 panel compares with analytical and finite-element (FEM) simulations of the same panel with local resonators. The guided experience begins with sample audio, a target-frequency control, a plain-language result, and optional technical detail. All processed outputs are simulations, not physical measurements; headphones are recommended.

## Run locally

Create and activate a virtual environment, then install the application dependencies and start Streamlit:

```bash
python -m venv .venv
# PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run streamlit_app.py --server.xsrfCookieSameSite=lax
```

The local override keeps uploads working over plain HTTP. The checked-in production setting uses
`SameSite=None; Secure` so uploads also work when the HTTPS app is embedded on the METAVISION site.

The bundle includes:

- the app and its required `secondment` modules under `src/`
- legacy FEM CSV overlay curves under `fem/results/`
- FEM cache config, base DAT, and SQLite cache for cached SOL 108 curves

The larger research-archive WAV collection is intentionally omitted to keep the deployment small. The app includes curated listening examples and accepts WAV uploads up to 20 MB.

## Listening feedback

After creating a listening comparison, visitors can report whether they heard a difference and
optionally leave a short comment. Copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` for local use, or add the same `[feedback]` settings to the deployed
app's Streamlit secrets. The Supabase `listening_feedback` table must grant the public role INSERT
access only; it must not grant SELECT, UPDATE, or DELETE access.

## Test

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

## Data and feedback privacy

Uploaded audio is processed transiently for the active Streamlit session and this application does not intentionally save it to persistent storage. Do not upload sensitive or personally identifying recordings to a public deployment; the hosting provider's processing and privacy terms still apply.

Submitted feedback stores the selected response, optional comment, anonymous session identifier,
sound-example label, target frequency, and compared-method labels. It does not store uploaded audio
or upload filenames. Visitors are asked not to put personal information in comments.

## Embed on the METAVISION site

The deployed app can be presented without most Streamlit chrome by using its embed URL:

```text
https://metavision.streamlit.app/?embed=true
```

Use that URL as the `src` of a responsive `iframe` on the project site. The server configuration permits cross-origin uploads only from `https://www.heu-metavision.eu` and `https://heu-metavision.eu`; add an origin to `server.corsAllowedOrigins` before embedding elsewhere. Keep a direct “Open demo” link as a fallback for small screens, restrictive browser settings, or hosts that block framing.
