# FEM Analytical Filter Window App

Minimal Streamlit Cloud bundle for `secondment.fem_analytical_filter_window_app`.

Deploy with:

```bash
streamlit run streamlit_app.py
```

Included:

- the app and its required `secondment` modules under `src/`
- legacy FEM CSV overlay curves under `fem/results/`
- FEM cache config, base DAT, and SQLite cache for cached SOL 108 curves

The curated example WAV files from the research archive are intentionally omitted to keep the bundle small. WAV upload still works.
