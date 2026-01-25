# Prediction SLA Explorer

This project ships a small Streamlit dashboard that works with your Postgres `predictions` and `actuals_date` tables. It:

- keeps the SQL queries in `config.py` so they can be adjusted without touching the UI logic.
- applies the required 15‑minute SLA adjustment when loading the prediction timestamps.
- surfaces a slider-driven 15‑minute window, prediction chart, min/max analytics, actuals lookups, and several strategy profit comparisons.

## Quick start

1. Create and activate the virtual environment (already created once per workspace):
   ```bash
   py -3 -m venv .venv
   .venv\\Scripts\\Activate.ps1  # PowerShell
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   streamlit run streamlit_app.py
   ```

## Configuration

- `config.py` contains the `PREDICTIONS_QUERY`, `ACTUALS_QUERY`, and the `SLA_INTERVAL_MINUTES` constant, so you can tweak SQL or the interval without touching the rest of the code.
- The dashboard depends on `common_util.postgres.connect_to_db_pg()` for the database connection, so ensure that module and its credentials are available in your environment.

## What the UI shows

1. A slider lets you sweep a fixed 15‑minute window across the prediction timeline.
2. Inside that window you see the prediction line, the min/max points, and the computed signal (BUY if the minimum comes first, SELL otherwise).
3. Actual values corresponding to the min and max events are displayed along with the prediction difference.
4. A comparison table shows the base strategy (min/max), exit-after-2–5-minute variants, and exits triggered when prediction values move by 5, 7, or 9 units.

