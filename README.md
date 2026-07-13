
# EDA Dashboard

Turn raw tabular data into insights in minutes. Upload a CSV and get automatic
data profiling, visual exploration, and rule-based "key findings" — no manual
charting required.

## Problem it solves

Businesses collect data but often lack the time or skills to explore it before
deciding what to build (a report, a model, a fix). This dashboard compresses
that first exploratory pass — the one analysts usually do by hand in a
notebook — into an interactive tool anyone can point at a CSV.

## Features

- **Upload any CSV** or explore the bundled sample e-commerce dataset
- **Overview**: shape, dtypes, missing values, summary statistics, data preview
- **Distributions**: histograms and box plots for numeric columns, value counts for categorical columns
- **Correlations**: heatmap + interactive scatter explorer
- **Trends**: time series aggregation by day/week/month (auto-detects date columns)
- **Categories**: group-by breakdowns and two-dimensional pivot heatmaps
- **Key Insights**: auto-generated plain-English findings — missing data warnings,
  strong correlations, outliers (IQR method), trend direction, top/bottom
  category performance, and skewed distributions
- **Sidebar filters**: date range and up to 3 categorical filters, applied live across every tab

## Tech stack

Python, pandas, NumPy, Seaborn, Plotly, Streamlit.

## Project structure

```
app.py                   # Streamlit app (UI, charts, filters)
insights.py              # Rule-based auto-insights engine
generate_data.py         # Script that generated the sample dataset
sample_sales_data.csv    # Bundled sample e-commerce dataset (5,000 orders)
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

## Using your own data

Click "Upload a CSV file" in the sidebar. The app auto-detects numeric,
categorical, and date columns — no configuration needed. Column names
containing "date" or "time" are automatically parsed as dates.

## Sample dataset

`sample_sales_data.csv` is a synthetic e-commerce dataset (5,000 orders,
2024–2025) with realistic patterns built in on purpose: seasonal holiday
spikes, a gradual revenue growth trend, category/region differences,
missing values in a few columns, and a handful of outlier orders — so the
Key Insights tab has real signal to surface. Regenerate or modify it with
`python generate_data.py`.

## Extending

- Swap the rule-based insights in `insights.py` for a model-based approach
  (e.g. anomaly detection, forecasting) once you have a specific use case
- Add a database/API data source alongside file upload
- Add export (PDF/PNG) of charts and the insights list
=======
# Insight-Lens-Automated-EDA-Dashboard-
Interactive EDA dashboard that auto-profiles any CSV, surfaces rule-based insights, and generates downloadable PDF reports — built with Python, Pandas, Streamlit, and Plotly.

