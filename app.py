"""
Interactive EDA Dashboard
-------------------------
Upload any tabular dataset (or use the bundled sample e-commerce data) and
get automatic profiling, visual exploration, and rule-based "key findings"
in minutes -- no manual charting required.

Run with:  streamlit run app.py
"""
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from insights import generate_all_insights
from report import generate_pdf_report

st.set_page_config(page_title="EDA Dashboard", layout="wide", page_icon="\U0001F4CA")

SAMPLE_DATA_PATH = "sample_sales_data.csv"


# ---------------------------------------------------------------------------
# Data loading & caching
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(file) -> pd.DataFrame:
    if isinstance(file, str):
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(file)
    # Best-effort auto-detect date columns
    for col in df.columns:
        if df[col].dtype == object and ("date" in col.lower() or "time" in col.lower()):
            try:
                df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError):
                pass
    return df


def get_column_types(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    date_cols = df.select_dtypes(include="datetime64").columns.tolist()
    categorical_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if df[c].nunique() <= 50  # avoid treating free-text/ID columns as categorical
    ]
    return numeric_cols, categorical_cols, date_cols


SEVERITY_ICON = {"info": "ℹ️", "warning": "⚠️", "success": "✅"}


# ---------------------------------------------------------------------------
# Sidebar: data source + filters
# ---------------------------------------------------------------------------
st.sidebar.title("📊 EDA Dashboard")
st.sidebar.caption("Explore, visualize, and find insights in minutes.")

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
use_sample = uploaded_file is None

if use_sample:
    st.sidebar.info("No file uploaded — showing the bundled sample e-commerce dataset.")
    df_raw = load_data(SAMPLE_DATA_PATH)
else:
    df_raw = load_data(uploaded_file)

numeric_cols, categorical_cols, date_cols = get_column_types(df_raw)

st.sidebar.markdown("---")
st.sidebar.subheader("🎚️ Filters")

df = df_raw.copy()

# Date range filter
selected_date_col = None
if date_cols:
    selected_date_col = st.sidebar.selectbox("Date column", date_cols, index=0)
    min_d, max_d = df[selected_date_col].min(), df[selected_date_col].max()
    date_range = st.sidebar.date_input(
        "Date range", value=(min_d.date(), max_d.date()),
        min_value=min_d.date(), max_value=max_d.date(),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        df = df[(df[selected_date_col] >= pd.Timestamp(start)) & (df[selected_date_col] <= pd.Timestamp(end))]

# Categorical filters (up to 3 to keep sidebar manageable)
for cat_col in categorical_cols[:3]:
    options = sorted(df_raw[cat_col].dropna().unique().tolist())
    selected = st.sidebar.multiselect(f"{cat_col}", options, default=[])
    if selected:
        df = df[df[cat_col].isin(selected)]

st.sidebar.markdown("---")
primary_metric = None
if numeric_cols:
    primary_metric = st.sidebar.selectbox(
        "Primary metric (used for trends & breakdowns)", numeric_cols,
        index=numeric_cols.index("revenue") if "revenue" in numeric_cols else 0,
    )

st.sidebar.caption(f"{len(df):,} rows selected out of {len(df_raw):,} total")


# ---------------------------------------------------------------------------
# Header + KPI row
# ---------------------------------------------------------------------------
st.title("📊 EDA Dashboard")

order_id_col = next((c for c in df.columns if "order" in c.lower() and "id" in c.lower()), None)
total_orders = df[order_id_col].nunique() if order_id_col else len(df)


def render_kpi_cards(cards: list[tuple[str, str]]) -> None:
    """Render KPI cards as equally-spaced boxes with rounded corners.

    Built as a single unbroken line of HTML (no leading whitespace or blank
    lines) so Streamlit's markdown parser treats it as one HTML block instead
    of misreading indented lines as a code block.
    """
    card_divs = "".join(
        '<div style="flex:1 1 0;background-color:var(--secondary-background-color,#f0f2f6);'
        'border-radius:16px;padding:18px 12px;text-align:center;'
        'box-shadow:0 1px 4px rgba(0,0,0,0.08);">'
        f'<div style="font-size:0.8rem;color:var(--text-color,#555);opacity:0.75;margin-bottom:6px;">{label}</div>'
        f'<div style="font-size:1.4rem;font-weight:700;color:var(--text-color,#111);">{value}</div>'
        "</div>"
        for label, value in cards
    )
    st.markdown(
        f'<div style="display:flex;gap:16px;margin-bottom:1.5rem;">{card_divs}</div>',
        unsafe_allow_html=True,
    )


kpi_cards = [
    ("🧾 Total Orders", f"{total_orders:,}"),
    ("🧮 Columns", f"{df.shape[1]}"),
]
if primary_metric:
    kpi_cards.append((f"💰 Total {primary_metric}", f"{df[primary_metric].sum():,.0f}"))
    kpi_cards.append((f"💵 Avg {primary_metric}", f"{df[primary_metric].mean():,.2f}"))

render_kpi_cards(kpi_cards)

st.markdown("---")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_distributions, tab_correlation, tab_trends, tab_categories, tab_insights, tab_report = st.tabs(
    ["🔍 Overview", "📈 Distributions", "🔗 Correlations", "📅 Trends", "🗂️ Categories", "💡 Key Insights", "📄 Report"]
)

# --- Overview tab ---
with tab_overview:
    st.subheader("👀 Data preview")
    st.dataframe(df.head(50), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🧬 Column types")
        dtype_df = pd.DataFrame({
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing_%": (df.isna().mean() * 100).round(1).values,
            "unique_values": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("📐 Summary statistics")
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
        else:
            st.write("No numeric columns detected.")

    if df.isna().sum().sum() > 0:
        st.subheader("❓ Missing values by column")
        miss = df.isna().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        fig = px.bar(x=miss.index, y=miss.values, labels={"x": "Column", "y": "Missing count"})
        st.plotly_chart(fig, use_container_width=True)

# --- Distributions tab ---
with tab_distributions:
    st.subheader("📊 Numeric distributions")
    if numeric_cols:
        col1, col2 = st.columns(2)
        dist_col = col1.selectbox("Column", numeric_cols, key="dist_col")
        chart_type = col2.radio("Chart type", ["Histogram", "Box plot"], horizontal=True, key="dist_type")

        if chart_type == "Histogram":
            fig = px.histogram(df, x=dist_col, nbins=40, marginal="box", title=f"Distribution of {dist_col}")
        else:
            fig = px.box(df, y=dist_col, points="outliers", title=f"Box plot of {dist_col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No numeric columns to plot.")

    st.subheader("🗂️ Categorical value counts")
    if categorical_cols:
        cat_col = st.selectbox("Column", categorical_cols, key="cat_dist_col")
        counts = df[cat_col].value_counts().reset_index()
        counts.columns = [cat_col, "count"]
        fig2 = px.bar(counts, x=cat_col, y="count", title=f"Value counts for {cat_col}")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.write("No categorical columns detected.")

# --- Correlation tab ---
with tab_correlation:
    st.subheader("🔥 Correlation heatmap")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)

        st.subheader("✨ Scatter explorer")
        c1, c2, c3 = st.columns(3)
        x_col = c1.selectbox("X axis", numeric_cols, index=0, key="scatter_x")
        y_col = c2.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols) - 1), key="scatter_y")
        color_col = c3.selectbox("Color by (optional)", [None] + categorical_cols, key="scatter_color")
        fig3 = px.scatter(df, x=x_col, y=y_col, color=color_col, opacity=0.6,
                           title=f"{y_col} vs {x_col}")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.write("Need at least 2 numeric columns for correlation analysis.")

# --- Trends tab ---
with tab_trends:
    st.subheader("📅 Time series trend")
    if selected_date_col and primary_metric:
        freq = st.radio("Aggregate by", ["Day", "Week", "Month"], index=2, horizontal=True)
        freq_map = {"Day": "D", "Week": "W", "Month": "M"}
        ts = df[[selected_date_col, primary_metric]].dropna().copy()
        ts = ts.set_index(selected_date_col).resample(freq_map[freq])[primary_metric].sum().reset_index()
        fig = px.line(ts, x=selected_date_col, y=primary_metric, markers=True,
                       title=f"{primary_metric} over time ({freq.lower()})")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Need a date column and a numeric metric to show trends.")

# --- Categories tab ---
with tab_categories:
    st.subheader("🏆 Metric by category")
    if categorical_cols and primary_metric:
        c1, c2 = st.columns(2)
        group_col = c1.selectbox("Group by", categorical_cols, key="group_col")
        agg_func = c2.selectbox("Aggregation", ["sum", "mean", "count", "median"], key="agg_func")
        grouped = df.groupby(group_col)[primary_metric].agg(agg_func).sort_values(ascending=False).reset_index()
        fig = px.bar(grouped, x=group_col, y=primary_metric,
                     title=f"{agg_func.title()} of {primary_metric} by {group_col}")
        st.plotly_chart(fig, use_container_width=True)

        if len(categorical_cols) >= 2:
            st.subheader("🧩 Two-dimensional breakdown")
            c3, c4 = st.columns(2)
            group_col2 = c3.selectbox("Second group", [c for c in categorical_cols if c != group_col], key="group_col2")
            pivot = df.pivot_table(index=group_col, columns=group_col2, values=primary_metric, aggfunc=agg_func, fill_value=0)
            fig4, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
            st.pyplot(fig4)
    else:
        st.write("Need at least one categorical column and a numeric metric.")

# --- Key Insights tab ---
with tab_insights:
    st.subheader("💡 Auto-generated key findings")
    st.caption("Rule-based analysis: missing data, correlations, outliers, trends, and category performance.")
    insights = generate_all_insights(
        df=df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        date_col=selected_date_col,
        primary_metric=primary_metric,
    )
    if not insights:
        st.write("No notable findings for the current filter selection.")
    else:
        for ins in insights:
            icon = SEVERITY_ICON.get(ins.severity, "•")
            st.markdown(f"{icon} {ins.text}")

# --- Report tab ---
with tab_report:
    st.subheader("📄 Generate EDA Report")
    st.caption(
        "Creates a PDF with dataset overview, summary statistics, key insights, "
        "and charts — based on your current filters and primary metric selection."
    )
    dataset_label = "Sample e-commerce data" if use_sample else (uploaded_file.name if uploaded_file else "Uploaded data")

    if st.button("🛠️ Generate PDF Report"):
        with st.spinner("Building your report..."):
            pdf_bytes = generate_pdf_report(
                df=df,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                date_col=selected_date_col,
                primary_metric=primary_metric,
                dataset_name=dataset_label,
            )
        st.session_state["pdf_report_bytes"] = pdf_bytes
        st.success("Report ready — click below to download.")

    if "pdf_report_bytes" in st.session_state:
        st.download_button(
            label="⬇️ Download PDF Report",
            data=st.session_state["pdf_report_bytes"],
            file_name=f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
        )

st.markdown("---")
