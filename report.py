"""
PDF EDA report generator.

Builds a self-contained PDF summarizing dataset overview, summary statistics,
auto-generated key insights, and a handful of charts (correlation heatmap,
primary metric distribution, category breakdown, and trend over time).
Uses fpdf2 (pure Python, no system dependencies) and matplotlib/seaborn for
chart images.
"""
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fpdf import FPDF

from insights import generate_all_insights

PAGE_WIDTH = 210  # A4 mm
MARGIN = 15
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN


def _safe(text) -> str:
    """Core PDF fonts only support latin-1; strip/replace anything else
    (emoji, smart quotes, etc.) so text never breaks PDF generation."""
    text = str(text).replace("**", "")
    return text.encode("latin-1", "replace").decode("latin-1")


def _fig_to_pdf(pdf: FPDF, fig, width: float = CONTENT_WIDTH) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    pdf.image(buf, x=MARGIN, w=width)
    plt.close(fig)


def _section_title(pdf: FPDF, text: str) -> None:
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 12, _safe(text), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(200, 200, 200)
    pdf.line(MARGIN, pdf.get_y(), PAGE_WIDTH - MARGIN, pdf.get_y())
    pdf.ln(4)


def generate_pdf_report(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    date_col: str | None,
    primary_metric: str | None,
    dataset_name: str = "dataset",
) -> bytes:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)

    # --- Title page ---
    pdf.add_page()
    pdf.ln(60)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 16, "EDA Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(90, 90, 90)
    pdf.ln(6)
    pdf.cell(0, 8, _safe(f"Dataset: {dataset_name}"), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    # --- 1. Dataset overview ---
    pdf.add_page()
    _section_title(pdf, "1. Dataset Overview")

    order_id_col = next((c for c in df.columns if "order" in c.lower() and "id" in c.lower()), None)
    total_orders = df[order_id_col].nunique() if order_id_col else len(df)

    pdf.set_font("Helvetica", "", 11)
    stats_lines = [
        f"Total orders: {total_orders:,}",
        f"Rows: {len(df):,}",
        f"Columns: {df.shape[1]}",
        f"Total missing values: {int(df.isna().sum().sum()):,}",
        f"Duplicate rows: {int(df.duplicated().sum()):,}",
    ]
    for line in stats_lines:
        pdf.cell(0, 7, _safe(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 10)
    headers = ["Column", "Type", "Missing %", "Unique"]
    widths = [70, 40, 35, 35]
    for h, w in zip(headers, widths):
        pdf.cell(w, 8, h, border=1, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for col in df.columns:
        row_vals = [
            _safe(col)[:38],
            _safe(df[col].dtype),
            f"{df[col].isna().mean() * 100:.1f}%",
            f"{df[col].nunique():,}",
        ]
        for val, w in zip(row_vals, widths):
            pdf.cell(w, 7, val, border=1, align="C")
        pdf.ln()

    # --- 2. Summary statistics ---
    if numeric_cols:
        pdf.add_page()
        _section_title(pdf, "2. Summary Statistics")
        desc = df[numeric_cols].describe().T.round(2)
        stat_cols = list(desc.columns)
        first_w = 42
        other_w = (CONTENT_WIDTH - first_w) / len(stat_cols)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(first_w, 8, "Column", border=1, align="C")
        for stat in stat_cols:
            pdf.cell(other_w, 8, _safe(stat), border=1, align="C")
        pdf.ln()
        pdf.set_font("Helvetica", "", 8)
        for idx, row in desc.iterrows():
            pdf.cell(first_w, 7, _safe(idx)[:22], border=1)
            for stat in stat_cols:
                pdf.cell(other_w, 7, f"{row[stat]:,.2f}", border=1, align="C")
            pdf.ln()

    # --- 3. Key insights ---
    pdf.add_page()
    _section_title(pdf, "3. Key Insights")
    insights = generate_all_insights(df, numeric_cols, categorical_cols, date_col, primary_metric)
    pdf.set_font("Helvetica", "", 11)
    if not insights:
        pdf.multi_cell(0, 7, "No notable findings for the current data selection.")
    else:
        for ins in insights:
            pdf.set_font("Helvetica", "B", 11)
            pdf.write(7, "-  ")
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 7, _safe(ins.text))
            pdf.ln(1)

    # --- 4. Correlation heatmap ---
    if len(numeric_cols) >= 2:
        pdf.add_page()
        _section_title(pdf, "4. Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        sns.heatmap(df[numeric_cols].corr(numeric_only=True), annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax)
        _fig_to_pdf(pdf, fig)

    # --- 5. Primary metric distribution ---
    if primary_metric:
        pdf.add_page()
        _section_title(pdf, f"5. Distribution of {primary_metric}")
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        sns.histplot(df[primary_metric].dropna(), kde=True, ax=ax, color="#4C78A8")
        ax.set_xlabel(primary_metric)
        _fig_to_pdf(pdf, fig)

    # --- 6. Category breakdown ---
    if categorical_cols and primary_metric:
        pdf.add_page()
        cat_col = categorical_cols[0]
        _section_title(pdf, f"6. {primary_metric} by {cat_col}")
        grouped = df.groupby(cat_col)[primary_metric].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        sns.barplot(x=grouped.index, y=grouped.values, ax=ax, color="#4C78A8")
        ax.set_ylabel(primary_metric)
        ax.set_xlabel(cat_col)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        _fig_to_pdf(pdf, fig)

    # --- 7. Trend over time ---
    if date_col and primary_metric:
        pdf.add_page()
        _section_title(pdf, f"7. {primary_metric} Trend Over Time")
        ts = df[[date_col, primary_metric]].dropna().copy()
        ts = ts.set_index(date_col).resample("ME")[primary_metric].sum()
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(ts.index, ts.values, marker="o", color="#4C78A8")
        ax.set_xlabel(date_col)
        ax.set_ylabel(primary_metric)
        fig.autofmt_xdate()
        _fig_to_pdf(pdf, fig)

    return bytes(pdf.output())
