"""
Generates a synthetic e-commerce sales dataset for the EDA dashboard demo.
Includes realistic patterns: seasonality, growth trend, regional/category
differences, occasional missing values, and a few outliers -- so the
auto-insights engine has real things to find.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 5000
start_date = pd.Timestamp("2024-01-01")
end_date = pd.Timestamp("2025-12-31")
date_range_days = (end_date - start_date).days

# --- Base dimensions ---
categories = ["Electronics", "Apparel", "Home & Garden", "Sports", "Beauty", "Toys"]
category_weights = [0.22, 0.20, 0.18, 0.15, 0.15, 0.10]

regions = ["North", "South", "East", "West", "Central"]
region_weights = [0.24, 0.22, 0.20, 0.19, 0.15]

channels = ["Online", "In-Store", "Mobile App"]
channel_weights = [0.5, 0.3, 0.2]

customer_segments = ["New", "Returning", "VIP"]
segment_weights = [0.35, 0.5, 0.15]

payment_methods = ["Credit Card", "Debit Card", "PayPal", "Gift Card"]

# Base price ranges per category (min, max)
price_ranges = {
    "Electronics": (50, 1200),
    "Apparel": (15, 200),
    "Home & Garden": (10, 500),
    "Sports": (10, 350),
    "Beauty": (5, 150),
    "Toys": (5, 120),
}

# --- Generate order dates with growth trend + seasonality (holiday bump in Nov/Dec) ---
day_offsets = rng.integers(0, date_range_days + 1, size=N)
order_dates = start_date + pd.to_timedelta(day_offsets, unit="D")

df = pd.DataFrame({
    "order_id": [f"ORD-{100000+i}" for i in range(N)],
    "order_date": order_dates,
})

df["category"] = rng.choice(categories, size=N, p=category_weights)
df["region"] = rng.choice(regions, size=N, p=region_weights)
df["channel"] = rng.choice(channels, size=N, p=channel_weights)
df["customer_segment"] = rng.choice(customer_segments, size=N, p=segment_weights)
df["payment_method"] = rng.choice(payment_methods, size=N)
df["customer_id"] = [f"CUST-{rng.integers(1000, 3500)}" for _ in range(N)]

# Unit price based on category range, with some noise
def sample_price(cat):
    lo, hi = price_ranges[cat]
    return round(rng.uniform(lo, hi), 2)

df["unit_price"] = df["category"].apply(sample_price)

# Quantity: mostly small, occasional bulk orders
df["quantity"] = rng.choice([1, 2, 3, 4, 5, 10], size=N, p=[0.45, 0.25, 0.15, 0.08, 0.05, 0.02])

# Discount percent: VIP/Returning customers get slightly better discounts on average
def sample_discount(segment):
    base = {"New": 0.03, "Returning": 0.07, "VIP": 0.12}[segment]
    return round(max(0, rng.normal(base, 0.05)), 3)

df["discount_pct"] = df["customer_segment"].apply(sample_discount).clip(0, 0.6)

# Revenue = price * qty * (1 - discount), plus seasonal boost in Nov/Dec and slow growth over time
month = df["order_date"].dt.month
seasonal_multiplier = month.map({11: 1.35, 12: 1.5}).fillna(1.0)
days_since_start = (df["order_date"] - start_date).dt.days
growth_multiplier = 1 + (days_since_start / date_range_days) * 0.35  # ~35% growth over 2 years

df["revenue"] = (
    df["unit_price"] * df["quantity"] * (1 - df["discount_pct"]) * seasonal_multiplier * growth_multiplier
).round(2)

# Customer satisfaction score (1-5), correlated loosely with discount and inversely with region latency (synthetic)
base_satisfaction = rng.normal(4.0, 0.6, size=N)
segment_bonus = df["customer_segment"].map({"New": -0.1, "Returning": 0.05, "VIP": 0.25})
df["satisfaction_score"] = (base_satisfaction + segment_bonus).clip(1, 5).round(1)

# Delivery days: correlated with region and channel
base_delivery = rng.normal(4, 1.5, size=N)
channel_adj = df["channel"].map({"Online": 1.0, "Mobile App": 0.8, "In-Store": -3.5})
df["delivery_days"] = (base_delivery + channel_adj).clip(0, 15).round(0)

# --- Inject some outliers (a handful of very large revenue orders) ---
outlier_idx = rng.choice(df.index, size=15, replace=False)
df.loc[outlier_idx, "revenue"] = df.loc[outlier_idx, "revenue"] * rng.uniform(5, 9, size=15)
df.loc[outlier_idx, "quantity"] = rng.integers(20, 50, size=15)

# --- Inject missing values in a few columns (realistic messiness) ---
for col, frac in [("satisfaction_score", 0.06), ("payment_method", 0.02), ("delivery_days", 0.04), ("discount_pct", 0.01)]:
    miss_idx = rng.choice(df.index, size=int(len(df) * frac), replace=False)
    df.loc[miss_idx, col] = np.nan

# Round/clean numeric types
df["unit_price"] = df["unit_price"].round(2)
df["revenue"] = df["revenue"].round(2)
df["discount_pct"] = (df["discount_pct"] * 100).round(1)  # store as percent
df = df.rename(columns={"discount_pct": "discount_pct_of_price"})

df = df.sort_values("order_date").reset_index(drop=True)

out_path = "sample_sales_data.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df.dtypes)
print(df.isna().sum())
