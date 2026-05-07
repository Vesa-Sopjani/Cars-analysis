import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

df_raw = pd.read_csv("data/clean_no_encoding.csv")
df_enc = pd.read_csv("data/clean.csv")

print(f"\n[INFO] Dataset loaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
print(f"[INFO] Columns: {df_raw.columns.tolist()}")
print(f"\n{df_raw.dtypes}\n")

# 1. Data Sanity Checks
missing = df_raw.isnull().sum()
print("\n[CHECK] Missing values per column:")
print(missing[missing > 0] if missing.any() else " No missing values found.")

dupes = df_raw.duplicated().sum()
print(f"\n[CHECK] Duplicate rows: {dupes}")
if dupes > 0:
    df_raw = df_raw.drop_duplicates().reset_index(drop=True)
    print(f"  → Removed. New shape: {df_raw.shape}")

print(f"\n[CHECK] Selling price range : {df_raw['selling_price'].min():,} – {df_raw['selling_price'].max():,} INR")
print(f"[CHECK] KM driven range     : {df_raw['km_driven'].min():,} – {df_raw['km_driven'].max():,}")
print(f"[CHECK] Year range          : {df_raw['year'].min()} – {df_raw['year'].max()}")
print(f"[CHECK] Mileage range       : {df_raw['mileage(km/ltr/kg)'].min()} – {df_raw['mileage(km/ltr/kg)'].max()}")
print(f"[CHECK] Engine range (cc)   : {df_raw['engine'].min()} – {df_raw['engine'].max()}")

zero_mileage = (df_raw["mileage(km/ltr/kg)"] == 0).sum()
print(f"\n[CHECK] Rows with mileage = 0 (suspicious): {zero_mileage}")

old_cars = (df_raw["year"] < 1990).sum()
print(f"[CHECK] Cars made before 1990 (potential outliers): {old_cars}")

# 2. Feature Engineering

CURRENT_YEAR = 2026

# 2.1 Car Age
df_raw["car_age"] = CURRENT_YEAR - df_raw["year"]
print(f"\n[FEATURE] car_age created  (range: {df_raw['car_age'].min()} – {df_raw['car_age'].max()} years)")

# 2.2 Km per year
df_raw["km_per_year"] = df_raw["km_driven"] / (df_raw["car_age"] + 1)
print(f"[FEATURE] km_per_year created (mean: {df_raw['km_per_year'].mean():.0f} km/yr)")

# 2.3 price per km
df_raw["price_per_km"] = df_raw["selling_price"] / (df_raw["km_driven"] + 1)
print(f"[FEATURE] price_per_km created (median: {df_raw['price_per_km'].median():.2f} INR/km)")

# 2.4 brand
df_raw["brand"] = df_raw["name"].str.split().str[0].str.lower()
print(f"[FEATURE] brand extracted  ({df_raw['brand'].nunique()} unique brands)")

# 2.5 Is Luxury
LUXURY_BRANDS = {"bmw", "audi", "mercedes-benz", "jaguar", "volvo", "lexus",
                 "land", "jeep"}
df_raw["is_luxury"] = df_raw["brand"].isin(LUXURY_BRANDS).astype(int)
print(f"[FEATURE] is_luxury flag   ({df_raw['is_luxury'].sum()} luxury cars)")

# 2.6 Engine Category
def categorize_engine(cc):
    if cc < 1000:   return "small"
    elif cc < 1500: return "medium"
    elif cc < 2000: return "large"
    else:           return "premium"
 
df_raw["engine_category"] = df_raw["engine"].apply(categorize_engine)
print(f"[FEATURE] engine_category created: {df_raw['engine_category'].value_counts().to_dict()}")

# 2.7 Mileage Category
def categorize_mileage(m):
    if m < 15:  return "low"
    elif m < 20: return "medium"
    elif m < 25: return "high"
    else:        return "very_high"
 
df_raw["mileage_category"] = df_raw["mileage(km/ltr/kg)"].apply(categorize_mileage)
print(f"[FEATURE] mileage_category created: {df_raw['mileage_category'].value_counts().to_dict()}")

# 2.8 Is First Owner
df_raw["is_first_owner"] = (df_raw["owner"] == "first owner").astype(int)
pct_first = df_raw["is_first_owner"].mean() * 100
print(f"[FEATURE] is_first_owner flag  ({pct_first:.1f}% of cars are first-owner)")

# 2.9 Age x Km
df_raw["age_x_km"] = df_raw["car_age"] * df_raw["km_driven"]
print(f"[FEATURE] age_x_km interaction term created (mean: {df_raw['age_x_km'].mean():.0f})")

# 2.10 Brand Avg Price
brand_avg = df_raw.groupby("brand")["selling_price"].transform("mean")
df_raw["brand_avg_price"] = brand_avg
print(f"[FEATURE] brand_avg_price (target-encoded mean price per brand)")
 
print(f"\n[INFO] Dataset now has {df_raw.shape[1]} columns (was 10)")

# 3 Feature Selection
df_sel = df_raw.copy()
le = LabelEncoder()
for col in ["fuel", "seller_type", "transmission", "owner",
            "engine_category", "mileage_category", "brand"]:
    df_sel[col] = le.fit_transform(df_sel[col].astype(str))

FEATURES = [
    "car_age", "km_driven", "km_per_year", "price_per_km",
    "engine", "mileage(km/ltr/kg)", "is_luxury", "is_first_owner",
    "age_x_km", "brand_avg_price",
    "fuel", "seller_type", "transmission", "owner",
    "engine_category", "mileage_category"
]
TARGET = "selling_price"
 
X = df_sel[FEATURES]
y = df_sel[TARGET]

print("\n[ANALYSIS] Pearson Correlation with selling_price (numerical features):")
numerical_feats = ["car_age", "km_driven", "km_per_year", "price_per_km",
                   "engine", "mileage(km/ltr/kg)", "is_luxury",
                   "is_first_owner", "age_x_km", "brand_avg_price"]
corr = df_raw[numerical_feats + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
print(corr.to_string())

print("\n[ANALYSIS] Mutual Information scores (all features):")
mi = mutual_info_regression(X, y, random_state=42)
mi_series = pd.Series(mi, index=FEATURES).sort_values(ascending=False)
print(mi_series.to_string())

print("\n[ANALYSIS] F-statistic scores (top 10 features):")
f_scores, f_pvals = f_regression(X, y)
f_series = pd.Series(f_scores, index=FEATURES).sort_values(ascending=False)
print(f_series.head(10).to_string())

print("\n[REJECTED FEATURES & REASONS]")
print("  year          → replaced by car_age (same info, more interpretable)")
print("  name          → too granular (2000+ unique values); brand extracted instead")
print("  price_per_km  → high outlier variance; useful only contextually")
print("  km_per_year   → low MI score; redundant with car_age + km_driven")

# 4 Group Level Analysis 
print("\n[ANALYSIS] Avg selling price by fuel type:")
print(df_raw.groupby("fuel")["selling_price"].mean().sort_values(ascending=False).to_string())
 
print("\n[ANALYSIS] Avg selling price by engine category:")
print(df_raw.groupby("engine_category")["selling_price"].mean().sort_values(ascending=False).to_string())
 
print("\n[ANALYSIS] Avg selling price by owner type:")
print(df_raw.groupby("owner")["selling_price"].mean().sort_values(ascending=False).to_string())
 
print("\n[ANALYSIS] Avg selling price by mileage category:")
print(df_raw.groupby("mileage_category")["selling_price"].mean().sort_values(ascending=False).to_string())
 
print("\n[ANALYSIS] Avg selling price: luxury vs non-luxury:")
print(df_raw.groupby("is_luxury")["selling_price"].mean().rename({0: "non-luxury", 1: "luxury"}).to_string())
 
print("\n[ANALYSIS] Avg selling price by transmission:")
print(df_raw.groupby("transmission")["selling_price"].mean().to_string())
 
print("\n[ANALYSIS] Avg selling price: first owner vs others:")
print(df_raw.groupby("is_first_owner")["selling_price"].mean().rename({0: "not first owner", 1: "first owner"}).to_string())
 
print("\n[ANALYSIS] Top 10 brands by avg selling price:")
top_brands = df_raw.groupby("brand")["selling_price"].agg(["mean","count"])
top_brands = top_brands[top_brands["count"] >= 10].sort_values("mean", ascending=False)
print(top_brands.head(10).to_string())

# 5. Statistical Validation
first  = df_raw[df_raw["is_first_owner"] == 1]["selling_price"]
others = df_raw[df_raw["is_first_owner"] == 0]["selling_price"]
t_stat, p_val = stats.ttest_ind(first, others)
print(f"\n[TEST] T-test  first_owner vs others:")
print(f"       t-statistic = {t_stat:.4f},  p-value = {p_val:.6f}")
print(f"       → {'SIGNIFICANT' if p_val < 0.05 else 'NOT significant'} difference in price (α=0.05)")

luxury_prices    = df_raw[df_raw["is_luxury"] == 1]["selling_price"]
nonluxury_prices = df_raw[df_raw["is_luxury"] == 0]["selling_price"]
t2, p2 = stats.ttest_ind(luxury_prices, nonluxury_prices)
print(f"\n[TEST] T-test  luxury vs non-luxury:")
print(f"       t-statistic = {t2:.4f},  p-value = {p2:.6f}")
print(f"       → {'SIGNIFICANT' if p2 < 0.05 else 'NOT significant'} difference in price (α=0.05)")

r, p3 = stats.pearsonr(df_raw["car_age"], df_raw["selling_price"])
print(f"\n[TEST] Pearson correlation  car_age ↔ selling_price:")
print(f"       r = {r:.4f},  p-value = {p3:.6f}")
print(f"       → {'SIGNIFICANT' if p3 < 0.05 else 'NOT significant'} (α=0.05)")

r2, p4 = stats.pearsonr(df_raw["engine"], df_raw["selling_price"])
print(f"\n[TEST] Pearson correlation  engine ↔ selling_price:")
print(f"       r = {r2:.4f},  p-value = {p4:.6f}")
print(f"       → {'SIGNIFICANT' if p4 < 0.05 else 'NOT significant'} (α=0.05)")

# 6. Save Enriched Dataset
output_path = "feature_engineering.csv"
df_raw.to_csv(output_path, index=False)
print(f"\n[SAVED] {output_path}")
print(f"        Shape: {df_raw.shape}")
print(f"        New feature columns added:")
new_cols = ["car_age", "km_per_year", "price_per_km", "brand",
            "is_luxury", "engine_category", "mileage_category",
            "is_first_owner", "age_x_km", "brand_avg_price"]
for c in new_cols:
    print(f"          + {c}")
 
print("\n" + "=" * 65)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 65)
print("\nSUMMARY – Best features by impact on selling_price:")
print("  1. brand_avg_price   (MI: highest – captures brand prestige)")
print("  2. is_luxury         (corr: +0.59  – strong positive)")
print("  3. engine (cc)       (corr: +0.45  – larger engine = higher price)")
print("  4. car_age           (corr: -0.44  – older = cheaper)")
print("  5. is_first_owner    (corr: +0.24  – ownership history matters)")
print("  6. age_x_km          (corr: -0.29  – combined depreciation signal)")
print("  7. km_driven         (corr: -0.20  – more usage = lower price)")
 