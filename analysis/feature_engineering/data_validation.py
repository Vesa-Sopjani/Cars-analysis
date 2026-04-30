import pandas as pd
import numpy as np

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["price","year","mileage","car_age","price_per_mile",
                "mileage_category","age_group","brand_name","brand_tier","is_salvage"]
    key_cols = [c for c in key_cols if c in df.columns]

    missing = df[key_cols].isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({
        "missing_count": missing,
        "missing_percent": missing_pct
    }).query("missing_count > 0").sort_values("missing_percent", ascending=False)

    if report.empty:
        print("[PASS] No missing values in key columns.")
    else:
        print("[WARNING] Missing values detected:")
        print(report.to_string())
        critical = report[report["missing_percent"] > 10]
        if not critical.empty:
            print(f"\n[CRITICAL] Columns with >10% missing: {critical.index.tolist()}")
        
    return report

def check_negative_prices(df: pd.DataFrame) -> int:
    bad = df[df["price"] <= 0]
    if len(bad) == 0:
        print("[PASS] All prices are positive.")
    else:
        print(f"[FAIL] {len(bad)} rows with price <= 0.")
        print(bad[["price"]].head(5).to_string())
    return len(bad)

def check_car_age(df: pd.DataFrame) -> int:
    if "car_age" not in df.columns:
        print("[SKIP] 'car_age' not found. Run feature_engineering.py first.")
        return 0
    bad = df[(df["car_age"] < 0) | (df["car_age"] > 80)]
    if len(bad) == 0:
        print("[PASS] All car_age values are in valid range (0-80).")
    else: 
        print(f"[FAIL] {len(bad)} rows with suspicious car_age:")
        print(bad[["year", "car_age"]].head(5).to_string())
    return len(bad)

def check_mileage(df: pd.DataFrame) -> int:
    bad = df[(df["mileage"] < 0) | (df["mileage"] > 500_000)]
    if len(bad) == 0:
        print("[PASS] All mileage values are in valid range (0–500,000).")
    else:
        print(f"[FAIL] {len(bad)} rows with suspicious mileage:")
        print(bad[["mileage"]].head(5).to_string())
    return len(bad)


def check_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
    Q1  = df["price"].quantile(0.25)
    Q3  = df["price"].quantile(0.75)
    IQR = Q3 - Q1
    lo  = Q1 - 3 * IQR
    hi  = Q3 + 3 * IQR
 
    outliers = df[(df["price"] < lo) | (df["price"] > hi)]
    pct = len(outliers) / len(df) * 100
    print(f"[OUTLIERS] Valid price range: ${lo:,.0f} – ${hi:,.0f}")
    print(f"[OUTLIERS] Extreme outliers flagged: {len(outliers)} rows ({pct:.1f}%)")
    if not outliers.empty:
        print(f"Min outlier: ${outliers['price'].min():,.0f}  |  Max outlier: ${outliers['price'].max():,.0f}")
    return outliers


def check_category_labels(df: pd.DataFrame) -> None:
    expected = {
        "mileage_category": ["Low", "Medium", "High", "Very High"],
        "age_group":        ["New", "Recent", "Old", "Classic"],
        "brand_tier":       ["Luxury", "Mid", "Economy"]
    }
    for col, allowed in expected.items():
        if col not in df.columns:
            print(f"[SKIP] '{col}' not found.")
            continue
        actual = df[col].dropna().unique().tolist()
        bad    = [v for v in actual if v not in allowed]
        if bad:
            print(f"[FAIL] '{col}' has unexpected values: {bad}")
        else:
            print(f"[PASS] '{col}' — all labels valid: {sorted(actual)}")

def generate_validation_report(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    print(f"Dataset: {df.shape[0]:,} rows  |  {df.shape[1]} columns")
 
    print("\n─── 1. Missing Values ───────────────────────────────")
    missing = check_missing_values(df)
 
    print("\n─── 2. Price Validity ───────────────────────────────")
    bad_prices = check_negative_prices(df)
 
    print("\n─── 3. Car Age Validity ─────────────────────────────")
    bad_ages = check_car_age(df)
 
    print("\n─── 4. Mileage Validity ─────────────────────────────")
    bad_mileage = check_mileage(df)
 
    print("\n─── 5. Price Outliers ───────────────────────────────")
    outliers = check_price_outliers(df)
 
    print("\n─── 6. Category Label Check ─────────────────────────")
    check_category_labels(df)
 
    total_issues = bad_prices + bad_ages + bad_mileage + len(missing)
    print("\n" + "=" * 60)
    status = "ALL CHECKS PASSED" if total_issues == 0 else f"{total_issues} issue type(s) found — review above"
    print(f"STATUS: {status}")
    print("=" * 60)
 
    return {
        "missing_values":  missing,
        "bad_price_rows":  bad_prices,
        "bad_age_rows":    bad_ages,
        "bad_mileage_rows": bad_mileage,
        "price_outliers":  outliers,
    }
 
 
if __name__ == "__main__":
    df = pd.read_csv("../data/featured_cars.csv")
    generate_validation_report(df)
