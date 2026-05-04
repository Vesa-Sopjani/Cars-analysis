import pandas as pd
from feature_engineering import run_feature_engineering
from data_validation import generate_validation_report
from feature_selection import run_feature_selection

INPUT_PATH = "../data/clean_no_encoding.csv"
OUTPUT_PATH = "../data/featured_cars.csv"

def main():
    print("=" * 60)
    print("Feature Engineering")
    print(" Car Price Analysis Project")
    print("=" * 60)

    print("\n[STEP 1/3] Running Feature Engineering...")
    df = run_feature_engineering(INPUT_PATH, OUTPUT_PATH)

    print("\n[STEP 2/3] Running Data Validation...")
    validation = generate_validation_report(df)

    print("\n[STEP 3/3] Running Feature Selection...")
    selection = run_feature_selection(df)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    new_cols = ["car_age", "price_per_mile", "mileage_category",
                "age_group", "brand_name", "brand_tier", "is_salvage"]
    existing = [c for c in new_cols if c in df.columns]
    print(f"\n  Enriched file saved to : {OUTPUT_PATH}")
    print(f"  Dataset shape           : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\n  New features created ({len(existing)}):")
    for col in existing:
        print(f"     • {col}")
 
    recommended = selection.get("final_recommended", [])
    print(f"\n  Top features for analysis : {recommended}")
 
    issues = (validation["bad_price_rows"] +
              validation["bad_age_rows"] +
              validation["bad_mileage_rows"] +
              len(validation["missing_values"]))
    print(f"\n   Validation issues found   : {issues}")
    print("\n[DONE]")
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()