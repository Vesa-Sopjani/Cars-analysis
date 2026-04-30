import pandas as pd
import numpy as np
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.preprocessing import LabelEncoder

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["price", "year", "mileage", "car_age", "price_per_mile", "is_salvage"]
    cat_cols = ["mileage_category", "age_group", "brand_tier"]
    available_numeric = [c for c in numeric_cols if c in df.columns]
    df_work = df[available_numeric].copy()

    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df_work[col] = le.fit_transform(df[col].astype(str))
            print(f"[ENCODE] '{col}' encoded for scoring.")

    return df_work

def correlation_with_price(df: pd.DataFrame) -> pd.DataFrame:
    df_work = prepare_features(df)
    correlations = df_work.corr()["price"].drop("price").abs().sort_values(ascending=False)

    corr_df = correlations.reset_index()
    corr_df.columns = ["feature", "correlation_with_price"]

    print("\n[CORRELATION] Features ranked by absolute correlation with price:")
    print(corr_df.to_string(index=False))
    return corr_df

def score_features_ftest(df: pd.DataFrame, top_k: int = 6) -> list:
    df_work = prepare_features(df).dropna()
    X = df_work.drop(columns=["price"])
    y = df_work["price"]

    selector = SelectKBest(score_func=f_regression, k=min(top_k, X.shape[1]))
    selector.fit(X, y)

    scores = pd.DataFrame({
        "feature": X.columns,
        "f_score": selector.scores_,
        "p_value": selector.pvalues_
    }).sort_values("f_score", ascending=False)

    print(f"\n[F-TEST] Features scored by F-statistic:")
    print(scores.to_string(index=False))

    significant = scores[scores["p_value"] < 0.05]["feature"].tolist()
    print(f"\n[SELECTED] Statistically significant (p < 0.05): {significant}")
    return significant

def remove_low_variance(df: pd.DataFrame, threshold: float = 0.01) -> list:
    df_work = prepare_features(df)
    variances = df_work.drop(columns=["price"]).var()

    passed = variances[variances > threshold].index.tolist()
    rejected = variances[variances <= threshold].index.tolist()

    print(f"\n[VARIANCE] Features with sufficient variance: {passed}")
    if rejected:
        print(f"[VARIANCE] Rejected (near-zero variance): {rejected}")
    return passed

def find_redundant_features(df: pd.DataFrame, threshold: float = 0.95) -> list:
    df_work = prepare_features(df).dropna()
    corr_matrix = df_work.drop(columns=["price"]).corr().abs()

    redundant = set()
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i +1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if val >= threshold:
                redundant.add(cols[j])
                print(f"[REDUNDANT] '{cols[j]}' highly correlated with '{cols[i]}' ({val:.2f}) — candidate to drop")
   
    return list(redundant) 

def run_feature_selection(df: pd.DataFrame) -> dict:
    print("=" * 60)
    print("FEATURE SELECTION REPORT")
    print("=" * 60)

    corr_df = correlation_with_price(df)
    top_features = score_features_ftest(df, top_k=6)
    good_variances = remove_low_variance(df)
    redundant = find_redundant_features(df)

    final = [f for f in top_features if f in good_variances and f not in redundant]

    print("\n" + "=" * 60)
    print(f"[FINAL] Recommended features for analysis:")
    for f in final:
        print(f"   • {f}")
    print("=" * 60 )

    return {
        "correlation_table":  corr_df,
        "top_features_ftest": top_features,
        "good_variance": good_variances,
        "redundant_features": redundant,
        "final_recommended": final
    }

if __name__ =="__main__":
    df = pd.read_csv("../data/featured_cars.csv")
    results = run_feature_selection(df)
