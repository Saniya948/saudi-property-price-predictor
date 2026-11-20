"""Train and compare regression models for property price/rent prediction.

Usage:
    python train.py --data data/SA_Aqar.csv --target price

Cleaning steps (all printed so you can report them in the README):
  1. drop free-text columns (default: details)
  2. strip whitespace from text columns
  3. drop exact duplicate listings (prevents train/test leakage)
  4. trim extreme outliers in the target and in `size` (1st-99th percentile)
The target is modelled on a log scale, but metrics are reported in original units.

Saves the best model to model.joblib and the results table to results.csv.
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean(df, target, drop_cols, size_col="size"):
    n0 = len(df)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    for c in df.select_dtypes(exclude="number").columns:
        df[c] = df[c].astype(str).str.strip()
    df = df.dropna(subset=[target])
    df = df.drop_duplicates()
    n1 = len(df)
    for col in [target, size_col]:
        if col in df.columns:
            lo, hi = df[col].quantile([0.01, 0.99])
            df = df[df[col].between(lo, hi)]
    print(f"Rows: {n0} raw -> {n1} after de-duplication -> {len(df)} after outlier trimming")
    return df


def build_preprocessor(X):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", default="price")
    ap.add_argument("--drop", nargs="*", default=["details"],
                    help="columns to drop (free text, IDs)")
    args = ap.parse_args()

    df = clean(pd.read_csv(args.data), args.target, args.drop)
    X, y = df.drop(columns=[args.target]), df[args.target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    # Baseline: always predict the training median. Real models must beat this.
    base = np.full(len(y_test), y_train.median())
    rows = [{"model": "Baseline (median)",
             "MAE": mean_absolute_error(y_test, base),
             "RMSE": mean_squared_error(y_test, base) ** 0.5,
             "R2": r2_score(y_test, base)}]

    best_name, best_pipe, best_r2 = None, None, -np.inf
    for name, model in models.items():
        reg = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
        pipe = Pipeline([("prep", build_preprocessor(X)), ("model", reg)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        r2 = r2_score(y_test, pred)
        rows.append({"model": name,
                     "MAE": mean_absolute_error(y_test, pred),
                     "RMSE": mean_squared_error(y_test, pred) ** 0.5,
                     "R2": r2})
        if r2 > best_r2:
            best_name, best_pipe, best_r2 = name, pipe, r2

    results = pd.DataFrame(rows).sort_values("R2", ascending=False).round(3)
    print(results.to_string(index=False))
    results.to_csv("results.csv", index=False)

    # Remember the allowed values for each text column so the app can offer a dropdown.
    cats = {c: sorted(X[c].unique().tolist()) for c in X.select_dtypes(exclude="number").columns}
    defaults = X.select_dtypes(include="number").median().to_dict()
    joblib.dump({"pipeline": best_pipe, "columns": X.columns.tolist(),
                 "categories": cats, "defaults": defaults}, "model.joblib")
    print(f"\nSaved best model: {best_name} (R2 = {best_r2:.3f})")


if __name__ == "__main__":
    main()
