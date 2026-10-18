import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import joblib

def run_lapse_model():
    clean_dir = "data/clean"
    orders = pd.read_csv(os.path.join(clean_dir, "orders.csv"))
    customers = pd.read_csv(os.path.join(clean_dir, "products.csv")) # placeholder or load products/customers
    customers = pd.read_csv(os.path.join(clean_dir, "customers.csv"))
    products = pd.read_csv(os.path.join(clean_dir, "products.csv"))

    cutoff = pd.to_datetime("2025-09-01")
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])

    # Pre-cutoff orders
    pre_orders = orders[orders["order_ts"] < cutoff].copy()
    cohort_cids = pre_orders["customer_id"].unique()

    # Define label: lapsed = 1 if NO completed order between 2025-09-01 and 2025-09-30
    sep_completed = orders[(orders["order_ts"] >= cutoff) & (orders["order_ts"] < pd.to_datetime("2025-10-01")) & (orders["status"] == "completed")]["customer_id"].unique()
    
    labels = []
    for cid in cohort_cids:
        lapsed = 0 if cid in sep_completed else 1
        labels.append({"customer_id": cid, "lapsed": lapsed})
    labels_df = pd.DataFrame(labels)

    # Feature engineering from pre-cutoff orders
    prod_cat = products.set_index("product_id")["category"].to_dict()
    pre_orders["category"] = pre_orders["product_id"].map(prod_cat)

    feat = pre_orders.groupby("customer_id").agg(
        recency=("order_ts", lambda x: (cutoff - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", lambda x: x[pre_orders["status"] == "completed"].sum()),
        avg_discount=("discount_pct", "mean"),
        avg_delivery=("delivery_minutes", "mean")
    ).reset_index()

    # Cancellation rate
    cancel_counts = pre_orders[pre_orders["status"].isin(["completed", "cancelled"])].groupby("customer_id").agg(
        total_orders=("order_id", "count"),
        cancelled_orders=("status", lambda x: (x == "cancelled").sum())
    ).reset_index()
    cancel_counts["cancel_rate"] = cancel_counts["cancelled_orders"] / cancel_counts["total_orders"]
    
    # Favorite category
    fav_cat = pre_orders.groupby(["customer_id", "category"]).size().reset_index(name="cnt")
    fav_cat = fav_cat.sort_values(["customer_id", "cnt"], ascending=[True, False]).drop_duplicates("customer_id")
    fav_cat = fav_cat.rename(columns={"category": "fav_category"})[["customer_id", "fav_category"]]

    # Merge features
    df_model = labels_df.merge(feat, on="customer_id").merge(cancel_counts[["customer_id", "cancel_rate"]], on="customer_id").merge(fav_cat, on="customer_id", how="left")
    df_model = df_model.merge(customers, on="customer_id", how="left")
    df_model["tenure"] = (cutoff - pd.to_datetime(df_model["signup_date"])).dt.days

    features = ["recency", "frequency", "monetary", "avg_discount", "avg_delivery", "cancel_rate", "fav_category", "city", "age", "gender", "tenure"]
    X = df_model[features]
    y = df_model["lapsed"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    numeric_features = ["recency", "frequency", "monetary", "avg_discount", "avg_delivery", "cancel_rate", "age", "tenure"]
    categorical_features = ["fav_category", "city", "gender"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
        ]
    )

    # Logistic Regression Pipeline & Tuning
    lr_pipeline = Pipeline(steps=[("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
    param_grid_lr = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    gs_lr = GridSearchCV(lr_pipeline, param_grid_lr, cv=5, scoring="f1")
    gs_lr.fit(X_train, y_train)

    # Random Forest Pipeline & Tuning
    rf_pipeline = Pipeline(steps=[("prep", preprocessor), ("clf", RandomForestClassifier(random_state=42))])
    param_grid_rf = {
        "clf__n_estimators": [100],
        "clf__max_depth": [3, 5, 8],
        "clf__min_samples_leaf": [5, 10, 20]
    }
    gs_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=5, scoring="f1")
    gs_rf.fit(X_train, y_train)

    print("Best LR F1:", gs_lr.best_score_, gs_lr.best_params_)
    print("Best RF F1:", gs_rf.best_score_, gs_rf.best_params_)

    best_model = gs_rf if gs_rf.best_score_ > gs_lr.best_score_ else gs_lr

    # Evaluation on test set
    y_pred = best_model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    print(f"Test Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

    # Save model
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(best_model, "outputs/lapse_model.joblib")

    # Retention list for premium customers
    segments = pd.read_csv("outputs/customer_segments.csv")
    test_cids = X_test.index
    test_df = df_model.loc[test_cids].merge(segments[["customer_id", "is_premium"]], on="customer_id")
    prem_test = test_df[test_df["is_premium"] == True].copy()
    
    if len(prem_test) > 0:
        probs = best_model.predict_proba(prem_test[features])[:, 1]
        prem_test["lapse_probability"] = probs
        prem_test["actually_lapsed"] = prem_test["lapsed"]
        retention_list = prem_test.sort_values(by="lapse_probability", ascending=False).head(10)
        retention_list[["customer_id", "lapse_probability", "actually_lapsed"]].to_csv("outputs/retention_list.csv", index=False)
        print("Saved outputs/retention_list.csv")

if __name__ == "__main__":
    run_lapse_model()
    