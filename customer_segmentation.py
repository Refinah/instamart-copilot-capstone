import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib

def run_segmentation():
    clean_dir = "data/clean"
    orders = pd.read_csv(os.path.join(clean_dir, "orders.csv"))
    customers = pd.read_csv(os.path.join(clean_dir, "customers.csv"))

    # Completed orders only for RFM
    comp = orders[orders["status"] == "completed"].copy()
    snapshot = pd.to_datetime("2025-10-01 00:00:00")
    comp["order_ts"] = pd.to_datetime(comp["order_ts"])

    rfm = comp.groupby("customer_id").agg(
        recency=("order_ts", lambda x: (snapshot - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()

    # K-means clustering evaluation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

    print("K-means Inertia & Silhouette Scores:")
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        print(f"K={k}: Inertia={km.inertia_:.1f}")

    # Choose K=4
    optimal_k = 4
    km_final = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
    rfm["cluster"] = km_final.fit_predict(X_scaled)

    # Profile segments to assign business names
    profile = rfm.groupby("cluster").agg(
        size=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean")
    ).reset_index()
    print("\nCluster Profiles:\n", profile)

    # Identify Premium segment (highest mean monetary)
    max_monetary_cluster = profile.loc[profile["mean_monetary"].idxmax(), "cluster"]

    def name_segment(c):
        if c == max_monetary_cluster:
            return "VIP Champions"
        elif profile.loc[profile["cluster"] == c, "mean_frequency"].values[0] > 10:
            return "Frequent Buyers"
        elif profile.loc[profile["cluster"] == c, "mean_recency"].values[0] > 60:
            return "At-Risk Lapsers"
        else:
            return "Casual Shoppers"

    rfm["segment_name"] = rfm["cluster"].apply(name_segment)
    rfm["is_premium"] = rfm["cluster"] == max_monetary_cluster
    rfm["segment_id"] = rfm["cluster"]

    # PCA projection
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_scaled)
    print("PCA Explained Variance Ratio:", pca.explained_variance_ratio_)

    # Save outputs
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv(
        os.path.join(out_dir, "customer_segments.csv"), index=False
    )
    print("Saved outputs/customer_segments.csv")

if __name__ == "__main__":
    run_segmentation()