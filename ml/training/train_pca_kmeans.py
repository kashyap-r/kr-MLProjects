import os, io, json, boto3, tarfile, numpy as np, pandas as pd
from datetime import datetime, timedelta
from ml.common.utils import load_parquet_dir, save_df_to_s3_csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import joblib

TRAINING_DAYS = int(os.environ.get("TRAINING_DAYS","90"))

def list_partitions(features_root):
    # expects layout features_root/dt=YYYY-MM-DD/
    parts = []
    for name in os.listdir(features_root):
        if name.startswith("dt="):
            parts.append(name.split("=",1)[1])
    return sorted(parts)

def main():
    features_root = "/opt/ml/processing/input/features_all"
    out_dir = "/opt/ml/processing/output/artifacts"
    parts = list_partitions(features_root)
    if not parts:
        raise RuntimeError("No feature partitions found")
    # last N days by lexical date
    parts = parts[-min(TRAINING_DAYS, len(parts)):]
    dfs = []
    for p in parts:
        local = os.path.join(features_root, f"dt={p}")
        df = load_parquet_dir(local)
        if not df.empty:
            dfs.append(df)
    Xdf = pd.concat(dfs, ignore_index=True)
    feature_cols = [
        'pct_trades_digital_90d','app_logins_30d','e_statement_flag','robo_advice_flag',
        'webinars_90d','pct_tickets_resolved_digital_90d','avg_notional_90d'
    ]
    Xdf = Xdf.fillna(0.0)
    X = Xdf[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    pca = PCA(n_components=min(6, Xs.shape[1]), random_state=42).fit(Xs)
    Xp = pca.transform(Xs)
    k = 4
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048).fit(Xp)
    sil = silhouette_score(Xp, km.labels_) if len(set(km.labels_))>1 else -1.0

    # Save artifacts
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(pca, os.path.join(out_dir, "pca.pkl"))
    joblib.dump(km, os.path.join(out_dir, "kmeans.pkl"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"silhouette": float(sil), "k": k}, f)

if __name__ == "__main__":
    main()
