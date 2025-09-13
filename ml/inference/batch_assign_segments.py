import os, json, joblib, pandas as pd, numpy as np, boto3
from ml.common.utils import load_parquet_dir, save_df_to_s3_csv

s3 = boto3.client('s3')

def pandas_concat(dfs):
    import pandas as pd
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def load_prev_segments(scores_bucket, as_of_date):
    try:
        import datetime
        y = (datetime.date.fromisoformat(as_of_date) - datetime.timedelta(days=1)).isoformat()
        prefix = f'cdi_segments_daily/dt={y}/'
        objs = s3.list_objects_v2(Bucket=scores_bucket, Prefix=prefix)
        csvs = []
        if 'Contents' in objs:
            for o in objs['Contents']:
                if o['Key'].endswith('.csv'):
                    local = os.path.join('/opt/ml/processing/input/prev', os.path.basename(o['Key']))
                    os.makedirs(os.path.dirname(local), exist_ok=True)
                    s3.download_file(scores_bucket, o['Key'], local)
                    csvs.append(pd.read_csv(local))
        return pandas_concat(csvs)
    except Exception as e:
        print('No previous segments found:', e)
        return pd.DataFrame()

def main():
    features_dir = "/opt/ml/processing/input/features_today"
    cdi_dir = "/opt/ml/processing/input/cdi_today"
    out_dir = "/opt/ml/processing/output/segments"
    model_dir = "/opt/ml/processing/input/model_artifacts"

    as_of = os.environ.get('AS_OF_DATE')
    scores_bucket = os.environ.get('SCORES_BUCKET')
    hysteresis_delta = float(os.environ.get('HYSTERESIS_DELTA', '10.0'))  # CDI points

    # Load features
    feat = load_parquet_dir(features_dir)
    if feat.empty:
        raise RuntimeError("No features for today")

    # Load CDI
    dfs = []
    for root, _, files in os.walk(cdi_dir):
        for f in files:
            if f.endswith(".csv"):
                dfs.append(pd.read_csv(os.path.join(root, f)))
    if not dfs:
        raise RuntimeError("No CDI csv found")
    cdi = pandas_concat(dfs)

    df = feat.merge(cdi[['customer_id','cdi']], on='customer_id', how='left').fillna({'cdi':0})

    # Load model artifacts
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    pca = joblib.load(os.path.join(model_dir, "pca.pkl"))
    km = joblib.load(os.path.join(model_dir, "kmeans.pkl"))

    feature_cols = [
        'pct_trades_digital_90d','app_logins_30d','e_statement_flag','robo_advice_flag',
        'webinars_90d','pct_tickets_resolved_digital_90d','avg_notional_90d'
    ]
    X = df[feature_cols].to_numpy(dtype=float)
    Xs = scaler.transform(X)
    Xp = pca.transform(Xs)
    labels_new = km.predict(Xp)

    # Hysteresis against yesterday
    prev = load_prev_segments(scores_bucket, as_of)
    if not prev.empty:
        prev = prev[['customer_id','segment','cdi']].rename(columns={'segment':'segment_prev','cdi':'cdi_prev'})
        merged = df[['customer_id','cdi']].copy()
        merged = merged.merge(prev, on='customer_id', how='left')
        merged['label_new'] = labels_new
        # If label changes but CDI change is small (< hysteresis_delta), keep previous
        change_mask = (merged['segment_prev'].notna()) & (merged['segment_prev'] != merged['label_new'])
        small_delta = (merged['cdi'] - merged['cdi_prev']).abs() < hysteresis_delta
        keep_prev = change_mask & small_delta
        final_labels = []
        for i, row in merged.iterrows():
            if keep_prev.iloc[i]:
                final_labels.append(int(row['segment_prev']))
            else:
                final_labels.append(int(row['label_new']))
        labels_final = np.array(final_labels, dtype=int)
    else:
        labels_final = labels_new

    out = df[['customer_id','cdi']].copy()
    out['segment'] = labels_final
    save_df_to_s3_csv(out, out_dir)

if __name__ == "__main__":
    main()
