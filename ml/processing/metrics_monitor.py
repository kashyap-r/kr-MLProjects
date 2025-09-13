import os, json, boto3, datetime, pandas as pd, numpy as np
from ml.common.utils import load_parquet_dir, put_metric, psi

s3 = boto3.client('s3')

def load_csvs_from_s3_prefix(bucket, prefix, local_root):
    os.makedirs(local_root, exist_ok=True)
    objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    rows = []
    if 'Contents' not in objs:
        return pd.DataFrame()
    for obj in objs['Contents']:
        key = obj['Key']
        if key.endswith('.csv'):
            local = os.path.join(local_root, os.path.basename(key))
            s3.download_file(bucket, key, local)
            rows.append(pd.read_csv(local))
    return pd.concat(rows) if rows else pd.DataFrame()

def main():
    as_of = os.environ.get('AS_OF_DATE')
    scores_bucket = os.environ.get('SCORES_BUCKET')
    seg_today_dir = '/opt/ml/processing/input/segments_today'

    # Load today's segments (csv or parquet)
    dfs = []
    for root, _, files in os.walk(seg_today_dir):
        for f in files:
            if f.endswith('.csv'):
                dfs.append(pd.read_csv(os.path.join(root, f)))
    today = pd.concat(dfs) if dfs else load_parquet_dir(seg_today_dir)
    if today.empty:
        raise RuntimeError('No segments data found')

    # Emit basic stats
    cdi_mean = float(today['cdi'].mean())
    cdi_std  = float(today['cdi'].std(ddof=0))
    put_metric('CDI_Mean', cdi_mean)
    put_metric('CDI_StdDev', cdi_std)
    for seg, cnt in today['segment'].value_counts().items():
        put_metric('Segment_Size', cnt, dims=[{'Name':'segment','Value':str(seg)}])

    # Compare with yesterday if available
    try:
        dt = datetime.date.fromisoformat(as_of)
        yday = (dt - datetime.timedelta(days=1)).isoformat()
        y_prefix = f'cdi_segments_daily/dt={yday}/'
        ydf = load_csvs_from_s3_prefix(scores_bucket, y_prefix, '/opt/ml/processing/input/segments_yday')
        if not ydf.empty:
            # PSI on CDI distribution
            psi_cdi = psi(ydf['cdi'].to_numpy(), today['cdi'].to_numpy(), bins=10)
            put_metric('PSI_CDI', psi_cdi)
            # Segment churn (% changed labels for overlapping customers)
            merged = today.merge(ydf[['customer_id','segment']].rename(columns={'segment':'segment_prev'}), on='customer_id', how='inner')
            if not merged.empty:
                churn = float((merged['segment'] != merged['segment_prev']).mean())
                put_metric('Segment_Churn', churn)
    except Exception as e:
        print('Drift calc skipped:', e)

    # Mark run completion
    put_metric('RunCompleted', 1)

if __name__ == '__main__':
    main()
