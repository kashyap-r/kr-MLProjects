import os, json, pandas as pd
from ml.common.utils import load_parquet_dir, save_df_to_s3_csv
from ml.common.cdi_formula import compute_cdi_row

AS_OF = os.environ.get("AS_OF_DATE")

def main():
    features_dir = "/opt/ml/processing/input/features"
    out_dir = "/opt/ml/processing/output/cdi"
    df = load_parquet_dir(features_dir)
    if df.empty:
        raise RuntimeError("No features found")
    # add alerts_enabled_flag if not present
    if 'alerts_enabled_flag' not in df.columns:
        df['alerts_enabled_flag'] = 0.0
    df['cdi'] = df.apply(lambda r: compute_cdi_row(r), axis=1)
    out = df[['customer_id','asof_date','cdi']]
    save_df_to_s3_csv(out, out_dir)

if __name__ == "__main__":
    main()
