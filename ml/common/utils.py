import json, math, os, io, sys, boto3, tarfile, datetime
import numpy as np
import pandas as pd
from typing import Tuple

CW_NAMESPACE = "CDI"

def put_metric(metric_name, value, unit="None", dims=None):
    cw = boto3.client("cloudwatch")
    dims = dims or []
    cw.put_metric_data(
        Namespace=CW_NAMESPACE,
        MetricData=[{
            "MetricName": metric_name,
            "Value": float(value),
            "Unit": unit,
            "Dimensions": dims
        }]
    )

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected_perc, edges = np.histogram(expected, bins=bins)
    actual_perc, _ = np.histogram(actual, bins=edges)
    expected_perc = expected_perc / max(expected_perc.sum(), 1)
    actual_perc = actual_perc / max(actual_perc.sum(), 1)
    eps = 1e-6
    psi_vals = (actual_perc - expected_perc) * np.log((actual_perc + eps) / (expected_perc + eps))
    return float(np.sum(psi_vals))

def load_parquet_dir(local_dir: str) -> pd.DataFrame:
    import pyarrow.parquet as pq
    import pyarrow as pa
    tables = []
    for root, _, files in os.walk(local_dir):
        for f in files:
            if f.endswith(".parquet"):
                tables.append(pq.read_table(os.path.join(root, f)))
    if not tables:
        return pd.DataFrame()
    return pq.concat_tables(tables).to_pandas()

def save_df_to_s3_csv(df: pd.DataFrame, out_dir: str, filename: str = "part-00000.csv"):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, filename)
    df.to_csv(p, index=False)
