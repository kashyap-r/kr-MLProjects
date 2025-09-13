import numpy as np

def scale01(x, lo, hi):
    if x is None:
        return 0.0
    try:
        return max(0.0, min(1.0, (float(x)-lo)/(hi-lo))) if hi>lo else 0.0
    except:
        return 0.0

def compute_cdi_row(row):
    # Simple explainable v1.0 formula
    chan = 0.5*scale01(row.get('pct_trades_digital_90d',0), 0, 1) +            0.5*scale01(row.get('app_logins_30d',0), 0, 30)
    prod = 0.6*row.get('e_statement_flag',0.0) + 0.4*row.get('robo_advice_flag',0.0)
    eng  = 0.5*scale01(row.get('webinars_90d',0), 0, 4) + 0.5*scale01(row.get('alerts_enabled_flag',0), 0, 1)
    svc  = scale01(row.get('pct_tickets_resolved_digital_90d',0), 0, 1)
    cdi_raw = 0.4*chan + 0.3*prod + 0.2*eng + 0.1*svc
    return int(round(100*cdi_raw))
