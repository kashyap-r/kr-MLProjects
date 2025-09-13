# CDI Segmentation (Batch) — End-to-End on AWS

This repository contains a complete *batch* pipeline to compute a **Customer Digitalization Index (CDI)**,
segment customers using **PCA + MiniBatch K-Means**, and publish daily scores to S3 — with monitoring hooked to **CloudWatch** and **(optional) Amazon Managed Grafana**.

## Major components
- **CloudFormation (IaC)**: S3 buckets, KMS, IAM, Glue DB, roles, Step Functions, AMG workspace (optional)
- **Glue (PySpark)**: raw→curated parquet, rolling window features
- **SageMaker Processing (scikit-learn image)**: CDI computation, model training, batch inference, metrics
- **Step Functions**: orchestrates daily batch
- **CloudWatch**: custom metrics (CDI mean/std, segment sizes), alarms
- **GitHub Actions (OIDC)**: deploy CFN, upload code artifacts to S3, kick the pipeline
- **Sample Data**: minimal CSVs under `data/sample/`

> ⚠️ You will need to set parameters (account, region, repo, branch, etc.). See _Deploy_ below.

---

## Deploy (high level)
1. **Create GitHub repo** and push this project.
2. **Foundation stack**: creates buckets, KMS, roles, OIDC trust, Grafana workspace (optional).
3. **Upload artifacts** (handled by GitHub Action) to the Artifact bucket.
4. **Pipeline stack**: creates Glue Jobs & Step Functions (points to your uploaded scripts).
5. **Run** the Step Functions execution to process the latest partitions.

See `.github/workflows/deploy.yml` for the automated path. You’ll need to set repo variables/secrets per the comments.

---

## Buckets (per environment)
- `<env>-cdi-raw` — raw CSV landing
- `<env>-cdi-curated` — curated parquet
- `<env>-cdi-features` — engineered features parquet
- `<env>-cdi-scores` — CDI scores & segments (published outputs)
- `<env>-cdi-artifacts` — code artifacts uploaded by CI/CD

All are KMS-encrypted.

---

## Step Functions (daily run)
1. **CurateRaw** (Glue)
2. **BuildFeatures** (Glue)
3. **ComputeCDI** (SM Processing)
4. **TrainModel** (SM Processing) — uses rolling ~90d features by default
5. **BatchInference** (SM Processing) — assigns segments, publishes to S3
6. **ComputeMetrics** (SM Processing) — pushes CloudWatch metrics
7. **(Optional)** Crawler/Partition updates / Exports

---

## Amazon Managed Grafana
A basic workspace resource is provisioned. Import `grafana/dashboard.json` into the workspace and set the **CloudWatch** data source.

---

## Notes
- The SageMaker image URI is a parameter. Use the **scikit-learn** CPU image for your region, e.g.:
  - `763104351884.dkr.ecr.<region>.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py310`
- The pipeline uses *SageMaker Processing Jobs* for both training and inference to avoid extra service types.
- All code is reference-grade; adjust for your org’s standards, VPC config, and security policies.


---

## Monitoring & Alarms (added)
- **RunCompletedAlarm**: alarms until the daily job pushes metric `RunCompleted=1` (TreatMissingData=breaching). If the pipeline doesn't finish, the alarm stays in ALARM.
- **PsiCdiAlarm**: alarms if `PSI_CDI > 0.2` comparing today's CDI vs yesterday's.
- **SegmentChurnAlarm**: alarms if daily segment churn > 25% (vs yesterday).

## Hysteresis (segment stability)
Batch inference now compares with yesterday:
- If the new segment differs but **|CDI_today - CDI_yesterday| < 10**, we **keep yesterday's segment** to reduce flip-flops.
- Override via env var `HYSTERESIS_DELTA` on the Processing job.


---

## Prereqs
- **AWS account** with permissions for CloudFormation, IAM, S3, Glue, SageMaker, Step Functions, CloudWatch, Athena, SNS.
- **GitHub** repo (Actions enabled).
- **AWS CLI v2** installed locally (optional) and configured (`aws configure sso` recommended).
- (Optional) **Amazon Managed Grafana** access (stack creates workspace).
- (Optional) **AWS Chatbot** linked to your Slack workspace (retrieve `SlackWorkspaceId`, `SlackChannelId`, and a role ARN).

## Variables you’ll set
- In **GitHub → Settings → Secrets and variables → Actions**:
  - Secrets: `AWS_ACCOUNT_ID`
  - Variables (optional): 
    - `QS_PRINCIPAL_ARN` (to auto-deploy QuickSight dataset)
    - `UPLOAD_SAMPLE_DATA` = `true` to push sample CSVs during deploy
- Workflow defaults: `AWS_REGION` (currently `ap-south-1`), schedule is 01:00 IST.

## One-time deploy (from GitHub Actions)
1. Create a GitHub repo and push this project.
2. Set **secret** `AWS_ACCOUNT_ID` in the repo.
3. (Optional) Decide alerting:
   - Email: run the Action with parameter `AlertEmail=you@example.com` (see below).
   - Slack: set `SlackWorkspaceId`, `SlackChannelId`, and `ChatbotIAMRoleArn` when deploying the foundation stack.
4. Run **Deploy CDI Pipeline** workflow (use `env=dev`).
5. If `UPLOAD_SAMPLE_DATA=true`, the action will load sample CSVs under today’s partition automatically. Otherwise, upload them manually:
   ```bash
   aws s3 cp --recursive data/sample/ s3://dev-cdi-raw-<acct>-<region>/ingest_date=$(date +%F)/
   ```
6. The workflow kicks a Step Functions execution for today.

## Verify
- **S3**: outputs under `.../cdi_daily/dt=YYYY-MM-DD/` and `.../cdi_segments_daily/dt=YYYY-MM-DD/`.
- **Athena**: run `SELECT * FROM dev_cdi_db.v_cdi_segments LIMIT 10;` (workgroup: `CDIWorkGroup`).
- **CloudWatch**: metrics in namespace `CDI` (CDI_Mean/StdDev, Segment_Size, PSI_CDI, Segment_Churn).
- **Alarms**: confirm `RunCompletedAlarm` flips to OK after the first successful run.
- **Grafana**: import `grafana/dashboard.json`, add CloudWatch data source; view CDI charts.
- **Notifications**: email/SNS subscription confirmation; Slack channel (if configured) gets alarm notifications.

