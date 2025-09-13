import sys
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import *
import datetime

args = getResolvedOptions(sys.argv, ['JOB_NAME','RAW_BUCKET','CURATED_BUCKET','DB_NAME','AS_OF_DATE'])
as_of = args['AS_OF_DATE'] if args.get('AS_OF_DATE') else datetime.date.today().isoformat()

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

raw = f's3://{args["RAW_BUCKET"]}/ingest_date={as_of}/'

# Minimal read of CSVs (assumes files present; in prod use Glue Catalog)
customers = spark.read.option('header', True).csv(raw + 'customers.csv')
channel = spark.read.option('header', True).csv(raw + 'channel_events.csv')
trades = spark.read.option('header', True).csv(raw + 'trades.csv')
tickets = spark.read.option('header', True).csv(raw + 'tickets.csv')
marketing = spark.read.option('header', True).csv(raw + 'marketing.csv')
statements = spark.read.option('header', True).csv(raw + 'statements.csv')
auth = spark.read.option('header', True).csv(raw + 'auth_events.csv')

# Normalize timestamps
for df_name, df in [('channel', channel), ('trades', trades), ('tickets', tickets), ('marketing', marketing), ('auth', auth)]:
    if 'ts' in df.columns:
        locals()[df_name] = df.withColumn('ts', F.to_timestamp('ts'))
    if 'opened_ts' in df.columns:
        locals()[df_name] = locals()[df_name].withColumn('opened_ts', F.to_timestamp('opened_ts'))

# Write curated parquet partitioned by date
base = f's3://{args["CURATED_BUCKET"]}/dt={as_of}/'
customers.write.mode('overwrite').parquet(base + 'customers')
channel.write.mode('overwrite').parquet(base + 'channel_events')
trades.write.mode('overwrite').parquet(base + 'trades')
tickets.write.mode('overwrite').parquet(base + 'tickets')
marketing.write.mode('overwrite').parquet(base + 'marketing')
statements.write.mode('overwrite').parquet(base + 'statements')
auth.write.mode('overwrite').parquet(base + 'auth_events')

job.commit()
