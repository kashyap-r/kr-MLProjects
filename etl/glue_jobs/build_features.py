import sys, datetime
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.window import Window

args = getResolvedOptions(sys.argv, ['JOB_NAME','CURATED_BUCKET','FEATURES_BUCKET','AS_OF_DATE'])
as_of = args['AS_OF_DATE'] if args.get('AS_OF_DATE') else datetime.date.today().isoformat()

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

cur = f's3://{args["CURATED_BUCKET"]}/dt={as_of}/'
customers = spark.read.parquet(cur + 'customers')
channel = spark.read.parquet(cur + 'channel_events')
trades = spark.read.parquet(cur + 'trades')
tickets = spark.read.parquet(cur + 'tickets')
marketing = spark.read.parquet(cur + 'marketing')
statements = spark.read.parquet(cur + 'statements')

# Simple aggregates (last 90d assumed in curated slice for sample)
trades = trades.withColumn('is_digital', F.when(F.col('channel').isin('web','app'), F.lit(1)).otherwise(F.lit(0)))
by_cust = trades.groupBy('customer_id').agg(
    F.count('*').alias('trades_total_90d'),
    F.sum('is_digital').alias('trades_digital_90d'),
    F.avg(F.col('notional').cast('double')).alias('avg_notional_90d')
).withColumn('pct_trades_digital_90d', F.when(F.col('trades_total_90d')>0, F.col('trades_digital_90d')/F.col('trades_total_90d')).otherwise(F.lit(0.0)))

logins = channel.filter(F.col('event')=='login').groupBy('customer_id').agg(F.count('*').alias('app_logins_30d'))
webinars = marketing.filter(F.col('event')=='webinar_attended').groupBy('customer_id').agg(F.count('*').alias('webinars_90d'))
e_stmt = statements.withColumn('e_statement_flag', F.when(F.col('delivery_mode')=='electronic', F.lit(1.0)).otherwise(F.lit(0.0))).select('customer_id','e_statement_flag')

# Resolve digital ticket resolution rate (dummy for sample)
tickets = tickets.withColumn('resolved_via_digital', F.when(F.col('resolved_via').isin('app','web','chat'), 1).otherwise(0))
svc = tickets.groupBy('customer_id').agg(
    F.sum('resolved_via_digital').alias('tickets_resolved_digital_90d'),
    F.count('*').alias('tickets_total_90d')
).withColumn('pct_tickets_resolved_digital_90d', F.when(F.col('tickets_total_90d')>0, F.col('tickets_resolved_digital_90d')/F.col('tickets_total_90d')).otherwise(F.lit(0.0)))

features = customers.select('customer_id','has_robo_advice').withColumn('robo_advice_flag', F.col('has_robo_advice').cast('double'))     .join(by_cust, 'customer_id', 'left')     .join(logins, 'customer_id', 'left')     .join(webinars, 'customer_id', 'left')     .join(e_stmt, 'customer_id', 'left')     .join(svc, 'customer_id', 'left')     .fillna({'app_logins_30d':0, 'webinars_90d':0, 'e_statement_flag':0.0, 'robo_advice_flag':0.0,
             'pct_tickets_resolved_digital_90d':0.0, 'pct_trades_digital_90d':0.0})

features = features.withColumn('asof_date', F.lit(as_of))

out = f's3://{args["FEATURES_BUCKET"]}/dt={as_of}/'
features.write.mode('overwrite').parquet(out)
job.commit()
