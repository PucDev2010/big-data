import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp, when, expr
from pyspark.sql.types import *

# Cấu hình Logging để dễ theo dõi
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo Spark Session với cấu hình Cassandra
spark = SparkSession.builder \
    .appName("KafkaToCassandra_ETL") \
    .config("spark.cassandra.connection.host", "cassandra") \
    .config("spark.cassandra.connection.port", "9042") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

#Định nghĩa Schema (Cấu trúc dữ liệu JSON từ Kafka)
schema = StructType([
    StructField("job_title", StringType()),
    StructField("job_type", StringType()),
    StructField("position_level", StringType()),
    StructField("city", StringType()),
    StructField("experience", StringType()),
    StructField("skills", StringType()),
    StructField("job_fields", StringType()),
    StructField("salary", StringType()),
    StructField("salary_min", DoubleType()),
    StructField("salary_max", DoubleType()),
    StructField("unit", StringType()),
    StructField("event_time", StringType()),
    StructField("event_type", StringType())
])

#Đọc dữ liệu từ Kafka (Stream)
logger.info("Dang khoi tao doc du lieu tu Kafka...")
raw_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "job_postings") \
    .option("startingOffsets", "latest") \
    .load()

#Parse JSON & ETL dữ liệu
parsed_df = raw_df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# Thực hiện các bước làm sạch và biến đổi dữ liệu
etl_df = parsed_df \
    .filter(col("job_title").isNotNull()) \
    .withColumn("event_time", to_timestamp("event_time")) \
    .withColumn("salary_avg", (col("salary_min") + col("salary_max")) / 2) \
    .withColumn("city", when(col("city") == "", "Unknown").otherwise(col("city"))) \
    .fillna(0, subset=["salary_min", "salary_max", "salary_avg"]) \
    .withColumn("id", expr("uuid()"))  # Tự sinh ID ngẫu nhiên (UUID)

#Ghi dữ liệu vào Cassandra
logger.info("Ghi du lieu vao Cassandra...")

#'checkpointLocation'để đảm bảo không mất dữ liệu khi restart
query = etl_df.writeStream \
    .format("org.apache.spark.sql.cassandra") \
    .option("keyspace", "job_analytics") \
    .option("table", "job_postings") \
    .option("checkpointLocation", "/opt/spark/work-dir/checkpoints/cassandra_uuid") \
    .outputMode("append") \
    .start()

query.awaitTermination()