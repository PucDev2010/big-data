import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, expr, lit,
    to_timestamp, lower, regexp_extract,
    from_json
)
from pyspark.sql.types import *

# Cấu hình Logging để dễ theo dõi
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo Spark Session với cấu hình Cassandra
spark = SparkSession.builder \
    .appName("KafkaToCassandra_ETL") \
    .config("spark.jars.packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0") \
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
    .option("failOnDataLoss", "false") \
    .load()

#Parse JSON & ETL dữ liệu
parsed_df = raw_df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# Thực hiện các bước làm sạch và biến đổi dữ liệu
etl_df = (
    parsed_df
    .filter(col("job_title").isNotNull())
    .withColumn("event_time", to_timestamp("event_time"))

    # ===== SALARY =====
    .withColumn(
        "salary_avg",
        when(
            col("salary_min").isNotNull() & col("salary_max").isNotNull(),
            (col("salary_min") + col("salary_max")) / 2
        ).otherwise(lit(None))
    )

    # ===== CITY =====
    .withColumn(
        "city",
        when(
            (col("city") == "") | col("city").isNull(),
            lit("Unknown")
        ).otherwise(col("city"))
    )

    # ===== EXPERIENCE ETL =====
    .withColumn("exp_raw", lower(col("experience")))

    # ---- exp_min_year ----
    .withColumn(
        "exp_min_year",
        when(col("exp_raw").contains("không yêu cầu"), lit(None))
        .when(col("exp_raw").contains("chưa có"), lit(0.0))
        .when(col("exp_raw").contains("mới tốt nghiệp"), lit(0.0))
        .when(col("exp_raw").contains("lên đến"), lit(0.0))
        .when(
            col("exp_raw").contains("trên"),
            regexp_extract(col("exp_raw"), r"(\d+)", 1).cast("double")
        )
        .when(
            col("exp_raw").rlike(r"\d+\s*-\s*\d+"),
            regexp_extract(col("exp_raw"), r"(\d+)\s*-\s*(\d+)", 1).cast("double")
        )
        .otherwise(lit(None))
    )

    # ---- exp_max_year ----
    .withColumn(
        "exp_max_year",
        when(col("exp_raw").contains("không yêu cầu"), lit(None))
        .when(col("exp_raw").contains("chưa có"), lit(0.0))
        .when(col("exp_raw").contains("mới tốt nghiệp"), lit(1.0))
        .when(
            col("exp_raw").contains("lên đến"),
            regexp_extract(col("exp_raw"), r"(\d+)", 1).cast("double")
        )
        .when(col("exp_raw").contains("trên"), lit(None))
        .when(
            col("exp_raw").rlike(r"\d+\s*-\s*\d+"),
            regexp_extract(col("exp_raw"), r"(\d+)\s*-\s*(\d+)", 2).cast("double")
        )
        .otherwise(lit(None))
    )

    # ---- exp_avg_year ----
    .withColumn(
        "exp_avg_year",
        when(
            col("exp_min_year").isNotNull() & col("exp_max_year").isNotNull(),
            (col("exp_min_year") + col("exp_max_year")) / 2
        ).otherwise(lit(None))
    )

    # ---- exp_type ----
    .withColumn(
        "exp_type",
        when(col("exp_raw").contains("không yêu cầu"), lit("no_requirement"))
        .when(col("exp_raw").contains("chưa có"), lit("no_experience"))
        .when(col("exp_raw").contains("mới tốt nghiệp"), lit("fresh_graduate"))
        .when(col("exp_raw").contains("lên đến"), lit("upper_bound"))
        .when(col("exp_raw").contains("trên"), lit("lower_bound"))
        .when(col("exp_raw").rlike(r"\d+\s*-\s*\d+"), lit("range"))
        .otherwise(lit("unknown"))
    )

    # ===== ID =====
    .withColumn("id", expr("uuid()"))
)

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