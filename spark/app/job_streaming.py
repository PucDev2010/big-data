import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, expr, lit,
    to_timestamp, lower, regexp_extract, regexp_replace,
    from_json, coalesce
)
from pyspark.sql.types import *

# ==========================================
# 1. CẤU HÌNH & KHỞI TẠO
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

spark = SparkSession.builder \
    .appName("KafkaToCassandra_ETL_NoHotColumn") \
    .config("spark.cassandra.connection.host", "cassandra") \
    .config("spark.cassandra.connection.port", "9042") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ==========================================
# 2. ĐỊNH NGHĨA SCHEMA
# ==========================================
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

# ==========================================
# 3. ĐỌC DỮ LIỆU TỪ KAFKA
# ==========================================
logger.info("Dang khoi tao doc du lieu tu Kafka...")
raw_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "job_postings") \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load()

parsed_df = raw_df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# ==========================================
# 4. ETL & CLEANING
# ==========================================
etl_df = (
    parsed_df
    .filter(col("job_title").isNotNull())
    .withColumn("event_time", to_timestamp("event_time"))

    # ==================================================
    # A. XỬ LÝ LƯƠNG (CHUẨN HÓA VỀ ĐƠN VỊ TRIỆU VNĐ)
    # ==================================================
    .withColumn("salary_clean", lower(col("salary")))
    .withColumn("raw_min", regexp_extract(col("salary_clean"), r"(\d+[.,\d]*)", 1))
    .withColumn("raw_max", regexp_extract(col("salary_clean"), r"-\s*(\d+[.,\d]*)", 1))
    
    # Xử lý dấu phân cách
    .withColumn(
        "val_min", 
        when(col("salary_clean").rlike("triệu|tr|m"), col("raw_min").cast("double"))
        .otherwise(regexp_replace(col("raw_min"), r"[.,]", "").cast("double"))
    )
    .withColumn(
        "val_max", 
        when(col("salary_clean").rlike("triệu|tr|m"), col("raw_max").cast("double"))
        .otherwise(regexp_replace(col("raw_max"), r"[.,]", "").cast("double"))
    )
    
    # Logic quy đổi đơn vị (USD/VND) -> Triệu
    .withColumn(
        "salary_min_final", 
        when(col("salary_clean").rlike("usd|\\$"), (col("val_min") * 25) / 1000)
        .when(col("val_min") >= 1000, col("val_min") / 1000000)
        .when((col("val_min") > 100) & (col("val_min") < 1000), col("val_min") / 1000)
        .otherwise(col("val_min"))
    )
    .withColumn(
        "salary_max_final", 
        when(col("salary_clean").rlike("usd|\\$"), (col("val_max") * 25) / 1000)
        .when(col("val_max") >= 1000, col("val_max") / 1000000)
        .when((col("val_max") > 100) & (col("val_max") < 1000), col("val_max") / 1000)
        .otherwise(col("val_max"))
    )

    # Gán vào cột chính
    .withColumn("salary_min", col("salary_min_final"))
    .withColumn("salary_max", col("salary_max_final"))
    .withColumn(
        "salary_avg", 
        when(col("salary_min").isNotNull() & col("salary_max").isNotNull(), (col("salary_min") + col("salary_max")) / 2)
        .when(col("salary_min").isNotNull(), col("salary_min"))
        .otherwise(lit(0.0))
    )

    # ==================================================
    # B. XỬ LÝ KINH NGHIỆM
    # ==================================================
    .withColumn("exp_raw", lower(col("experience")))
    
    .withColumn("exp_min_year",
        when(col("exp_raw").contains("không yêu cầu"), lit(None))
        .when(col("exp_raw").rlike("chưa có|mới tốt nghiệp|intern"), lit(0.0))
        .when(col("exp_raw").rlike(r"(từ|from|at least|tối thiểu|min)\s*(\d+)"), regexp_extract(col("exp_raw"), r"(?:từ|from|at least|tối thiểu|min)\s*(\d+)", 1).cast("double"))
        .when(col("exp_raw").rlike(r"(\d+)\s*\+"), regexp_extract(col("exp_raw"), r"(\d+)", 1).cast("double"))
        .when(col("exp_raw").rlike(r"(\d+)\s*(năm|year|yoe|kn)"), regexp_extract(col("exp_raw"), r"(\d+)", 1).cast("double"))
        .when(col("exp_raw").rlike(r"\d+\s*-\s*\d+"), regexp_extract(col("exp_raw"), r"(\d+)\s*-\s*(\d+)", 1).cast("double"))
        .otherwise(lit(None))
    )
    .withColumn("exp_max_year",
        when(col("exp_raw").rlike(r"\d+\s*-\s*\d+"), regexp_extract(col("exp_raw"), r"(\d+)\s*-\s*(\d+)", 2).cast("double"))
        .otherwise(lit(None))
    )
    
    # Tính trung bình & Lọc nhiễu
    .withColumn("exp_temp", coalesce(col("exp_min_year"), lit(0.0)))
    .withColumn("exp_avg_year", 
         when(col("exp_temp") > 40, lit(None)).otherwise(col("exp_temp"))
    )
    
    # Tạo exp_type (Giữ nguyên logic cũ cho đầy đủ)
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

    # ==================================================
    # C. DỌN DẸP & ID
    # ==================================================
    .withColumn("city", when((col("city") == "") | col("city").isNull(), lit("Unknown")).otherwise(col("city")))
    .withColumn("id", expr("uuid()"))
    # Xóa cột tạm
    .drop("salary_clean", "raw_min", "raw_max", "val_min", "val_max", "salary_min_final", "salary_max_final", "exp_raw", "exp_temp")
)

# ==========================================
# 5. GHI VÀO CASSANDRA
# ==========================================
logger.info("Ghi du lieu vao Cassandra...")

# Lưu ý: Đổi checkpointLocation để tránh xung đột với lần chạy trước
query = etl_df.writeStream \
    .format("org.apache.spark.sql.cassandra") \
    .option("keyspace", "job_analytics") \
    .option("table", "job_postings") \
    .option("checkpointLocation", "/opt/spark/work-dir/checkpoints/cassandra_live_v4") \
    .outputMode("append") \
    .start()
logger.info("Da ghi du lieu vao Cassandra...")

query.awaitTermination()