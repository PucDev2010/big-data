from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

# ====================================================
# 1. KHỞI TẠO SPARK
# ====================================================
spark = SparkSession.builder \
    .appName("JobAttractiveness_Test_Model") \
    .config("spark.cassandra.connection.host", "cassandra") \
    .config("spark.cassandra.connection.port", "9042") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print(">>> Khởi tạo Spark thành công")

# ====================================================
# 2. LOAD MODEL ĐÃ TRAIN
# ====================================================
model_path = "/opt/spark/work-dir/models/job_attractiveness_lr_v1"
print(f">>> Load model từ: {model_path}")

model = PipelineModel.load(model_path)
print(">>> Load model thành công")

# ====================================================
# 3. ĐỌC DỮ LIỆU TỪ CASSANDRA
# ====================================================
print(">>> Đang đọc dữ liệu từ Cassandra...")

df = spark.read \
    .format("org.apache.spark.sql.cassandra") \
    .options(table="job_postings", keyspace="job_analytics") \
    .load()

print(f">>> Tổng số bản ghi: {df.count()}")

# ====================================================
# 4. TIỀN XỬ LÝ NHẸ (KHỚP PIPELINE)
# ====================================================
df = df.filter(col("job_title").isNotNull())

df = df.fillna({
    "skills": "",
    "job_fields": "",
    "job_title": "",
    "city": "Unknown",
    "position_level": "Unknown",
    "experience": ""
})

# ====================================================
# 5. CHẠY DỰ ĐOÁN
# ====================================================
print(">>> Đang chạy dự đoán...")

predictions = model.transform(df)

predictions.select(
    "job_title",
    "city",
    "position_level",
    "probability",
    "prediction"
).show(10, truncate=False)

# ====================================================
# 6. ĐÁNH GIÁ MODEL
# ====================================================
print(">>> Đang đánh giá model...")

# AUC - ROC
auc_evaluator = BinaryClassificationEvaluator(
    labelCol="is_attractive",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = auc_evaluator.evaluate(predictions)

# Accuracy
acc_evaluator = MulticlassClassificationEvaluator(
    labelCol="is_attractive",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = acc_evaluator.evaluate(predictions)

print("\n================= KẾT QUẢ =================")
print(f"AUC (Area Under ROC): {auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print("==========================================")

# ====================================================
# 7. PHÂN TÍCH NHANH TỶ LỆ JOB HẤP DẪN
# ====================================================
summary = predictions.groupBy("prediction").count()
summary.show()

spark.stop()
