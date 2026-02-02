from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# ====================================================
# 1. KHỞI TẠO SPARK SESSION
# ====================================================
spark = SparkSession.builder \
    .appName("SkillHotScore_GBT_Regressor") \
    .config("spark.cassandra.connection.host", "cassandra") \
    .config("spark.cassandra.connection.port", "9042") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ====================================================
# 2. ĐỌC DỮ LIỆU TỪ CASSANDRA
# ====================================================
print(">>> Đang đọc dữ liệu từ Cassandra...")
df = spark.read \
    .format("org.apache.spark.sql.cassandra") \
    .options(table="job_postings", keyspace="job_analytics") \
    .load()

print(f">>> Tổng số jobs: {df.count()}")

# ====================================================
# 3. TIỀN XỬ LÝ DỮ LIỆU
# ====================================================
print(">>> Đang tiền xử lý dữ liệu...")

df = df.select(
    "id",
    lower(col("city")).alias("city"),
    col("salary_avg").cast("double"),
    col("exp_avg_year").cast("double"),
    col("skills"),
    "event_time"
)

df = df.fillna({
    "salary_avg": 0.0,
    "exp_avg_year": 0.0,
    "skills": ""
})

# Lọc bỏ jobs không có skills
df = df.filter(col("skills") != "")

# ====================================================
# 4. TÁCH VÀ PHÂN TÍCH TỪNG KỸ NĂNG
# ====================================================
print(">>> Đang phân tích kỹ năng...")

# Explode skills thành từng dòng riêng
skill_df = df.withColumn(
    "skill",
    explode(split(lower(col("skills")), ","))
)

# Làm sạch skill
skill_df = skill_df.withColumn("skill", trim(col("skill"))) \
                   .filter(col("skill") != "") \
                   .filter(length(col("skill")) > 1)  # Bỏ skill quá ngắn

# Thêm feature: thành phố lớn
skill_df = skill_df.withColumn(
    "is_big_city",
    when(col("city").rlike("hồ chí minh|hà nội|hcm|ha noi"), 1.0).otherwise(0.0)
)

# ====================================================
# 5. TỔNG HỢP THỐNG KÊ CHO TỪNG KỸ NĂNG
# ====================================================
print(">>> Đang tổng hợp thống kê...")

skill_agg = skill_df.groupBy("skill").agg(
    count("*").alias("job_count"),
    avg("salary_avg").alias("avg_salary"),
    avg("exp_avg_year").alias("avg_exp"),
    avg("is_big_city").alias("big_city_ratio")
)

# Lọc chỉ giữ các skill phổ biến (ít nhất 10 jobs)
skill_agg = skill_agg.filter(col("job_count") >= 10)

print(f">>> Số kỹ năng phân tích: {skill_agg.count()}")

# ====================================================
# 6. TÍNH SKILL HOT SCORE (LABEL)
# ====================================================
# Công thức: hot_score = 0.4*salary + 0.3*demand - 0.2*exp + 0.1*city
# Giải thích:
# - Lương cao (40%): Kỹ năng trả lương cao = hấp dẫn
# - Demand cao (30%): Nhiều jobs yêu cầu = hấp dẫn
# - Kinh nghiệm thấp (-20%): Dễ học = hấp dẫn cho người mới
# - Thành phố lớn (10%): Có ở thành phố lớn = hấp dẫn

skill_agg = skill_agg.withColumn(
    "salary_norm", col("avg_salary") / 100.0  # Chuẩn hóa về 0-1
).withColumn(
    "demand_norm", least(col("job_count") / 100.0, lit(1.0))  # Chuẩn hóa, max = 1
).withColumn(
    "exp_norm", col("avg_exp") / 10.0  # Chuẩn hóa về 0-1
)

skill_agg = skill_agg.withColumn(
    "skill_hot_score",
    0.4 * col("salary_norm") +
    0.3 * col("demand_norm") -
    0.2 * col("exp_norm") +
    0.1 * col("big_city_ratio")
)

# ====================================================
# 7. CHUẨN BỊ FEATURES CHO ML
# ====================================================
feature_cols = ["avg_salary", "job_count", "avg_exp", "big_city_ratio"]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw"
)

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True
)

# ====================================================
# 8. CHIA TRAIN/TEST (80/20)
# ====================================================
print("\n>>> CHIA DỮ LIỆU TRAIN/TEST (80/20):")
train_data, test_data = skill_agg.randomSplit([0.8, 0.2], seed=42)
print(f"    - Tổng số skills: {skill_agg.count()}")
print(f"    - Train set (80%): {train_data.count()} skills")
print(f"    - Test set (20%): {test_data.count()} skills")

# ====================================================
# 9. XÂY DỰNG MODEL GBT REGRESSOR
# ====================================================
gbt = GBTRegressor(
    featuresCol="features",
    labelCol="skill_hot_score",
    maxIter=50,
    maxDepth=5,
    seed=42
)

# Tạo Pipeline
pipeline = Pipeline(stages=[assembler, scaler, gbt])

# ====================================================
# 10. TRAIN MODEL
# ====================================================
print("\n>>> Đang train GBT Regressor...")
model = pipeline.fit(train_data)
print(">>> Train xong!")

# ====================================================
# 11. DỰ ĐOÁN VÀ ĐÁNH GIÁ
# ====================================================
# Dự đoán trên test data
predictions = model.transform(test_data)

# Đánh giá bằng RMSE và R²
evaluator_rmse = RegressionEvaluator(
    labelCol="skill_hot_score",
    predictionCol="prediction",
    metricName="rmse"
)
rmse = evaluator_rmse.evaluate(predictions)

evaluator_r2 = RegressionEvaluator(
    labelCol="skill_hot_score",
    predictionCol="prediction",
    metricName="r2"
)
r2 = evaluator_r2.evaluate(predictions)

evaluator_mae = RegressionEvaluator(
    labelCol="skill_hot_score",
    predictionCol="prediction",
    metricName="mae"
)
mae = evaluator_mae.evaluate(predictions)

print("\n" + "="*50)
print("KẾT QUẢ ĐÁNH GIÁ MODEL")
print("="*50)
print(f"RMSE (Root Mean Square Error): {rmse:.4f}")
print(f"MAE (Mean Absolute Error):     {mae:.4f}")
print(f"R² (Coefficient of Determination): {r2:.4f}")
print("(R² càng gần 1 càng tốt)")

# ====================================================
# 12. DỰ ĐOÁN CHO TOÀN BỘ KỸ NĂNG
# ====================================================
print("\n>>> Dự đoán hot score cho toàn bộ kỹ năng...")
all_predictions = model.transform(skill_agg)

# ====================================================
# 13. TOP KỸ NĂNG HẤP DẪN NHẤT
# ====================================================
print("\n" + "="*50)
print("TOP 20 KỸ NĂNG HẤP DẪN NHẤT")
print("="*50)
all_predictions.select(
    "skill",
    round(col("prediction"), 4).alias("predicted_hot_score"),
    "job_count",
    round(col("avg_salary"), 1).alias("avg_salary"),
    round(col("avg_exp"), 1).alias("avg_exp")
).orderBy(desc("predicted_hot_score")).show(20, truncate=False)

# ====================================================
# 14. LƯU KẾT QUẢ VÀO CASSANDRA
# ====================================================
print("\n>>> Đang lưu kết quả vào Cassandra...")

result_df = all_predictions.select(
    col("skill"),
    col("job_count").cast("int"),
    col("avg_salary"),
    col("avg_exp"),
    col("big_city_ratio"),
    col("skill_hot_score"),
    col("prediction").alias("predicted_hot_score")
)

result_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .option("keyspace", "job_analytics") \
    .option("table", "skill_hot_scores") \
    .option("confirm.truncate", "true") \
    .mode("overwrite") \
    .save()

print(">>> Đã lưu vào table job_analytics.skill_hot_scores!")

# ====================================================
# 15. LƯU MODEL
# ====================================================
model_path = "/opt/spark/work-dir/models/skill_hot_gbt"
model.write().overwrite().save(model_path)
print(f"\n>>> Đã lưu model tại: {model_path}")

# ====================================================
# 16. PHÂN TÍCH FEATURE IMPORTANCE
# ====================================================
print("\n" + "="*50)
print("FEATURE IMPORTANCE (ĐỘ QUAN TRỌNG CỦA FEATURES)")
print("="*50)

gbt_model = model.stages[-1]
importances = gbt_model.featureImportances.toArray()

feature_importance = list(zip(feature_cols, importances))
feature_importance.sort(key=lambda x: x[1], reverse=True)

for feature, importance in feature_importance:
    bar = "█" * int(importance * 50)
    print(f"{feature:.<20} {importance:.4f} {bar}")

spark.stop()
print("\n✅ HOÀN THÀNH!")
