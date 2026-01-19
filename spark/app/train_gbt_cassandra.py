from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, when, lower, regexp_replace
from pyspark.ml.feature import (
    VectorAssembler, StringIndexer, OneHotEncoder,
    Tokenizer, HashingTF, IDF, StopWordsRemover
)
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# ====================================================
# 1. KHỞI TẠO SPARK & KẾT NỐI CASSANDRA
# ====================================================
spark = SparkSession.builder \
    .appName("JobSalary_GBT_Cassandra") \
    .config("spark.cassandra.connection.host", "cassandra") \
    .config("spark.cassandra.connection.port", "9042") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print(">>> Đang đọc dữ liệu từ Cassandra...")
# Đọc dữ liệu từ bảng job_postings trong keyspace job_analytics
raw_df = spark.read \
    .format("org.apache.spark.sql.cassandra") \
    .options(table="job_postings", keyspace="job_analytics") \
    .load()

# Kiểm tra dữ liệu
if raw_df.count() == 0:
    print("!!! Lỗi: Bảng Cassandra trống rỗng. Hãy chạy Streaming trước.")
    spark.stop()
    exit()

print("=== SCHEMA TỪ CASSANDRA ===")
raw_df.printSchema()

# ====================================================
# 2. XỬ LÝ DỮ LIỆU (ETL)
# ====================================================

# A. Xử lý Lương & Tính Trung Bình
# Trong Cassandra, salary_min/max thường đã là Double, nhưng ta fillna(0) cho chắc ăn
df = raw_df.fillna(0, subset=["salary_min", "salary_max"])

df = df.withColumn(
    "avg_salary",
    when((col("salary_min") > 0) & (col("salary_max") > 0), (col("salary_min") + col("salary_max")) / 2)
    .when((col("salary_min") > 0) & (col("salary_max") == 0), col("salary_min"))
    .when((col("salary_min") == 0) & (col("salary_max") > 0), col("salary_max"))
    .otherwise(0)
)

# B. Lọc sạch dữ liệu (Data Cleaning) - QUAN TRỌNG
# 1. Loại bỏ các job trùng lặp (dựa trên job_title)
# 2. Chỉ lấy các job có lương hợp lý (ví dụ: > 0) để Model học tốt hơn
# 3. Điền giá trị rỗng cho các cột text để tránh lỗi NullPointerException khi chạy Tokenizer
df_clean = df.filter(col("avg_salary") > 0) \
    .filter(col("job_title").isNotNull()) \
    .fillna({
        'skills': '', 
        'job_title': '', 
        'job_fields': '', 
        'city': 'Unknown', 
        'position_level': 'Unknown',
        'experience': 'Không yêu cầu'
    })

print(f">>> Số lượng bản ghi sạch dùng để train: {df_clean.count()}")

# C. Xử lý kinh nghiệm (Regex -> Số năm)
df_clean = df_clean.withColumn(
    "experience_years",
    when(col("experience").rlike("chưa có|không yêu cầu"), 0)
    .when(col("experience").rlike("lên đến 1"), 1)
    .when(col("experience").rlike("trên 1"), 2)
    .when(col("experience").rlike("1 - 2"), 1.5)
    .when(col("experience").rlike("1 - 3"), 2)
    .when(col("experience").rlike("2 - 5"), 3.5)
    .when(col("experience").rlike("3 - 5"), 4)
    .when(col("experience").rlike("5 - 7"), 6)
    .when(col("experience").rlike("5 - 15"), 10)
    .otherwise(2)
)

# D. Tạo Full Text Features (Gộp cột)
df_clean = df_clean.withColumn(
    "full_text_features", 
    expr("concat(job_title, ' ', skills, ' ', job_fields)")
)

# Chia tập dữ liệu Train/Test
train_data, test_data = df_clean.randomSplit([0.8, 0.2], seed=42)

# ====================================================
# 3. XÂY DỰNG PIPELINE ML
# ====================================================

# City Handling
city_indexer = StringIndexer(inputCol="city", outputCol="city_idx", handleInvalid="keep")
city_encoder = OneHotEncoder(inputCols=["city_idx"], outputCols=["city_vec"])

# Position Handling
pos_indexer = StringIndexer(inputCol="position_level", outputCol="pos_idx", handleInvalid="keep")
pos_encoder = OneHotEncoder(inputCols=["pos_idx"], outputCols=["pos_vec"])

# Text Handling (TF-IDF)
tokenizer = Tokenizer(inputCol="full_text_features", outputCol="words_raw")
vi_stopwords = [
    "của", "và", "các", "có", "làm", "tại", "trong", "được", "với", "là", 
    "người", "nhân viên", "công ty", "tuyển", "hcm", "hn", "lương", "tháng", 
    "yêu cầu", "mô tả", "chi nhánh", "trách nhiệm", "quyền lợi"
]
remover = StopWordsRemover(inputCol="words_raw", outputCol="words_clean", stopWords=vi_stopwords)
hashingTF = HashingTF(inputCol="words_clean", outputCol="tf_features", numFeatures=3000)
idf = IDF(inputCol="tf_features", outputCol="text_vec")

# Vector Assembler
assembler = VectorAssembler(
    inputCols=["experience_years", "city_vec", "pos_vec", "text_vec"],
    outputCol="features"
)

# GBT Regressor Model
gbt = GBTRegressor(
    labelCol="avg_salary",
    featuresCol="features",
    maxIter=100,   # Giảm xuống 50 để train nhanh hơn demo (thực tế để 100)
    maxDepth=8,
    stepSize=0.05
)

# Pipeline
pipeline = Pipeline(stages=[
    city_indexer, city_encoder,
    pos_indexer, pos_encoder,
    tokenizer, remover, hashingTF, idf,
    assembler,
    gbt
])

# ====================================================
# 4. TRAIN & EVALUATE
# ====================================================
print(">>> Đang huấn luyện GBT Regressor...")
model = pipeline.fit(train_data)
print(">>> Huấn luyện xong!")

# Lưu model (Để tái sử dụng sau này)
model_save_path = "/opt/spark/work-dir/models/gbt_salary_model"
print(f">>> Đang lưu model vào: {model_save_path}")
model.write().overwrite().save(model_save_path)

# Dự báo trên tập Test
predictions = model.transform(test_data)

# Đánh giá
evaluator_rmse = RegressionEvaluator(labelCol="avg_salary", predictionCol="prediction", metricName="rmse")
rmse = evaluator_rmse.evaluate(predictions)

evaluator_r2 = RegressionEvaluator(labelCol="avg_salary", predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(predictions)

print(f"\n================ KẾT QUẢ ===================")
print(f"RMSE (Sai số trung bình): {rmse:.2f} (Triệu VNĐ)")
print(f"R2 (Độ phù hợp mô hình): {r2:.2f}")
print(f"============================================")

print("\n--- TOP 10 DỰ BÁO VS THỰC TẾ (Lấy từ Cassandra) ---")
predictions.select("job_title", "city", "experience_years", "avg_salary", "prediction").show(10, truncate=False)

spark.stop()