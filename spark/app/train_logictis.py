from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lower, lit, coalesce, regexp_extract

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# ====================================================
# 1. KHỞI TẠO SPARK SESSION
# ====================================================
spark = SparkSession.builder \
    .appName("JobAttractiveness_Logistic_Optimization") \
    .config("spark.cassandra.connection.host", "cassandra") \
    .config("spark.cassandra.connection.port", "9042") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ====================================================
# 2. ĐỌC DỮ LIỆU TỪ CASSANDRA
# ====================================================
print(">>> Đang đọc dữ liệu...")
df_raw = spark.read \
    .format("org.apache.spark.sql.cassandra") \
    .options(table="job_postings", keyspace="job_analytics") \
    .load()

# Lọc bỏ các bản ghi rác nếu cần
df = df_raw.filter(col("job_title").isNotNull())

# ====================================================
# 3. DATA CLEANING & PRE-PROCESSING
# ====================================================

# 3.1. Xử lý Lương (Salary): Ưu tiên salary_avg, nếu null thì lấy trung bình min/max, cuối cùng là 0
df = df.withColumn(
    "salary_final",
    coalesce(
        col("salary_avg"), 
        (col("salary_min") + col("salary_max")) / 2, 
        lit(0.0)
    )
)

# 3.2. Xử lý Kinh nghiệm (Experience): Ưu tiên exp_avg, nếu null thì 0
df = df.withColumn(
    "exp_final",
    coalesce(col("exp_avg_year"), col("exp_min_year"), lit(0.0))
)

# 3.3. Xử lý chuỗi (Lower case để so sánh chuỗi chính xác)
df = df.withColumn("city_lower", lower(col("city"))) \
       .withColumn("job_fields_lower", lower(col("job_fields"))) \
       .withColumn("job_title_lower", lower(col("job_title"))) \
       .withColumn("pos_lower", lower(col("position_level")))

# ====================================================
# 4. LABEL ENGINEERING (TẠO NHÃN HOT/NO HOT)
# ====================================================
# Logic thảo luận:
# 1. Hot kiểu hiệu suất: Lương >= 15tr VÀ Kinh nghiệm <= 2 năm
# 2. Hot kiểu High-end: Lương >= 30tr (bất chấp kinh nghiệm)
# Còn lại là 0

df = df.withColumn(
    "is_hot",
    when(
        ((col("salary_final") >= 15.0) & (col("exp_final") <= 2.0)), 1.0
    ).when(
        (col("salary_final") >= 30.0), 1.0
    ).otherwise(0.0)
)

# Kiểm tra phân bố nhãn
print(">>> Phân bố nhãn (0: No Hot, 1: Hot):")
df.groupBy("is_hot").count().show()

# ====================================================
# 5. FEATURE ENGINEERING (TẠO BIẾN ĐẦU VÀO X)
# ====================================================

# Feature 1: Hiệu suất Lương (Salary per Experience Ratio)
# Công thức: Lương / (Năm kinh nghiệm + 1). Cộng 1 để tránh chia cho 0.
df = df.withColumn("salary_per_exp", col("salary_final") / (col("exp_final") + 1.0))

# Feature 2: Big City (Tier 1)
# 1 nếu ở HCM/HN, 0 nếu ở tỉnh khác
df = df.withColumn(
    "is_big_city",
    when(col("city_lower").rlike("hồ chí minh|hà nội|hcm|ha noi"), 1.0).otherwise(0.0)
)

# Feature 3: Is Manager/Lead (Dựa trên Job Title hoặc Position Level)
df = df.withColumn(
    "is_manager",
    when(
        col("job_title_lower").rlike("trưởng|quản lý|giám đốc|manager|lead|head") | 
        col("pos_lower").rlike("trưởng|quản lý|giám đốc"), 
        1.0
    ).otherwise(0.0)
)

# Feature 4: Is Tech Job (Ngành công nghệ)
df = df.withColumn(
    "is_tech",
    when(col("job_fields_lower").rlike("cntt|it|phần mềm|developer|kỹ sư|data|ai"), 1.0).otherwise(0.0)
)

# Feature 5: Is Sales/Biz Job (Ngành Sales/Kinh doanh - thường biến động cao)
df = df.withColumn(
    "is_sales",
    when(col("job_fields_lower").rlike("kinh doanh|bán hàng|sales|tiếp thị|marketing"), 1.0).otherwise(0.0)
)

# Chọn các cột Features để đưa vào Vector
feature_cols = [
    "salary_final",     # Lương gốc
    "exp_final",        # Kinh nghiệm gốc
    "salary_per_exp",   # Chỉ số phái sinh quan trọng
    "is_big_city",      # Địa điểm
    "is_manager",       # Cấp bậc
    "is_tech",          # Ngành Tech
    "is_sales"          # Ngành Sales
]

# Chia tập train/test
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# ====================================================
# 6. ML PIPELINE
# ====================================================

# Bước 1: Gom các feature thành 1 vector
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw"
)

# Bước 2: Chuẩn hóa dữ liệu (StandardScaler)
# Logistic Regression rất nhạy cảm với biên độ dữ liệu (Lương 20.000.000 vs Feature là 0/1)
# Cần đưa về cùng 1 scale để thuật toán hội tụ tốt hơn.
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True
)

# Bước 3: Model Logistic Regression
lr = LogisticRegression(
    labelCol="is_hot",
    featuresCol="features",
    maxIter=100,
    regParam=0.01,       # L2 Regularization để tránh Overfitting
    elasticNetParam=0.0  # 0 là L2 (Ridge), 1 là L1 (Lasso)
)

pipeline = Pipeline(stages=[assembler, scaler, lr])

# ====================================================
# 7. TRAIN & EVALUATE
# ====================================================
print(">>> Bắt đầu huấn luyện...")
model = pipeline.fit(train_df)
print(">>> Huấn luyện xong!")

# Dự đoán trên tập test
predictions = model.transform(test_df)

# Đánh giá
evaluator_auc = BinaryClassificationEvaluator(labelCol="is_hot", metricName="areaUnderROC")
auc = evaluator_auc.evaluate(predictions)

evaluator_acc = MulticlassClassificationEvaluator(labelCol="is_hot", metricName="accuracy")
acc = evaluator_acc.evaluate(predictions)

print("\n" + "="*30)
print(f"KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
print("="*30)
print(f"Area Under ROC (AUC): {auc:.4f}")
print(f"Accuracy:           {acc:.4f}")

# ====================================================
# 8. PHÂN TÍCH TRỌNG SỐ (COEFFICIENTS)
# ====================================================
# Phần này cực kỳ quan trọng để bạn giải thích cho Đồ án
# Nó cho biết yếu tố nào ảnh hưởng nhất đến việc 1 job là HOT

lr_model = model.stages[-1] # Lấy model LR từ pipeline
coeffs = lr_model.coefficients.toArray()
intercept = lr_model.intercept

print("\n" + "="*30)
print("GIẢI THÍCH TRỌNG SỐ (IMPORTANCE)")
print("="*30)
print("(Dương: Tác động tích cực tới độ HOT | Âm: Tác động tiêu cực)")

feature_importance = list(zip(feature_cols, coeffs))
# Sắp xếp theo độ lớn tuyệt đối của trọng số
feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

for feature, weight in feature_importance:
    impact = "TĂNG cơ hội Hot" if weight > 0 else "GIẢM cơ hội Hot"
    print(f"{feature:.<20} {weight:.4f}  --> {impact}")

print(f"\nIntercept (Hệ số chặn): {intercept:.4f}")

# ====================================================
# 9. LƯU MÔ HÌNH
# ====================================================
model_path = "/opt/spark/work-dir/models/job_attractiveness_logistic_v2"
model.write().overwrite().save(model_path)
print(f"\n>>> Đã lưu model tại: {model_path}")

spark.stop()