from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lower, lit, coalesce
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import ClusteringEvaluator

# ====================================================
# 1. KHỞI TẠO SPARK SESSION
# ====================================================
spark = SparkSession.builder \
    .appName("JobClustering_KMeans") \
    .config("spark.cassandra.connection.host", "cassandra") \
    .config("spark.cassandra.connection.port", "9042") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ====================================================
# 2. ĐỌC DỮ LIỆU TỪ CASSANDRA
# ====================================================
print(">>> Đang đọc dữ liệu từ Cassandra...")
df_raw = spark.read \
    .format("org.apache.spark.sql.cassandra") \
    .options(table="job_postings", keyspace="job_analytics") \
    .load()

print(f">>> Tổng số jobs: {df_raw.count()}")

# ====================================================
# 3. DATA PREPROCESSING
# ====================================================
print(">>> Đang xử lý dữ liệu...")

# Lọc bỏ records rác
df = df_raw.filter(col("job_title").isNotNull())

# 3.1. Xử lý Lương
df = df.withColumn(
    "salary_final",
    coalesce(
        col("salary_avg"), 
        (col("salary_min") + col("salary_max")) / 2, 
        lit(0.0)
    )
)

# 3.2. Xử lý Kinh nghiệm
df = df.withColumn(
    "exp_final",
    coalesce(col("exp_avg_year"), col("exp_min_year"), lit(0.0))
)

# 3.3. Tạo features từ text columns
# City features
df = df.withColumn("city_lower", lower(col("city")))
df = df.withColumn(
    "is_hcm",
    when(col("city_lower").rlike("hồ chí minh|hcm"), 1.0).otherwise(0.0)
)
df = df.withColumn(
    "is_hanoi",
    when(col("city_lower").rlike("hà nội|ha noi|hanoi"), 1.0).otherwise(0.0)
)

# Job fields features
df = df.withColumn("job_fields_lower", lower(col("job_fields")))
df = df.withColumn(
    "is_it",
    when(col("job_fields_lower").rlike("it|phần mềm|developer|lập trình|data|ai"), 1.0).otherwise(0.0)
)
df = df.withColumn(
    "is_sales",
    when(col("job_fields_lower").rlike("bán hàng|kinh doanh|sales|tiếp thị|marketing"), 1.0).otherwise(0.0)
)
df = df.withColumn(
    "is_finance",
    when(col("job_fields_lower").rlike("tài chính|ngân hàng|kế toán|finance|banking"), 1.0).otherwise(0.0)
)
df = df.withColumn(
    "is_education",
    when(col("job_fields_lower").rlike("giáo dục|đào tạo|giáo viên|education"), 1.0).otherwise(0.0)
)

# Position level features
df = df.withColumn("pos_lower", lower(col("position_level")))
df = df.withColumn(
    "is_manager",
    when(col("pos_lower").rlike("trưởng|quản lý|giám đốc|manager|lead|head"), 1.0).otherwise(0.0)
)
df = df.withColumn(
    "is_senior",
    when(col("pos_lower").rlike("senior|chuyên gia|chuyên viên cao cấp"), 1.0).otherwise(0.0)
)

# Lọc bỏ jobs có dữ liệu bất hợp lý
# - salary = 0: Không có thông tin lương
# - salary > 200: Lương > 200 triệu/tháng (không thực tế)
# - exp > 30: Kinh nghiệm > 30 năm (data bị lỗi)
df = df.filter(
    (col("salary_final") > 0) & 
    (col("salary_final") <= 200) &
    (col("exp_final") >= 0) &
    (col("exp_final") <= 30)
)

print(f">>> Số jobs sau khi lọc dữ liệu bất hợp lý: {df.count()}")

# ====================================================
# 4. FEATURE ENGINEERING CHO CLUSTERING
# ====================================================
feature_cols = [
    "salary_final",     # Lương
    "exp_final",        # Kinh nghiệm
    "is_hcm",           # Thành phố HCM
    "is_hanoi",         # Thành phố Hà Nội
    "is_it",            # Ngành IT
    "is_sales",         # Ngành Sales
    "is_finance",       # Ngành Tài chính
    "is_education",     # Ngành Giáo dục
    "is_manager",       # Vị trí quản lý
    "is_senior"         # Vị trí senior
]

# Fill null values với 0
for col_name in feature_cols:
    df = df.fillna({col_name: 0.0})

# ====================================================
# 5. ML PIPELINE
# ====================================================
print(">>> Đang xây dựng pipeline...")

# Bước 1: Gom features thành vector
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw"
)

# Bước 2: Chuẩn hóa dữ liệu (quan trọng cho K-Means)
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True
)

# Bước 3: K-Means với 5 clusters
# Số K có thể điều chỉnh dựa trên Elbow method hoặc Silhouette score
NUM_CLUSTERS = 5
kmeans = KMeans(
    k=NUM_CLUSTERS,
    featuresCol="features",
    predictionCol="cluster",
    seed=42
)

pipeline = Pipeline(stages=[assembler, scaler, kmeans])

# ====================================================
# 6. TRAIN MODEL
# ====================================================
print(f">>> Đang train K-Means với K={NUM_CLUSTERS}...")
model = pipeline.fit(df)
print(">>> Train xong!")

# ====================================================
# 7. ĐÁNH GIÁ MODEL
# ====================================================
# Dự đoán cluster cho toàn bộ data
predictions = model.transform(df)

# Đánh giá bằng Silhouette Score
evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="cluster",
    metricName="silhouette"
)
silhouette = evaluator.evaluate(predictions)

print("\n" + "="*50)
print("KẾT QUẢ ĐÁNH GIÁ MODEL")
print("="*50)
print(f"Silhouette Score: {silhouette:.4f}")
print("(Giá trị càng gần 1 càng tốt, > 0.5 là khá tốt)")

# ====================================================
# 8. PHÂN TÍCH CÁC CLUSTER
# ====================================================
print("\n" + "="*50)
print("PHÂN BỐ JOBS THEO CLUSTER")
print("="*50)
predictions.groupBy("cluster").count().orderBy("cluster").show()

# Phân tích đặc điểm từng cluster
print("\n" + "="*50)
print("ĐẶC ĐIỂM TRUNG BÌNH CỦA TỪNG CLUSTER")
print("="*50)

cluster_stats = predictions.groupBy("cluster").agg(
    {"salary_final": "avg", 
     "exp_final": "avg",
     "is_hcm": "avg",
     "is_hanoi": "avg",
     "is_it": "avg",
     "is_sales": "avg",
     "is_finance": "avg",
     "is_education": "avg",
     "is_manager": "avg",
     "is_senior": "avg"}
).orderBy("cluster")

cluster_stats.show(truncate=False)

# In mô tả từng cluster
print("\n" + "="*50)
print("MÔ TẢ TỪNG CLUSTER (DỰA TRÊN ĐẶC ĐIỂM)")
print("="*50)

# Lấy dữ liệu để phân tích
stats_pd = cluster_stats.toPandas()
for _, row in stats_pd.iterrows():
    cluster_id = int(row['cluster'])
    salary = row['avg(salary_final)']
    exp = row['avg(exp_final)']
    
    # Xác định đặc điểm nổi bật
    features = []
    if row['avg(is_hcm)'] > 0.5:
        features.append("HCM")
    if row['avg(is_hanoi)'] > 0.5:
        features.append("Hà Nội")
    if row['avg(is_it)'] > 0.3:
        features.append("IT")
    if row['avg(is_sales)'] > 0.3:
        features.append("Sales")
    if row['avg(is_finance)'] > 0.3:
        features.append("Finance")
    if row['avg(is_education)'] > 0.3:
        features.append("Education")
    if row['avg(is_manager)'] > 0.3:
        features.append("Manager")
    if row['avg(is_senior)'] > 0.3:
        features.append("Senior")
    
    # Phân loại theo lương
    if salary >= 25:
        salary_level = "Lương cao"
    elif salary >= 15:
        salary_level = "Lương trung bình"
    else:
        salary_level = "Lương thấp"
    
    # Phân loại theo kinh nghiệm
    if exp >= 3:
        exp_level = "Kinh nghiệm cao (3+ năm)"
    elif exp >= 1:
        exp_level = "Kinh nghiệm trung bình (1-3 năm)"
    else:
        exp_level = "Entry-level/Fresher"
    
    feature_str = ", ".join(features) if features else "Đa dạng"
    
    print(f"\n📌 Cluster {cluster_id}:")
    print(f"   - {salary_level} (~{salary:.1f} triệu)")
    print(f"   - {exp_level} (~{exp:.1f} năm)")
    print(f"   - Đặc điểm: {feature_str}")

# ====================================================
# 9. LƯU KẾT QUẢ VÀO CASSANDRA
# ====================================================
print("\n>>> Đang lưu kết quả clustering vào Cassandra...")

# Chọn các cột cần lưu
result_df = predictions.select(
    "id", "job_title", "city", "salary_final", "exp_final", 
    "job_fields", "position_level", "cluster"
)

# Lưu vào table mới
result_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .option("keyspace", "job_analytics") \
    .option("table", "job_clusters") \
    .option("confirm.truncate", "true") \
    .mode("overwrite") \
    .save()

print(">>> Đã lưu kết quả vào table job_analytics.job_clusters!")

# ====================================================
# 10. LƯU MODEL
# ====================================================
model_path = "/opt/spark/work-dir/models/job_clustering_kmeans"
model.write().overwrite().save(model_path)
print(f"\n>>> Đã lưu model tại: {model_path}")

# ====================================================
# 11. HIỂN THỊ MẪU KẾT QUẢ
# ====================================================
print("\n" + "="*50)
print("MẪU JOBS TRONG TỪNG CLUSTER")
print("="*50)

for i in range(NUM_CLUSTERS):
    print(f"\n--- Cluster {i} (5 jobs mẫu) ---")
    predictions.filter(col("cluster") == i) \
        .select("job_title", "city", "salary_final", "exp_final", "job_fields") \
        .show(5, truncate=50)

spark.stop()
print("\n✅ HOÀN THÀNH!")
