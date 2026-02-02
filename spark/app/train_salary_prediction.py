from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lower, lit, coalesce
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# ====================================================
# 1. KHỞI TẠO SPARK SESSION
# ====================================================
spark = SparkSession.builder \
    .appName("SalaryPrediction_RandomForest") \
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

# 3.1. Xử lý Lương (TARGET - Label)
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
df = df.withColumn(
    "is_danang",
    when(col("city_lower").rlike("đà nẵng|da nang"), 1.0).otherwise(0.0)
)

# Job fields features
df = df.withColumn("job_fields_lower", lower(col("job_fields")))
df = df.withColumn(
    "is_it",
    when(col("job_fields_lower").rlike("it|phần mềm|developer|lập trình|data|ai|software"), 1.0).otherwise(0.0)
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
df = df.withColumn(
    "is_engineering",
    when(col("job_fields_lower").rlike("kỹ thuật|cơ khí|điện|xây dựng|engineer"), 1.0).otherwise(0.0)
)

# Position level features - 7 cấp bậc theo thị trường lao động VN
df = df.withColumn("pos_lower", lower(col("position_level")))

# 1. Thực tập sinh (Intern)
df = df.withColumn(
    "is_intern",
    when(col("pos_lower").rlike("thực tập|intern|internship"), 1.0).otherwise(0.0)
)

# 2. Fresher (Mới ra trường, < 1 năm)
df = df.withColumn(
    "is_fresher",
    when(col("pos_lower").rlike("fresher|mới ra trường|sinh viên mới"), 1.0).otherwise(0.0)
)

# 3. Junior (1-2 năm kinh nghiệm)
df = df.withColumn(
    "is_junior",
    when(col("pos_lower").rlike("junior"), 1.0).otherwise(0.0)
)

# 4. Nhân viên/Chuyên viên (Staff - 2-4 năm)
df = df.withColumn(
    "is_staff",
    when(col("pos_lower").rlike("nhân viên|chuyên viên|staff|employee"), 1.0).otherwise(0.0)
)

# 5. Senior (4-7 năm)
df = df.withColumn(
    "is_senior",
    when(col("pos_lower").rlike("senior|chuyên gia|chuyên viên cao cấp"), 1.0).otherwise(0.0)
)

# 6. Trưởng nhóm (Team Lead - 5-8 năm)
df = df.withColumn(
    "is_team_lead",
    when(col("pos_lower").rlike("trưởng nhóm|team lead|leader|tech lead"), 1.0).otherwise(0.0)
)

# 7. Quản lý/Trưởng phòng (Manager - 7+ năm)
df = df.withColumn(
    "is_manager",
    when(col("pos_lower").rlike("trưởng phòng|quản lý|giám đốc|manager|head|director"), 1.0).otherwise(0.0)
)

# ====================================================
# 4. LỌC DỮ LIỆU HỢP LỆ
# ====================================================
# Chỉ giữ jobs có lương hợp lệ (> 0 triệu) và kinh nghiệm hợp lệ (0-30 năm)
df = df.filter(
    (col("salary_final") > 0) &         # Bỏ filter >= 5 triệu để giữ lại intern
    (col("salary_final") <= 200) &
    (col("exp_final") >= 0) &
    (col("exp_final") <= 30)
)

print(f">>> Số jobs sau khi lọc: {df.count()}")

# ====================================================
# 5. FEATURE ENGINEERING - 16 features
# ====================================================
feature_cols = [
    "exp_final",        # Kinh nghiệm (năm)
    "is_hcm",           # Thành phố HCM
    "is_hanoi",         # Thành phố Hà Nội
    "is_danang",        # Thành phố Đà Nẵng
    "is_it",            # Ngành IT
    "is_sales",         # Ngành Sales
    "is_finance",       # Ngành Tài chính
    "is_education",     # Ngành Giáo dục
    "is_engineering",   # Ngành Kỹ thuật
    "is_intern",        # Cấp 1: Thực tập sinh
    "is_fresher",       # Cấp 2: Fresher
    "is_junior",        # Cấp 3: Junior
    "is_staff",         # Cấp 4: Nhân viên/Chuyên viên
    "is_senior",        # Cấp 5: Senior
    "is_team_lead",     # Cấp 6: Trưởng nhóm
    "is_manager"        # Cấp 7: Quản lý/Trưởng phòng
]

# Fill null values với 0
for col_name in feature_cols:
    df = df.fillna({col_name: 0.0})

# ====================================================
# 6. CHIA TRAIN/TEST (80/20)
# ====================================================
print("\n>>> CHIA DỮ LIỆU TRAIN/TEST (80/20):")
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"    - Tổng số jobs: {df.count()}")
print(f"    - Train set (80%): {train_df.count()} jobs")
print(f"    - Test set (20%): {test_df.count()} jobs")

# ====================================================
# 7. ML PIPELINE
# ====================================================
print("\n>>> Đang xây dựng pipeline...")

# Bước 1: Gom features thành vector
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw"
)

# Bước 2: Chuẩn hóa dữ liệu
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True
)

# Bước 3: Random Forest Regressor
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="salary_final",
    numTrees=100,           # Số cây trong rừng
    maxDepth=10,            # Độ sâu tối đa mỗi cây
    seed=42
)

pipeline = Pipeline(stages=[assembler, scaler, rf])

# ====================================================
# 8. TRAIN MODEL
# ====================================================
print(">>> Đang train Random Forest Regressor...")
model = pipeline.fit(train_df)
print(">>> Train xong!")

# ====================================================
# 9. ĐÁNH GIÁ MODEL
# ====================================================
# Dự đoán trên tập test
predictions = model.transform(test_df)

# Đánh giá bằng RMSE, MAE, R²
evaluator_rmse = RegressionEvaluator(
    labelCol="salary_final",
    predictionCol="prediction",
    metricName="rmse"
)
rmse = evaluator_rmse.evaluate(predictions)

evaluator_mae = RegressionEvaluator(
    labelCol="salary_final",
    predictionCol="prediction",
    metricName="mae"
)
mae = evaluator_mae.evaluate(predictions)

evaluator_r2 = RegressionEvaluator(
    labelCol="salary_final",
    predictionCol="prediction",
    metricName="r2"
)
r2 = evaluator_r2.evaluate(predictions)

print("\n" + "="*50)
print("KẾT QUẢ ĐÁNH GIÁ MODEL")
print("="*50)
print(f"RMSE (Root Mean Square Error): {rmse:.2f} triệu")
print(f"MAE (Mean Absolute Error):     {mae:.2f} triệu")
print(f"R² (Coefficient of Determination): {r2:.4f}")
print("(R² càng gần 1 càng tốt, MAE/RMSE càng thấp càng tốt)")

# ====================================================
# 10. FEATURE IMPORTANCE
# ====================================================
print("\n" + "="*50)
print("FEATURE IMPORTANCE (ĐỘ QUAN TRỌNG CỦA FEATURES)")
print("="*50)

rf_model = model.stages[-1]
importances = rf_model.featureImportances.toArray()

feature_importance = list(zip(feature_cols, importances))
feature_importance.sort(key=lambda x: x[1], reverse=True)

for feature, importance in feature_importance:
    bar = "█" * int(importance * 100)
    print(f"{feature:.<20} {importance:.4f} {bar}")

# ====================================================
# 11. HIỂN THỊ MẪU DỰ ĐOÁN
# ====================================================
print("\n" + "="*50)
print("MẪU DỰ ĐOÁN VS THỰC TẾ")
print("="*50)

predictions.select(
    "job_title",
    "city",
    "exp_final",
    col("salary_final").alias("actual_salary"),
    col("prediction").alias("predicted_salary")
).show(20, truncate=30)

# ====================================================
# 12. THỐNG KÊ THEO NGÀNH
# ====================================================
print("\n" + "="*50)
print("LƯƠNG TRUNG BÌNH THEO NGÀNH (DỰ ĐOÁN)")
print("="*50)

# Tạo dataframe tổng hợp
all_predictions = model.transform(df)
industry_stats = all_predictions.groupBy("is_it", "is_sales", "is_finance", "is_education", "is_engineering").agg(
    {"prediction": "avg", "salary_final": "avg"}
)
industry_stats.show(10)

# ====================================================
# 13. LƯU MODEL
# ====================================================
model_path = "/opt/spark/work-dir/models/salary_prediction_rf"
model.write().overwrite().save(model_path)
print(f"\n>>> Đã lưu model tại: {model_path}")

# ====================================================
# 14. LƯU KẾT QUẢ ĐÁNH GIÁ
# ====================================================
print("\n" + "="*50)
print("TÓM TẮT MODEL")
print("="*50)
print(f"""
📊 SALARY PREDICTION MODEL - RANDOM FOREST
├── Thuật toán: Random Forest Regressor
├── Số cây: 100
├── Độ sâu tối đa: 10
├── Train/Test: 80/20
│
├── KẾT QUẢ:
│   ├── RMSE: {rmse:.2f} triệu
│   ├── MAE:  {mae:.2f} triệu
│   └── R²:   {r2:.4f}
│
└── TOP FEATURES:
    ├── {feature_importance[0][0]}: {feature_importance[0][1]:.4f}
    ├── {feature_importance[1][0]}: {feature_importance[1][1]:.4f}
    └── {feature_importance[2][0]}: {feature_importance[2][1]:.4f}
""")

spark.stop()
print("\n✅ HOÀN THÀNH!")
