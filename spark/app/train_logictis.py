from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lower, concat, lit, expr, count
)
from pyspark.ml.feature import (
    VectorAssembler, StringIndexer, OneHotEncoder,
    Tokenizer, HashingTF, IDF, StopWordsRemover
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# ====================================================
# PHẦN 1: KẾT NỐI CASSANDRA & ĐỌC DỮ LIỆU
# ====================================================
spark = SparkSession.builder \
    .appName("JobAttractiveness_Cassandra_Train") \
    .config("spark.cassandra.connection.host", "cassandra") \
    .config("spark.cassandra.connection.port", "9042") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print(">>> Đang đọc dữ liệu từ Cassandra (job_analytics.job_postings)...")

# Đọc dữ liệu từ bảng Cassandra
raw_df = spark.read \
    .format("org.apache.spark.sql.cassandra") \
    .options(table="job_postings", keyspace="job_analytics") \
    .load()

# Kiểm tra nếu không có dữ liệu
if raw_df.count() == 0:
    print("!!! Bảng Cassandra đang trống. Hãy chạy Streaming để đẩy dữ liệu vào trước !!!")
    spark.stop()
    exit()

print(f">>> Tìm thấy {raw_df.count()} bản ghi. Bắt đầu xử lý...")

# ====================================================
# PHẦN 2: DATA CLEANING & GÁN NHÃN (LABELING)
# ====================================================

# 1. Xử lý Lương
# Lưu ý: Trong Cassandra cột salary_min/max đã là Double, nhưng ta fillna cho chắc chắn
df = raw_df.fillna(0, subset=["salary_min", "salary_max"])

# Tính lại avg_salary (Dù ETL có thể đã tính, ta tính lại ở đây để đảm bảo logic P70/P50 nhất quán)
df = df.withColumn(
    "avg_salary_calc",
    when(
        (col("salary_min") > 0) & (col("salary_max") > 0),
        (col("salary_min") + col("salary_max")) / 2
    ).otherwise(col("salary_min"))
)

# 2. Xử lý Kinh nghiệm (Regex -> Số năm)
# Cần xử lý null cho cột experience trước khi chạy rlike
df = df.fillna("", subset=["experience"])

df = df.withColumn(
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
    .otherwise(2) # Mặc định
)

# 3. Phân nhóm vị trí
df = df.fillna("Unknown", subset=["position_level"]) # Xử lý null
df = df.withColumn(
    "position_group_label",
    when(lower(col("position_level")).rlike("quản lý|giám đốc|phó giám đốc"), "SENIOR")
    .when(lower(col("position_level")).rlike("trưởng nhóm|giám sát"), "MID")
    .otherwise("ENTRY")
)

# 4. Tính ngưỡng lương (P50, P70)
quantiles = df.stat.approxQuantile("avg_salary_calc", [0.5, 0.7], 0.01)
P50, P70 = quantiles[0], quantiles[1]
print(f"--- NGƯỠNG LƯƠNG TỪ DB: Median={P50}, Top30%={P70} ---")

# 5. Tạo cột LABEL: is_attractive
df = df.withColumn(
    "is_attractive",
    when(col("avg_salary_calc") >= P70, 1)  # Lương cao
    .when(col("position_group_label").isin("MID", "SENIOR"), 1) # Chức vụ cao
    .when((col("experience_years") <= 1) & (col("avg_salary_calc") >= P50), 1) # Lương tốt cho người mới
    .otherwise(0)
)

# ====================================================
# PHẦN 3: FEATURE ENGINEERING
# ====================================================

# 1. Làm sạch dữ liệu đầu vào cho Model
df_clean = df.filter(col("job_title").isNotNull()) \
             .fillna({
                 'skills': '',
                 'job_title': '',
                 'job_fields': '',
                 'city': 'Unknown',
                 'position_level': 'Unknown'
             })

# 2. Gộp Text Features
df_clean = df_clean.withColumn(
    "full_text_features",
    expr("concat(job_title, ' ', skills, ' ', job_fields)")
)

# 3. Chia dữ liệu Train/Test
train_data, test_data = df_clean.randomSplit([0.8, 0.2], seed=42)

# ====================================================
# PHẦN 4: XÂY DỰNG ML PIPELINE
# ====================================================

# A. Categorical Features
city_indexer = StringIndexer(inputCol="city", outputCol="city_idx", handleInvalid="keep")
city_encoder = OneHotEncoder(inputCols=["city_idx"], outputCols=["city_vec"])

pos_indexer = StringIndexer(inputCol="position_level", outputCol="pos_idx", handleInvalid="keep")
pos_encoder = OneHotEncoder(inputCols=["pos_idx"], outputCols=["pos_vec"])

# B. Text Features
tokenizer = Tokenizer(inputCol="full_text_features", outputCol="words_raw")
vi_stopwords = [
    "của", "và", "các", "có", "làm", "tại", "trong", "được", "với", "là",
    "người", "những", "cho", "về", "nhân viên", "công ty", "tuyển", "gấp",
    "hcm", "hn", "lương", "tháng", "nam", "nữ", "mô tả", "yêu cầu", "chi nhánh"
]
remover = StopWordsRemover(inputCol="words_raw", outputCol="words_clean", stopWords=vi_stopwords)
hashingTF = HashingTF(inputCol="words_clean", outputCol="tf_features", numFeatures=3000)
idf = IDF(inputCol="tf_features", outputCol="text_vec")

# C. Assemble Vectors
assembler = VectorAssembler(
    inputCols=["experience_years", "city_vec", "pos_vec", "text_vec"],
    outputCol="features"
)

# D. Model Logistic Regression
lr = LogisticRegression(
    labelCol="is_attractive",
    featuresCol="features",
    regParam=0.01,
    elasticNetParam=0.8
)

# E. Pipeline
pipeline = Pipeline(stages=[
    city_indexer, city_encoder,
    pos_indexer, pos_encoder,
    tokenizer, remover, hashingTF, idf,
    assembler,
    lr
])

# ====================================================
# PHẦN 5: TRAIN & ĐÁNH GIÁ
# ====================================================

print(">>> Đang huấn luyện mô hình...")
model = pipeline.fit(train_data)
print(">>> Huấn luyện xong!")

# ----------------------------------------------------
# [MỚI] PHẦN LƯU MODEL (SAVE)
# ----------------------------------------------------
# Đường dẫn lưu: Lưu vào thư mục 'models' nằm trong 'app'
# Trong Docker nó là /opt/spark/work-dir/models/...
# Trên Windows nó sẽ hiện ở: D:\UIT\...\project\spark\app\models\...
model_path = "/opt/spark/work-dir/models/job_attractiveness_v1"

print(f">>> Đang lưu model vào: {model_path}")

# overwrite(): Ghi đè nếu folder đã tồn tại (tiện khi chạy lại nhiều lần)
model.write().overwrite().save(model_path)

print(">>> Đã lưu model thành công!")

# ----------------------------------------------------
# Đánh giá (Giữ nguyên)
# ----------------------------------------------------
predictions = model.transform(test_data)
evaluator_auc = BinaryClassificationEvaluator(labelCol="is_attractive", metricName="areaUnderROC")
auc = evaluator_auc.evaluate(predictions)

print(f"\n================ KẾT QUẢ ===================")
print(f"Area Under ROC (AUC): {auc:.4f}")
print(f"Model saved at: {model_path}")
print(f"============================================")

spark.stop()