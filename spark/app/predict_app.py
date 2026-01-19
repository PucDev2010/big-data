from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import expr, col, lit

# 1. Khởi tạo Spark (Không cần config Cassandra nếu chỉ dự báo dữ liệu nhập tay)
spark = SparkSession.builder \
    .appName("JobPrediction_Inference") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ====================================================
# BƯỚC 1: TẢI MODEL ĐÃ TRAIN
# ====================================================
# Đường dẫn phải KHỚP với đường dẫn bạn đã lưu ở file train
model_path = "/opt/spark/work-dir/models/job_attractiveness_v1"

print(f">>> Đang tải model từ: {model_path} ...")
try:
    # Lưu ý: Dùng PipelineModel vì lúc train ta lưu cả Pipeline
    loaded_model = PipelineModel.load(model_path)
    print(">>> Tải model thành công!")
except Exception as e:
    print(f"!!! LỖI: Không tìm thấy model. {e}")
    spark.stop()
    exit()

# ====================================================
# BƯỚC 2: CHUẨN BỊ DỮ LIỆU MỚI (INPUT DATA)
# ====================================================
print(">>> Đang chuẩn bị dữ liệu mẫu để test...")

# Giả sử đây là dữ liệu nhập từ Web Form hoặc API gửi xuống
input_data = [
    # Case 1: Lương thấp, exp cao -> Dự đoán: Không hấp dẫn (0)
    ("Nhân viên nhập liệu", "Excel, Word", "Hành chính", "Hồ Chí Minh", "1 năm", "Staff", 0), 
    
    # Case 2: IT, Lương (ẩn), Skill xịn -> Dự đoán: Hấp dẫn (1)
    ("Senior Java Developer", "Spring Boot, Microservices, Kafka", "IT Software", "Hà Nội", "5 năm", "Senior", 0),
    
    # Case 3: Giám đốc, exp khủng -> Dự đoán: Hấp dẫn (1)
    ("Giám đốc kinh doanh vùng", "B2B Sales, Strategy", "Sales", "Đà Nẵng", "10 năm", "Director", 0)
]

# Tạo DataFrame
schema = ["job_title", "skills", "job_fields", "city", "experience", "position_level", "experience_years"] 
# Lưu ý: 'experience_years' ở đây mình giả định đã xử lý sơ bộ. 
# Nếu muốn chuẩn chỉ, bạn phải copy logic regex từ file train sang đây để biến đổi '5 năm' -> 5.0

df_new = spark.createDataFrame(input_data, schema)

# ====================================================
# BƯỚC 3: TIỀN XỬ LÝ (QUAN TRỌNG !!!)
# ====================================================
# Model Pipeline của bạn bắt đầu từ các cột: "experience_years", "city", "position_level", "full_text_features".
# Nhưng dữ liệu gốc chưa có cột "full_text_features" (do bước này nằm NGOÀI pipeline lúc train).
# => BẠN PHẢI TÁI TẠO LẠI BƯỚC NÀY.

print(">>> Đang tiền xử lý dữ liệu...")
df_ready = df_new.fillna("") \
    .withColumn("full_text_features", expr("concat(job_title, ' ', skills, ' ', job_fields)"))

# ====================================================
# BƯỚC 4: DỰ BÁO (PREDICT)
# ====================================================
# Hàm transform sẽ tự động chạy qua: StringIndexer -> Tokenizer -> HashingTF -> LogisticRegression
predictions = loaded_model.transform(df_ready)

# ====================================================
# BƯỚC 5: HIỂN THỊ KẾT QUẢ
# ====================================================
print("\n" + "="*50)
print("KẾT QUẢ DỰ BÁO ĐỘ HẤP DẪN CÔNG VIỆC")
print("="*50)

# Chọn các cột cần xem
result = predictions.select(
    col("job_title"),
    col("city"),
    col("probability"), # Cột này chứa xác suất [xác suất là 0, xác suất là 1]
    col("prediction")   # 1.0 = Hấp dẫn, 0.0 = Không hấp dẫn
)

# ...
print(">>> Tải model thành công!", flush=True) 
# ...
print("KẾT QUẢ DỰ BÁO ĐỘ HẤP DẪN CÔNG VIỆC", flush=True)
# ...
result.show(truncate=False) # .show() của Spark thường tự flush, nhưng print thì cần ép

spark.stop()