# 📊 SLIDE TRÌNH BÀY ĐỒ ÁN BIG DATA
## Phân tích xu hướng thị trường việc làm

---

## SLIDE 1: TRANG BÌA

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│              ĐẠI HỌC CÔNG NGHỆ THÔNG TIN                   │
│                   KHOA KHOA HỌC DỮ LIỆU                    │
│                                                             │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                             │
│                      ĐỒ ÁN MÔN HỌC                         │
│                       BIG DATA                              │
│                                                             │
│     📊 PHÂN TÍCH XU HƯỚNG THỊ TRƯỜNG VIỆC LÀM             │
│        VÀ DỰ ĐOÁN MỨC ĐỘ HẤP DẪN CỦA KỸ NĂNG              │
│                                                             │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                             │
│                  Giảng viên hướng dẫn:                     │
│                  [Tên giảng viên]                          │
│                                                             │
│                  Nhóm: [Tên nhóm]                          │
│                  Thành viên:                               │
│                  • [Họ tên] - [MSSV]                       │
│                  • [Họ tên] - [MSSV]                       │
│                  • [Họ tên] - [MSSV]                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 2: MỤC TIÊU ĐỀ TÀI

### 🎯 Mục tiêu chính

1. **Xây dựng hệ thống Big Data** xử lý dữ liệu việc làm real-time
2. **Phân tích xu hướng** thị trường lao động Việt Nam
3. **Áp dụng Machine Learning** để:
   - Phân nhóm việc làm (Clustering)
   - Dự đoán độ "hot" của kỹ năng
   - Dự đoán mức lương

### 📈 Đầu ra mong đợi

| Output | Mô tả |
|--------|-------|
| Dashboard | Trực quan hóa dữ liệu real-time |
| ML Models | 3 thuật toán: K-Means, GBT, Random Forest |
| Insights | Phân tích thị trường việc làm |

---

## SLIDE 3: MÔ TẢ BÀI TOÁN

### 📋 Bài toán

**Input:**
- ~85,000 bài đăng tuyển dụng
- Thông tin: Vị trí, Lương, Kinh nghiệm, Kỹ năng, Thành phố...

**Output:**
- Phân nhóm 5 clusters thị trường việc làm
- Xếp hạng kỹ năng theo độ hấp dẫn
- Dự đoán mức lương theo đặc điểm công việc

### ❓ Câu hỏi nghiên cứu

1. Thị trường việc làm có những phân khúc nào?
2. Kỹ năng nào đang "hot" nhất?
3. Các yếu tố nào ảnh hưởng đến mức lương?

---

## SLIDE 4: QUY TRÌNH THỰC HIỆN

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE                           │
└─────────────────────────────────────────────────────────────┘

   📁 CSV          🔄 KAFKA         ⚡ SPARK        💾 CASSANDRA
  ─────────►     ─────────────►   ─────────────►   ─────────────►
   Data           Streaming        Processing       Storage
   Source                          & ETL

                                       │
                                       ▼
                              ┌────────────────┐
                              │   ML MODELS    │
                              │  • K-Means     │
                              │  • GBT         │
                              │  • Random Forest│
                              └───────┬────────┘
                                      │
                                      ▼
                              ┌────────────────┐
                              │   STREAMLIT    │
                              │   Dashboard    │
                              └────────────────┘
```

---

## SLIDE 5: KIẾN TRÚC HỆ THỐNG

```
┌──────────────────────────────────────────────────────────────┐
│                    DOCKER CONTAINERS                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐   ┌─────────┐   ┌─────────────────┐            │
│  │Zookeeper│   │  Kafka  │   │  Spark Cluster  │            │
│  │  :2181  │◄──│  :9092  │──►│ Master + Worker │            │
│  └─────────┘   └─────────┘   └────────┬────────┘            │
│                                        │                     │
│                                        ▼                     │
│  ┌─────────────┐              ┌────────────────┐            │
│  │  Cassandra  │◄─────────────│   ML Models    │            │
│  │   :9042     │              └────────────────┘            │
│  └──────┬──────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │  Streamlit  │ ◄── User Interface                         │
│  │   :8501     │                                            │
│  └─────────────┘                                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## SLIDE 6: CÔNG NGHỆ SỬ DỤNG

| Layer | Công nghệ | Vai trò |
|-------|-----------|---------|
| **Ingestion** | Apache Kafka | Message streaming |
| **Processing** | Apache Spark | Distributed computing |
| **Storage** | Apache Cassandra | NoSQL database |
| **ML** | PySpark MLlib | Machine Learning |
| **Visualization** | Streamlit, Plotly | Dashboard |
| **Container** | Docker Compose | Deployment |

### Tại sao chọn stack này?

- ✅ **Scalable**: Xử lý hàng triệu records
- ✅ **Real-time**: Streaming data liên tục
- ✅ **Distributed**: Chạy trên cluster
- ✅ **Industry Standard**: Dùng ở Grab, Shopee, VNG...

---

## SLIDE 7: THUẬT TOÁN MACHINE LEARNING

### 1️⃣ K-Means Clustering (Unsupervised)

| Thông số | Giá trị |
|----------|---------|
| Số clusters | K = 5 |
| Features | 10 (salary, exp, city, industry...) |
| Metric | Silhouette Score = **0.296** |

### 2️⃣ GBT Regressor (Supervised)

| Thông số | Giá trị |
|----------|---------|
| Target | Skill Hotness Score |
| Max Iterations | 50 |
| Max Depth | 5 |

### 3️⃣ Random Forest Regressor (Supervised)

| Thông số | Giá trị |
|----------|---------|
| Target | Salary Prediction |
| Số cây | 100 |
| RMSE | **7.91** triệu |
| R² | **0.26** |

---

## SLIDE 8: CÀI ĐẶT & DEMO

### Cấu trúc Source Code

```
project/
├── docker-compose.yml      # Container orchestration
├── spark/app/
│   ├── job_streaming.py    # Spark Streaming ETL
│   ├── train_kmeans.py     # K-Means clustering
│   ├── train_gbt_cassandra.py  # GBT model
│   ├── train_salary_prediction.py  # Random Forest
│   ├── streamlit_app.py    # Dashboard UI
│   └── real_time_data_simulation.py  # Kafka producer
└── data/
    └── jobs.csv            # Dataset ~85,000 jobs
```

### Lệnh chạy

```bash
# Khởi động containers
docker-compose up -d

# Chạy streaming
docker exec spark-master spark-submit job_streaming.py

# Train ML models
docker exec spark-master spark-submit train_kmeans.py
```

---

## SLIDE 9: KẾT QUẢ THỰC NGHIỆM (1)

### 📊 Kết quả K-Means Clustering

| Cluster | Tên | Số jobs | Đặc điểm |
|---------|-----|---------|----------|
| 0 | Entry Hà Nội | 33,148 | Sales, entry-level |
| 1 | Entry HCM | 33,958 | Entry-level HCM |
| 2 | IT Specialist | 3,068 | 100% IT, lương cao |
| 3 | Manager | 10,856 | 96% quản lý, 25tr+ |
| 4 | Education | 3,153 | 100% giáo dục |

### 💡 Insight

- **79%** việc làm là entry-level
- IT tách riêng cluster với lương cao hơn 40%
- Manager có mức lương gấp đôi nhân viên

---

## SLIDE 10: KẾT QUẢ THỰC NGHIỆM (2)

### 🔥 Top Kỹ năng Hot nhất

| Rank | Skill | Hot Score | Lương TB |
|------|-------|-----------|----------|
| 1 | Python | 0.85 | 28 triệu |
| 2 | SQL | 0.82 | 25 triệu |
| 3 | Java | 0.78 | 27 triệu |
| 4 | JavaScript | 0.75 | 24 triệu |
| 5 | Data Analysis | 0.72 | 26 triệu |

### 📈 Salary Prediction

| Metric | Giá trị |
|--------|---------|
| RMSE | 7.91 triệu |
| MAE | 4.96 triệu |
| R² | 0.26 |

**Top Features:** Experience (42%), Manager (38%), Sales (6%)

---

## SLIDE 11: NHẬN XÉT & HƯỚNG PHÁT TRIỂN

### ✅ Đã đạt được

- Xây dựng pipeline Big Data hoàn chỉnh
- 3 models ML với kết quả đánh giá
- Dashboard trực quan real-time

### ⚠️ Hạn chế

- R² của salary prediction còn thấp (0.26)
- Chưa có data crawl real-time thực sự
- Chưa deploy lên cloud

### 🚀 Hướng phát triển

1. **Cải thiện model**: Thêm features từ job description (NLP)
2. **Real-time crawling**: Tích hợp crawler tự động
3. **Cloud deployment**: AWS/GCP cluster
4. **Mobile app**: Ứng dụng tìm việc với AI

---

## SLIDE 12: CẢM ƠN

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                                                             │
│                    CẢM ƠN THẦY/CÔ                          │
│                   VÀ CÁC BẠN ĐÃ LẮNG NGHE                  │
│                                                             │
│                         ❓                                  │
│                   HỎI ĐÁP & THẢO LUẬN                      │
│                                                             │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                             │
│                  📧 Email: [email]                         │
│                  📱 GitHub: [github link]                  │
│                                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 CHECKLIST TRƯỚC KHI THUYẾT TRÌNH

- [ ] Chạy demo dashboard thành công
- [ ] Chuẩn bị video backup nếu demo fail
- [ ] In handout cho giám khảo
- [ ] Test thời gian trình bày (10-15 phút)
- [ ] Chuẩn bị câu hỏi có thể được hỏi
