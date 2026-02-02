# 🚀 Hướng dẫn chạy đồ án Big Data

## Yêu cầu
- Docker Desktop đã cài đặt và đang chạy

---

## Bước 1: Khởi động Docker containers

```bash
cd d:\UIT\hoc-ki-3\big-data\do-an\project
docker-compose up -d
```

Kiểm tra containers đang chạy:
```bash
docker ps
```

Phải thấy 4 containers: `zookeeper`, `kafka`, `spark-master`, `spark-worker`, `cassandra`

---

## Bước 2: Đợi Cassandra khởi động (~30-60 giây)

```bash
docker exec -it cassandra cqlsh -e "DESCRIBE KEYSPACES;"
```

Nếu thấy lỗi "Connection refused", đợi thêm và thử lại.

---

## Bước 3: Tạo Keyspace và Table (chạy lần đầu)

```bash
docker exec -it cassandra cqlsh
```

Chạy các lệnh CQL:
```sql
CREATE KEYSPACE IF NOT EXISTS job_analytics
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

USE job_analytics;

CREATE TABLE IF NOT EXISTS job_postings (
    id UUID PRIMARY KEY,
    job_title TEXT,
    job_type TEXT,
    position_level TEXT,
    city TEXT,
    experience TEXT,
    skills TEXT,
    job_fields TEXT,
    salary TEXT,
    salary_min DOUBLE,
    salary_max DOUBLE,
    salary_avg DOUBLE,
    unit TEXT,
    exp_min_year DOUBLE,
    exp_max_year DOUBLE,
    exp_avg_year DOUBLE,
    exp_type TEXT,
    event_time TIMESTAMP,
    event_type TEXT
);

-- Table lưu kết quả clustering
CREATE TABLE IF NOT EXISTS job_clusters (
    id UUID PRIMARY KEY,
    job_title TEXT,
    city TEXT,
    salary_final DOUBLE,
    exp_final DOUBLE,
    job_fields TEXT,
    position_level TEXT,
    cluster INT
);

EXIT;
```

---

## Bước 4: Chạy Streaming ETL (đọc từ Kafka, ghi vào Cassandra)

```bash
docker exec -it spark-master /opt/spark/bin/spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3,com.datastax.spark:spark-cassandra-connector_2.12:3.5.0 \
  /opt/spark/work-dir/job_streaming.py
```

---

## Bước 5: Mô phỏng dữ liệu (terminal khác)

```bash
docker exec -it spark-master python3 /opt/spark/work-dir/real_time_data_simulation.py
```

---

## Bước 6: Train model ML

### Option A: K-Means Clustering (Khuyến khích)
```bash
docker exec spark-master /opt/spark/bin/spark-submit \
  --packages com.datastax.spark:spark-cassandra-connector_2.12:3.5.0 \
  /opt/spark/work-dir/train_kmeans.py
```

### Option B: Logistic Regression
```bash
docker exec spark-master /opt/spark/bin/spark-submit \
  --packages com.datastax.spark:spark-cassandra-connector_2.12:3.5.0 \
  /opt/spark/work-dir/train_logictis.py
```

---

## Bước 7: Chạy Streamlit Dashboard (trên máy host)

```bash
# Cài đặt dependencies
cd d:\UIT\hoc-ki-3\big-data\do-an\project\spark\app
pip install -r requirements.txt

# Chạy Streamlit
streamlit run streamlit_app.py
```

Truy cập: http://localhost:8501

---

## Truy cập các dịch vụ

| Dịch vụ | URL/Port |
|---------|----------|
| **Streamlit Dashboard** | http://localhost:8501 |
| Spark Master UI | http://localhost:8080 |
| Spark Worker UI | http://localhost:8081 |
| Cassandra | localhost:9042 (DataGrip/DBeaver) |
| Kafka | localhost:29092 |

---

## Dừng tất cả containers

```bash
docker-compose down
```

Xóa cả data:
```bash
docker-compose down -v
```
