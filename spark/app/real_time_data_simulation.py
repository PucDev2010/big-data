"""
Real-time Data Simulation - Kafka Producer
Gửi 1 job mỗi giây để demo streaming

Usage:
    python real_time_data_simulation.py
    
    Nhấn Ctrl+C để dừng
"""

import json
import time
import pandas as pd
from kafka import KafkaProducer
from datetime import datetime

# ====================================================
# CẤU HÌNH
# ====================================================
HOST_KAFKA = "localhost:29092"  # Hoặc "kafka:9092" nếu chạy trong Docker
TOPIC_NAME = "job_postings"
DELAY_SECONDS = 1  # Delay giữa mỗi message (1 giây)

# ====================================================
# KHỞI TẠO PRODUCER
# ====================================================
print("=" * 60)
print("🚀 REAL-TIME JOB STREAMING SIMULATOR")
print("=" * 60)
print(f"📡 Kafka Host: {HOST_KAFKA}")
print(f"📝 Topic: {TOPIC_NAME}")
print(f"⏱️  Tốc độ: {DELAY_SECONDS} giây/job")
print("=" * 60)

try:
    producer = KafkaProducer(
        bootstrap_servers=HOST_KAFKA,
        key_serializer=lambda k: k.encode("utf-8"),
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
    )
    print("✅ Kết nối Kafka thành công!")
except Exception as e:
    print(f"❌ Không thể kết nối Kafka: {e}")
    exit(1)

# ====================================================
# LOAD DATA
# ====================================================
df = pd.read_csv("jobs.csv")
total_jobs = len(df)
print(f"📊 Loaded {total_jobs:,} job records")
print("=" * 60)
print("\n🔴 BẮT ĐẦU STREAMING... (Ctrl+C để dừng)\n")

# ====================================================
# STREAMING LOOP - 1 JOB MỖI GIÂY
# ====================================================
sent_count = 0

try:
    for idx, row in df.iterrows():
        # Tạo event_time theo thời gian thực
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = {
            "job_title": str(row.get("job_title", "")),
            "job_type": str(row.get("job_type", "")),
            "position_level": str(row.get("position_level", "")),
            "city": str(row.get("city", "")),
            "experience": str(row.get("experience", "")),
            "skills": "" if pd.isna(row.get("skills")) else str(row.get("skills")),
            "job_fields": str(row.get("job_fields", "")),
            "salary": str(row.get("salary", "")),
            "salary_min": float(row.get("salary_min", 0) or 0),
            "salary_max": float(row.get("salary_max", 0) or 0),
            "unit": str(row.get("unit", "")),
            "event_time": current_time,  # Thời gian thực
            "event_type": "JOB_CREATED"
        }

        # Gửi message
        producer.send(
            TOPIC_NAME,
            key=message["city"],
            value=message
        )
        producer.flush()
        
        sent_count += 1
        
        # Hiển thị progress
        job_title = message["job_title"][:40] + "..." if len(message["job_title"]) > 40 else message["job_title"]
        print(f"[{current_time}] 📤 {sent_count:,}/{total_jobs:,} | {job_title} | {message['city']}")
        
        # Delay 1 giây
        time.sleep(DELAY_SECONDS)

except KeyboardInterrupt:
    print("\n\n⏹️  DỪNG STREAMING!")
    print(f"📊 Đã gửi: {sent_count:,} jobs")

finally:
    producer.close()
    print("✅ Đã đóng kết nối Kafka")
    print("=" * 60)
