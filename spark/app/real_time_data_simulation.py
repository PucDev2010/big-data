import json
import time
import pandas as pd
from kafka import KafkaProducer

HOST_KAFKA = "localhost:29092"
TOPIC_NAME = "job_postings"

producer = KafkaProducer(
    bootstrap_servers=HOST_KAFKA,
    key_serializer=lambda k: k.encode("utf-8"),
    value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
    linger_ms=50,          # cho phép Kafka batch
    batch_size=16384
)

df = pd.read_csv("jobs.csv")
print(f"Loaded {len(df)} job records")

event_time = time.strftime("%Y-%m-%d %H:%M:%S")

for idx, row in df.iterrows():
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
        "event_time": event_time,
        "event_type": "JOB_CREATED"
    }

    producer.send(
        TOPIC_NAME,
        key=message["city"],
        value=message
    )

producer.flush()
producer.close()

print("🚀 Done sending ALL job data to Kafka")
