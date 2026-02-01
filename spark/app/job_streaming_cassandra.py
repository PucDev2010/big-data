"""
Spark Streaming ETL Job
Consumes data from Kafka, performs ETL transformations, and stores in Cassandra
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp, when, lit, current_timestamp, regexp_replace, trim, split, size, expr, md5, concat_ws, length
from pyspark.sql.types import *

# Create Spark Session with Cassandra connector
spark = SparkSession.builder \
    .appName("KafkaToCassandraETL") \
    .config("spark.jars.packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0") \
    .config("spark.cassandra.connection.host", "cassandra") \
    .config("spark.cassandra.connection.port", "9042") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Define schema for Kafka messages
schema = StructType([
    StructField("job_title", StringType()),
    StructField("job_type", StringType()),
    StructField("position_level", StringType()),
    StructField("city", StringType()),
    StructField("experience", StringType()),
    StructField("skills", StringType()),
    StructField("job_fields", StringType()),
    StructField("salary", StringType()),
    StructField("salary_min", DoubleType()),
    StructField("salary_max", DoubleType()),
    StructField("unit", StringType()),
    StructField("event_time", StringType()),
    StructField("event_type", StringType())
])

print("=" * 60)
print("STARTING KAFKA STREAM CONSUMPTION")
print("=" * 60)
print(f"Kafka server: kafka:9092")
print(f"Topic: job_postings")
print(f"Starting offset: latest (will resume from checkpoint if exists)")
print("=" * 60)

# Read from Kafka
# Note: If checkpoint exists, Spark will resume from checkpoint offset
# If no checkpoint, 'latest' will start from new messages only
raw_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "job_postings") \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load()

print("Kafka stream created successfully!")
print(f"Stream schema: {raw_df.schema}")

# Parse JSON and extract data
print("Parsing JSON messages...")
parsed_df = raw_df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

print("JSON parsing configured!")
print(f"Parsed schema: {parsed_df.schema}")

# ETL Transformations
print("Applying ETL transformations...")

transformed_df = parsed_df \
    .withColumn("event_time", to_timestamp("event_time", "yyyy-MM-dd HH:mm:ss")) \
    .withColumn("id", expr("uuid()")) \
    .withColumn("city", when(col("city").isNull(), lit("unknown")).otherwise(col("city"))) \
    .withColumn("skills", when(col("skills").isNull(), lit("")).otherwise(col("skills"))) \
    .withColumn("job_fields", when(col("job_fields").isNull(), lit("")).otherwise(col("job_fields"))) \
    .withColumn("salary_min", when(col("salary_min").isNull() | (col("salary_min") < 0), lit(0.0)).otherwise(col("salary_min"))) \
    .withColumn("salary_max", when(col("salary_max").isNull() | (col("salary_max") < 0), lit(0.0)).otherwise(col("salary_max"))) \
    .withColumn("avg_salary", (col("salary_min") + col("salary_max")) / 2) \
    .withColumn("num_skills", when(col("skills") != "", 
        when(col("skills").contains(","), 
            size(split(col("skills"), ","))
        ).otherwise(lit(1))
    ).otherwise(lit(0))) \
    .withColumn("num_fields", when(col("job_fields") != "", 
        when(col("job_fields").contains(","), 
            size(split(col("job_fields"), ","))
        ).otherwise(lit(1))
    ).otherwise(lit(0))) \
    .withColumn("title_length", length(col("job_title"))) \
    .withColumn("processed_at", current_timestamp()) \
    .withColumn("salary_range", col("salary_max") - col("salary_min")) \
    .filter(
        col("job_title").isNotNull()
    )

# Select columns for Cassandra (matching table schema)
cassandra_df = transformed_df.select(
    "id",
    "job_title",
    "job_type",
    "position_level",
    "city",
    "experience",
    "skills",
    "job_fields",
    "salary",
    "salary_min",
    "salary_max",
    "unit",
    "avg_salary",
    "salary_range",
    "num_skills",
    "num_fields",
    "title_length",
    "event_time",
    "event_type",
    "processed_at"
)

# Write to Cassandra
print("Writing to Cassandra...")

def write_to_cassandra(batch_df, batch_id):
    """Write each batch to Cassandra"""
    try:
        row_count = batch_df.count()
        print(f"Batch {batch_id}: Processing {row_count} rows...")
        
        if row_count > 0:
            print(f"Batch {batch_id}: Sample data - {batch_df.select('job_title', 'city', 'avg_salary').limit(2).collect()}")
            batch_df.write \
                .format("org.apache.spark.sql.cassandra") \
                .mode("append") \
                .options(table="job_postings", keyspace="job_analytics") \
                .save()
            print(f"Batch {batch_id}: Successfully wrote {row_count} rows to Cassandra")
        else:
            print(f"Batch {batch_id}: No rows to write (all filtered out)")
    except Exception as e:
        print(f"Error writing batch {batch_id}: {str(e)}")
        import traceback
        traceback.print_exc()

# Start streaming query with console output for verification
print("Starting streaming query...")
print("Mode: foreachBatch (writing to Cassandra)")

# Function to write to Cassandra with detailed logging
def write_to_cassandra_and_console(batch_df, batch_id):
    """Write each batch to Cassandra and console"""
    try:
        row_count = batch_df.count()
        print(f"\n{'='*60}")
        print(f"BATCH {batch_id}: Processing {row_count} rows")
        print(f"{'='*60}")
        
        if row_count > 0:
            # Show sample data
            print(f"\nSample data (first 3 rows):")
            sample = batch_df.select("job_title", "city", "avg_salary", "salary_min", "salary_max").limit(3).collect()
            for i, row in enumerate(sample, 1):
                print(f"  {i}. {row['job_title'][:50]}... | City: {row['city']} | Salary: {row['salary_min']:.0f}-{row['salary_max']:.0f}M")
            
            # Write to Cassandra
            batch_df.write \
                .format("org.apache.spark.sql.cassandra") \
                .mode("append") \
                .options(table="job_postings", keyspace="job_analytics") \
                .save()
            print(f"\n✓ Successfully wrote {row_count} rows to Cassandra")
        else:
            print("⚠ No rows to write (all filtered out or empty batch)")
    except Exception as e:
        print(f"\n✗ ERROR writing batch {batch_id}: {str(e)}")
        import traceback
        traceback.print_exc()

# Start the Cassandra write query
query = cassandra_df.writeStream \
    .foreachBatch(write_to_cassandra_and_console) \
    .option("checkpointLocation", "/opt/spark/work-dir/checkpoints_cassandra") \
    .trigger(processingTime="30 seconds") \
    .start()

print("\n" + "="*60)
print("Streaming query started. Writing to Cassandra...")
print("Keyspace: job_analytics, Table: job_postings")
print("Checkpoint: /opt/spark/work-dir/checkpoints_cassandra")
print("Trigger: Every 30 seconds")
print("="*60)
print("\nWaiting for messages from Kafka...")

query.awaitTermination()
