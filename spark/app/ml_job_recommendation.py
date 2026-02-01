"""
Job Recommendation System using Content-Based Filtering with PySpark ML
Recommends similar jobs based on job features using Spark's distributed ML capabilities
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, Normalizer, RegexTokenizer
from pyspark.ml import Pipeline
from pyspark.sql.functions import (
    col, when, concat_ws, udf, lit, row_number, array, explode, 
    collect_list, struct, desc, size, split, lower, regexp_replace
)
from pyspark.sql.types import DoubleType, ArrayType, FloatType
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors, VectorUDT, SparseVector, DenseVector
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def compute_dot_product_with_broadcast(target_indices, target_values, target_size):
    """
    Create a UDF function that computes dot product with a broadcasted sparse vector.
    This avoids serialization issues by only passing primitive types.
    """
    def dot_product(v):
        if v is None:
            return 0.0
        
        # Reconstruct target vector components
        t_indices = target_indices
        t_values = target_values
        
        # Get components from input vector
        if isinstance(v, SparseVector):
            v_indices = v.indices
            v_values = v.values
        elif isinstance(v, DenseVector):
            v_array = v.toArray()
            v_indices = np.where(v_array != 0)[0]
            v_values = v_array[v_indices]
        else:
            return 0.0
        
        # Compute dot product using set intersection of indices
        result = 0.0
        t_idx_set = set(t_indices)
        for i, idx in enumerate(v_indices):
            if idx in t_idx_set:
                t_pos = np.where(t_indices == idx)[0]
                if len(t_pos) > 0:
                    result += v_values[i] * t_values[t_pos[0]]
        
        return float(result)
    
    return dot_product


class JobRecommenderPySpark:
    def __init__(self, cassandra_host='cassandra', cassandra_port=9042, spark_master='local[*]'):
        """Initialize Spark session with Cassandra connector"""
        self.cassandra_host = cassandra_host
        self.cassandra_port = cassandra_port
        
        # Determine correct Scala version for Cassandra connector
        import pyspark
        spark_version = pyspark.__version__
        if spark_version.startswith('4.'):
            # Spark 4.x uses Scala 2.13
            connector_package = "com.datastax.spark:spark-cassandra-connector_2.13:3.5.0"
        else:
            # Spark 3.x uses Scala 2.12
            connector_package = "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0"
        
        self.spark = SparkSession.builder \
            .appName("JobRecommendationSystem") \
            .master(spark_master) \
            .config("spark.jars.packages", connector_package) \
            .config("spark.cassandra.connection.host", cassandra_host) \
            .config("spark.cassandra.connection.port", str(cassandra_port)) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.session.timeZone", "UTC") \
            .getOrCreate()
        
        # Set log levels
        self.spark.sparkContext.setLogLevel("WARN")
        log4j = self.spark.sparkContext._jvm.org.apache.log4j
        log4j.LogManager.getLogger("com.datastax.spark.connector").setLevel(log4j.Level.WARN)
        log4j.LogManager.getLogger("com.datastax.oss.driver").setLevel(log4j.Level.WARN)
        log4j.LogManager.getLogger("org.apache.cassandra").setLevel(log4j.Level.WARN)
        log4j.LogManager.getLogger("com.datastax.oss.driver.internal.core.cql.CqlRequestHandler").setLevel(log4j.Level.ERROR)
        
        self.df = None
        self.tfidf_model = None
        self.features_df = None
        
    def load_data_from_cassandra(self, keyspace='job_analytics', table='job_postings', limit=None):
        """Load job posting data from Cassandra"""
        print(f"\n{'='*60}")
        print(f"Loading data from {keyspace}.{table}...")
        print(f"{'='*60}")
        
        try:
            # Read directly from Cassandra using Spark Cassandra connector
            self.df = self.spark.read \
                .format("org.apache.spark.sql.cassandra") \
                .options(table=table, keyspace=keyspace) \
                .load()
            
            if limit:
                self.df = self.df.limit(limit)
            
            count = self.df.count()
            print(f"✓ Loaded {count} job postings from Cassandra")
            print(f"  Columns: {len(self.df.columns)}")
            
            # Handle missing values
            self.df = self.df.fillna({
                'skills': '',
                'job_fields': '',
                'city': 'unknown',
                'job_title': '',
                'position_level': ''
            })
            print("✓ Missing values handled")
            
            # Add row index for identification
            window = Window.orderBy(lit(1))
            self.df = self.df.withColumn("job_index", row_number().over(window) - 1)
            
            return self
            
        except Exception as e:
            import traceback
            print(f"Error loading data: {e}")
            traceback.print_exc()
            return None
    
    def prepare_features(self, num_features=500):
        """Create TF-IDF features for similarity calculation using PySpark ML"""
        print(f"\n{'='*60}")
        print("Preparing features with PySpark ML...")
        print(f"{'='*60}")
        
        # Combine text features into a single column
        print("Combining text features...")
        self.df = self.df.withColumn(
            'combined_text',
            concat_ws(' ',
                col('job_title'),
                col('skills'),
                col('job_fields'),
                col('position_level'),
                col('city')
            )
        )
        
        # Clean text: lowercase and remove special characters
        self.df = self.df.withColumn(
            'combined_text',
            lower(regexp_replace(col('combined_text'), '[^a-zA-Z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\\s]', ' '))
        )
        
        # Build TF-IDF pipeline
        print("Building TF-IDF pipeline...")
        
        # Tokenize text
        tokenizer = RegexTokenizer(
            inputCol="combined_text",
            outputCol="words",
            pattern="\\s+",
            minTokenLength=2
        )
        
        # HashingTF for term frequency
        hashing_tf = HashingTF(
            inputCol="words",
            outputCol="raw_features",
            numFeatures=num_features
        )
        
        # IDF for inverse document frequency
        idf = IDF(
            inputCol="raw_features",
            outputCol="tfidf_features",
            minDocFreq=2
        )
        
        # Normalize vectors for cosine similarity
        normalizer = Normalizer(
            inputCol="tfidf_features",
            outputCol="normalized_features",
            p=2.0
        )
        
        # Build and fit pipeline
        pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf, normalizer])
        print("Fitting TF-IDF model...")
        self.tfidf_model = pipeline.fit(self.df)
        
        # Transform data
        self.features_df = self.tfidf_model.transform(self.df)
        
        print(f"✓ Created TF-IDF features with {num_features} dimensions")
        print(f"  Total jobs: {self.features_df.count()}")
        
        return self
    
    def _extract_vector_components(self, vec):
        """Extract indices and values from a sparse/dense vector for serialization"""
        if vec is None:
            return np.array([]), np.array([]), 0
        
        if isinstance(vec, SparseVector):
            return np.array(vec.indices), np.array(vec.values), vec.size
        elif isinstance(vec, DenseVector):
            arr = vec.toArray()
            indices = np.where(arr != 0)[0]
            values = arr[indices]
            return indices, values, len(arr)
        else:
            return np.array([]), np.array([]), 0
    
    def find_similar_jobs(self, job_index, top_n=10):
        """
        Find similar jobs to a given job using cosine similarity
        
        Args:
            job_index: Index of the job
            top_n: Number of similar jobs to return
            
        Returns:
            DataFrame with similar jobs and similarity scores
        """
        print(f"\nFinding similar jobs for job index {job_index}...")
        
        # Get the target job's feature vector
        target_job = self.features_df.filter(col("job_index") == job_index).first()
        
        if target_job is None:
            raise ValueError(f"Job index {job_index} not found")
        
        target_vector = target_job["normalized_features"]
        target_title = target_job["job_title"]
        
        print(f"Target job: {target_title}")
        
        # Extract vector components (serializable primitive types)
        target_indices, target_values, target_size = self._extract_vector_components(target_vector)
        
        # Create UDF with primitive types only (no SparkContext reference)
        dot_product_func = compute_dot_product_with_broadcast(target_indices, target_values, target_size)
        dot_product_udf = udf(dot_product_func, DoubleType())
        
        # Calculate similarity scores
        similarity_df = self.features_df.withColumn(
            "similarity_score",
            dot_product_udf(col("normalized_features"))
        )
        
        # Filter out the target job and get top N similar jobs
        similar_jobs = similarity_df \
            .filter(col("job_index") != job_index) \
            .orderBy(desc("similarity_score")) \
            .limit(top_n) \
            .select(
                col("job_title"),
                col("city"),
                col("position_level"),
                col("experience"),
                col("salary_min"),
                col("salary_max"),
                col("unit"),
                col("similarity_score"),
                col("skills")
            )
        
        return similar_jobs
    
    def recommend_by_query(self, query_text, top_n=10):
        """
        Recommend jobs based on a text query using TF-IDF similarity
        
        Args:
            query_text: Text description of desired job
            top_n: Number of recommendations
            
        Returns:
            DataFrame with recommended jobs
        """
        print(f"\nRecommending jobs for query: '{query_text}'")
        
        # Create a DataFrame with the query
        query_df = self.spark.createDataFrame([(query_text,)], ["combined_text"])
        
        # Clean query text
        query_df = query_df.withColumn(
            'combined_text',
            lower(regexp_replace(col('combined_text'), '[^a-zA-Z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\\s]', ' '))
        )
        
        # Transform query using the fitted model
        query_features = self.tfidf_model.transform(query_df)
        query_vector = query_features.first()["normalized_features"]
        
        # Extract vector components (serializable primitive types)
        query_indices, query_values, query_size = self._extract_vector_components(query_vector)
        
        # Create UDF with primitive types only (no SparkContext reference)
        dot_product_func = compute_dot_product_with_broadcast(query_indices, query_values, query_size)
        dot_product_udf = udf(dot_product_func, DoubleType())
        
        # Calculate similarity scores
        similarity_df = self.features_df.withColumn(
            "similarity_score",
            dot_product_udf(col("normalized_features"))
        )
        
        # Get top N recommendations
        recommendations = similarity_df \
            .orderBy(desc("similarity_score")) \
            .limit(top_n) \
            .select(
                col("job_title"),
                col("city"),
                col("position_level"),
                col("experience"),
                col("salary_min"),
                col("salary_max"),
                col("unit"),
                col("similarity_score"),
                col("skills")
            )
        
        return recommendations
    
    def close(self):
        """Close Spark session"""
        if self.spark:
            self.spark.stop()
            print("\nSpark session closed")


def main():
    """Example usage"""
    print("="*60)
    print("JOB RECOMMENDATION SYSTEM (PySpark ML)")
    print("="*60)
    
    # Initialize recommender
    # Use 'cassandra' hostname when running in Docker containers
    # Use '127.0.0.1' when running locally
    recommender = JobRecommenderPySpark(cassandra_host='cassandra', cassandra_port=9042)
    
    # Load data from Cassandra
    result = recommender.load_data_from_cassandra(keyspace='job_analytics', limit=80000)
    
    if result is None or recommender.df is None or recommender.df.count() == 0:
        print("No data found in Cassandra. Make sure data is being streamed.")
        recommender.close()
        return
    
    # Prepare features
    recommender.prepare_features(num_features=500)
    
    # Example 1: Find similar jobs to a specific job
    print("\n" + "="*60)
    print("EXAMPLE 1: Find similar jobs")
    print("="*60)
    
    # Pick a sample job
    sample_job_idx = 100
    total_jobs = recommender.features_df.count()
    if sample_job_idx >= total_jobs:
        sample_job_idx = 0
    
    sample_job = recommender.features_df.filter(col("job_index") == sample_job_idx).first()
    
    if sample_job:
        print(f"\nOriginal Job:")
        print(f"  Title: {sample_job['job_title']}")
        print(f"  City: {sample_job['city']}")
        print(f"  Position: {sample_job['position_level']}")
        salary_min = sample_job['salary_min'] if sample_job['salary_min'] else 0
        salary_max = sample_job['salary_max'] if sample_job['salary_max'] else 0
        unit = sample_job['unit'] if sample_job['unit'] else ''
        print(f"  Salary: {salary_min:.0f}-{salary_max:.0f} {unit}")
        
        print(f"\nTop 5 Similar Jobs:")
        similar_jobs = recommender.find_similar_jobs(sample_job_idx, top_n=5)
        similar_jobs_list = similar_jobs.collect()
        
        for idx, row in enumerate(similar_jobs_list, 1):
            print(f"\n  {idx}. {row['job_title']}")
            s_min = row['salary_min'] if row['salary_min'] else 0
            s_max = row['salary_max'] if row['salary_max'] else 0
            s_unit = row['unit'] if row['unit'] else ''
            print(f"     City: {row['city']} | Salary: {s_min:.0f}-{s_max:.0f} {s_unit}")
            print(f"     Similarity: {row['similarity_score']:.3f}")
    
    # Example 2: Query-based recommendation
    print("\n" + "="*60)
    print("EXAMPLE 2: Query-based recommendation")
    print("="*60)
    
    query = "nhân viên kinh doanh marketing digital"
    print(f"\nQuery: '{query}'")
    
    recommendations = recommender.recommend_by_query(query, top_n=5)
    recommendations_list = recommendations.collect()
    
    print(f"\nTop 5 Recommendations:")
    for idx, row in enumerate(recommendations_list, 1):
        print(f"\n  {idx}. {row['job_title']}")
        s_min = row['salary_min'] if row['salary_min'] else 0
        s_max = row['salary_max'] if row['salary_max'] else 0
        s_unit = row['unit'] if row['unit'] else ''
        print(f"     City: {row['city']} | Position: {row['position_level']}")
        print(f"     Salary: {s_min:.0f}-{s_max:.0f} {s_unit}")
        print(f"     Similarity: {row['similarity_score']:.3f}")
    
    print("\n" + "="*60)
    print("Recommendation system ready!")
    print("="*60)
    
    recommender.close()


if __name__ == "__main__":
    main()
