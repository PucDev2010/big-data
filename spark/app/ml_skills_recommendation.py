"""
Skills Extraction & Recommendation System using PySpark ML
Features:
- Word2Vec for skill embeddings and similarity
- LDA Topic Modeling for skill clusters
- TF-IDF for skill importance ranking
- Skill autocomplete and recommendations
"""

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Word2Vec, Word2VecModel, 
    CountVectorizer, IDF, 
    RegexTokenizer, StopWordsRemover,
    Normalizer
)
from pyspark.ml.clustering import LDA
from pyspark.sql.functions import (
    col, explode, split, trim, lower, regexp_replace,
    collect_list, collect_set, count, desc, asc,
    size, array_distinct, lit, udf, concat_ws,
    row_number, when, length, array, struct
)
from pyspark.sql.types import (
    DoubleType, ArrayType, StringType, FloatType,
    StructType, StructField, IntegerType
)
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors, SparseVector, DenseVector
import numpy as np
import uuid
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def compute_cosine_similarity(target_vec_list):
    """
    Create a UDF function that computes cosine similarity with a target vector.
    Uses list representation to avoid serialization issues.
    """
    target_arr = np.array(target_vec_list)
    target_norm = np.linalg.norm(target_arr)
    
    def cosine_sim(v):
        if v is None or target_norm == 0:
            return 0.0
        
        if isinstance(v, (list, np.ndarray)):
            v_arr = np.array(v)
        elif hasattr(v, 'toArray'):
            v_arr = v.toArray()
        else:
            return 0.0
        
        v_norm = np.linalg.norm(v_arr)
        if v_norm == 0:
            return 0.0
        
        return float(np.dot(target_arr, v_arr) / (target_norm * v_norm))
    
    return cosine_sim


class SkillsRecommenderPySpark:
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
            .appName("SkillsRecommendationSystem") \
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
        self.skills_df = None
        self.word2vec_model = None
        self.lda_model = None
        self.skill_vectors = None
        self.skill_clusters = None
        
    def load_data_from_cassandra(self, keyspace='job_analytics', table='job_postings', limit=None):
        """Load job posting data from Cassandra"""
        print(f"\n{'='*60}")
        print(f"Loading data from {keyspace}.{table}...")
        print(f"{'='*60}")
        
        try:
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
            
            return self
            
        except Exception as e:
            import traceback
            print(f"Error loading data: {e}")
            traceback.print_exc()
            return None
    
    def extract_skills(self):
        """Extract and process skills from job postings"""
        print(f"\n{'='*60}")
        print("Extracting skills from job postings...")
        print(f"{'='*60}")
        
        # Filter jobs with skills
        jobs_with_skills = self.df.filter(
            (col('skills').isNotNull()) & 
            (length(col('skills')) > 0)
        )
        
        print(f"Jobs with skills: {jobs_with_skills.count()}")
        
        # Explode skills (comma-separated) into individual rows
        skills_exploded = jobs_with_skills.select(
            col('job_title'),
            col('position_level'),
            col('job_fields'),
            col('city'),
            col('avg_salary'),
            explode(split(col('skills'), ',')).alias('skill_raw')
        )
        
        # Clean skills: trim, lowercase, remove special chars
        skills_cleaned = skills_exploded.withColumn(
            'skill',
            trim(lower(regexp_replace(col('skill_raw'), '[^a-zA-Z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\\s\\+\\#\\.\\-]', '')))
        )
        
        # Filter empty skills and too short skills
        skills_cleaned = skills_cleaned.filter(
            (length(col('skill')) >= 2) &
            (col('skill') != '')
        )
        
        # Get skill statistics
        self.skills_df = skills_cleaned.groupBy('skill').agg(
            count('*').alias('frequency'),
            collect_set('position_level').alias('position_levels'),
            collect_set('job_fields').alias('related_fields')
        ).orderBy(desc('frequency'))
        
        # Cache for faster access
        self.skills_df.cache()
        
        total_skills = self.skills_df.count()
        print(f"✓ Extracted {total_skills} unique skills")
        
        # Show top skills
        print("\nTop 15 most common skills:")
        top_skills = self.skills_df.limit(15).collect()
        for i, row in enumerate(top_skills, 1):
            print(f"  {i:2d}. {row['skill'][:40]:40s} (frequency: {row['frequency']})")
        
        return self
    
    def train_word2vec(self, vector_size=100, min_count=5, window_size=5):
        """Train Word2Vec model on skill sequences"""
        print(f"\n{'='*60}")
        print("Training Word2Vec model for skill embeddings...")
        print(f"{'='*60}")
        
        # Prepare skill sequences from job postings
        # Each job's skills form a "sentence" for Word2Vec
        skill_sequences = self.df.filter(
            (col('skills').isNotNull()) & 
            (length(col('skills')) > 0)
        ).select(
            split(lower(col('skills')), ',').alias('skill_tokens')
        )
        
        # Clean tokens
        clean_tokens_udf = udf(
            lambda tokens: [t.strip() for t in tokens if t.strip() and len(t.strip()) >= 2],
            ArrayType(StringType())
        )
        skill_sequences = skill_sequences.withColumn(
            'skill_tokens', 
            clean_tokens_udf(col('skill_tokens'))
        )
        
        # Filter sequences with at least 2 skills
        skill_sequences = skill_sequences.filter(size(col('skill_tokens')) >= 2)
        
        print(f"Training on {skill_sequences.count()} skill sequences")
        print(f"  Parameters:")
        print(f"    - vectorSize: {vector_size}")
        print(f"    - minCount: {min_count}")
        print(f"    - windowSize: {window_size}")
        
        # Train Word2Vec
        word2vec = Word2Vec(
            inputCol="skill_tokens",
            outputCol="skill_vector",
            vectorSize=vector_size,
            minCount=min_count,
            windowSize=window_size,
            seed=42
        )
        
        import time
        start_time = time.time()
        self.word2vec_model = word2vec.fit(skill_sequences)
        training_time = time.time() - start_time
        
        print(f"✓ Word2Vec model trained in {training_time:.2f} seconds")
        
        # Get vocabulary size
        vocab_size = self.word2vec_model.getVectors().count()
        print(f"  Vocabulary size: {vocab_size} skills")
        
        # Store skill vectors for later use
        self.skill_vectors = self.word2vec_model.getVectors()
        self.skill_vectors.cache()
        
        return self
    
    def train_lda_topic_model(self, num_topics=10, max_iter=20):
        """Train LDA model for skill topic/cluster discovery"""
        print(f"\n{'='*60}")
        print("Training LDA Topic Model for skill clusters...")
        print(f"{'='*60}")
        
        # Prepare skill documents (each job is a document)
        skill_docs = self.df.filter(
            (col('skills').isNotNull()) & 
            (length(col('skills')) > 0)
        ).select(
            split(lower(col('skills')), ',').alias('skill_tokens')
        )
        
        # Clean tokens
        clean_tokens_udf = udf(
            lambda tokens: [t.strip() for t in tokens if t.strip() and len(t.strip()) >= 2],
            ArrayType(StringType())
        )
        skill_docs = skill_docs.withColumn(
            'skill_tokens', 
            clean_tokens_udf(col('skill_tokens'))
        )
        
        # CountVectorizer for LDA
        count_vectorizer = CountVectorizer(
            inputCol="skill_tokens",
            outputCol="skill_counts",
            minDF=5,
            vocabSize=1000
        )
        
        cv_model = count_vectorizer.fit(skill_docs)
        skill_docs_vectorized = cv_model.transform(skill_docs)
        
        print(f"Training LDA with {num_topics} topics...")
        print(f"  Vocabulary size: {len(cv_model.vocabulary)}")
        print(f"  Max iterations: {max_iter}")
        
        # Train LDA
        lda = LDA(
            k=num_topics,
            maxIter=max_iter,
            featuresCol="skill_counts",
            seed=42
        )
        
        import time
        start_time = time.time()
        self.lda_model = lda.fit(skill_docs_vectorized)
        training_time = time.time() - start_time
        
        print(f"✓ LDA model trained in {training_time:.2f} seconds")
        
        # Get log likelihood and perplexity
        ll = self.lda_model.logLikelihood(skill_docs_vectorized)
        lp = self.lda_model.logPerplexity(skill_docs_vectorized)
        print(f"  Log Likelihood: {ll:.2f}")
        print(f"  Log Perplexity: {lp:.2f}")
        
        # Display topics
        print(f"\n{'='*60}")
        print("DISCOVERED SKILL CLUSTERS (Topics)")
        print(f"{'='*60}")
        
        topics = self.lda_model.describeTopics(maxTermsPerTopic=8)
        vocabulary = cv_model.vocabulary
        
        self.skill_clusters = []
        for topic in topics.collect():
            topic_id = topic['topic']
            term_indices = topic['termIndices']
            term_weights = topic['termWeights']
            
            terms = [vocabulary[i] for i in term_indices]
            cluster_info = {
                'topic_id': topic_id,
                'skills': terms,
                'weights': term_weights
            }
            self.skill_clusters.append(cluster_info)
            
            print(f"\nCluster {topic_id + 1}:")
            for skill, weight in zip(terms, term_weights):
                print(f"  - {skill:30s} ({weight:.4f})")
        
        return self
    
    def find_similar_skills(self, skill_name, top_n=10):
        """Find skills similar to a given skill using Word2Vec"""
        print(f"\n{'='*60}")
        print(f"Finding skills similar to: '{skill_name}'")
        print(f"{'='*60}")
        
        if self.word2vec_model is None:
            print("Error: Word2Vec model not trained. Call train_word2vec() first.")
            return None
        
        skill_name_clean = skill_name.lower().strip()
        
        try:
            # Use findSynonyms to get similar words
            synonyms = self.word2vec_model.findSynonyms(skill_name_clean, top_n)
            
            print(f"\nTop {top_n} similar skills:")
            results = synonyms.collect()
            for i, row in enumerate(results, 1):
                print(f"  {i:2d}. {row['word']:40s} (similarity: {row['similarity']:.4f})")
            
            return synonyms
            
        except Exception as e:
            print(f"Skill '{skill_name}' not found in vocabulary.")
            print(f"Try one of these common skills:")
            common_skills = self.skill_vectors.limit(10).collect()
            for row in common_skills:
                print(f"  - {row['word']}")
            return None
    
    def autocomplete_skills(self, prefix, top_n=10):
        """Autocomplete skills based on prefix"""
        print(f"\n{'='*60}")
        print(f"Autocomplete for: '{prefix}'")
        print(f"{'='*60}")
        
        prefix_lower = prefix.lower().strip()
        
        # Search in skills DataFrame
        matches = self.skills_df.filter(
            col('skill').startswith(prefix_lower)
        ).orderBy(desc('frequency')).limit(top_n)
        
        results = matches.collect()
        
        if results:
            print(f"\nSuggested skills:")
            for i, row in enumerate(results, 1):
                print(f"  {i:2d}. {row['skill']:40s} (used in {row['frequency']} jobs)")
        else:
            print(f"No skills found starting with '{prefix}'")
        
        return matches
    
    def recommend_skills_for_job(self, job_title, current_skills=None, top_n=10):
        """Recommend skills based on job title and current skills"""
        print(f"\n{'='*60}")
        print(f"Recommending skills for: '{job_title}'")
        if current_skills:
            print(f"Current skills: {current_skills}")
        print(f"{'='*60}")
        
        job_title_lower = job_title.lower()
        
        # Find similar job postings
        similar_jobs = self.df.filter(
            (col('skills').isNotNull()) & 
            (length(col('skills')) > 0) &
            (lower(col('job_title')).contains(job_title_lower))
        )
        
        job_count = similar_jobs.count()
        print(f"Found {job_count} similar job postings")
        
        if job_count == 0:
            print("No matching jobs found. Try a different job title.")
            return None
        
        # Extract all skills from similar jobs
        all_skills = similar_jobs.select(
            explode(split(col('skills'), ',')).alias('skill_raw')
        ).withColumn(
            'skill',
            trim(lower(col('skill_raw')))
        ).filter(
            length(col('skill')) >= 2
        )
        
        # Count skill frequencies
        skill_freq = all_skills.groupBy('skill').agg(
            count('*').alias('frequency')
        ).orderBy(desc('frequency'))
        
        # Filter out current skills if provided
        if current_skills:
            current_skills_lower = [s.lower().strip() for s in current_skills]
            skill_freq = skill_freq.filter(~col('skill').isin(current_skills_lower))
        
        recommendations = skill_freq.limit(top_n)
        results = recommendations.collect()
        
        print(f"\nRecommended skills:")
        for i, row in enumerate(results, 1):
            pct = (row['frequency'] / job_count) * 100
            print(f"  {i:2d}. {row['skill']:40s} ({pct:.1f}% of similar jobs)")
        
        return recommendations
    
    def analyze_skill_gap(self, current_skills, target_job_title, top_n=10):
        """Analyze skill gap between current skills and target job"""
        print(f"\n{'='*60}")
        print("SKILL GAP ANALYSIS")
        print(f"{'='*60}")
        print(f"Target position: {target_job_title}")
        print(f"Current skills: {', '.join(current_skills)}")
        
        # Get required skills for target job
        target_skills = self.recommend_skills_for_job(
            target_job_title, 
            current_skills=None, 
            top_n=30
        )
        
        if target_skills is None:
            return None
        
        target_skills_list = [row['skill'] for row in target_skills.collect()]
        current_skills_lower = [s.lower().strip() for s in current_skills]
        
        # Find missing skills
        missing_skills = [s for s in target_skills_list if s not in current_skills_lower]
        matching_skills = [s for s in target_skills_list if s in current_skills_lower]
        
        print(f"\n✓ Skills you already have ({len(matching_skills)}):")
        for skill in matching_skills[:10]:
            print(f"  ✓ {skill}")
        
        print(f"\n✗ Skills to acquire ({len(missing_skills)}):")
        for i, skill in enumerate(missing_skills[:top_n], 1):
            print(f"  {i:2d}. {skill}")
        
        # Calculate readiness score
        if len(target_skills_list) > 0:
            readiness = (len(matching_skills) / len(target_skills_list)) * 100
            print(f"\n📊 Career Readiness Score: {readiness:.1f}%")
        
        return {
            'matching_skills': matching_skills,
            'missing_skills': missing_skills[:top_n],
            'readiness_score': readiness if len(target_skills_list) > 0 else 0
        }
    
    def get_career_path_recommendations(self, current_position, current_skills, top_n=5):
        """Recommend career paths based on current position and skills"""
        print(f"\n{'='*60}")
        print("CAREER PATH RECOMMENDATIONS")
        print(f"{'='*60}")
        print(f"Current position: {current_position}")
        print(f"Current skills: {', '.join(current_skills[:5])}...")
        
        current_skills_lower = set(s.lower().strip() for s in current_skills)
        
        # Find jobs that require similar skills
        jobs_with_skills = self.df.filter(
            (col('skills').isNotNull()) & 
            (length(col('skills')) > 0)
        ).select(
            'job_title',
            'position_level',
            'avg_salary',
            'skills'
        )
        
        # Calculate skill match score for each job
        def calculate_skill_match(skills_str):
            if not skills_str:
                return 0.0
            job_skills = set(s.lower().strip() for s in skills_str.split(','))
            if len(job_skills) == 0:
                return 0.0
            overlap = len(current_skills_lower.intersection(job_skills))
            return float(overlap) / len(job_skills)
        
        skill_match_udf = udf(calculate_skill_match, DoubleType())
        
        jobs_scored = jobs_with_skills.withColumn(
            'skill_match',
            skill_match_udf(col('skills'))
        )
        
        # Filter jobs with good match but different from current
        from pyspark.sql.functions import avg as spark_avg
        
        career_options = jobs_scored.filter(
            (col('skill_match') >= 0.3) &
            (col('skill_match') < 1.0) &
            (~lower(col('job_title')).contains(current_position.lower()))
        ).groupBy('job_title', 'position_level').agg(
            count('*').alias('job_count'),
            spark_avg('skill_match').alias('avg_skill_match'),
            spark_avg('avg_salary').alias('avg_salary')
        ).orderBy(desc('avg_skill_match'), desc('avg_salary')) \
         .limit(top_n * 3)
        
        results = career_options.collect()
        
        print(f"\nPotential career paths:")
        shown = 0
        for row in results:
            if shown >= top_n:
                break
            job_title = row['job_title']
            if job_title and len(job_title) > 3:
                shown += 1
                match_pct = row['avg_skill_match'] * 100
                salary = row['avg_salary'] if row['avg_salary'] else 0
                print(f"\n  {shown}. {job_title[:50]}")
                print(f"     Level: {row['position_level'] or 'N/A'}")
                print(f"     Skill Match: {match_pct:.1f}%")
                if salary > 0:
                    print(f"     Avg Salary: {salary:.1f}M VND")
        
        return career_options
    
    def save_model(self, model_path="models/skills_recommender"):
        """Save Word2Vec and LDA models to disk"""
        print(f"\n{'='*60}")
        print(f"Saving models to {model_path}...")
        print(f"{'='*60}")
        
        import os
        
        # Create directories if they don't exist
        os.makedirs(model_path, exist_ok=True)
        
        saved_paths = {}
        
        try:
            # Save Word2Vec model
            if self.word2vec_model is not None:
                word2vec_path = os.path.join(model_path, "word2vec")
                self.word2vec_model.write().overwrite().save(word2vec_path)
                saved_paths['word2vec'] = word2vec_path
                print(f"✓ Word2Vec model saved to {word2vec_path}")
                
                # Verify the model was saved
                metadata_path = os.path.join(word2vec_path, "metadata")
                if os.path.exists(metadata_path):
                    print(f"  ✓ Word2Vec metadata directory created")
            else:
                print("⚠ Word2Vec model not trained, skipping save")
            
            # Save LDA model
            if self.lda_model is not None:
                lda_path = os.path.join(model_path, "lda")
                self.lda_model.write().overwrite().save(lda_path)
                saved_paths['lda'] = lda_path
                print(f"✓ LDA model saved to {lda_path}")
                
                # Verify the model was saved
                metadata_path = os.path.join(lda_path, "metadata")
                if os.path.exists(metadata_path):
                    print(f"  ✓ LDA metadata directory created")
            else:
                print("⚠ LDA model not trained, skipping save")
            
            # Save skill vectors as Parquet for fast lookup
            if self.skill_vectors is not None:
                vectors_path = os.path.join(model_path, "skill_vectors")
                self.skill_vectors.write.mode("overwrite").parquet(vectors_path)
                saved_paths['skill_vectors'] = vectors_path
                print(f"✓ Skill vectors saved to {vectors_path}")
            
            # Save skills DataFrame
            if self.skills_df is not None:
                skills_path = os.path.join(model_path, "skills_df")
                self.skills_df.write.mode("overwrite").parquet(skills_path)
                saved_paths['skills_df'] = skills_path
                print(f"✓ Skills DataFrame saved to {skills_path}")
            
            # Save skill clusters to JSON
            if self.skill_clusters:
                import json
                clusters_path = os.path.join(model_path, "skill_clusters.json")
                with open(clusters_path, 'w', encoding='utf-8') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    clusters_serializable = []
                    for cluster in self.skill_clusters:
                        clusters_serializable.append({
                            'topic_id': cluster['topic_id'],
                            'skills': cluster['skills'],
                            'weights': [float(w) for w in cluster['weights']]
                        })
                    json.dump(clusters_serializable, f, indent=2, ensure_ascii=False)
                saved_paths['skill_clusters'] = clusters_path
                print(f"✓ Skill clusters saved to {clusters_path}")
            
            print(f"\n✓ All models saved successfully to {model_path}")
            return model_path, saved_paths
            
        except Exception as e:
            import traceback
            print(f"✗ Error saving models: {e}")
            traceback.print_exc()
            return None, None
    
    def load_model(self, model_path="models/skills_recommender"):
        """Load Word2Vec and LDA models from disk"""
        print(f"\n{'='*60}")
        print(f"Loading models from {model_path}...")
        print(f"{'='*60}")
        
        import os
        
        if not os.path.exists(model_path):
            print(f"Error: Model path does not exist: {model_path}")
            return False
        
        loaded_models = {}
        
        try:
            # Load Word2Vec model
            word2vec_path = os.path.join(model_path, "word2vec")
            if os.path.exists(word2vec_path):
                self.word2vec_model = Word2VecModel.load(word2vec_path)
                loaded_models['word2vec'] = True
                print(f"✓ Word2Vec model loaded from {word2vec_path}")
                
                # Reload skill vectors from model
                self.skill_vectors = self.word2vec_model.getVectors()
                self.skill_vectors.cache()
                print(f"  ✓ Vocabulary size: {self.skill_vectors.count()}")
            else:
                print(f"⚠ Word2Vec model not found at {word2vec_path}")
            
            # Load LDA model
            lda_path = os.path.join(model_path, "lda")
            if os.path.exists(lda_path):
                from pyspark.ml.clustering import DistributedLDAModel, LocalLDAModel
                # Try DistributedLDAModel first (default), then LocalLDAModel
                try:
                    self.lda_model = DistributedLDAModel.load(lda_path)
                    loaded_models['lda'] = True
                    print(f"✓ LDA model loaded from {lda_path} (Distributed)")
                except Exception:
                    try:
                        self.lda_model = LocalLDAModel.load(lda_path)
                        loaded_models['lda'] = True
                        print(f"✓ LDA model loaded from {lda_path} (Local)")
                    except Exception as e:
                        print(f"⚠ Could not load LDA model: {e}")
            else:
                print(f"⚠ LDA model not found at {lda_path}")
            
            # Load skill vectors (fallback if Word2Vec reload didn't work)
            vectors_path = os.path.join(model_path, "skill_vectors")
            if os.path.exists(vectors_path) and self.skill_vectors is None:
                self.skill_vectors = self.spark.read.parquet(vectors_path)
                self.skill_vectors.cache()
                loaded_models['skill_vectors'] = True
                print(f"✓ Skill vectors loaded from {vectors_path}")
            
            # Load skills DataFrame
            skills_path = os.path.join(model_path, "skills_df")
            if os.path.exists(skills_path):
                self.skills_df = self.spark.read.parquet(skills_path)
                self.skills_df.cache()
                loaded_models['skills_df'] = True
                print(f"✓ Skills DataFrame loaded from {skills_path}")
            else:
                print(f"⚠ Skills DataFrame not found at {skills_path}")
            
            # Load skill clusters
            import json
            clusters_path = os.path.join(model_path, "skill_clusters.json")
            if os.path.exists(clusters_path):
                with open(clusters_path, 'r', encoding='utf-8') as f:
                    self.skill_clusters = json.load(f)
                loaded_models['skill_clusters'] = True
                print(f"✓ Skill clusters loaded from {clusters_path}")
            else:
                print(f"⚠ Skill clusters not found at {clusters_path}")
            
            success = len(loaded_models) > 0
            if success:
                print(f"\n✓ Models loaded successfully from {model_path}")
            else:
                print(f"\n✗ No models found at {model_path}")
            
            return success
            
        except Exception as e:
            import traceback
            print(f"✗ Error loading models: {e}")
            traceback.print_exc()
            return False
    
    def save_model_metadata(self, model_name, model_path="", metrics=None):
        """Save model metadata to Cassandra"""
        print(f"\n{'='*60}")
        print("Saving model metadata to Cassandra...")
        print(f"{'='*60}")
        
        try:
            from pyspark.sql import Row
            
            model_id = str(uuid.uuid4())
            training_date = datetime.now()
            
            vocab_size = self.skill_vectors.count() if self.skill_vectors else 0
            num_clusters = len(self.skill_clusters) if self.skill_clusters else 0
            
            # Use provided model_path or default
            if not model_path:
                model_path = "/opt/spark/work-dir/models/skills_recommender"
            
            metadata_row = Row(
                model_id=model_id,
                model_name=str(model_name),
                model_type="Word2Vec_LDA_Skills",
                training_date=training_date,
                accuracy=float(vocab_size),  # Using vocab size as a metric
                mae=float(0),
                rmse=float(0),
                r2_score=float(num_clusters),  # Using num clusters
                feature_columns=['skills', 'job_title', 'job_fields'],
                model_path=str(model_path),
                version=int(1)
            )
            
            metadata_df = self.spark.createDataFrame([metadata_row])
            
            metadata_df.write \
                .format("org.apache.spark.sql.cassandra") \
                .options(table="ml_models", keyspace="jobdb") \
                .mode("append") \
                .save()
            
            print(f"✓ Model metadata saved with ID: {model_id}")
            print(f"  Model Path: {model_path}")
            print(f"  Vocabulary Size: {vocab_size}")
            print(f"  Num Clusters: {num_clusters}")
            return model_id
            
        except Exception as e:
            import traceback
            print(f"Error saving model metadata: {e}")
            traceback.print_exc()
            return None
    
    def get_latest_model_from_cassandra(self, keyspace='jobdb', table='ml_models'):
        """Get the latest skills recommender model metadata from Cassandra"""
        try:
            print(f"\nFetching latest skills model from {keyspace}.{table}...")
            
            df = self.spark.read \
                .format("org.apache.spark.sql.cassandra") \
                .options(table=table, keyspace=keyspace) \
                .load()
            
            # Filter for skills recommender models only
            df = df.filter(col('model_type') == 'Word2Vec_LDA_Skills')
            
            if df.count() == 0:
                print("No skills recommender models found in Cassandra")
                return None
            
            # Get the latest model by training_date
            from pyspark.sql.functions import desc
            latest = df.orderBy(desc('training_date')).first()
            
            if latest:
                model_info = {
                    'model_id': str(latest['model_id']),
                    'model_name': latest['model_name'],
                    'model_type': latest['model_type'],
                    'training_date': latest['training_date'],
                    'accuracy': latest['accuracy'],  # vocab size
                    'r2_score': latest['r2_score'],  # num clusters
                    'model_path': latest['model_path'],
                    'version': latest['version']
                }
                print(f"✓ Found model: {model_info['model_name']}")
                return model_info
            
            return None
            
        except Exception as e:
            import traceback
            print(f"Error loading model metadata: {e}")
            traceback.print_exc()
            return None
    
    def load_model_from_cassandra(self, keyspace='jobdb', table='ml_models'):
        """Load model from disk using path stored in Cassandra"""
        try:
            print(f"\n{'='*60}")
            print("Loading Skills Recommender from Cassandra")
            print(f"{'='*60}")
            
            # Get model metadata
            model_info = self.get_latest_model_from_cassandra(keyspace=keyspace, table=table)
            
            if not model_info:
                print("No skills recommender model found in Cassandra")
                return False, None
            
            print(f"\n✓ Found model metadata:")
            print(f"   Model ID: {model_info['model_id']}")
            print(f"   Model Name: {model_info['model_name']}")
            print(f"   Model Path: {model_info['model_path']}")
            print(f"   Vocabulary Size: {model_info['accuracy']:.0f}")
            print(f"   Num Clusters: {model_info['r2_score']:.0f}")
            
            model_path = model_info['model_path']
            
            # Load models from disk
            success = self.load_model(model_path)
            
            if success:
                print(f"\n✓ Skills Recommender loaded successfully!")
                return True, model_info
            else:
                print(f"\n✗ Failed to load models from {model_path}")
                return False, model_info
            
        except Exception as e:
            import traceback
            print(f"Error loading model from Cassandra: {e}")
            traceback.print_exc()
            return False, None
    
    def close(self):
        """Close Spark session"""
        if self.spark:
            self.spark.stop()
            print("\nSpark session closed")


def main():
    """Main training and demonstration pipeline"""
    print("="*60)
    print("SKILLS EXTRACTION & RECOMMENDATION SYSTEM")
    print("Using Word2Vec, TF-IDF, and LDA Topic Modeling")
    print("="*60)
    
    # Initialize recommender
    recommender = SkillsRecommenderPySpark(cassandra_host='cassandra', cassandra_port=9042)
    
    # Load data from Cassandra
    result = recommender.load_data_from_cassandra(keyspace='job_analytics', limit=50000)
    
    if result is None or recommender.df is None or recommender.df.count() == 0:
        print("No data found in Cassandra. Make sure data is being streamed.")
        recommender.close()
        return
    
    # Extract skills
    recommender.extract_skills()
    
    # Train Word2Vec model
    recommender.train_word2vec(vector_size=100, min_count=5, window_size=5)
    
    # Train LDA topic model
    recommender.train_lda_topic_model(num_topics=8, max_iter=15)
    
    # Demo 1: Find similar skills
    print("\n" + "="*60)
    print("DEMO 1: Find Similar Skills")
    print("="*60)
    
    demo_skills = ['python', 'excel', 'marketing', 'java']
    for skill in demo_skills:
        recommender.find_similar_skills(skill, top_n=5)
    
    # Demo 2: Skill autocomplete
    print("\n" + "="*60)
    print("DEMO 2: Skill Autocomplete")
    print("="*60)
    
    prefixes = ['java', 'mark', 'python']
    for prefix in prefixes:
        recommender.autocomplete_skills(prefix, top_n=5)
    
    # Demo 3: Skills recommendation for job
    print("\n" + "="*60)
    print("DEMO 3: Skills Recommendation for Job")
    print("="*60)
    
    recommender.recommend_skills_for_job(
        job_title="developer",
        current_skills=['html', 'css'],
        top_n=10
    )
    
    # Demo 4: Skill gap analysis
    print("\n" + "="*60)
    print("DEMO 4: Skill Gap Analysis")
    print("="*60)
    
    recommender.analyze_skill_gap(
        current_skills=['python', 'sql', 'excel'],
        target_job_title="data analyst",
        top_n=10
    )
    
    # Demo 5: Career path recommendations
    print("\n" + "="*60)
    print("DEMO 5: Career Path Recommendations")
    print("="*60)
    
    recommender.get_career_path_recommendations(
        current_position="junior developer",
        current_skills=['python', 'javascript', 'html', 'css', 'git'],
        top_n=5
    )
    
    # Save models to disk
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(script_dir, "models", "skills_recommender")
    
    # For Docker, use absolute path
    if os.path.exists("/.dockerenv"):
        model_save_path = "/opt/spark/work-dir/models/skills_recommender"
    
    saved_path, saved_components = recommender.save_model(model_path=model_save_path)
    
    if saved_path:
        # Save model metadata to Cassandra with the actual path
        recommender.save_model_metadata(
            model_name="Skills_Recommender_W2V_LDA",
            model_path=saved_path
        )
    
    print("\n" + "="*60)
    print("Skills Recommendation System Ready!")
    print("="*60)
    print(f"Models saved to: {saved_path}")
    
    recommender.close()


if __name__ == "__main__":
    main()
