"""
ML Training Script using PySpark MLlib
Reads data from Cassandra and trains model using Spark's distributed ML capabilities
"""

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, when, isnan, isnull, length, size, split, lit, mean as spark_mean
from pyspark.sql.types import DoubleType, IntegerType, StringType, TimestampType
import uuid
from datetime import datetime
import warnings
import os
import sys
warnings.filterwarnings('ignore')

# Fix for Windows: Set HADOOP_HOME if not set and disable native IO
if sys.platform == 'win32':
    # Disable Hadoop native IO to avoid UnsatisfiedLinkError on Windows
    # These must be set BEFORE SparkSession is created
    hadoop_opts = os.environ.get('HADOOP_OPTS', '')
    if '-Djava.library.path=' not in hadoop_opts:
        os.environ['HADOOP_OPTS'] = hadoop_opts + ' -Djava.library.path='
    if '-Dhadoop.native.lib=false' not in hadoop_opts:
        os.environ['HADOOP_OPTS'] = os.environ['HADOOP_OPTS'] + ' -Dhadoop.native.lib=false'
    if '-Dhadoop.io.native.lib.available=false' not in hadoop_opts:
        os.environ['HADOOP_OPTS'] = os.environ['HADOOP_OPTS'] + ' -Dhadoop.io.native.lib.available=false'
    if '-Dio.native.lib.available=false' not in hadoop_opts:
        os.environ['HADOOP_OPTS'] = os.environ['HADOOP_OPTS'] + ' -Dio.native.lib.available=false'
    os.environ['HADOOP_COMMON_LIB_NATIVE_DIR'] = ''
    # Disable native code loader
    os.environ['HADOOP_USE_NATIVE'] = 'false'
    
    if 'HADOOP_HOME' not in os.environ:
        # Try to find hadoop directory in the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        hadoop_home = os.path.join(script_dir, 'hadoop')
        bin_dir = os.path.join(hadoop_home, 'bin')
        winutils_path = os.path.join(bin_dir, 'winutils.exe')
        
        # If winutils exists in script directory, use it
        if os.path.exists(winutils_path):
            os.environ['HADOOP_HOME'] = hadoop_home
            os.environ['hadoop.home.dir'] = hadoop_home
        else:
            # Fallback: use temp directory (may not work without winutils)
            import tempfile
            hadoop_home = os.path.join(tempfile.gettempdir(), 'hadoop')
            bin_dir = os.path.join(hadoop_home, 'bin')
            os.makedirs(bin_dir, exist_ok=True)
            os.environ['HADOOP_HOME'] = hadoop_home
            os.environ['hadoop.home.dir'] = hadoop_home
            print(f"[WARNING] HADOOP_HOME set to {hadoop_home} but winutils.exe not found.")
            print("  Run 'python spark/app/setup_windows_hadoop.py' to download winutils.exe")


class MLTrainerFromCassandraPySpark:
    def __init__(self, cassandra_host='cassandra', cassandra_port=9042, spark_master='local[*]'):
        """Initialize Spark session with Cassandra connector"""
        self.cassandra_host = cassandra_host
        self.cassandra_port = cassandra_port
        
        # Build SparkSession configuration
        # Try Scala 2.13 connector first (for Spark 4.x), fallback to 2.12 (for Spark 3.x)
        import pyspark
        spark_version = pyspark.__version__
        if spark_version.startswith('4.'):
            # Spark 4.x uses Scala 2.13
            connector_package = "com.datastax.spark:spark-cassandra-connector_2.13:3.5.0"
        else:
            # Spark 3.x uses Scala 2.12
            connector_package = "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0"
        
        builder = SparkSession.builder \
            .appName("MLTrainingFromCassandra") \
            .master(spark_master) \
            .config("spark.jars.packages", connector_package) \
            .config("spark.cassandra.connection.host", cassandra_host) \
            .config("spark.cassandra.connection.port", str(cassandra_port)) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.session.timeZone", "UTC") \
            .config("spark.sql.parquet.int96RebaseModeInRead", "LEGACY") \
            .config("spark.sql.parquet.int96RebaseModeInWrite", "LEGACY")
        
        # Windows-specific configurations
        if sys.platform == 'win32':
            # Set hadoop.home.dir in Spark config
            if 'HADOOP_HOME' in os.environ:
                builder = builder.config("spark.hadoop.hadoop.home.dir", os.environ['HADOOP_HOME'])
            # Disable Hadoop file system checks for local mode
            builder = builder.config("spark.hadoop.fs.defaultFS", "file:///")
            # Suppress temp file deletion warnings on Windows
            builder = builder.config("spark.local.dir", os.path.join(os.environ.get('TEMP', os.environ.get('TMP', '/tmp')), 'spark-temp'))
            # Disable Hadoop native IO on Windows (fixes UnsatisfiedLinkError when loading models)
            # Set Java system properties to disable native IO
            builder = builder.config("spark.driver.extraJavaOptions", 
                "-Djava.library.path= -Dhadoop.native.lib=false -Dhadoop.io.native.lib.available=false -Dio.native.lib.available=false")
            builder = builder.config("spark.executor.extraJavaOptions", 
                "-Djava.library.path= -Dhadoop.native.lib=false -Dhadoop.io.native.lib.available=false -Dio.native.lib.available=false")
            builder = builder.config("spark.hadoop.io.native.lib.available", "false")
            builder = builder.config("spark.hadoop.hadoop.native.lib", "false")
            # Use Java file system instead of native Hadoop
            builder = builder.config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
        
        self.spark = builder.getOrCreate()
        self._disable_hadoop_native_io()
        
        # Suppress SparkEnv cleanup warnings on Windows
        if sys.platform == 'win32':
            try:
                log4j = self.spark.sparkContext._jvm.org.apache.log4j
                log4j.LogManager.getLogger("org.apache.spark.SparkEnv").setLevel(log4j.Level.ERROR)
                log4j.LogManager.getLogger("org.apache.spark.util.SparkFileUtils").setLevel(log4j.Level.ERROR)
                # Suppress Hadoop NativeIO warnings
                log4j.LogManager.getLogger("org.apache.hadoop.io.nativeio").setLevel(log4j.Level.ERROR)
            except:
                pass  # Ignore if log4j not available
        
        # Set log levels to reduce Cassandra connector verbosity
        self.spark.sparkContext.setLogLevel("WARN")
        # Suppress Cassandra connector and driver logs (including CqlRequestHandler query logs)
        log4j = self.spark.sparkContext._jvm.org.apache.log4j
        log4j.LogManager.getLogger("com.datastax.spark.connector").setLevel(log4j.Level.WARN)
        log4j.LogManager.getLogger("com.datastax.oss.driver").setLevel(log4j.Level.WARN)
        log4j.LogManager.getLogger("org.apache.cassandra").setLevel(log4j.Level.WARN)
        # Specifically suppress CqlRequestHandler query logs
        log4j.LogManager.getLogger("com.datastax.oss.driver.internal.core.cql.CqlRequestHandler").setLevel(log4j.Level.ERROR)
        self.model = None
        self.scaler = None
        self.pipeline = None

    def _disable_hadoop_native_io(self):
        """Best-effort disable of Hadoop native IO on Windows."""
        if sys.platform != 'win32':
            return
        try:
            self.spark.conf.set("spark.hadoop.io.native.lib.available", "false")
            self.spark.conf.set("spark.hadoop.hadoop.native.lib", "false")
            hconf = self.spark.sparkContext._jsc.hadoopConfiguration()
            hconf.set("io.native.lib.available", "false")
            hconf.set("hadoop.native.lib", "false")
        except Exception:
            pass
        
    def load_data_from_cassandra(self, keyspace='job_analytics', table='job_postings', limit=None):
        """Load data from Cassandra directly into Spark DataFrame"""
        print(f"\n{'='*60}")
        print(f"Loading data from {keyspace}.{table}...")
        print(f"{'='*60}")
        
        try:
            # Read directly from Cassandra using Spark Cassandra connector
            df = self.spark.read \
                .format("org.apache.spark.sql.cassandra") \
                .options(table=table, keyspace=keyspace) \
                .load()
            
            if limit:
                df = df.limit(limit)
            
            count = df.count()
            print(f"✓ Loaded {count} records from Cassandra")
            print(f"  Columns: {len(df.columns)}")
            
            return df
            
        except Exception as e:
            import traceback
            print(f"Error loading data: {e}")
            traceback.print_exc()
            return None
    
    def preprocess_data(self, df):
        """Preprocess data for ML training"""
        print(f"\n{'='*60}")
        print("Preprocessing data...")
        print(f"{'='*60}")
        initial_count = df.count()
        print(f"Initial records: {initial_count}")
        
        # Handle missing values
        print("Handling missing values...")
        df = df.fillna({'skills': '', 'job_fields': '', 'city': 'unknown'})
        print("✓ Missing values handled")
        
        # Filter valid data
        print("Filtering valid salary data...")
        before_filter = df.count()
        df = df.filter(
            (col('avg_salary') > 0) & 
            (col('salary_min') > 0) & 
            (col('salary_max') > 0) &
            (col('salary_min') <= col('salary_max'))
        )
        filtered_out = before_filter - df.count()
        print(f"✓ Filtered out {filtered_out} invalid records")
        
        # Remove outliers (top and bottom 1%)
        print("Removing outliers (top and bottom 1%)...")
        quantiles = df.approxQuantile('avg_salary', [0.01, 0.99], 0.25)
        q1, q99 = quantiles[0], quantiles[1]
        before_outlier = df.count()
        df = df.filter((col('avg_salary') >= q1) & (col('avg_salary') <= q99))
        outliers_removed = before_outlier - df.count()
        print(f"✓ Removed {outliers_removed} outliers")
        
        final_count = df.count()
        print(f"\nFinal records after preprocessing: {final_count} ({final_count/initial_count*100:.1f}% of original)")
        return df
    
    def prepare_features(self, df):
        """Prepare features for ML model using PySpark transformers"""
        print(f"\n{'='*60}")
        print("Preparing features...")
        print(f"{'='*60}")
        
        # Create feature columns if they don't exist
        if 'num_skills' not in df.columns:
            df = df.withColumn('num_skills', 
                when(col('skills').isNotNull(), size(split(col('skills'), ',')))
                .otherwise(lit(0)))
        
        if 'num_fields' not in df.columns:
            df = df.withColumn('num_fields',
                when(col('job_fields').isNotNull(), size(split(col('job_fields'), ',')))
                .otherwise(lit(0)))
        
        if 'title_length' not in df.columns:
            df = df.withColumn('title_length', length(col('job_title')))
        
        # String indexers for categorical variables
        print("Encoding categorical variables...")
        city_indexer = StringIndexer(inputCol='city', outputCol='city_encoded', handleInvalid='keep')
        job_type_indexer = StringIndexer(inputCol='job_type', outputCol='job_type_encoded', handleInvalid='keep')
        position_indexer = StringIndexer(inputCol='position_level', outputCol='position_encoded', handleInvalid='keep')
        experience_indexer = StringIndexer(inputCol='experience', outputCol='experience_encoded', handleInvalid='keep')
        
        # Fit indexers
        df = city_indexer.fit(df).transform(df)
        print(f"  ✓ Encoded city")
        df = job_type_indexer.fit(df).transform(df)
        print(f"  ✓ Encoded job_type")
        df = position_indexer.fit(df).transform(df)
        print(f"  ✓ Encoded position_level")
        df = experience_indexer.fit(df).transform(df)
        print(f"  ✓ Encoded experience")
        
        # Feature columns
        feature_cols = [
            'city_encoded',
            'job_type_encoded',
            'position_encoded',
            'experience_encoded',
            'num_skills',
            'num_fields',
            'title_length'
        ]
        
        # Assemble features into a vector
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol='features',
            handleInvalid='skip'
        )
        
        df = assembler.transform(df)
        
        # Get statistics
        stats = df.select(
            spark_mean('avg_salary').alias('avg_salary_mean'),
            spark_mean('num_skills').alias('num_skills_mean')
        ).collect()[0]
        
        print(f"\n✓ Feature preparation complete:")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Samples: {df.count()}")
        print(f"  Avg salary: {stats['avg_salary_mean']:.2f}")
        
        return df, feature_cols
    
    def train_model(self, df, test_size=0.2):
        """Train Random Forest model using PySpark MLlib"""
        print(f"\n{'='*60}")
        print("Training Random Forest Model with PySpark...")
        print(f"{'='*60}")
        
        # Split data
        print("Splitting data into train/test sets...")
        train_df, test_df = df.randomSplit([1 - test_size, test_size], seed=42)
        train_count = train_df.count()
        test_count = test_df.count()
        total = train_count + test_count
        print(f"  Training set: {train_count} samples ({train_count/total*100:.1f}%)")
        print(f"  Test set: {test_count} samples ({test_count/total*100:.1f}%)")
        
        # Scale features
        print("\nScaling features...")
        scaler = StandardScaler(
            inputCol='features',
            outputCol='scaled_features',
            withStd=True,
            withMean=True
        )
        scaler_model = scaler.fit(train_df)
        self.scaler = scaler_model  # Store scaler for later use
        train_df = scaler_model.transform(train_df)
        test_df = scaler_model.transform(test_df)
        print("✓ Features scaled")
        
        # Determine optimal parameters based on dataset size
        n_samples = train_count
        if n_samples < 500:
            num_trees = 30
            max_depth = 8
        elif n_samples < 1000:
            num_trees = 40
            max_depth = 10
        elif n_samples < 2000:
            num_trees = 50
            max_depth = 12
        elif n_samples < 5000:
            num_trees = 60
            max_depth = 15
        else:
            num_trees = 75
            max_depth = 18
        
        # Train model
        print("\nTraining Random Forest...")
        print("  Parameters (optimized for speed):")
        print(f"    - numTrees: {num_trees}")
        print(f"    - maxDepth: {max_depth}")
        print(f"    - maxBins: 32")
        print(f"    - seed: 42")
        
        rf = RandomForestRegressor(
            featuresCol='scaled_features',
            labelCol='avg_salary',
            numTrees=num_trees,
            maxDepth=max_depth,
            maxBins=32,
            seed=42
        )
        
        print("\nTraining progress:")
        import time
        start_time = time.time()
        self.model = rf.fit(train_df)
        training_time = time.time() - start_time
        print(f"✓ Model training completed in {training_time:.2f} seconds!")
        
        # Predictions
        print("\nGenerating predictions...")
        print("  Predicting on training set...", end=" ", flush=True)
        train_predictions = self.model.transform(train_df)
        print("✓")
        print("  Predicting on test set...", end=" ", flush=True)
        test_predictions = self.model.transform(test_df)
        print("✓")
        
        # Evaluate
        print("\nEvaluating model performance...")
        evaluator_mae = RegressionEvaluator(labelCol='avg_salary', predictionCol='prediction', metricName='mae')
        evaluator_rmse = RegressionEvaluator(labelCol='avg_salary', predictionCol='prediction', metricName='rmse')
        evaluator_r2 = RegressionEvaluator(labelCol='avg_salary', predictionCol='prediction', metricName='r2')
        
        train_mae = evaluator_mae.evaluate(train_predictions)
        test_mae = evaluator_mae.evaluate(test_predictions)
        train_rmse = evaluator_rmse.evaluate(train_predictions)
        test_rmse = evaluator_rmse.evaluate(test_predictions)
        train_r2 = evaluator_r2.evaluate(train_predictions)
        test_r2 = evaluator_r2.evaluate(test_predictions)
        print("✓ Evaluation complete")
        
        metrics = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        return self.model, scaler_model, metrics, test_predictions, test_df
    
    def save_model_metadata(self, model_name, model_type, metrics, feature_cols, model_path=""):
        """Save model metadata to Cassandra using Spark"""
        print("\nSaving model metadata to Cassandra...")
        
        try:
            model_id = str(uuid.uuid4())
            training_date = datetime.now()
            
            # Create a DataFrame with model metadata
            from pyspark.sql import Row
            
            # Convert feature_cols list to list of strings for Cassandra LIST<TEXT>
            feature_cols_list = [str(col) for col in feature_cols] if isinstance(feature_cols, list) else feature_cols
            
            metadata_row = Row(
                model_id=model_id,  # Spark will convert string to UUID for Cassandra
                model_name=str(model_name),
                model_type=str(model_type),
                training_date=training_date,
                accuracy=float(metrics['test_r2']),
                mae=float(metrics['test_mae']),
                rmse=float(metrics['test_rmse']),
                r2_score=float(metrics['test_r2']),
                feature_columns=feature_cols_list,  # LIST<TEXT> in Cassandra
                model_path=str(model_path),
                version=int(1)
            )
            
            # Create DataFrame from single row
            metadata_df = self.spark.createDataFrame([metadata_row])
            
            # Write to Cassandra using Spark Cassandra connector
            metadata_df.write \
                .format("org.apache.spark.sql.cassandra") \
                .options(table="ml_models", keyspace="jobdb") \
                .mode("append") \
                .save()
            
            print(f"✓ Model metadata saved with ID: {model_id}")
            return model_id
        except Exception as e:
            import traceback
            print(f"Error saving model metadata: {e}")
            traceback.print_exc()
            return None
    
    def check_cassandra_connection(self, keyspace='jobdb', table='ml_models'):
        """Check if Cassandra connection and table exist
        
        Returns:
            tuple: (success: bool, message: str, count: int)
        """
        try:
            # Try to read from the table (this will fail if keyspace/table doesn't exist)
            df = self.spark.read \
                .format("org.apache.spark.sql.cassandra") \
                .options(table=table, keyspace=keyspace) \
                .load()
            
            count = df.count()
            return True, f"Connection successful! Table exists with {count} row(s)", count
        except Exception as e:
            error_msg = str(e).lower()
            
            if "keyspace" in error_msg or "does not exist" in error_msg:
                return False, f"Keyspace '{keyspace}' or table '{table}' does not exist. Please create them first.", 0
            elif "connection" in error_msg or "refused" in error_msg:
                return False, f"Cannot connect to Cassandra at {self.cassandra_host}:{self.cassandra_port}", 0
            else:
                return False, f"Error: {str(e)}", 0
    
    def get_models_from_cassandra(self, keyspace='jobdb', table='ml_models', limit=None):
        """Load model metadata from Cassandra"""
        try:
            print(f"\nLoading model metadata from Cassandra ({keyspace}.{table})...")
            print(f"Cassandra host: {self.cassandra_host}:{self.cassandra_port}")
            
            # First, try to check if keyspace exists by listing tables
            try:
                # Try to read from the table
                df = self.spark.read \
                    .format("org.apache.spark.sql.cassandra") \
                    .options(table=table, keyspace=keyspace) \
                    .load()
                
                if limit:
                    df = df.limit(limit)
                
                count = df.count()
                print(f"✓ Found {count} model(s) in Cassandra")
                
                if count > 0:
                    # Show first row for debugging
                    print("\nFirst model metadata:")
                    first_row = df.first()
                    print(f"  Model ID: {first_row.model_id}")
                    print(f"  Model Name: {first_row.model_name}")
                    print(f"  Model Path: {first_row.model_path}")
                
                return df
            except Exception as read_error:
                error_msg = str(read_error)
                print(f"\nError reading from Cassandra table {keyspace}.{table}:")
                print(f"  {error_msg}")
                
                # Check if it's a keyspace/table not found error
                if "keyspace" in error_msg.lower() or "does not exist" in error_msg.lower():
                    print(f"\n⚠ Keyspace '{keyspace}' or table '{table}' may not exist.")
                    print("  Please create the keyspace and table first:")
                    print(f"  CREATE KEYSPACE IF NOT EXISTS {keyspace} WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}};")
                    print(f"  USE {keyspace};")
                    print(f"  CREATE TABLE IF NOT EXISTS {table} (")
                    print("    model_id UUID PRIMARY KEY,")
                    print("    model_name TEXT,")
                    print("    model_type TEXT,")
                    print("    training_date TIMESTAMP,")
                    print("    accuracy DOUBLE,")
                    print("    mae DOUBLE,")
                    print("    rmse DOUBLE,")
                    print("    r2_score DOUBLE,")
                    print("    feature_columns LIST<TEXT>,")
                    print("    model_path TEXT,")
                    print("    version INT")
                    print("  );")
                
                raise read_error
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"\nError loading models from Cassandra: {e}")
            
            # Check for version compatibility issues
            if "NoSuchMethodError" in error_msg or "CollectionConverters" in error_msg:
                print("\n" + "="*60)
                print("VERSION COMPATIBILITY ISSUE DETECTED")
                print("="*60)
                print("The Cassandra connector version is incompatible with your Spark version.")
                print("\nRecommended solutions:")
                print("1. Use Docker (recommended):")
                print("   docker exec -e JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 spark-master \\")
                print("     /opt/spark/bin/spark-submit --packages com.datastax.spark:spark-cassandra-connector_2.12:3.5.0 \\")
                print("     /opt/spark/work-dir/ml_train_from_cassandra_pyspark.py")
                print("\n2. Or downgrade PySpark to 3.x:")
                print("   pip install pyspark==3.5.3")
                print("="*60)
            
            traceback.print_exc()
            return None
    
    def get_latest_model_from_cassandra(self, keyspace='jobdb', table='ml_models'):
        """Get the latest trained model metadata from Cassandra"""
        try:
            df = self.get_models_from_cassandra(keyspace=keyspace, table=table)
            
            if df is None or df.count() == 0:
                return None
            
            # Get the latest model by training_date
            from pyspark.sql.functions import desc
            latest_model = df.orderBy(desc("training_date")).first()
            
            if latest_model:
                model_info = {
                    'model_id': str(latest_model.model_id),
                    'model_name': latest_model.model_name,
                    'model_type': latest_model.model_type,
                    'model_path': latest_model.model_path,
                    'training_date': latest_model.training_date,
                    'accuracy': latest_model.accuracy,
                    'r2_score': latest_model.r2_score,
                    'mae': latest_model.mae,
                    'rmse': latest_model.rmse,
                    'feature_columns': latest_model.feature_columns,
                    'version': latest_model.version
                }
                return model_info
            return None
        except Exception as e:
            import traceback
            print(f"Error getting latest model: {e}")
            traceback.print_exc()
            return None
    
    def get_model_by_id(self, model_id, keyspace='jobdb', table='ml_models'):
        """Get model metadata by model_id"""
        try:
            df = self.get_models_from_cassandra(keyspace=keyspace, table=table)
            
            if df is None:
                return None
            
            from pyspark.sql.functions import col
            model_df = df.filter(col("model_id") == model_id)
            
            if model_df.count() == 0:
                return None
            
            model_row = model_df.first()
            model_info = {
                'model_id': str(model_row.model_id),
                'model_name': model_row.model_name,
                'model_type': model_row.model_type,
                'model_path': model_row.model_path,
                'training_date': model_row.training_date,
                'accuracy': model_row.accuracy,
                'r2_score': model_row.r2_score,
                'mae': model_row.mae,
                'rmse': model_row.rmse,
                'feature_columns': model_row.feature_columns,
                'version': model_row.version
            }
            return model_info
        except Exception as e:
            import traceback
            print(f"Error getting model by ID: {e}")
            traceback.print_exc()
            return None
    
    def load_model(self, model_path):
        """Load a saved RandomForest model from disk"""
        try:
            from pyspark.ml.regression import RandomForestRegressionModel
            import os
            
            # Check if model directory exists
            if not os.path.exists(model_path):
                print(f"Error: Model path does not exist: {model_path}")
                return None
            
            # Check if metadata directory exists (required for Spark ML models)
            metadata_path = os.path.join(model_path, "metadata")
            if not os.path.exists(metadata_path):
                print(f"Error: Model metadata directory not found at: {metadata_path}")
                print("The model may not have been saved correctly. Please retrain the model.")
                return None
            
            print(f"Loading model from {model_path}...")
            
            # On Windows, try to work around Hadoop NativeIO issues
            if sys.platform == 'win32':
                try:
                    self._disable_hadoop_native_io()
                    # Try to disable native IO in the JVM if possible
                    jvm = self.spark.sparkContext._jvm
                    # Set system property to disable native IO
                    try:
                        jvm.System.setProperty("hadoop.io.native.lib.available", "false")
                        jvm.System.setProperty("hadoop.native.lib", "false")
                        jvm.System.setProperty("io.native.lib.available", "false")
                        jvm.System.setProperty("java.library.path", "")
                    except:
                        pass  # Ignore if can't set properties
                except:
                    pass  # Ignore if JVM not accessible
            
            model = RandomForestRegressionModel.load(model_path)
            self.model = model
            
            # Also try to load the scaler (if it exists)
            scaler_path = model_path + "_scaler"
            if os.path.exists(scaler_path):
                from pyspark.ml.feature import StandardScalerModel
                try:
                    self.scaler = StandardScalerModel.load(scaler_path)
                    print(f"✓ Scaler loaded successfully")
                except Exception as e:
                    print(f"⚠ Warning: Could not load scaler: {e}")
                    self.scaler = None
            else:
                print(f"⚠ Warning: Scaler not found at {scaler_path}. Predictions may fail.")
                self.scaler = None
            
            print(f"✓ Model loaded successfully")
            return model
        except Exception as e:
            error_msg = str(e)
            print(f"Error loading model from {model_path}: {e}")
            
            # Check if it's the Windows NativeIO error
            if sys.platform == 'win32' and "UnsatisfiedLinkError" in error_msg and "NativeIO" in error_msg:
                print("\n⚠ Windows Hadoop NativeIO error detected.")
                print("This is a known issue with Spark on Windows.")
                print("\nPossible solutions:")
                print("1. Use Docker to run Spark (recommended):")
                print("   docker exec -e JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 spark-master \\")
                print("     /opt/spark/bin/spark-submit --packages com.datastax.spark:spark-cassandra-connector_2.12:3.5.0 \\")
                print("     /opt/spark/work-dir/ml_train_from_cassandra_pyspark.py")
                print("\n2. Ensure winutils/hadoop native libs are installed:")
                print("   python spark/app/setup_windows_hadoop.py")
                print("\n3. Or try loading the model using a different method")
                print("   (The model file exists but Spark can't read it due to Hadoop native library issues)")
            
            import traceback
            traceback.print_exc()
            return None
    
    def load_model_from_cassandra(self, keyspace='jobdb', table='ml_models', model_id=None):
        """Load model from disk using path stored in Cassandra"""
        try:
            print(f"\n{'='*60}")
            print(f"Loading model from Cassandra")
            print(f"{'='*60}")
            print(f"Keyspace: {keyspace}")
            print(f"Table: {table}")
            print(f"Model ID: {model_id if model_id else 'Latest'}")
            
            # First check if connection and table exist
            connection_ok, connection_msg, count = self.check_cassandra_connection(keyspace=keyspace, table=table)
            print(f"\nConnection check: {connection_msg}")
            if not connection_ok:
                print("\n❌ Cannot proceed: Cassandra connection or table check failed")
                return None, None
            if count == 0:
                print(f"\n⚠ Table exists but is empty ({count} rows)")
            
            # Get model metadata from Cassandra
            if model_id:
                print(f"\nFetching model by ID: {model_id}...")
                model_info = self.get_model_by_id(model_id, keyspace=keyspace, table=table)
            else:
                print(f"\nFetching latest model...")
                model_info = self.get_latest_model_from_cassandra(keyspace=keyspace, table=table)
            
            if not model_info:
                print("\n❌ No model found in Cassandra")
                print(f"   Keyspace: {keyspace}")
                print(f"   Table: {table}")
                print("\nPossible reasons:")
                print("  1. No models have been trained yet")
                print("  2. Keyspace/table doesn't exist")
                print("  3. Connection issue with Cassandra")
                return None, None
            
            print(f"\n✓ Found model metadata:")
            print(f"   Model ID: {model_info['model_id']}")
            print(f"   Model Name: {model_info['model_name']}")
            print(f"   Model Path: {model_info['model_path']}")
            
            model_path = model_info['model_path']
            print(f"\nLoading model from disk path: {model_path}")
            
            # Load the actual model from disk
            model = self.load_model(model_path)
            
            if model:
                print(f"\n✓ Model loaded successfully!")
                print(f"   Model Name: {model_info['model_name']}")
                print(f"   Model ID: {model_info['model_id']}")
                print(f"   R² Score: {model_info['r2_score']:.4f}")
                print(f"   MAE: {model_info['mae']:.2f}")
                print(f"   RMSE: {model_info['rmse']:.2f}")
            else:
                print(f"\n❌ Failed to load model from disk path: {model_path}")
                print("   The model file may not exist at this path.")
                print("   Please check if the model was saved correctly during training.")
            
            return model, model_info
        except Exception as e:
            import traceback
            print(f"\n❌ Error loading model from Cassandra: {e}")
            print(f"   Keyspace: {keyspace}")
            print(f"   Table: {table}")
            traceback.print_exc()
            return None, None
    
    def predict_salary(self, model, city, job_type, position_level, experience, 
                      num_skills, num_fields, title_length, indexers=None):
        """Make a salary prediction for given features"""
        try:
            if model is None:
                return None
            
            # Create a single row DataFrame with the input features
            from pyspark.sql import Row
            from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
            
            # Encode categorical features (simple encoding if indexers not available)
            city_encoded = 0
            job_type_encoded = 0
            position_encoded = 0
            experience_encoded = 0
            
            if indexers:
                # Use provided indexers if available
                city_map = indexers.get('city', {})
                job_type_map = indexers.get('job_type', {})
                position_map = indexers.get('position_level', {})
                experience_map = indexers.get('experience', {})
                
                city_encoded = city_map.get(city.lower(), 0)
                job_type_encoded = job_type_map.get(job_type.lower(), 0)
                position_encoded = position_map.get(position_level.lower(), 0)
                experience_encoded = experience_map.get(experience.lower(), 0)
            else:
                # Simple fallback encoding
                city_map = {'hồ chí minh': 0, 'hà nội': 1, 'đà nẵng': 2}
                job_type_map = {'nhân viên chính thức': 0, 'part-time': 1, 'freelance': 2}
                position_map = {'nhân viên': 0, 'chuyên viên': 1, 'trưởng phòng': 2, 'giám đốc': 3}
                experience_map = {'không yêu cầu': 0, '1-2 năm': 1, '3-5 năm': 2, '5+ năm': 3}
                
                city_encoded = city_map.get(city.lower(), 0)
                job_type_encoded = job_type_map.get(job_type.lower(), 0)
                position_encoded = position_map.get(position_level.lower(), 0)
                experience_encoded = experience_map.get(experience.lower(), 0)
            
            # Create feature vector (matching training features)
            row = Row(
                city_encoded=city_encoded,
                job_type_encoded=job_type_encoded,
                position_encoded=position_encoded,
                experience_encoded=experience_encoded,
                num_skills=num_skills,
                num_fields=num_fields,
                title_length=title_length
            )
            
            # Create DataFrame
            input_df = self.spark.createDataFrame([row])
            
            # Apply feature transformations (same as training)
            # Step 1: Assemble features into a vector
            from pyspark.ml.feature import VectorAssembler
            feature_cols = [
                'city_encoded',
                'job_type_encoded',
                'position_encoded',
                'experience_encoded',
                'num_skills',
                'num_fields',
                'title_length'
            ]
            
            assembler = VectorAssembler(
                inputCols=feature_cols,
                outputCol='features',
                handleInvalid='skip'
            )
            input_df = assembler.transform(input_df)
            
            # Step 2: Scale features (if scaler is available)
            if hasattr(self, 'scaler') and self.scaler is not None:
                input_df = self.scaler.transform(input_df)
            else:
                # If scaler not available, use features directly (may cause issues)
                # Try to create a dummy scaler or use features as-is
                print("⚠ Warning: Scaler not available. Using unscaled features.")
                input_df = input_df.withColumn('scaled_features', col('features'))
            
            # Make prediction (model expects 'scaled_features' column)
            prediction = model.transform(input_df)
            result = prediction.select("prediction").first()
            
            if result:
                return float(result.prediction)
            return None
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def close(self):
        """Close Spark session"""
        if self.spark:
            try:
                # Suppress cleanup warnings on Windows
                if sys.platform == 'win32':
                    try:
                        log4j = self.spark.sparkContext._jvm.org.apache.log4j
                        log4j.LogManager.getLogger("org.apache.spark.SparkEnv").setLevel(log4j.Level.ERROR)
                        log4j.LogManager.getLogger("org.apache.spark.util.SparkFileUtils").setLevel(log4j.Level.ERROR)
                    except:
                        pass
                
                self.spark.stop()
                print("\nSpark session closed")
            except Exception as e:
                # Ignore cleanup errors on Windows (file locking issues are common)
                if sys.platform == 'win32' and ("delete" in str(e).lower() or "IOException" in str(e)):
                    print("\nSpark session closed (some temp files may remain - this is normal on Windows)")
                else:
                    print(f"\nError closing Spark session: {e}")


def main():
    """Main training pipeline"""
    print("="*60)
    print("ML TRAINING FROM CASSANDRA (PySpark MLlib)")
    print("="*60)
    
    # Initialize trainer
    # Use 'cassandra' hostname when running in Docker containers
    # Use '127.0.0.1' when running locally
    trainer = MLTrainerFromCassandraPySpark(cassandra_host='cassandra', cassandra_port=9042)
    
    # Load data from Cassandra
    df = trainer.load_data_from_cassandra(keyspace='job_analytics', limit=10000)
    
    if df is None or df.count() == 0:
        print("No data found in Cassandra. Make sure data is being streamed.")
        trainer.close()
        return
    
    # Preprocess
    df = trainer.preprocess_data(df)
    
    if df.count() == 0:
        print("No valid data after preprocessing.")
        trainer.close()
        return
    
    # Prepare features
    df, feature_cols = trainer.prepare_features(df)
    
    # Train model
    model, scaler, metrics, test_predictions, test_df = trainer.train_model(df)
    
    # Print results
    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(f"\nTraining Set:")
    print(f"  MAE:  {metrics['train_mae']:.2f} million VND")
    print(f"  RMSE: {metrics['train_rmse']:.2f} million VND")
    print(f"  R²:   {metrics['train_r2']:.4f}")
    
    print(f"\nTest Set:")
    print(f"  MAE:  {metrics['test_mae']:.2f} million VND")
    print(f"  RMSE: {metrics['test_rmse']:.2f} million VND")
    print(f"  R²:   {metrics['test_r2']:.4f}")
    
    # Feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)
    feature_importance = model.featureImportances.toArray()
    feature_names = ['city_encoded', 'job_type_encoded', 'position_encoded', 'experience_encoded', 
                     'num_skills', 'num_fields', 'title_length']
    
    importance_df = list(zip(feature_names, feature_importance))
    importance_df.sort(key=lambda x: x[1], reverse=True)
    
    for name, importance in importance_df:
        print(f"  {name:20s}: {importance:.4f}")
    
    # Example predictions
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    sample_predictions = test_predictions.select('job_title', 'city', 'avg_salary', 'prediction').limit(5)
    samples = sample_predictions.collect()
    
    for i, row in enumerate(samples, 1):
        actual = row['avg_salary']
        predicted = row['prediction']
        job_title = row['job_title'][:50] if row['job_title'] else 'N/A'
        city = row['city'] if row['city'] else 'N/A'
        print(f"\n{i}. Job: {job_title}")
        print(f"   City: {city}")
        print(f"   Actual Salary: {actual:.2f}M VND")
        print(f"   Predicted: {predicted:.2f}M VND")
        print(f"   Error: {abs(actual - predicted):.2f}M VND")
    
    # Save model to disk
    # Use Docker path if running in container, otherwise use local path
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists("/opt/spark/work-dir"):
        # Running in Docker
        model_save_path = "/opt/spark/work-dir/models/salary_rf_pyspark"
    else:
        # Running locally
        model_save_path = os.path.join(script_dir, "models", "salary_rf_pyspark")
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Saving model to {model_save_path}...")
    print(f"{'='*60}")
    try:
        model.write().overwrite().save(model_save_path)
        print(f"✓ Model saved successfully to {model_save_path}")
        
        # Also save the scaler (needed for predictions)
        scaler_save_path = model_save_path + "_scaler"
        scaler.write().overwrite().save(scaler_save_path)
        print(f"✓ Scaler saved successfully to {scaler_save_path}")
        
        # Verify the model was saved correctly
        metadata_path = os.path.join(model_save_path, "metadata")
        if os.path.exists(metadata_path):
            print(f"✓ Model metadata directory created successfully")
        else:
            print(f"⚠ Warning: Model metadata directory not found after save")
    except Exception as e:
        print(f"✗ Error saving model: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Save model metadata to Cassandra
    model_id = trainer.save_model_metadata(
        model_name="Salary_Prediction_RF_PySpark",
        model_type="RandomForest_PySpark",
        metrics=metrics,
        feature_cols=feature_cols,
        model_path=model_save_path
    )
    
    trainer.close()
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
