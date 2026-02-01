"""
Streamlit UI for ML Model Training from Cassandra
Provides interactive interface for training and monitoring ML models
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import subprocess
import json
import os
import sys
import time
from collections import Counter

# Try to import Docker SDK, fallback to subprocess if not available
try:
    import docker
    DOCKER_SDK_AVAILABLE = True
except ImportError:
    DOCKER_SDK_AVAILABLE = False

# Configure page - MUST be first Streamlit command
st.set_page_config(
    page_title="ML Model Training Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🤖 ML Model Training Dashboard</h1>', unsafe_allow_html=True)

# Windows warning
if sys.platform == 'win32':
    with st.expander("⚠️ **Important Notice for Windows Users**", expanded=True):
        st.warning("""
        **Model Loading Limitation on Windows**
        
        Due to Hadoop NativeIO limitations, Spark cannot load model files from disk when running locally on Windows.
        This is a known issue with Spark 4.x on Windows.
        
        **Solutions:**
        1. **Use Docker (Recommended)**: All Spark operations work perfectly in Docker
        2. **Use WSL2**: Run Spark in Windows Subsystem for Linux
        3. **Train and predict in Docker**: Use Docker for both training and prediction
        
        **Note**: Model training and Cassandra operations work fine. Only loading saved models from disk has this limitation.
        """)

st.markdown("---")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Detect if running in Docker (check for common Docker environment indicators)
default_cassandra_host = os.environ.get("CASSANDRA_HOST", "cassandra" if os.path.exists("/.dockerenv") else "127.0.0.1")

# Configuration options
cassandra_host = st.sidebar.text_input("Cassandra Host", value=default_cassandra_host)
cassandra_port = st.sidebar.number_input("Cassandra Port", value=9042, min_value=1, max_value=65535)

# Show Docker networking hint
if os.path.exists("/.dockerenv") or default_cassandra_host == "cassandra":
    st.sidebar.info("🐳 **Docker Mode**: Using service name 'cassandra' for container networking")

keyspace = st.sidebar.selectbox("Keyspace", ["job_analytics", "jobdb"], index=0)
table_name = st.sidebar.text_input("Table Name", value="job_postings")
data_limit = st.sidebar.number_input("Data Limit", value=10000, min_value=100, max_value=100000, step=1000)
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)

# Docker configuration
st.sidebar.subheader("🐳 Docker Configuration")
docker_container = st.sidebar.text_input("Docker Container", value="spark-master")
spark_script_path = st.sidebar.text_input("Script Path in Container", value="/opt/spark/work-dir/ml_train_from_cassandra_pyspark.py")

# Show Docker SDK status
if DOCKER_SDK_AVAILABLE:
    st.sidebar.success("✅ Docker SDK available (recommended)")
else:
    st.sidebar.warning("⚠️ Docker SDK not available, using CLI fallback")

# Initialize session state
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'training_output' not in st.session_state:
    st.session_state.training_output = None
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'training_log' not in st.session_state:
    st.session_state.training_log = ""

# Job Recommendation session state
if 'recommender' not in st.session_state:
    st.session_state.recommender = None
if 'recommender_ready' not in st.session_state:
    st.session_state.recommender_ready = False
if 'recommendation_results' not in st.session_state:
    st.session_state.recommendation_results = None
if 'sample_jobs' not in st.session_state:
    st.session_state.sample_jobs = None
if 'jobs_features_pandas' not in st.session_state:
    st.session_state.jobs_features_pandas = None

# Skills Recommendation session state
if 'skills_recommender' not in st.session_state:
    st.session_state.skills_recommender = None
if 'skills_recommender_ready' not in st.session_state:
    st.session_state.skills_recommender_ready = False
if 'skill_clusters' not in st.session_state:
    st.session_state.skill_clusters = None
if 'top_skills' not in st.session_state:
    st.session_state.top_skills = None


def check_spark_session_active(recommender):
    """Check if the SparkContext in the recommender is still active"""
    try:
        if recommender is None:
            return False
        if recommender.spark is None:
            return False
        # Try to access SparkContext - this will fail if it's stopped
        sc = recommender.spark.sparkContext
        if sc._jsc is None:
            return False
        # Try a simple operation to verify the session is working
        sc.getConf().get("spark.app.name")
        return True
    except Exception:
        return False


def get_or_rebuild_recommender(cassandra_host, cassandra_port, keyspace, rec_data_limit, num_features):
    """Get existing recommender or rebuild if SparkContext is no longer active"""
    from ml_job_recommendation import JobRecommenderPySpark
    
    # Check if existing recommender is still active
    if st.session_state.recommender is not None and check_spark_session_active(st.session_state.recommender):
        return st.session_state.recommender, False  # Return existing, not rebuilt
    
    # Need to rebuild
    recommender = JobRecommenderPySpark(
        cassandra_host=cassandra_host,
        cassandra_port=cassandra_port
    )
    
    result = recommender.load_data_from_cassandra(
        keyspace=keyspace,
        limit=rec_data_limit
    )
    
    if result is None or recommender.df is None or recommender.df.count() == 0:
        return None, False
    
    recommender.prepare_features(num_features=num_features)
    
    # Update session state
    st.session_state.recommender = recommender
    
    # Get sample jobs for dropdown
    sample_jobs_df = recommender.features_df.select(
        "job_index", "job_title", "city", "position_level", 
        "salary_min", "salary_max", "unit"
    ).limit(500).toPandas()
    st.session_state.sample_jobs = sample_jobs_df
    
    # Store full features as pandas for fallback
    full_features_df = recommender.features_df.select(
        "job_index", "job_title", "city", "position_level",
        "salary_min", "salary_max", "unit", "skills", "experience"
    ).toPandas()
    st.session_state.jobs_features_pandas = full_features_df
    
    return recommender, True  # Return recommender, was rebuilt


def get_or_rebuild_skills_recommender(cassandra_host, cassandra_port, keyspace, data_limit, vector_size=100, num_topics=8, force_retrain=False):
    """Get existing skills recommender or load from disk if available"""
    from ml_skills_recommendation import SkillsRecommenderPySpark
    import os
    
    # Check if existing recommender is still active
    if st.session_state.skills_recommender is not None and check_spark_session_active(st.session_state.skills_recommender):
        return st.session_state.skills_recommender, False  # Return existing, not rebuilt
    
    # Create new recommender instance
    recommender = SkillsRecommenderPySpark(
        cassandra_host=cassandra_host,
        cassandra_port=cassandra_port
    )
    
    # Try to load saved models first (unless force_retrain is True)
    if not force_retrain:
        # Check multiple possible model paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(script_dir, "models", "skills_recommender"),
            "./models/skills_recommender",
            "/opt/spark/work-dir/models/skills_recommender",
            "models/skills_recommender"
        ]
        
        model_loaded = False
        for model_path in possible_paths:
            if os.path.exists(model_path):
                print(f"Found saved models at: {model_path}")
                success = recommender.load_model(model_path)
                if success:
                    model_loaded = True
                    print(f"✓ Models loaded successfully from {model_path}")
                    
                    # Also load data from Cassandra for operations that need it
                    print("Loading data from Cassandra for query operations...")
                    result = recommender.load_data_from_cassandra(
                        keyspace=keyspace,
                        limit=data_limit
                    )
                    
                    if result is None or recommender.df is None:
                        print("⚠ Warning: Could not load data from Cassandra")
                    
                    # Update session state
                    st.session_state.skills_recommender = recommender
                    st.session_state.skills_recommender_ready = True
                    
                    # Store skill clusters
                    if recommender.skill_clusters:
                        st.session_state.skill_clusters = recommender.skill_clusters
                    
                    # Get top skills for display
                    if recommender.skills_df is not None:
                        top_skills_df = recommender.skills_df.limit(100).toPandas()
                        st.session_state.top_skills = top_skills_df
                    
                    return recommender, False  # Loaded from disk, not retrained
                break
        
        if model_loaded:
            return recommender, False
    
    # If no saved models found or force_retrain, load data and train
    result = recommender.load_data_from_cassandra(
        keyspace=keyspace,
        limit=data_limit
    )
    
    if result is None or recommender.df is None or recommender.df.count() == 0:
        return None, False
    
    # Extract skills and train models
    recommender.extract_skills()
    recommender.train_word2vec(vector_size=vector_size, min_count=5, window_size=5)
    recommender.train_lda_topic_model(num_topics=num_topics, max_iter=15)
    
    # Update session state
    st.session_state.skills_recommender = recommender
    st.session_state.skills_recommender_ready = True
    
    # Store skill clusters
    if recommender.skill_clusters:
        st.session_state.skill_clusters = recommender.skill_clusters
    
    # Get top skills for display
    if recommender.skills_df:
        top_skills_df = recommender.skills_df.limit(100).toPandas()
        st.session_state.top_skills = top_skills_df
    
    return recommender, True  # Retrained


# Check for saved results file
RESULTS_FILE = "training_results.json"

def load_saved_results():
    """Load results from saved JSON file if exists"""
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None
    return None

# ============================================================================
# VISUALIZATION HELPER FUNCTIONS
# ============================================================================

def display_enhanced_metrics(df):
    """Display comprehensive metrics cards"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_jobs = len(df)
        st.metric("Total Jobs", f"{total_jobs:,}", help="Total number of job postings")
    
    with col2:
        if 'avg_salary' in df.columns:
            salary_data = pd.to_numeric(df['avg_salary'], errors='coerce').dropna()
            if len(salary_data) > 0:
                avg_salary = salary_data.mean()
                median_salary = salary_data.median()
                if not (pd.isna(avg_salary) or pd.isna(median_salary)):
                    st.metric("Avg Salary", f"{avg_salary:.1f}M VND", 
                             delta=f"{median_salary:.1f}M median")
                else:
                    st.metric("Avg Salary", "N/A")
            else:
                st.metric("Avg Salary", "N/A")
        elif 'salary_max' in df.columns:
            salary_data = pd.to_numeric(df['salary_max'], errors='coerce').dropna()
            if len(salary_data) > 0:
                avg_salary = salary_data.mean()
                if not pd.isna(avg_salary):
                    st.metric("Avg Salary", f"{avg_salary:.1f}M VND")
                else:
                    st.metric("Avg Salary", "N/A")
            else:
                st.metric("Avg Salary", "N/A")
        else:
            st.metric("Avg Salary", "N/A")
    
    with col3:
        unique_cities = df['city'].nunique() if 'city' in df.columns else 0
        st.metric("Cities", unique_cities, help="Number of unique cities")
    
    with col4:
        if 'salary_max' in df.columns:
            salary_data = pd.to_numeric(df['salary_max'], errors='coerce').dropna()
            if len(salary_data) > 0:
                max_salary = salary_data.max()
                if not pd.isna(max_salary):
                    st.metric("Max Salary", f"{max_salary:.1f}M VND")
                else:
                    st.metric("Max Salary", "N/A")
            else:
                st.metric("Max Salary", "N/A")
        else:
            st.metric("Max Salary", "N/A")
    
    with col5:
        if 'exp_avg_year' in df.columns:
            exp_data = pd.to_numeric(df['exp_avg_year'], errors='coerce').dropna()
            if len(exp_data) > 0:
                avg_exp = exp_data.mean()
                if not pd.isna(avg_exp):
                    st.metric("Avg Experience", f"{avg_exp:.1f} years")
                else:
                    st.metric("Avg Experience", "N/A")
            else:
                st.metric("Avg Experience", "N/A")
        else:
            st.metric("Avg Experience", "N/A")


def salary_distribution_by_city(df):
    """Box plot showing salary distribution across cities"""
    if 'city' not in df.columns:
        return None
    
    salary_col = 'avg_salary' if 'avg_salary' in df.columns else 'salary_max'
    if salary_col not in df.columns:
        return None
    
    # Filter out NaN values
    df_clean = df[[salary_col, 'city']].copy()
    df_clean[salary_col] = pd.to_numeric(df_clean[salary_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[salary_col, 'city'])
    
    if len(df_clean) == 0:
        return None
    
    fig = px.box(df_clean, x='city', y=salary_col,
                 title='📊 Salary Distribution by City',
                 labels={salary_col: 'Salary (Million VND)', 'city': 'City'},
                 color='city',
                 points='outliers')
    fig.update_layout(showlegend=False, height=500)
    return fig


def salary_vs_experience_scatter(df):
    """Scatter plot showing relationship between experience and salary"""
    exp_col = 'exp_avg_year' if 'exp_avg_year' in df.columns else None
    salary_col = 'avg_salary' if 'avg_salary' in df.columns else 'salary_max'
    
    if not exp_col or salary_col not in df.columns:
        return None
    
    # Filter out NaN values
    cols_to_keep = [exp_col, salary_col]
    if 'city' in df.columns:
        cols_to_keep.append('city')
    if 'num_skills' in df.columns:
        cols_to_keep.append('num_skills')
    if 'job_title' in df.columns:
        cols_to_keep.append('job_title')
    
    df_clean = df[cols_to_keep].copy()
    df_clean[exp_col] = pd.to_numeric(df_clean[exp_col], errors='coerce')
    df_clean[salary_col] = pd.to_numeric(df_clean[salary_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[exp_col, salary_col])
    
    if len(df_clean) == 0:
        return None
    
    fig = px.scatter(df_clean, x=exp_col, y=salary_col,
                     color='city' if 'city' in df_clean.columns else None,
                     size='num_skills' if 'num_skills' in df_clean.columns else None,
                     hover_data=['job_title'] if 'job_title' in df_clean.columns else None,
                     trendline='ols',
                     title='💼 Experience vs Salary Analysis',
                     labels={exp_col: 'Years of Experience', 
                            salary_col: 'Salary (Million VND)'})
    fig.update_layout(height=500)
    return fig


def jobs_by_city_bar(df):
    """Bar chart showing job count by city"""
    if 'city' not in df.columns:
        return None
    
    city_counts = df['city'].value_counts().reset_index()
    city_counts.columns = ['City', 'Count']
    
    fig = px.bar(city_counts, x='City', y='Count',
                 title='🏙️ Job Postings by City',
                 labels={'Count': 'Number of Jobs'},
                 color='Count',
                 color_continuous_scale='Blues')
    fig.update_layout(height=400)
    return fig


def city_salary_comparison(df):
    """Compare average salary and job count by city"""
    if 'city' not in df.columns:
        return None
    
    salary_col = 'avg_salary' if 'avg_salary' in df.columns else 'salary_max'
    if salary_col not in df.columns:
        return None
    
    # Filter out NaN values
    df_clean = df[['city', salary_col]].copy()
    df_clean[salary_col] = pd.to_numeric(df_clean[salary_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[salary_col, 'city'])
    
    if len(df_clean) == 0:
        return None
    
    city_stats = df_clean.groupby('city').agg({
        salary_col: ['mean', 'count']
    }).reset_index()
    city_stats.columns = ['City', 'Avg Salary', 'Job Count']
    
    # Remove any remaining NaN values
    city_stats = city_stats.dropna()
    
    if len(city_stats) == 0:
        return None
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=city_stats['City'], y=city_stats['Job Count'],
               name='Job Count', marker_color='#1f77b4'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=city_stats['City'], y=city_stats['Avg Salary'],
                  name='Avg Salary', marker_color='#ff7f0e', mode='lines+markers'),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="City")
    fig.update_yaxes(title_text="Job Count", secondary_y=False)
    fig.update_yaxes(title_text="Avg Salary (Million VND)", secondary_y=True)
    fig.update_layout(title='📊 City Comparison: Job Count vs Average Salary', height=500)
    
    return fig


def top_skills_analysis(df):
    """Analyze most common skills"""
    if 'skills' not in df.columns:
        return None
    
    # Simple word frequency
    all_skills = ' '.join(df['skills'].dropna().astype(str)).lower()
    skills_list = all_skills.split()
    skill_counts = Counter(skills_list)
    
    top_skills = pd.DataFrame(skill_counts.most_common(20), columns=['Skill', 'Count'])
    
    fig = px.bar(top_skills, x='Count', y='Skill',
                 orientation='h',
                 title='🔧 Top 20 Skills in Demand',
                 labels={'Count': 'Frequency'},
                 color='Count',
                 color_continuous_scale='Viridis')
    fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
    return fig


def experience_distribution(df):
    """Distribution of experience requirements"""
    if 'experience' not in df.columns:
        return None
    
    exp_counts = df['experience'].value_counts().reset_index()
    exp_counts.columns = ['Experience', 'Count']
    
    fig = px.pie(exp_counts, values='Count', names='Experience',
                title='📚 Experience Requirements Distribution',
                hole=0.4)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500)
    return fig


def prediction_vs_actual_scatter(predictions_df):
    """Scatter plot of predicted vs actual values"""
    if 'prediction' not in predictions_df.columns:
        return None
    
    actual_col = 'avg_salary' if 'avg_salary' in predictions_df.columns else None
    if not actual_col:
        return None
    
    fig = px.scatter(predictions_df, x=actual_col, y='prediction',
                     title='🎯 Predicted vs Actual Salary',
                     labels={actual_col: 'Actual Salary (Million VND)',
                            'prediction': 'Predicted Salary (Million VND)'},
                     trendline='ols',
                     hover_data=['job_title'] if 'job_title' in predictions_df.columns else None)
    
    # Add perfect prediction line
    max_val = max(predictions_df[actual_col].max(), predictions_df['prediction'].max())
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                            mode='lines', name='Perfect Prediction',
                            line=dict(dash='dash', color='red')))
    
    fig.update_layout(height=500)
    return fig


def residual_analysis(predictions_df):
    """Residual analysis plots"""
    if 'prediction' not in predictions_df.columns:
        return None
    
    actual_col = 'avg_salary' if 'avg_salary' in predictions_df.columns else None
    if not actual_col:
        return None
    
    predictions_df = predictions_df.copy()
    predictions_df['residuals'] = predictions_df[actual_col] - predictions_df['prediction']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Residuals vs Predicted', 'Residual Distribution'),
        specs=[[{"type": "scatter"}, {"type": "histogram"}]]
    )
    
    fig.add_trace(
        go.Scatter(x=predictions_df['prediction'], y=predictions_df['residuals'],
                  mode='markers', name='Residuals'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=predictions_df['residuals'], name='Distribution'),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Predicted Salary", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Residuals", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_layout(height=400, title_text="Residual Analysis", showlegend=False)
    
    return fig


# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏠 Dashboard", "📊 Data Overview", "🎯 Model Training", "🔮 Predictions", "🔍 Job Recommendations", "🛠️ Skills Analysis"])

with tab1:
    st.header("📈 Training Dashboard")
    
    # Load data overview
    st.subheader("📊 Data Overview")
    try:
        from ml_train_from_cassandra_pyspark import MLTrainerFromCassandraPySpark
        trainer = MLTrainerFromCassandraPySpark(cassandra_host=cassandra_host, cassandra_port=cassandra_port)
        
        # Load sample data for overview
        df_sample = trainer.load_data_from_cassandra(keyspace=keyspace, table=table_name, limit=1000)
        
        if df_sample and df_sample.count() > 0:
            # Convert to pandas for visualization
            df_pandas = df_sample.limit(1000).toPandas()
            
            # Display metrics
            display_enhanced_metrics(df_pandas)
            
            st.markdown("---")
            
            # Quick visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig_city = jobs_by_city_bar(df_pandas)
                if fig_city:
                    st.plotly_chart(fig_city, use_container_width=True)
            
            with col2:
                fig_exp = experience_distribution(df_pandas)
                if fig_exp:
                    st.plotly_chart(fig_exp, use_container_width=True)
            
            trainer.close()
        else:
            st.info("No data available. Please check your Cassandra connection and ensure data is loaded.")
    except Exception as e:
        st.warning(f"Could not load data overview: {str(e)}")
        st.info("💡 Make sure Cassandra is running and contains data.")
    
    st.markdown("---")
    
    # Model Performance Section
    st.subheader("🤖 Model Performance")
    
    # Try to load saved results
    saved_results = load_saved_results()
    results = st.session_state.training_results or saved_results
    
    if results:
        metrics = results.get('metrics', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Test R² Score", f"{metrics.get('test_r2', 0):.4f}", 
                     delta=f"{metrics.get('test_r2', 0)*100:.1f}%")
        
        with col2:
            st.metric("Test MAE", f"{metrics.get('test_mae', 0):.2f}M VND")
        
        with col3:
            st.metric("Test RMSE", f"{metrics.get('test_rmse', 0):.2f}M VND")
        
        with col4:
            training_time = results.get('training_time', 0)
            st.metric("Training Time", f"{training_time:.2f}s")
        
        # Model performance comparison
        st.subheader("📊 Model Performance Comparison")
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Training Set',
            x=['MAE', 'RMSE', 'R² (%)'],
            y=[metrics.get('train_mae', 0), metrics.get('train_rmse', 0), metrics.get('train_r2', 0) * 100],
            marker_color='#1f77b4'
        ))
        
        fig.add_trace(go.Bar(
            name='Test Set',
            x=['MAE', 'RMSE', 'R² (%)'],
            y=[metrics.get('test_mae', 0), metrics.get('test_rmse', 0), metrics.get('test_r2', 0) * 100],
            marker_color='#ff7f0e'
        ))
        
        fig.update_layout(
            title="Model Performance Metrics",
            xaxis_title="Metric",
            yaxis_title="Value",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Show models from Cassandra
    st.subheader("📦 Models in Cassandra")
    try:
        from ml_train_from_cassandra_pyspark import MLTrainerFromCassandraPySpark
        trainer = MLTrainerFromCassandraPySpark(cassandra_host=cassandra_host, cassandra_port=cassandra_port)
        
        models_df = trainer.get_models_from_cassandra(keyspace='jobdb', table='ml_models')
        
        if models_df and models_df.count() > 0:
            # Convert timestamp to string before converting to pandas to avoid timezone issues
            from pyspark.sql.functions import date_format, col
            # Order by original timestamp first, then format
            models_df_ordered = models_df.orderBy(col("training_date"), ascending=False)
            models_df_formatted = models_df_ordered.select(
                col("model_id"),
                col("model_name"),
                col("model_type"),
                date_format(col("training_date"), "yyyy-MM-dd HH:mm:ss").alias("training_date"),
                col("r2_score"),
                col("mae"),
                col("rmse"),
                col("model_path"),
                col("version")
            )
            
            # Convert to pandas for display
            models_pandas = models_df_formatted.toPandas()
            
            # Format the dataframe for display
            models_pandas['model_id'] = models_pandas['model_id'].astype(str)
            models_pandas['r2_score'] = models_pandas['r2_score'].round(4)
            models_pandas['mae'] = models_pandas['mae'].round(2)
            models_pandas['rmse'] = models_pandas['rmse'].round(2)
            
            st.dataframe(models_pandas, use_container_width=True)
            
            # Show latest model info
            latest_model = trainer.get_latest_model_from_cassandra(keyspace='jobdb', table='ml_models')
            if latest_model:
                st.info(f"📌 **Latest Model:** {latest_model['model_name']} (ID: {latest_model['model_id'][:8]}...) | R²: {latest_model['r2_score']:.4f}")
        else:
            st.warning("No models found in Cassandra. Train a model first.")
        
        trainer.close()
    except Exception as e:
        st.error(f"Error loading models from Cassandra: {str(e)}")
        import traceback
        with st.expander("Show Error Details"):
            st.code(traceback.format_exc())
    
    if not results:
        st.info("💡 Train a model in the 'Model Training' tab to see metrics here.")
    
    # Feature importance visualization (if results exist)
    if results and 'feature_importance' in results:
            st.subheader("🎯 Feature Importance")
            feature_importance = pd.DataFrame(results['feature_importance'])
            
            fig_importance = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Importance Ranking",
                labels={'importance': 'Importance Score', 'feature': 'Feature'},
                color='importance',
                color_continuous_scale='Blues'
            )
            fig_importance.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Model Diagnostics (if test predictions available)
            if 'test_predictions' in results and results['test_predictions']:
                st.subheader("🔬 Model Diagnostics")
                
                try:
                    # Convert test predictions to DataFrame if it's a list
                    if isinstance(results['test_predictions'], list):
                        test_df = pd.DataFrame(results['test_predictions'])
                    else:
                        test_df = results['test_predictions']
                    
                    if 'prediction' in test_df.columns:
                        col_diag1, col_diag2 = st.columns(2)
                        
                        with col_diag1:
                            fig_pred = prediction_vs_actual_scatter(test_df)
                            if fig_pred:
                                st.plotly_chart(fig_pred, use_container_width=True)
                        
                        with col_diag2:
                            fig_resid = residual_analysis(test_df)
                            if fig_resid:
                                st.plotly_chart(fig_resid, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display model diagnostics: {str(e)}")
    else:
        st.info("👈 Start training a model to see metrics here")
        
        # Show sample metrics placeholder
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test R² Score", "—")
        with col2:
            st.metric("Test MAE", "—")
        with col3:
            st.metric("Test RMSE", "—")
        with col4:
            st.metric("Training Time", "—")
        
        # Show sample chart
        st.subheader("📊 Sample Performance Chart")
        sample_fig = go.Figure()
        sample_fig.add_trace(go.Bar(
            name='Training',
            x=['MAE', 'RMSE', 'R² (%)'],
            y=[5, 7, 75],
            marker_color='#1f77b4',
            opacity=0.3
        ))
        sample_fig.add_trace(go.Bar(
            name='Test',
            x=['MAE', 'RMSE', 'R² (%)'],
            y=[6, 8, 70],
            marker_color='#ff7f0e',
            opacity=0.3
        ))
        sample_fig.update_layout(
            title="Example Metrics (Train a model to see real data)",
            barmode='group',
            height=350
        )
        st.plotly_chart(sample_fig, use_container_width=True)

with tab2:
    st.header("📊 Data Overview & Exploration")
    
    # Load data from Cassandra
    try:
        from ml_train_from_cassandra_pyspark import MLTrainerFromCassandraPySpark
        trainer = MLTrainerFromCassandraPySpark(cassandra_host=cassandra_host, cassandra_port=cassandra_port)
        
        with st.spinner("Loading data from Cassandra..."):
            df = trainer.load_data_from_cassandra(keyspace=keyspace, table=table_name, limit=data_limit)
        
        if df and df.count() > 0:
            # Convert to pandas for visualization
            df_pandas = df.limit(data_limit).toPandas()
            
            st.success(f"✅ Loaded {len(df_pandas):,} records from {keyspace}.{table_name}")
            
            # Interactive Filters
            st.subheader("🔍 Filters")
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            
            with col_filter1:
                if 'city' in df_pandas.columns:
                    cities = ['All'] + sorted(df_pandas['city'].unique().tolist())
                    selected_cities = st.multiselect("Select Cities", cities, default=['All'])
                    if 'All' not in selected_cities:
                        df_pandas = df_pandas[df_pandas['city'].isin(selected_cities)]
            
            with col_filter2:
                salary_col = 'avg_salary' if 'avg_salary' in df_pandas.columns else 'salary_max'
                if salary_col in df_pandas.columns:
                    # Filter out NaN values and ensure we have valid numeric data
                    salary_data = pd.to_numeric(df_pandas[salary_col], errors='coerce').dropna()
                    if len(salary_data) > 0:
                        min_sal, max_sal = float(salary_data.min()), float(salary_data.max())
                        # Ensure valid range
                        if not (pd.isna(min_sal) or pd.isna(max_sal)) and min_sal < max_sal:
                            salary_range = st.slider("Salary Range (Million VND)", 
                                                     min_sal, max_sal, (min_sal, max_sal))
                            df_pandas = df_pandas[(pd.to_numeric(df_pandas[salary_col], errors='coerce') >= salary_range[0]) & 
                                                 (pd.to_numeric(df_pandas[salary_col], errors='coerce') <= salary_range[1])]
                        else:
                            st.info("No valid salary data to filter")
                    else:
                        st.info("No valid salary data available")
            
            with col_filter3:
                if 'exp_avg_year' in df_pandas.columns:
                    # Filter out NaN values and ensure we have valid numeric data
                    exp_data = pd.to_numeric(df_pandas['exp_avg_year'], errors='coerce').dropna()
                    if len(exp_data) > 0:
                        min_exp, max_exp = float(exp_data.min()), float(exp_data.max())
                        # Ensure valid range
                        if not (pd.isna(min_exp) or pd.isna(max_exp)) and min_exp < max_exp:
                            exp_range = st.slider("Experience Range (Years)", 
                                                 min_exp, max_exp, (min_exp, max_exp))
                            df_pandas = df_pandas[(pd.to_numeric(df_pandas['exp_avg_year'], errors='coerce') >= exp_range[0]) & 
                                                 (pd.to_numeric(df_pandas['exp_avg_year'], errors='coerce') <= exp_range[1])]
                        else:
                            st.info("No valid experience data to filter")
                    else:
                        st.info("No valid experience data available")
            
            st.write(f"**Showing {len(df_pandas):,} filtered records**")
            st.markdown("---")
            
            # Enhanced Metrics
            display_enhanced_metrics(df_pandas)
            
            st.markdown("---")
            
            # Salary Analysis Section
            st.subheader("💰 Salary Analysis")
            col_sal1, col_sal2 = st.columns(2)
            
            with col_sal1:
                fig_box = salary_distribution_by_city(df_pandas)
                if fig_box:
                    st.plotly_chart(fig_box, use_container_width=True)
            
            with col_sal2:
                fig_scatter = salary_vs_experience_scatter(df_pandas)
                if fig_scatter:
                    st.plotly_chart(fig_scatter, use_container_width=True)
            
            # City Comparison
            fig_city_comp = city_salary_comparison(df_pandas)
            if fig_city_comp:
                st.plotly_chart(fig_city_comp, use_container_width=True)
            
            st.markdown("---")
            
            # Geographic & Skills Analysis
            st.subheader("🏙️ Geographic & Skills Analysis")
            col_geo1, col_geo2 = st.columns(2)
            
            with col_geo1:
                fig_jobs_city = jobs_by_city_bar(df_pandas)
                if fig_jobs_city:
                    st.plotly_chart(fig_jobs_city, use_container_width=True)
            
            with col_geo2:
                fig_exp_dist = experience_distribution(df_pandas)
                if fig_exp_dist:
                    st.plotly_chart(fig_exp_dist, use_container_width=True)
            
            # Skills Analysis
            fig_skills = top_skills_analysis(df_pandas)
            if fig_skills:
                st.plotly_chart(fig_skills, use_container_width=True)
            
            st.markdown("---")
            
            # Data Table
            st.subheader("📋 Data Table")
            st.dataframe(df_pandas.head(100), use_container_width=True, height=400)
            
            # Download button
            csv = df_pandas.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv,
                file_name=f"job_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            trainer.close()
        else:
            st.warning("No data found in Cassandra. Please ensure:")
            st.write("1. Cassandra is running")
            st.write("2. Data has been loaded into the database")
            st.write("3. Keyspace and table names are correct")
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        import traceback
        with st.expander("Show Error Details"):
            st.code(traceback.format_exc())
        
        st.info("💡 **Troubleshooting:**")
        st.write("- Check Cassandra connection settings in sidebar")
        st.write("- Verify keyspace and table exist")
        st.write("- Ensure data streaming has populated the database")

with tab3:
    st.header("🎯 Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Training Configuration")
        
        st.write(f"""
        **Current Settings:**
        - Cassandra Host: `{cassandra_host}:{cassandra_port}`
        - Keyspace: `{keyspace}`
        - Table: `{table_name}`
        - Data Limit: `{data_limit:,}` records
        - Test Size: `{test_size*100:.0f}%`
        """)
        
        st.info("📝 Training uses PySpark ML Random Forest Regressor")
        
        # Show training command
        st.subheader("🖥️ Training Command")
        training_cmd = f"""docker exec -e JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 {docker_container} /opt/spark/bin/spark-submit \\
    --packages com.datastax.spark:spark-cassandra-connector_2.12:3.5.0 \\
    {spark_script_path}"""
        
        st.code(training_cmd, language="bash")
    
    with col2:
        st.subheader("Quick Actions")
        
        # Show training status
        if st.session_state.training_in_progress:
            st.warning("🔄 Training in progress...")
            if st.button("⏹️ Stop Training", type="secondary"):
                st.session_state.training_in_progress = False
                st.session_state.training_log = ""
                st.rerun()
        else:
            if st.button("🚀 Run Training Script", type="primary"):
                st.session_state.training_in_progress = True
                st.session_state.training_log = ""
                st.rerun()
        
        if st.button("🔄 Reload Results", type="secondary"):
            saved = load_saved_results()
            if saved:
                st.session_state.training_results = saved
                st.success("✅ Results loaded!")
                st.rerun()
            else:
                st.warning("No saved results found")
    
    # Real-time training log display
    if st.session_state.training_in_progress:
        st.subheader("📜 Live Training Log")
        status_placeholder = st.empty()
        log_placeholder = st.empty()
        
        # Show initial status
        status_placeholder.info("🔄 Starting training process...")
        
        # Run training and capture output
        try:
            if DOCKER_SDK_AVAILABLE:
                # Use Docker SDK (more reliable, no CLI version issues)
                try:
                    docker_client = docker.from_env()
                    status_placeholder.info("🔄 Connecting to Docker...")
                    
                    # Execute command in container
                    exec_result = docker_client.containers.get(docker_container).exec_run(
                        cmd=[
                            "/opt/spark/bin/spark-submit",
                            "--packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0",
                            spark_script_path
                        ],
                        environment={"JAVA_HOME": "/usr/lib/jvm/java-17-openjdk-amd64"},
                        stream=True,
                        stdout=True,
                        stderr=True
                    )
                    
                    log_lines = []
                    max_lines = 2000
                    line_count = 0
                    
                    status_placeholder.info("🔄 Training in progress... Reading output...")
                    
                    # Show initial empty log
                    log_placeholder.text_area(
                        "Training Output",
                        value="Waiting for output...",
                        height=400,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                    
                    # Read output stream
                    for chunk in exec_result.output:
                        if chunk:
                            # Decode bytes if needed
                            if isinstance(chunk, bytes):
                                chunk = chunk.decode('utf-8', errors='replace')
                            
                            # Split into lines
                            for line in chunk.split('\n'):
                                if line.strip():
                                    log_lines.append(line.rstrip())
                                    line_count += 1
                                    
                                    # Keep only last max_lines
                                    if len(log_lines) > max_lines:
                                        log_lines = log_lines[-max_lines:]
                                    
                                    # Update status
                                    status_placeholder.info(f"🔄 Training in progress... ({line_count} lines of output)")
                                    
                                    # Update log display
                                    log_text = '\n'.join(log_lines)
                                    st.session_state.training_log = log_text
                                    display_lines = log_lines[-500:] if len(log_lines) > 500 else log_lines
                                    display_text = '\n'.join(display_lines)
                                    if len(log_lines) > 500:
                                        display_text = f"... ({len(log_lines) - 500} earlier lines) ...\n" + display_text
                                    
                                    log_placeholder.text_area(
                                        "Training Output",
                                        value=display_text,
                                        height=400,
                                        disabled=True,
                                        label_visibility="collapsed"
                                    )
                    
                    # Check exit code
                    exit_code = exec_result.exit_code if hasattr(exec_result, 'exit_code') else 0
                    
                except docker.errors.NotFound:
                    st.error(f"❌ Container '{docker_container}' not found. Please check the container name.")
                    st.session_state.training_in_progress = False
                    status_placeholder.empty()
                    log_placeholder.text_area(
                        "Training Output",
                        value="Container not found. Please check the container name.",
                        height=400,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                    # Skip rest of processing
                    exit_code = None
                except docker.errors.APIError as e:
                    st.error(f"❌ Docker API Error: {str(e)}")
                    st.info("💡 Make sure Docker socket is accessible: `/var/run/docker.sock`")
                    st.session_state.training_in_progress = False
                    status_placeholder.empty()
                    log_placeholder.text_area(
                        "Training Output",
                        value=f"Docker API Error: {str(e)}\n\nMake sure Docker socket is mounted: /var/run/docker.sock",
                        height=400,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                    # Skip rest of processing
                    exit_code = None
                else:
                    # Only process results if training completed successfully
                    if 'exit_code' in locals() and exit_code is not None:
                        # Get final return code
                        return_code = exit_code
                        
                        # Final output
                        final_output = '\n'.join(log_lines) if 'log_lines' in locals() else st.session_state.training_log
                        st.session_state.training_output = final_output
                        st.session_state.training_in_progress = False
                        
                        status_placeholder.empty()
                        
                        if return_code == 0:
                            st.success("✅ Training completed successfully!")
                            # Try to load results
                            saved = load_saved_results()
                            if saved:
                                st.session_state.training_results = saved
                        else:
                            st.error(f"❌ Training failed with return code {return_code}")
                        
                        # Show final log
                        log_placeholder.text_area(
                            "Training Output (Final)",
                            value=final_output,
                            height=400,
                            disabled=True,
                            label_visibility="collapsed"
                        )
            else:
                # Fallback to subprocess (for environments without docker SDK)
                docker_cmd = [
                    "docker", "exec", "-e", "JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64",
                    docker_container,
                    "/opt/spark/bin/spark-submit",
                    "--packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0",
                    spark_script_path
                ]
                
                # Start process
                process = subprocess.Popen(
                    docker_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1,
                    universal_newlines=True,
                    cwd=os.path.dirname(__file__)
                )
                
                # Read output line by line and accumulate
                log_lines = []
                max_lines = 2000  # Limit to prevent memory issues
                line_count = 0
                
                status_placeholder.info("🔄 Training in progress... Reading output...")
                
                # Show initial empty log
                log_placeholder.text_area(
                    "Training Output",
                    value="Waiting for output...",
                    height=400,
                    disabled=True,
                    label_visibility="collapsed"
                )
                
                # Read all output
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        line = output.rstrip()
                        log_lines.append(line)
                        line_count += 1
                        # Keep only last max_lines
                        if len(log_lines) > max_lines:
                            log_lines = log_lines[-max_lines:]
                        
                        # Update status with line count
                        status_placeholder.info(f"🔄 Training in progress... ({line_count} lines of output)")
                        
                        # Update log display (accumulated)
                        log_text = '\n'.join(log_lines)
                        st.session_state.training_log = log_text
                        # Use text_area for better scrolling - show last 500 lines for performance
                        display_lines = log_lines[-500:] if len(log_lines) > 500 else log_lines
                        display_text = '\n'.join(display_lines)
                        if len(log_lines) > 500:
                            display_text = f"... ({len(log_lines) - 500} earlier lines) ...\n" + display_text
                        
                        log_placeholder.text_area(
                            "Training Output",
                            value=display_text,
                            height=400,
                            disabled=True,
                            label_visibility="collapsed"
                        )
                
                exit_code = process.poll()
                
                # Get final return code
                return_code = exit_code
                
                # Final output
                final_output = '\n'.join(log_lines) if 'log_lines' in locals() else st.session_state.training_log
                st.session_state.training_output = final_output
                st.session_state.training_in_progress = False
                
                status_placeholder.empty()
                
                if return_code == 0:
                    st.success("✅ Training completed successfully!")
                    # Try to load results
                    saved = load_saved_results()
                    if saved:
                        st.session_state.training_results = saved
                else:
                    st.error(f"❌ Training failed with return code {return_code}")
                
                # Show final log
                log_placeholder.text_area(
                    "Training Output (Final)",
                    value=final_output,
                    height=400,
                    disabled=True,
                    label_visibility="collapsed"
                )
            
        except FileNotFoundError:
            st.error("❌ Docker command not found. Make sure Docker is installed and in PATH.")
            st.session_state.training_in_progress = False
            status_placeholder.empty()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.code(str(e))
            st.session_state.training_in_progress = False
            status_placeholder.empty()
    
    # Show training output if available (after completion)
    elif st.session_state.training_output:
        st.subheader("📜 Training Output")
        st.text_area(
            "Training Log",
            value=st.session_state.training_output,
            height=400,
            disabled=True,
            label_visibility="collapsed"
        )
    
    # Display results if available
    results = st.session_state.training_results or load_saved_results()
    if results:
        st.subheader("📈 Training Results")
        
        metrics = results.get('metrics', {})
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric("Training R²", f"{metrics.get('train_r2', 0):.4f}")
            st.metric("Training MAE", f"{metrics.get('train_mae', 0):.2f}M VND")
            st.metric("Training RMSE", f"{metrics.get('train_rmse', 0):.2f}M VND")
        
        with res_col2:
            st.metric("Test R²", f"{metrics.get('test_r2', 0):.4f}")
            st.metric("Test MAE", f"{metrics.get('test_mae', 0):.2f}M VND")
            st.metric("Test RMSE", f"{metrics.get('test_rmse', 0):.2f}M VND")
        
        # Metrics comparison chart
        fig_metrics = go.Figure()
        
        fig_metrics.add_trace(go.Scatter(
            x=['MAE', 'RMSE', 'R² (%)'],
            y=[metrics.get('train_mae', 0), metrics.get('train_rmse', 0), metrics.get('train_r2', 0) * 100],
            mode='lines+markers',
            name='Training',
            line=dict(color='#1f77b4', width=3)
        ))
        
        fig_metrics.add_trace(go.Scatter(
            x=['MAE', 'RMSE', 'R² (%)'],
            y=[metrics.get('test_mae', 0), metrics.get('test_rmse', 0), metrics.get('test_r2', 0) * 100],
            mode='lines+markers',
            name='Test',
            line=dict(color='#ff7f0e', width=3)
        ))
        
        fig_metrics.update_layout(
            title="Training vs Test Performance",
            xaxis_title="Metric",
            yaxis_title="Value",
            height=400
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True)

with tab4:
    st.header("🔮 Make Predictions")
    
    # Check if model files exist (more reliable than checking results JSON)
    # Check relative to the script location first, then Docker paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_model_paths = [
        os.path.join(script_dir, "models", "salary_rf_model"),  # Local: ./spark/app/models/salary_rf_model
        "./models/salary_rf_model",  # Relative to current working directory
        "./spark/app/models/salary_rf_model",  # From project root
        "/opt/spark/work-dir/models/salary_rf_model",  # Docker container path
        "models/salary_rf_model"  # Simple relative path
    ]
    
    model_exists = False
    model_path = None
    for path in possible_model_paths:
        if os.path.exists(path):
            model_exists = True
            model_path = path
            break
    
    # Also check for results (for displaying metrics)
    results = st.session_state.training_results or load_saved_results()
    
    # Show model status
    if model_exists:
        st.success(f"✅ Model found - predictions available (Model: {model_path})")
    elif results:
        st.info("ℹ️ Model metadata found, but model file not detected. Please check model path.")
    else:
        st.warning("⚠️ No model found. Please train a model first.")
    
    # Show sample predictions if available
    if results and 'sample_predictions' in results:
        st.subheader("📋 Sample Predictions from Test Set")
        predictions_df = pd.DataFrame(results['sample_predictions'])
        st.dataframe(predictions_df, use_container_width=True)
        
        # Prediction accuracy visualization
        if 'actual' in predictions_df.columns and 'predicted' in predictions_df.columns:
            fig_pred = go.Figure()
            
            fig_pred.add_trace(go.Scatter(
                x=predictions_df['actual'],
                y=predictions_df['predicted'],
                mode='markers',
                name='Predictions',
                marker=dict(size=10, color='#1f77b4')
            ))
            
            # Perfect prediction line
            max_val = max(predictions_df['actual'].max(), predictions_df['predicted'].max())
            min_val = min(predictions_df['actual'].min(), predictions_df['predicted'].min())
            
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            fig_pred.update_layout(
                title="Actual vs Predicted Salaries",
                xaxis_title="Actual Salary (Million VND)",
                yaxis_title="Predicted Salary (Million VND)",
                height=500
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
    
    # Prediction form - always show
    st.subheader("🔧 Make a New Prediction")
    
    if sys.platform == 'win32':
        st.info("""
        💡 **Tip for Windows Users**: If model loading fails, you can run predictions through Docker:
        ```bash
        docker exec -e JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 spark-master \\
          /opt/spark/bin/spark-submit \\
          --packages com.datastax.spark:spark-cassandra-connector_2.12:3.5.0 \\
          /opt/spark/work-dir/predict_app.py
        ```
        """)
    
    with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                city = st.selectbox("City", ["hồ chí minh", "hà nội", "đà nẵng", "other"])
                job_type = st.selectbox("Job Type", ["nhân viên chính thức", "part-time", "freelance"])
                position_level = st.selectbox("Position Level", ["nhân viên", "chuyên viên", "trưởng phòng", "giám đốc"])
            
            with col2:
                num_skills = st.number_input("Number of Skills", min_value=0, max_value=20, value=5)
                num_fields = st.number_input("Number of Job Fields", min_value=0, max_value=10, value=2)
                experience = st.selectbox("Experience", ["không yêu cầu", "1-2 năm", "3-5 năm", "5+ năm"])
            
            submitted = st.form_submit_button("🔮 Predict Salary", type="primary")
            
            if submitted:
                try:
                    # Load model and make prediction
                    from ml_train_from_cassandra_pyspark import MLTrainerFromCassandraPySpark
                    import pickle
                    
                    # Initialize trainer
                    trainer = MLTrainerFromCassandraPySpark(cassandra_host=cassandra_host, cassandra_port=cassandra_port)
                    
                    # Try loading from Cassandra first
                    model = None
                    model_info = None
                    model_path_used = None
                    
                    st.info("🔍 Looking for model in Cassandra...")
                    
                    # First, check if Cassandra connection works
                    with st.spinner("Checking Cassandra connection..."):
                        try:
                            # Check connection - returns (success, message, count)
                            connection_ok, connection_msg, count = trainer.check_cassandra_connection(keyspace='jobdb', table='ml_models')
                            
                            if not connection_ok:
                                st.error(f"❌ {connection_msg}")
                                st.info("""
                                **To fix this:**
                                1. Make sure Cassandra is running: `docker ps | grep cassandra`
                                2. Create the keyspace and table:
                                   ```bash
                                   docker exec -it cassandra cqlsh -f /opt/spark/work-dir/create_ml_models_schema.cql
                                   ```
                                3. Or train a model first - it will create the table automatically
                                """)
                                model = None
                                model_info = None
                            elif count == 0:
                                st.warning("⚠️ Connected to Cassandra, but the table is empty (no models found).")
                                st.info("Please train a model first - it will be saved to Cassandra automatically.")
                                model = None
                                model_info = None
                            else:
                                st.success(f"✓ Found {count} model(s) in Cassandra. Loading latest...")
                                # Load the model
                                model, model_info = trainer.load_model_from_cassandra(keyspace='jobdb', table='ml_models')
                        except Exception as e:
                            st.error(f"❌ Error checking Cassandra: {str(e)}")
                            st.exception(e)
                            model = None
                            model_info = None
                    
                    if model and model_info:
                        model_path_used = f"Cassandra (Model ID: {model_info['model_id']})"
                        st.success(f"✅ Model loaded from Cassandra: {model_info['model_name']}")
                        st.info(f"📊 R² Score: {model_info['r2_score']:.4f} | MAE: {model_info['mae']:.2f} | RMSE: {model_info['rmse']:.2f}")
                    elif model_info and not model:
                        # Model metadata found but model file doesn't exist
                        st.warning(f"⚠️ Model metadata found but model file not found at: {model_info['model_path']}")
                        st.info(f"**Model Info:** {model_info['model_name']} (ID: {model_info['model_id'][:8]}...)")
                        st.info("Trying alternative paths...")
                        
                        # Try alternative paths
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        stored_path = model_info['model_path']
                        possible_model_paths = [
                            stored_path,  # Try stored path first
                            os.path.join(script_dir, "models", "salary_rf_pyspark"),
                            os.path.join(script_dir, "models", "salary_rf_model"),
                            "./models/salary_rf_pyspark",
                            "./models/salary_rf_model",
                            "./spark/app/models/salary_rf_pyspark",
                            "/opt/spark/work-dir/models/salary_rf_pyspark",
                            "models/salary_rf_pyspark"
                        ]
                        
                        model_loaded = False
                        for path in possible_model_paths:
                            if os.path.exists(path):
                                loaded_model = trainer.load_model(path)
                                if loaded_model:
                                    model = loaded_model
                                    model_path_used = path
                                    st.success(f"✅ Model loaded from alternative path: {path}")
                                    model_loaded = True
                                    break
                        
                        if not model_loaded:
                            # Check if it's the Windows NativeIO error
                            st.error("❌ Could not load model file due to Windows Hadoop NativeIO error.")
                            st.warning("""
                            **This is a known limitation of Spark on Windows.**
                            
                            **Recommended Solution: Use Docker**
                            
                            The model files exist, but Spark on Windows cannot read them due to Hadoop native library issues.
                            
                            **To use predictions, please:**
                            1. Run the prediction through Docker:
                               ```bash
                               docker exec -e JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 spark-master \\
                                 /opt/spark/bin/spark-submit \\
                                 --packages com.datastax.spark:spark-cassandra-connector_2.12:3.5.0 \\
                                 /opt/spark/work-dir/predict_app.py
                               ```
                            
                            2. Or use WSL2 (Windows Subsystem for Linux) to run Spark
                            
                            3. Or train and use models only in Docker environment
                            """)
                            model = None
                    elif model is None and model_info is None:
                        # No model metadata found
                        st.info("🔍 Model not found in Cassandra. Trying disk paths...")
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        possible_model_paths = [
                            os.path.join(script_dir, "models", "salary_rf_model"),  # Local: ./spark/app/models/salary_rf_model
                            "./models/salary_rf_model",  # Relative to current working directory
                            "./spark/app/models/salary_rf_model",  # From project root
                            "/opt/spark/work-dir/models/salary_rf_model",  # Docker container path
                            "models/salary_rf_model"  # Simple relative path
                        ]
                        
                        for path in possible_model_paths:
                            if os.path.exists(path):
                                model = trainer.load_model(path)
                                if model:
                                    model_path_used = path
                                    break
                    
                    if not model:
                        if model_info:
                            # Model metadata exists but file not found or can't be loaded
                            stored_path = model_info['model_path']
                            st.error(f"❌ Cannot load model file from: `{stored_path}`")
                            
                            # Check if it's likely a Windows NativeIO issue
                            if sys.platform == 'win32':
                                st.warning("""
                                **⚠️ Windows Hadoop NativeIO Error Detected**
                                
                                This is a known limitation when running Spark locally on Windows.
                                Spark cannot read model files due to Hadoop native library issues.
                                
                                **Solutions:**
                                1. **Use Docker (Recommended)**: Run predictions in Docker where Spark works properly
                                   ```bash
                                   docker exec -e JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 spark-master \\
                                     /opt/spark/bin/spark-submit \\
                                     --packages com.datastax.spark:spark-cassandra-connector_2.12:3.5.0 \\
                                     /opt/spark/work-dir/predict_app.py
                                   ```
                                
                                2. **Use WSL2**: Run Spark in Windows Subsystem for Linux
                                
                                3. **Train and predict in same environment**: Use Docker for both training and prediction
                                
                                **Note:** The model files exist, but Windows Spark cannot access them due to native library limitations.
                                """)
                            else:
                                st.warning("""
                                **Possible solutions:**
                                1. The model was trained in Docker but you're running locally (or vice versa)
                                2. The model file was deleted or moved
                                3. Path mismatch between training and prediction environments
                                
                                **Try:**
                                - Retrain the model in the same environment where you'll use it
                                - Or manually copy the model files to the expected location
                                """)
                        else:
                            st.error("❌ Model not found. Please train a model first.")
                            st.info("Expected model locations:")
                            st.write("  • Cassandra: jobdb.ml_models table")
                            st.write("  • Disk paths:")
                            script_dir = os.path.dirname(os.path.abspath(__file__))
                            possible_model_paths = [
                                os.path.join(script_dir, "models", "salary_rf_pyspark"),
                                os.path.join(script_dir, "models", "salary_rf_model"),
                                "./models/salary_rf_pyspark",
                                "./models/salary_rf_model",
                                "./spark/app/models/salary_rf_pyspark",
                                "/opt/spark/work-dir/models/salary_rf_pyspark",
                                "models/salary_rf_pyspark"
                            ]
                            for path in possible_model_paths:
                                exists = "✅" if os.path.exists(path) else "❌"
                                st.write(f"    {exists} {path}")
                        trainer.close()
                    else:
                        if not model_path_used.startswith("Cassandra"):
                            st.success(f"✅ Model loaded from disk: {model_path_used}")
                        # Load indexers
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        possible_indexer_paths = [
                            os.path.join(script_dir, "models", "indexers.pkl"),  # Local: ./spark/app/models/indexers.pkl
                            "./models/indexers.pkl",  # Relative to current working directory
                            "./spark/app/models/indexers.pkl",  # From project root
                            "/opt/spark/work-dir/models/indexers.pkl",  # Docker container path
                            "models/indexers.pkl"  # Simple relative path
                        ]
                        
                        indexers = None
                        for path in possible_indexer_paths:
                            if os.path.exists(path):
                                try:
                                    with open(path, 'rb') as f:
                                        indexers = pickle.load(f)
                                    break
                                except:
                                    continue
                        
                        if indexers is None:
                            st.warning("⚠️ Indexers not found. Using fallback encoding (may be less accurate).")
                        
                        # Make prediction
                        # Estimate title_length (average)
                        title_length = len(job_type) + len(position_level) + 10  # Rough estimate
                        
                        with st.spinner("Making prediction..."):
                            prediction = trainer.predict_salary(
                                model=model,
                                city=city,
                                job_type=job_type,
                                position_level=position_level,
                                experience=experience,
                                num_skills=num_skills,
                                num_fields=num_fields,
                                title_length=title_length,
                                indexers=indexers
                            )
                        
                        if prediction is not None:
                            st.success(f"💰 **Predicted Salary: {prediction:.2f} Million VND**")
                            st.info(f"""
**Input Features:**
- City: {city}
- Job Type: {job_type}
- Position Level: {position_level}
- Experience: {experience}
- Number of Skills: {num_skills}
- Number of Job Fields: {num_fields}
- Title Length: {title_length}
                            """)
                        else:
                            st.error("❌ Failed to make prediction. Check the logs for errors.")
                        
                        trainer.close()
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    import traceback
                    with st.expander("Show Error Details"):
                        st.code(traceback.format_exc())
    
    # Show diagnostic information if model not found
    # if not model_exists:
        
        # for path in possible_model_paths:
        #     exists = "✅ Found" if os.path.exists(path) else "❌ Not found"
        #     st.write(f"- `{path}`: {exists}")
        

with tab5:
    st.header("🔍 Job Recommendation System")
    st.markdown("Find similar jobs using content-based filtering with TF-IDF similarity")
    
    # Sidebar for recommendation settings
    st.sidebar.subheader("🔍 Recommendation Settings")
    rec_data_limit = st.sidebar.number_input("Recommendation Data Limit", value=10000, min_value=100, max_value=100000, step=1000, key="rec_limit")
    num_features = st.sidebar.number_input("TF-IDF Features", value=500, min_value=100, max_value=2000, step=100, key="tfidf_features")
    top_n = st.sidebar.slider("Number of Recommendations", 3, 20, 5, key="top_n")
    
    # Build Recommender Section
    st.subheader("1. Build Recommendation Model")
    
    col_build1, col_build2 = st.columns([2, 1])
    
    with col_build1:
        st.info("""
        **How it works:**
        1. Loads job data from Cassandra
        2. Creates TF-IDF features from job text (title, skills, fields, city)
        3. Normalizes vectors for cosine similarity calculation
        4. Enables fast similarity search across all jobs
        """)
    
    with col_build2:
        if st.session_state.recommender_ready:
            st.success("✅ Recommender Ready!")
            if st.session_state.sample_jobs is not None:
                st.metric("Jobs Loaded", f"{len(st.session_state.sample_jobs):,}")
        
        build_button = st.button("🔧 Build Recommender", type="primary", key="build_rec")
    
    if build_button:
        try:
            with st.spinner("Building recommendation model... This may take a minute..."):
                from ml_job_recommendation import JobRecommenderPySpark
                
                # Initialize recommender
                recommender = JobRecommenderPySpark(
                    cassandra_host=cassandra_host, 
                    cassandra_port=cassandra_port
                )
                
                # Load data
                result = recommender.load_data_from_cassandra(
                    keyspace=keyspace, 
                    limit=rec_data_limit
                )
                
                if result is None or recommender.df is None or recommender.df.count() == 0:
                    st.error("No data found in Cassandra. Please ensure data is available.")
                else:
                    # Prepare features
                    recommender.prepare_features(num_features=num_features)
                    
                    # Store in session state
                    st.session_state.recommender = recommender
                    st.session_state.recommender_ready = True
                    
                    # Get sample jobs for dropdown (include salary info)
                    sample_jobs_df = recommender.features_df.select(
                        "job_index", "job_title", "city", "position_level",
                        "salary_min", "salary_max", "unit"
                    ).limit(500).toPandas()
                    st.session_state.sample_jobs = sample_jobs_df
                    
                    # Store full features as pandas for fallback
                    full_features_df = recommender.features_df.select(
                        "job_index", "job_title", "city", "position_level",
                        "salary_min", "salary_max", "unit", "skills", "experience"
                    ).toPandas()
                    st.session_state.jobs_features_pandas = full_features_df
                    
                    st.success(f"✅ Recommender built successfully with {recommender.features_df.count():,} jobs!")
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Error building recommender: {str(e)}")
            import traceback
            with st.expander("Show Error Details"):
                st.code(traceback.format_exc())
    
    st.markdown("---")
    
    # Two recommendation methods
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.subheader("2. Find Similar Jobs")
        st.markdown("Select a job to find similar positions")
        
        if st.session_state.recommender_ready and st.session_state.sample_jobs is not None:
            # Create job options for dropdown
            sample_df = st.session_state.sample_jobs
            job_options = [
                f"{row['job_index']}: {row['job_title']} ({row['city']})"
                for _, row in sample_df.iterrows()
            ]
            
            selected_job = st.selectbox(
                "Select a Job",
                options=job_options,
                key="similar_job_select"
            )
            
            if st.button("🔍 Find Similar Jobs", key="find_similar"):
                try:
                    # Extract job index from selection
                    job_idx = int(selected_job.split(":")[0])
                    
                    with st.spinner("Finding similar jobs... (rebuilding model if needed)"):
                        # Get or rebuild recommender
                        recommender, was_rebuilt = get_or_rebuild_recommender(
                            cassandra_host, cassandra_port, keyspace, 
                            rec_data_limit, num_features
                        )
                        
                        if recommender is None:
                            st.error("Failed to build recommender. Check Cassandra connection.")
                        else:
                            if was_rebuilt:
                                st.info("Spark session was rebuilt automatically.")
                            
                            similar_jobs = recommender.find_similar_jobs(job_idx, top_n=top_n)
                            similar_jobs_list = similar_jobs.collect()
                    
                            # Get selected job info from pandas cache
                            if st.session_state.sample_jobs is not None:
                                selected_job_row = st.session_state.sample_jobs[
                                    st.session_state.sample_jobs['job_index'] == job_idx
                                ]
                                if len(selected_job_row) > 0:
                                    job_info = selected_job_row.iloc[0]
                                    st.markdown("**Selected Job:**")
                                    s_min = job_info.get('salary_min', 0) or 0
                                    s_max = job_info.get('salary_max', 0) or 0
                                    s_unit = job_info.get('unit', '') or ''
                                    st.info(f"""
                                    **{job_info['job_title']}**
                                    - City: {job_info['city']}
                                    - Position: {job_info['position_level']}
                                    - Salary: {s_min:.0f}-{s_max:.0f} {s_unit}
                                    """)
                            
                            # Display similar jobs
                            st.markdown(f"**Top {len(similar_jobs_list)} Similar Jobs:**")
                            
                            for idx, row in enumerate(similar_jobs_list, 1):
                                s_min = row['salary_min'] if row['salary_min'] else 0
                                s_max = row['salary_max'] if row['salary_max'] else 0
                                s_unit = row['unit'] if row['unit'] else ''
                                similarity = row['similarity_score']
                                
                                # Color based on similarity
                                if similarity >= 0.9:
                                    color = "green"
                                elif similarity >= 0.7:
                                    color = "orange"
                                else:
                                    color = "red"
                                
                                st.markdown(f"""
                                **{idx}. {row['job_title']}** 
                                <span style='color: {color}; font-weight: bold;'>Similarity: {similarity:.3f}</span>
                                """, unsafe_allow_html=True)
                                
                                st.write(f"   📍 {row['city']} | 💼 {row['position_level']} | 💰 {s_min:.0f}-{s_max:.0f} {s_unit}")
                                
                                with st.expander(f"Skills for job #{idx}"):
                                    st.write(row['skills'] if row['skills'] else "No skills listed")
                        
                except Exception as e:
                    st.error(f"Error finding similar jobs: {str(e)}")
                    import traceback
                    with st.expander("Show Error Details"):
                        st.code(traceback.format_exc())
        else:
            st.warning("⚠️ Please build the recommender first (click 'Build Recommender' above)")
    
    with rec_col2:
        st.subheader("3. Search by Query")
        st.markdown("Enter job description to find matching positions")
        
        if st.session_state.recommender_ready:
            query_text = st.text_area(
                "Enter job description or keywords",
                value="nhân viên kinh doanh marketing digital",
                height=100,
                key="query_input"
            )
            
            if st.button("🔎 Search Jobs", key="search_query"):
                try:
                    with st.spinner("Searching for matching jobs... (rebuilding model if needed)"):
                        # Get or rebuild recommender
                        recommender, was_rebuilt = get_or_rebuild_recommender(
                            cassandra_host, cassandra_port, keyspace,
                            rec_data_limit, num_features
                        )
                        
                        if recommender is None:
                            st.error("Failed to build recommender. Check Cassandra connection.")
                        else:
                            if was_rebuilt:
                                st.info("Spark session was rebuilt automatically.")
                            
                            recommendations = recommender.recommend_by_query(query_text, top_n=top_n)
                            recommendations_list = recommendations.collect()
                    
                    st.markdown(f"**Query:** *{query_text}*")
                    st.markdown(f"**Top {len(recommendations_list)} Recommendations:**")
                    
                    for idx, row in enumerate(recommendations_list, 1):
                        s_min = row['salary_min'] if row['salary_min'] else 0
                        s_max = row['salary_max'] if row['salary_max'] else 0
                        s_unit = row['unit'] if row['unit'] else ''
                        similarity = row['similarity_score']
                        
                        # Color based on similarity
                        if similarity >= 0.9:
                            color = "green"
                        elif similarity >= 0.7:
                            color = "orange"
                        else:
                            color = "red"
                        
                        st.markdown(f"""
                        **{idx}. {row['job_title']}**
                        <span style='color: {color}; font-weight: bold;'>Match: {similarity:.3f}</span>
                        """, unsafe_allow_html=True)
                        
                        st.write(f"   📍 {row['city']} | 💼 {row['position_level']} | 💰 {s_min:.0f}-{s_max:.0f} {s_unit}")
                        
                        with st.expander(f"Skills for job #{idx}"):
                            st.write(row['skills'] if row['skills'] else "No skills listed")
                    
                    # Visualize similarity scores
                    if recommendations_list:
                        st.markdown("---")
                        st.markdown("**Similarity Score Distribution:**")
                        
                        rec_df = pd.DataFrame([{
                            'Job Title': row['job_title'][:30] + "..." if len(row['job_title']) > 30 else row['job_title'],
                            'Similarity': row['similarity_score'],
                            'City': row['city']
                        } for row in recommendations_list])
                        
                        fig_sim = px.bar(
                            rec_df, 
                            x='Similarity', 
                            y='Job Title',
                            orientation='h',
                            color='Similarity',
                            color_continuous_scale='RdYlGn',
                            title='Similarity Scores'
                        )
                        fig_sim.update_layout(
                            height=300,
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        st.plotly_chart(fig_sim, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error searching jobs: {str(e)}")
                    import traceback
                    with st.expander("Show Error Details"):
                        st.code(traceback.format_exc())
        else:
            st.warning("⚠️ Please build the recommender first (click 'Build Recommender' above)")
    
    st.markdown("---")
    
    # Quick Examples Section
    st.subheader("💡 Quick Examples")
    
    example_queries = [
        "lập trình viên java backend",
        "marketing manager digital",
        "kế toán trưởng",
        "nhân viên bán hàng",
        "data scientist machine learning",
        "quản lý dự án IT"
    ]
    
    st.write("**Try these example queries:**")
    cols = st.columns(3)
    for i, query in enumerate(example_queries):
        with cols[i % 3]:
            if st.button(f"🔎 {query}", key=f"example_{i}"):
                if st.session_state.recommender_ready:
                    try:
                        with st.spinner(f"Searching for '{query}'..."):
                            # Get or rebuild recommender
                            recommender, was_rebuilt = get_or_rebuild_recommender(
                                cassandra_host, cassandra_port, keyspace,
                                rec_data_limit, num_features
                            )
                            
                            if recommender is None:
                                st.error("Failed to build recommender.")
                            else:
                                recommendations = recommender.recommend_by_query(query, top_n=5)
                                recommendations_list = recommendations.collect()
                                
                                st.markdown(f"**Results for:** *{query}*")
                                for idx, row in enumerate(recommendations_list, 1):
                                    s_min = row['salary_min'] if row['salary_min'] else 0
                                    s_max = row['salary_max'] if row['salary_max'] else 0
                                    st.write(f"{idx}. **{row['job_title']}** ({row['city']}) - {row['similarity_score']:.3f}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.warning("Build recommender first!")
    
    # Run via Docker Section
    st.markdown("---")
    st.subheader("🐳 Run via Docker")
    
    docker_rec_cmd = f"""docker exec -e JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 {docker_container} /opt/spark/bin/spark-submit \\
    --packages com.datastax.spark:spark-cassandra-connector_2.12:3.5.0 \\
    /opt/spark/work-dir/ml_job_recommendation.py"""
    
    st.code(docker_rec_cmd, language="bash")
    
    if st.button("🚀 Run Recommendation Script in Docker", key="run_rec_docker"):
        try:
            with st.spinner("Running recommendation script in Docker..."):
                if DOCKER_SDK_AVAILABLE:
                    docker_client = docker.from_env()
                    exec_result = docker_client.containers.get(docker_container).exec_run(
                        cmd=[
                            "/opt/spark/bin/spark-submit",
                            "--packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0",
                            "/opt/spark/work-dir/ml_job_recommendation.py"
                        ],
                        environment={"JAVA_HOME": "/usr/lib/jvm/java-17-openjdk-amd64"},
                        stdout=True,
                        stderr=True
                    )
                    output = exec_result.output.decode('utf-8', errors='replace') if exec_result.output else "No output"
                else:
                    result = subprocess.run(
                        [
                            "docker", "exec", "-e", "JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64",
                            docker_container,
                            "/opt/spark/bin/spark-submit",
                            "--packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0",
                            "/opt/spark/work-dir/ml_job_recommendation.py"
                        ],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    output = result.stdout + result.stderr
                
                st.text_area("Docker Output", value=output, height=400)
                
        except Exception as e:
            st.error(f"Error running Docker command: {str(e)}")
            import traceback
            with st.expander("Show Error Details"):
                st.code(traceback.format_exc())

with tab6:
    st.header("🛠️ Skills Analysis & Recommendation")
    st.markdown("Analyze skills using Word2Vec embeddings and LDA topic modeling")
    
    # Sidebar for skills settings
    st.sidebar.subheader("🛠️ Skills Analysis Settings")
    skills_data_limit = st.sidebar.number_input("Skills Data Limit", value=20000, min_value=1000, max_value=100000, step=1000, key="skills_limit")
    vector_size = st.sidebar.number_input("Word2Vec Vector Size", value=100, min_value=50, max_value=300, step=50, key="vector_size")
    num_topics = st.sidebar.number_input("LDA Topics (Clusters)", value=8, min_value=3, max_value=20, step=1, key="num_topics")
    
    # Build Skills Recommender Section
    st.subheader("1. Build Skills Recommender")
    
    col_build1, col_build2 = st.columns([2, 1])
    
    with col_build1:
        st.info("""
        **How it works:**
        1. **Load Saved Models** (Fast): Loads pre-trained Word2Vec and LDA models from disk
        2. **Train New Models** (Slow): Extracts skills and trains new models from Cassandra data
        
        Models are saved at: `models/skills_recommender/`
        """)
        
        # Check if saved models exist
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(script_dir, "models", "skills_recommender"),
            "/opt/spark/work-dir/models/skills_recommender"
        ]
        saved_models_exist = any(os.path.exists(p) for p in possible_paths)
        
        if saved_models_exist:
            st.success("✅ Saved models found on disk")
        else:
            st.warning("⚠️ No saved models found - train new models first")
    
    with col_build2:
        if st.session_state.skills_recommender_ready:
            st.success("✅ Skills Recommender Ready!")
            if st.session_state.top_skills is not None:
                st.metric("Unique Skills", f"{len(st.session_state.top_skills):,}")
            if st.session_state.skill_clusters is not None:
                st.metric("Skill Clusters", len(st.session_state.skill_clusters))
        
        # Two buttons: Load saved or Train new
        load_saved_button = st.button("📂 Load Saved Models", type="primary", key="load_skills", 
                                       disabled=not saved_models_exist)
        train_new_button = st.button("🔧 Train New Models", type="secondary", key="train_skills")
    
    # Load saved models
    if load_saved_button:
        try:
            with st.spinner("Loading saved models from disk..."):
                from ml_skills_recommendation import SkillsRecommenderPySpark
                
                recommender = SkillsRecommenderPySpark(
                    cassandra_host=cassandra_host,
                    cassandra_port=cassandra_port
                )
                
                # Try to load from possible paths
                model_loaded = False
                for model_path in possible_paths:
                    if os.path.exists(model_path):
                        success = recommender.load_model(model_path)
                        if success:
                            model_loaded = True
                            st.success(f"✓ Models loaded from {model_path}")
                            break
                
                if model_loaded:
                    # Also need to load data for some operations (like autocomplete)
                    result = recommender.load_data_from_cassandra(
                        keyspace=keyspace,
                        limit=skills_data_limit
                    )
                    
                    # Store in session state
                    st.session_state.skills_recommender = recommender
                    st.session_state.skills_recommender_ready = True
                    
                    # Store skill clusters
                    if recommender.skill_clusters:
                        st.session_state.skill_clusters = recommender.skill_clusters
                    
                    # Get top skills
                    if recommender.skills_df is not None:
                        top_skills_df = recommender.skills_df.limit(100).toPandas()
                        st.session_state.top_skills = top_skills_df
                    
                    st.success("✅ Skills Recommender loaded successfully!")
                    st.rerun()
                else:
                    st.error("Failed to load models. Try training new models.")
                    
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            import traceback
            with st.expander("Show Error Details"):
                st.code(traceback.format_exc())
    
    # Train new models
    if train_new_button:
        try:
            with st.spinner("Training new models... This may take 1-2 minutes..."):
                from ml_skills_recommendation import SkillsRecommenderPySpark
                
                recommender = SkillsRecommenderPySpark(
                    cassandra_host=cassandra_host,
                    cassandra_port=cassandra_port
                )
                
                result = recommender.load_data_from_cassandra(
                    keyspace=keyspace,
                    limit=skills_data_limit
                )
                
                if result is None or recommender.df is None or recommender.df.count() == 0:
                    st.error("No data found in Cassandra. Please ensure data is available.")
                else:
                    # Extract skills
                    st.info("Extracting skills from job postings...")
                    recommender.extract_skills()
                    
                    # Train Word2Vec
                    st.info("Training Word2Vec model...")
                    recommender.train_word2vec(vector_size=vector_size, min_count=5, window_size=5)
                    
                    # Train LDA
                    st.info("Training LDA topic model...")
                    recommender.train_lda_topic_model(num_topics=num_topics, max_iter=15)
                    
                    # Save models to disk
                    st.info("Saving models to disk...")
                    model_save_path = os.path.join(script_dir, "models", "skills_recommender")
                    if os.path.exists("/.dockerenv"):
                        model_save_path = "/opt/spark/work-dir/models/skills_recommender"
                    
                    saved_path, saved_components = recommender.save_model(model_path=model_save_path)
                    if saved_path:
                        st.success(f"✓ Models saved to {saved_path}")
                    
                    # Store in session state
                    st.session_state.skills_recommender = recommender
                    st.session_state.skills_recommender_ready = True
                    
                    # Store skill clusters
                    if recommender.skill_clusters:
                        st.session_state.skill_clusters = recommender.skill_clusters
                    
                    # Get top skills
                    if recommender.skills_df:
                        top_skills_df = recommender.skills_df.limit(100).toPandas()
                        st.session_state.top_skills = top_skills_df
                    
                    st.success("✅ Skills Recommender trained and saved successfully!")
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Error training models: {str(e)}")
            import traceback
            with st.expander("Show Error Details"):
                st.code(traceback.format_exc())
    
    st.markdown("---")
    
    # Display Skill Clusters
    if st.session_state.skill_clusters:
        st.subheader("📊 Discovered Skill Clusters")
        
        cluster_cols = st.columns(4)
        for i, cluster in enumerate(st.session_state.skill_clusters):
            with cluster_cols[i % 4]:
                with st.expander(f"Cluster {cluster['topic_id'] + 1}", expanded=(i < 4)):
                    for skill, weight in zip(cluster['skills'][:5], cluster['weights'][:5]):
                        st.write(f"• {skill} ({weight:.3f})")
        
        st.markdown("---")
    
    # Skills Analysis Features
    skills_col1, skills_col2 = st.columns(2)
    
    with skills_col1:
        st.subheader("2. Find Similar Skills")
        st.markdown("Find skills similar to a given skill using Word2Vec")
        
        if st.session_state.skills_recommender_ready:
            skill_input = st.text_input("Enter a skill", value="python", key="similar_skill_input")
            similar_top_n = st.slider("Number of results", 3, 15, 5, key="similar_skill_n")
            
            if st.button("🔍 Find Similar Skills", key="find_similar_skills"):
                try:
                    with st.spinner("Finding similar skills..."):
                        recommender, was_rebuilt = get_or_rebuild_skills_recommender(
                            cassandra_host, cassandra_port, keyspace,
                            skills_data_limit, vector_size, num_topics
                        )
                        
                        if recommender is None:
                            st.error("Failed to build skills recommender.")
                        else:
                            if was_rebuilt:
                                st.info("Spark session was rebuilt automatically.")
                            
                            similar_skills = recommender.find_similar_skills(skill_input, top_n=similar_top_n)
                            
                            if similar_skills:
                                results = similar_skills.collect()
                                st.markdown(f"**Skills similar to '{skill_input}':**")
                                
                                # Create DataFrame for display
                                similar_df = pd.DataFrame([{
                                    'Skill': row['word'],
                                    'Similarity': row['similarity']
                                } for row in results])
                                
                                # Bar chart
                                fig = px.bar(
                                    similar_df,
                                    x='Similarity',
                                    y='Skill',
                                    orientation='h',
                                    color='Similarity',
                                    color_continuous_scale='Viridis',
                                    title=f'Skills Similar to "{skill_input}"'
                                )
                                fig.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"Skill '{skill_input}' not found in vocabulary.")
                                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("⚠️ Please build the skills recommender first")
    
    with skills_col2:
        st.subheader("3. Skill Autocomplete")
        st.markdown("Autocomplete skills based on prefix")
        
        if st.session_state.skills_recommender_ready:
            prefix_input = st.text_input("Enter skill prefix", value="java", key="autocomplete_input")
            autocomplete_n = st.slider("Number of suggestions", 3, 15, 5, key="autocomplete_n")
            
            if st.button("🔎 Autocomplete", key="autocomplete_skill"):
                try:
                    with st.spinner("Finding suggestions..."):
                        recommender, was_rebuilt = get_or_rebuild_skills_recommender(
                            cassandra_host, cassandra_port, keyspace,
                            skills_data_limit, vector_size, num_topics
                        )
                        
                        if recommender is None:
                            st.error("Failed to build skills recommender.")
                        else:
                            matches = recommender.autocomplete_skills(prefix_input, top_n=autocomplete_n)
                            results = matches.collect()
                            
                            if results:
                                st.markdown(f"**Suggestions for '{prefix_input}':**")
                                for i, row in enumerate(results, 1):
                                    st.write(f"{i}. **{row['skill']}** (used in {row['frequency']} jobs)")
                            else:
                                st.warning(f"No skills found starting with '{prefix_input}'")
                                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("⚠️ Please build the skills recommender first")
    
    st.markdown("---")
    
    # Skills Recommendation for Job
    st.subheader("4. Skills Recommendation for Job Title")
    
    if st.session_state.skills_recommender_ready:
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            job_title_input = st.text_input("Enter job title", value="developer", key="job_title_skills")
        
        with rec_col2:
            current_skills_input = st.text_input("Your current skills (comma-separated)", value="html, css", key="current_skills")
        
        skills_top_n = st.slider("Number of recommendations", 5, 20, 10, key="skills_rec_n")
        
        if st.button("🎯 Get Skill Recommendations", key="get_skill_recs"):
            try:
                with st.spinner("Finding skill recommendations..."):
                    recommender, was_rebuilt = get_or_rebuild_skills_recommender(
                        cassandra_host, cassandra_port, keyspace,
                        skills_data_limit, vector_size, num_topics
                    )
                    
                    if recommender is None:
                        st.error("Failed to build skills recommender.")
                    else:
                        current_skills = [s.strip() for s in current_skills_input.split(',') if s.strip()]
                        recommendations = recommender.recommend_skills_for_job(
                            job_title_input,
                            current_skills=current_skills if current_skills else None,
                            top_n=skills_top_n
                        )
                        
                        if recommendations:
                            results = recommendations.collect()
                            
                            # Get job count for percentage calculation
                            similar_jobs = recommender.df.filter(
                                recommender.df.job_title.contains(job_title_input.lower())
                            ).count()
                            
                            st.markdown(f"**Recommended skills for '{job_title_input}':**")
                            
                            # Create DataFrame
                            rec_df = pd.DataFrame([{
                                'Skill': row['skill'],
                                'Frequency': row['frequency'],
                                'Percentage': (row['frequency'] / max(similar_jobs, 1)) * 100
                            } for row in results])
                            
                            # Bar chart
                            fig = px.bar(
                                rec_df,
                                x='Percentage',
                                y='Skill',
                                orientation='h',
                                color='Percentage',
                                color_continuous_scale='Blues',
                                title=f'Recommended Skills for "{job_title_input}"'
                            )
                            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"No job postings found matching '{job_title_input}'")
                            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                with st.expander("Show Error Details"):
                    st.code(traceback.format_exc())
    else:
        st.warning("⚠️ Please build the skills recommender first")
    
    st.markdown("---")
    
    # Skill Gap Analysis
    st.subheader("5. Skill Gap Analysis")
    
    if st.session_state.skills_recommender_ready:
        gap_col1, gap_col2 = st.columns(2)
        
        with gap_col1:
            target_job = st.text_input("Target job title", value="data analyst", key="target_job")
        
        with gap_col2:
            your_skills = st.text_input("Your skills (comma-separated)", value="python, sql, excel", key="your_skills")
        
        if st.button("📊 Analyze Skill Gap", key="analyze_gap"):
            try:
                with st.spinner("Analyzing skill gap..."):
                    recommender, was_rebuilt = get_or_rebuild_skills_recommender(
                        cassandra_host, cassandra_port, keyspace,
                        skills_data_limit, vector_size, num_topics
                    )
                    
                    if recommender is None:
                        st.error("Failed to build skills recommender.")
                    else:
                        skills_list = [s.strip() for s in your_skills.split(',') if s.strip()]
                        gap_result = recommender.analyze_skill_gap(
                            current_skills=skills_list,
                            target_job_title=target_job,
                            top_n=10
                        )
                        
                        if gap_result:
                            # Display results
                            gap_col_a, gap_col_b, gap_col_c = st.columns(3)
                            
                            with gap_col_a:
                                st.metric("Career Readiness", f"{gap_result['readiness_score']:.1f}%")
                            
                            with gap_col_b:
                                st.metric("Skills You Have", len(gap_result['matching_skills']))
                            
                            with gap_col_c:
                                st.metric("Skills to Acquire", len(gap_result['missing_skills']))
                            
                            # Two columns for matching and missing
                            match_col, miss_col = st.columns(2)
                            
                            with match_col:
                                st.markdown("**✅ Skills You Have:**")
                                if gap_result['matching_skills']:
                                    for skill in gap_result['matching_skills']:
                                        st.success(f"✓ {skill}")
                                else:
                                    st.info("No matching skills found")
                            
                            with miss_col:
                                st.markdown("**📚 Skills to Acquire:**")
                                for i, skill in enumerate(gap_result['missing_skills'], 1):
                                    st.warning(f"{i}. {skill}")
                            
                            # Progress bar
                            st.progress(min(gap_result['readiness_score'] / 100, 1.0))
                        else:
                            st.warning(f"Could not analyze gap for '{target_job}'")
                            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                with st.expander("Show Error Details"):
                    st.code(traceback.format_exc())
    else:
        st.warning("⚠️ Please build the skills recommender first")
    
    st.markdown("---")
    
    # Career Path Recommendations
    st.subheader("6. Career Path Recommendations")
    
    if st.session_state.skills_recommender_ready:
        career_col1, career_col2 = st.columns(2)
        
        with career_col1:
            current_position = st.text_input("Current position", value="junior developer", key="current_position")
        
        with career_col2:
            career_skills = st.text_input("Your skills (comma-separated)", value="python, javascript, html, css, git", key="career_skills")
        
        career_top_n = st.slider("Number of career paths", 3, 10, 5, key="career_n")
        
        if st.button("🚀 Find Career Paths", key="find_careers"):
            try:
                with st.spinner("Finding career paths..."):
                    recommender, was_rebuilt = get_or_rebuild_skills_recommender(
                        cassandra_host, cassandra_port, keyspace,
                        skills_data_limit, vector_size, num_topics
                    )
                    
                    if recommender is None:
                        st.error("Failed to build skills recommender.")
                    else:
                        skills_list = [s.strip() for s in career_skills.split(',') if s.strip()]
                        career_options = recommender.get_career_path_recommendations(
                            current_position=current_position,
                            current_skills=skills_list,
                            top_n=career_top_n
                        )
                        
                        if career_options:
                            results = career_options.collect()
                            
                            st.markdown(f"**Career paths from '{current_position}':**")
                            
                            shown = 0
                            for row in results:
                                if shown >= career_top_n:
                                    break
                                job_title = row['job_title']
                                if job_title and len(job_title) > 3:
                                    shown += 1
                                    match_pct = row['avg_skill_match'] * 100
                                    salary = row['avg_salary'] if row['avg_salary'] else 0
                                    
                                    with st.expander(f"**{shown}. {job_title[:50]}** - {match_pct:.1f}% match"):
                                        st.write(f"**Position Level:** {row['position_level'] or 'N/A'}")
                                        st.write(f"**Skill Match:** {match_pct:.1f}%")
                                        if salary > 0:
                                            st.write(f"**Avg Salary:** {salary:.1f}M VND")
                                        st.progress(match_pct / 100)
                        else:
                            st.warning("No career paths found")
                            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                with st.expander("Show Error Details"):
                    st.code(traceback.format_exc())
    else:
        st.warning("⚠️ Please build the skills recommender first")
    
    st.markdown("---")
    
    # Run via Docker Section
    st.subheader("🐳 Run via Docker")
    
    docker_skills_cmd = f"""docker exec -e JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 {docker_container} /opt/spark/bin/spark-submit \\
    --packages com.datastax.spark:spark-cassandra-connector_2.12:3.5.0 \\
    /opt/spark/work-dir/ml_skills_recommendation.py"""
    
    st.code(docker_skills_cmd, language="bash")
    
    if st.button("🚀 Run Skills Script in Docker", key="run_skills_docker"):
        try:
            with st.spinner("Running skills recommendation script in Docker..."):
                if DOCKER_SDK_AVAILABLE:
                    docker_client = docker.from_env()
                    exec_result = docker_client.containers.get(docker_container).exec_run(
                        cmd=[
                            "/opt/spark/bin/spark-submit",
                            "--packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0",
                            "/opt/spark/work-dir/ml_skills_recommendation.py"
                        ],
                        environment={"JAVA_HOME": "/usr/lib/jvm/java-17-openjdk-amd64"},
                        stdout=True,
                        stderr=True
                    )
                    output = exec_result.output.decode('utf-8', errors='replace') if exec_result.output else "No output"
                else:
                    result = subprocess.run(
                        [
                            "docker", "exec", "-e", "JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64",
                            docker_container,
                            "/opt/spark/bin/spark-submit",
                            "--packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0",
                            "/opt/spark/work-dir/ml_skills_recommendation.py"
                        ],
                        capture_output=True,
                        text=True,
                        timeout=600
                    )
                    output = result.stdout + result.stderr
                
                st.text_area("Docker Output", value=output, height=400)
                
        except Exception as e:
            st.error(f"Error running Docker command: {str(e)}")
            import traceback
            with st.expander("Show Error Details"):
                st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        ML Model Training Dashboard | Built with Streamlit & PySpark ML
    </div>
""", unsafe_allow_html=True)
