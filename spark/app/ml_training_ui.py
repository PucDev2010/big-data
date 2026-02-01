"""
Streamlit UI for ML Model Training from Cassandra
Provides interactive interface for training and monitoring ML models
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import subprocess
import json
import os
import sys
import time

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

# Configuration options
cassandra_host = st.sidebar.text_input("Cassandra Host", value="127.0.0.1")
cassandra_port = st.sidebar.number_input("Cassandra Port", value=9042, min_value=1, max_value=65535)
keyspace = st.sidebar.selectbox("Keyspace", ["job_analytics", "jobdb"], index=0)
table_name = st.sidebar.text_input("Table Name", value="job_postings")
data_limit = st.sidebar.number_input("Data Limit", value=10000, min_value=100, max_value=100000, step=1000)
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)

# Docker configuration
st.sidebar.subheader("🐳 Docker Configuration")
docker_container = st.sidebar.text_input("Docker Container", value="spark-master")
spark_script_path = st.sidebar.text_input("Script Path in Container", value="/opt/spark/work-dir/ml_train_from_cassandra_pyspark.py")

# Initialize session state
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'training_output' not in st.session_state:
    st.session_state.training_output = None
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'training_log' not in st.session_state:
    st.session_state.training_log = ""

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

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "📊 Data Overview", "🎯 Model Training", "🔮 Predictions"])

with tab1:
    st.header("📈 Training Dashboard")
    
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
    st.header("📊 Data Overview")
    
    st.write("Preview the expected data structure and run queries.")
    
    # Show sample data structure
    st.subheader("📋 Expected Data Structure")
    sample_data = pd.DataFrame({
        'id': ['uuid-1', 'uuid-2', 'uuid-3', 'uuid-4', 'uuid-5'],
        'job_title': ['Software Engineer', 'Data Analyst', 'Product Manager', 'DevOps Engineer', 'ML Engineer'],
        'city': ['Ho Chi Minh', 'Ha Noi', 'Da Nang', 'Ho Chi Minh', 'Ha Noi'],
        'salary_min': [15.0, 12.0, 20.0, 18.0, 25.0],
        'salary_max': [25.0, 18.0, 35.0, 30.0, 40.0],
        'experience': ['2-3 years', '1-2 years', '5+ years', '3-5 years', '3-5 years'],
        'position_level': ['Senior', 'Junior', 'Manager', 'Senior', 'Senior']
    })
    st.dataframe(sample_data, use_container_width=True)
    
    st.subheader("📈 Sample Salary Distribution")
    fig_dist = px.histogram(
        sample_data,
        x='salary_max',
        nbins=10,
        title="Sample Salary Distribution",
        labels={'salary_max': 'Maximum Salary (Million VND)'}
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    st.subheader("🏙️ Sample Jobs by City")
    city_counts = sample_data['city'].value_counts()
    fig_city = px.bar(
        x=city_counts.index,
        y=city_counts.values,
        title="Jobs by City (Sample)",
        labels={'x': 'City', 'y': 'Number of Jobs'}
    )
    st.plotly_chart(fig_city, use_container_width=True)

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
            
            # Get final return code
            return_code = process.poll()
            
            # Final output
            final_output = '\n'.join(log_lines)
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
    if not model_exists:
        st.info("💡 **Model Detection Info:**")
        st.write("Checking for model files in:")
        for path in possible_model_paths:
            exists = "✅ Found" if os.path.exists(path) else "❌ Not found"
            st.write(f"- `{path}`: {exists}")
        
        st.write("\n**Note:** If training completed in Docker, ensure model files are accessible from this location.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        ML Model Training Dashboard | Built with Streamlit & PySpark ML
    </div>
""", unsafe_allow_html=True)
