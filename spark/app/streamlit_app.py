import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ====================================================
# CẤU HÌNH TRANG
# ====================================================
st.set_page_config(
    page_title="Job Attractiveness Analyzer",
    page_icon="🎯",
    layout="wide"
)

# ====================================================
# KẾT NỐI CASSANDRA
# ====================================================
import os

CASSANDRA_HOST = os.getenv('CASSANDRA_HOST', 'localhost')  # Docker: cassandra, Local: localhost

@st.cache_resource
def get_cassandra_session():
    """Kết nối tới Cassandra"""
    try:
        from cassandra.cluster import Cluster
        cluster = Cluster([CASSANDRA_HOST], port=9042)
        session = cluster.connect('job_analytics')
        return session
    except Exception as e:
        return None

@st.cache_data(ttl=60)
def load_data():
    """Đọc dữ liệu từ Cassandra hoặc tạo demo data"""
    session = get_cassandra_session()
    
    if session is not None:
        try:
            query = "SELECT * FROM job_postings"
            rows = session.execute(query)
            df = pd.DataFrame(list(rows))
            if not df.empty:
                return df, True  # True = real data
        except Exception as e:
            pass
    
    # Nếu không kết nối được Cassandra, tạo demo data
    import numpy as np
    np.random.seed(42)
    n = 500
    
    demo_df = pd.DataFrame({
        'job_title': [f'Job {i}' for i in range(n)],
        'city': np.random.choice(['Hồ Chí Minh', 'Hà Nội', 'Đà Nẵng', 'Bình Dương', 'Cần Thơ'], n),
        'salary_avg': np.random.uniform(5, 50, n),
        'salary_min': np.random.uniform(3, 30, n),
        'salary_max': np.random.uniform(20, 60, n),
        'exp_avg_year': np.random.uniform(0, 10, n),
        'exp_min_year': np.random.uniform(0, 5, n),
        'job_fields': np.random.choice(['IT', 'Sales', 'Marketing', 'Finance', 'HR'], n),
    })
    
    return demo_df, False  # False = demo data

@st.cache_data(ttl=60)
def load_clusters():
    """Đọc dữ liệu clustering từ Cassandra"""
    session = get_cassandra_session()
    
    if session is not None:
        try:
            query = "SELECT * FROM job_clusters"
            rows = session.execute(query)
            df = pd.DataFrame(list(rows))
            if not df.empty:
                return df, True
        except Exception as e:
            pass
    
    # Demo data nếu không có
    import numpy as np
    np.random.seed(42)
    n = 500
    demo_df = pd.DataFrame({
        'job_title': [f'Job {i}' for i in range(n)],
        'city': np.random.choice(['Hồ Chí Minh', 'Hà Nội', 'Đà Nẵng'], n),
        'salary_final': np.random.uniform(5, 50, n),
        'exp_final': np.random.uniform(0, 10, n),
        'job_fields': np.random.choice(['IT', 'Sales', 'Finance'], n),
        'cluster': np.random.choice([0, 1, 2, 3, 4], n),
    })
    return demo_df, False

@st.cache_data(ttl=60)
def load_skill_scores():
    """Đọc dữ liệu skill hot scores từ Cassandra"""
    session = get_cassandra_session()
    
    if session is not None:
        try:
            query = "SELECT * FROM skill_hot_scores"
            rows = session.execute(query)
            df = pd.DataFrame(list(rows))
            if not df.empty:
                return df, True
        except Exception as e:
            pass
    
    # Demo data nếu không có
    import numpy as np
    np.random.seed(42)
    skills = ['Python', 'Java', 'JavaScript', 'SQL', 'React', 'AWS', 'Docker', 'Excel', 'C++', 'Node.js',
              'Angular', 'PHP', 'Machine Learning', 'Data Analysis', 'Project Management']
    n = len(skills)
    demo_df = pd.DataFrame({
        'skill': skills,
        'job_count': np.random.randint(50, 5000, n),
        'avg_salary': np.random.uniform(10, 40, n),
        'avg_exp': np.random.uniform(0.5, 5, n),
        'big_city_ratio': np.random.uniform(0.3, 0.9, n),
        'skill_hot_score': np.random.uniform(0.2, 0.9, n),
        'predicted_hot_score': np.random.uniform(0.2, 0.9, n),
    })
    return demo_df, False

# ====================================================
# HÀM DỰ ĐOÁN
# ====================================================
def predict_job_attractiveness(salary, experience):
    """
    Dự đoán job có hấp dẫn không dựa trên logic đã định nghĩa:
    - Hot: Lương >= 15tr VÀ KN <= 2 năm
    - Hot: Lương >= 30tr
    """
    if salary >= 15 and experience <= 2:
        return True, "Lương tốt cho người ít kinh nghiệm"
    elif salary >= 30:
        return True, "Lương cao, ai cũng muốn"
    else:
        return False, "Chưa đủ điều kiện hấp dẫn"

# ====================================================
# GIAO DIỆN CHÍNH
# ====================================================
# st.title("🎯 Job Attractiveness Analyzer")
st.markdown("**Phân tích xu hướng việc làm và dự đoán mức độ hấp dẫn của các kỹ năng trên thị trường lao động**")

st.divider()

# Load dữ liệu
df, is_real_data = load_data()

if not is_real_data:
    st.warning("⚠️ Đang sử dụng **demo data** vì không kết nối được Cassandra. Hãy chạy Docker containers trước!")

# Tạo các cột tính toán nếu chưa có
if len(df) > 0:
    if 'salary_final' not in df.columns:
        df['salary_final'] = df['salary_avg'].fillna(
            (df['salary_min'].fillna(0) + df['salary_max'].fillna(0)) / 2
        ).fillna(0)
    if 'exp_final' not in df.columns:
        df['exp_final'] = df['exp_avg_year'].fillna(df['exp_min_year'].fillna(0))

# ====================================================
# TAB LAYOUT
# ====================================================
tab1, tab4, tab5, tab7 = st.tabs([
    "📊 Thống kê", "🎯 Phân cụm Job", 
    "🔥 Skill Hot", "🔮 Dự đoán Lương"
])

# ====================================================
# TAB 1: THỐNG KÊ TỔNG QUAN
# ====================================================
with tab1:
    st.header("📊 Thống kê tổng quan")
    
    if len(df) == 0:
        st.warning("⚠️ Không có dữ liệu. Hãy chạy ETL pipeline trước!")
    else:
        # Tính toán thống kê cơ bản
        total_jobs = len(df)
        
        # Hiển thị tổng số jobs
        st.metric(
            label="📋 Tổng số Job trong hệ thống",
            value=f"{total_jobs:,} jobs"
        )
        
        st.divider()
        
        # Thống kê theo thành phố
        st.subheader("📍 Phân bố theo Thành phố")
        city_counts = df['city'].value_counts().head(10)
        fig_city = px.bar(
            x=city_counts.index, 
            y=city_counts.values,
            labels={'x': 'Thành phố', 'y': 'Số lượng Job'},
            color=city_counts.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_city, use_container_width=True)





# ====================================================
# FOOTER
# ====================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    🎓 Đồ án Big Data - Job Attractiveness Analysis<br>
    Built with Streamlit, Spark, Cassandra, Kafka
</div>
""", unsafe_allow_html=True)

# ====================================================
# TAB 4: CLUSTERING
# ====================================================
with tab4:
    # st.header("🎯 Kết quả K-Means Clustering")
    
    df_clusters, is_real_clusters = load_clusters()
    
    if not is_real_clusters:
        st.warning("⚠️ Đang dùng demo data. Hãy chạy `train_kmeans.py` để có kết quả thực!")
    
    if len(df_clusters) == 0:
        st.error("Không có dữ liệu clustering!")
    else:
        # Sử dụng tên nhóm mặc định (Nhóm 0, Nhóm 1...) để đảm bảo tính chính xác
        # vì mỗi lần train lại model, thứ tự các cluster có thể thay đổi
        df_clusters['cluster_name'] = df_clusters['cluster'].apply(lambda x: f"Nhóm {x}")
        
        # Thống kê số jobs mỗi cluster
        st.subheader("📊 Phân bố Jobs theo Nhóm")
        cluster_counts = df_clusters['cluster'].value_counts().sort_index()
        cluster_names_list = [f"Nhóm {i}" for i in cluster_counts.index]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Biểu đồ cột
            fig_bar = px.bar(
                x=cluster_names_list,
                y=cluster_counts.values,
                labels={'x': 'Nhóm công việc', 'y': 'Số lượng Jobs'},
                color=cluster_names_list,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Biểu đồ tròn
            fig_pie = px.pie(
                names=cluster_names_list,
                values=cluster_counts.values,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Đặc điểm từng cluster
        st.subheader("📋 Đặc điểm trung bình từng Nhóm")
        cluster_stats = df_clusters.groupby('cluster').agg({
            'salary_final': 'mean',
            'exp_final': 'mean'
        }).round(2)
        cluster_stats['Tên nhóm'] = [f"Nhóm {i}" for i in cluster_stats.index]
        cluster_stats.columns = ['Lương TB (triệu)', 'KN TB (năm)', 'Tên nhóm']
        cluster_stats['Số Jobs'] = cluster_counts.values
        cluster_stats = cluster_stats[['Tên nhóm', 'Lương TB (triệu)', 'KN TB (năm)', 'Số Jobs']]
        st.dataframe(cluster_stats, use_container_width=True)
        
        # Scatter plot
        st.subheader("🔍 Phân bố Lương vs Kinh nghiệm theo Nhóm")
        fig_scatter = px.scatter(
            df_clusters,
            x='salary_final',
            y='exp_final',
            color='cluster_name',
            labels={'salary_final': 'Lương (triệu)', 'exp_final': 'Kinh nghiệm (năm)', 'cluster_name': 'Nhóm'},
            color_discrete_sequence=px.colors.qualitative.Set2,
            opacity=0.6
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Top jobs mỗi cluster
        st.subheader("📝 Mẫu Jobs trong mỗi Nhóm")
        
        # Dropdown với tên tiếng Việt
        col1, col2 = st.columns([2, 1])
        
        with col1:
            cluster_options = {f"Nhóm {c}": c for c in sorted(df_clusters['cluster'].unique())}
            selected_name = st.selectbox("Chọn Nhóm:", list(cluster_options.keys()))
            selected_cluster = cluster_options[selected_name]
        
        with col2:
            show_all = st.checkbox("Xem tất cả", value=False)
            if not show_all:
                num_rows = st.slider("Số jobs hiển thị:", min_value=10, max_value=1000, value=50, step=10)
            else:
                num_rows = len(cluster_data)
        
        # Lọc data theo cluster
        cluster_data = df_clusters[df_clusters['cluster'] == selected_cluster]
        
        # Hiển thị thông tin tổng quan
        showing_text = "tất cả" if show_all else f"{min(num_rows, len(cluster_data)):,}"
        st.info(f"📊 Đang hiển thị **{showing_text}** / **{len(cluster_data):,}** jobs trong nhóm **{selected_name}**")
        
        # Hiển thị bảng với số lượng đã chọn
        st.dataframe(
            cluster_data[['job_title', 'city', 'salary_final', 'exp_final', 'job_fields']].head(num_rows).rename(columns={
                'job_title': 'Tên công việc',
                'city': 'Thành phố',
                'salary_final': 'Lương (triệu)',
                'exp_final': 'Kinh nghiệm (năm)',
                'job_fields': 'Lĩnh vực'
            }),
            use_container_width=True,
            height=400
        )

# ====================================================
# TAB 5: SKILL HOT SCORE
# ====================================================
with tab5:
    st.header("🔥 Phân tích độ hấp dẫn của Kỹ năng")
    
    df_skills, is_real_skills = load_skill_scores()
    
    if not is_real_skills:
        st.warning("⚠️ Đang dùng demo data. Hãy chạy `train_gbt_cassandra.py` để có kết quả thực!")
    
    if len(df_skills) == 0:
        st.error("Không có dữ liệu skill!")
    else:
        # Sắp xếp theo hot score
        df_skills = df_skills.sort_values('predicted_hot_score', ascending=False)
        
        # Thống kê tổng quan
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Tổng số kỹ năng", f"{len(df_skills):,}")
        with col2:
            st.metric("🔥 Hot Score cao nhất", f"{df_skills['predicted_hot_score'].max():.2f}")
        with col3:
            top_skill = df_skills.iloc[0]['skill'] if len(df_skills) > 0 else "N/A"
            st.metric("🏆 Kỹ năng hot nhất", top_skill)
        
        st.divider()
        
        # Top kỹ năng hấp dẫn nhất
        st.subheader("🏆 Top 20 Kỹ năng hấp dẫn nhất")
        
        top_n = st.slider("Số kỹ năng hiển thị:", min_value=10, max_value=50, value=20)
        top_skills = df_skills.head(top_n)
        
        # Bar chart
        fig_bar = px.bar(
            top_skills,
            x='skill',
            y='predicted_hot_score',
            color='predicted_hot_score',
            color_continuous_scale='Reds',
            labels={'skill': 'Kỹ năng', 'predicted_hot_score': 'Hot Score'},
            title=f'Top {top_n} Kỹ năng Hot nhất'
        )
        fig_bar.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Chi tiết từng kỹ năng
        st.subheader("📋 Chi tiết Kỹ năng")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Bảng thống kê
            display_df = df_skills[['skill', 'predicted_hot_score', 'job_count', 'avg_salary', 'avg_exp']].copy()
            display_df.columns = ['Kỹ năng', 'Hot Score', 'Số Jobs', 'Lương TB', 'KN TB']
            display_df['Hot Score'] = display_df['Hot Score'].round(3)
            display_df['Lương TB'] = display_df['Lương TB'].round(1)
            display_df['KN TB'] = display_df['KN TB'].round(1)
            st.dataframe(display_df, use_container_width=True, height=400)
        
        with col2:
            # Scatter plot: Lương vs Số jobs, màu = Hot score
            fig_scatter = px.scatter(
                df_skills,
                x='avg_salary',
                y='job_count',
                size='predicted_hot_score',
                color='predicted_hot_score',
                hover_name='skill',
                color_continuous_scale='Reds',
                labels={
                    'avg_salary': 'Lương trung bình (triệu)',
                    'job_count': 'Số lượng Jobs',
                    'predicted_hot_score': 'Hot Score'
                },
                title='Phân bố kỹ năng theo Lương và Nhu cầu'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # So sánh kỹ năng
        st.subheader("⚖️ So sánh Kỹ năng")
        
        available_skills = df_skills['skill'].tolist()
        selected_skills = st.multiselect(
            "Chọn kỹ năng để so sánh:",
            available_skills,
            default=available_skills[:5] if len(available_skills) >= 5 else available_skills
        )
        
        if selected_skills:
            compare_df = df_skills[df_skills['skill'].isin(selected_skills)]
            
            # Radar chart
            fig_radar = go.Figure()
            
            for _, row in compare_df.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row['predicted_hot_score']*100, row['avg_salary'], row['job_count']/100, (1-row['avg_exp']/10)*10, row['big_city_ratio']*100],
                    theta=['Hot Score', 'Lương TB', 'Nhu cầu', 'Dễ vào nghề', 'TP Lớn'],
                    fill='toself',
                    name=row['skill']
                ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=True,
                title="So sánh các kỹ năng"
            )
            st.plotly_chart(fig_radar, use_container_width=True)



# ====================================================
# TAB 7: SALARY PREDICTION (Using pre-trained model)
# ====================================================
with tab7:
    st.header("🔮 Dự đoán Lương")
    # st.markdown("**Dự đoán mức lương dựa trên Random Forest model đã train**")
    
    st.divider()
    st.markdown("Nhập thông tin công việc để dự đoán mức lương ")
    
    # Session state for loaded model
    if 'salary_model_loaded' not in st.session_state:
        st.session_state.salary_model_loaded = None
    
    # Experience range mapping by position level
    EXP_RANGES = {
        "🎓 Thực tập sinh": (0, 0, 0),          # min, max, default
        "🌱 Fresher": (0, 1, 0),
        "📚 Junior": (1, 2, 1),
        "👤 Nhân viên/Chuyên viên": (2, 4, 3),
        "⭐ Senior": (4, 7, 5),
        "👥 Trưởng nhóm": (5, 10, 6),
        "👔 Quản lý/Trưởng phòng": (7, 20, 10),
    }
    
    # Initialize session state for position
    if 'selected_position' not in st.session_state:
        st.session_state.selected_position = "🎓 Thực tập sinh"
    
    st.markdown("##### 📍 Thông tin vị trí")
    pred_col1, pred_col2 = st.columns(2)
    
    with pred_col2:
        pred_position = st.selectbox("Cấp bậc", 
            list(EXP_RANGES.keys()),
            key="pred_position")
        st.session_state.selected_position = pred_position
    
    # Get experience range based on selected position
    exp_min, exp_max, exp_default = EXP_RANGES[pred_position]
    
    with pred_col1:
        pred_city = st.selectbox("Thành phố", 
            ["Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Khác"],
            key="pred_city")
        
        # Dynamic slider based on position
        if exp_min == exp_max:
            # Fixed value (e.g., Intern = 0)
            pred_experience = exp_min
            st.info(f"⏱️ Kinh nghiệm: **{exp_min} năm** (cố định cho {pred_position})")
        else:
            pred_experience = st.slider(
                f"Số năm kinh nghiệm ({exp_min}-{exp_max} năm)", 
                exp_min, exp_max, exp_default, 
                key=f"pred_exp_{pred_position}"  # Unique key per position
            )
    
    st.markdown("##### 💼 Lĩnh vực công việc")
    field_col1, field_col2, field_col3 = st.columns(3)
    
    with field_col1:
        is_it = st.checkbox("🖥️ IT/Phần mềm", key="is_it")
        is_finance = st.checkbox("💰 Tài chính/Ngân hàng", key="is_finance")
    with field_col2:
        is_sales = st.checkbox("📈 Sales/Marketing", key="is_sales")
        is_education = st.checkbox("📚 Giáo dục", key="is_education")
    with field_col3:
        is_engineering = st.checkbox("🔧 Kỹ thuật/Engineering", key="is_engineering")
    
    predict_submitted = st.button("🔮 Dự đoán Lương", type="primary")
    
    if predict_submitted:
        try:
            from pyspark.sql import SparkSession
            from pyspark.ml import PipelineModel
            from pyspark.sql.types import StructType, StructField, DoubleType
            
            with st.spinner("Đang load model và dự đoán..."):
                # Convert inputs to features
                is_hcm = 1.0 if "Hồ Chí Minh" in pred_city else 0.0
                is_hanoi = 1.0 if "Hà Nội" in pred_city else 0.0
                is_danang = 1.0 if "Đà Nẵng" in pred_city else 0.0
                
                # Position features - 7 levels (mutually exclusive)
                is_intern = 1.0 if "Thực tập" in pred_position else 0.0
                is_fresher = 1.0 if "Fresher" in pred_position else 0.0
                is_junior = 1.0 if "Junior" in pred_position else 0.0
                is_staff = 1.0 if "Nhân viên" in pred_position else 0.0
                is_senior = 1.0 if "Senior" in pred_position else 0.0
                is_team_lead = 1.0 if "Trưởng nhóm" in pred_position else 0.0
                is_manager = 1.0 if "Quản lý" in pred_position else 0.0
                
                exp_final = float(pred_experience)
                
                # Create Spark session
                spark = SparkSession.builder \
                    .appName("SalaryPrediction_UI") \
                    .config("spark.driver.memory", "1g") \
                    .getOrCreate()
                
                # Load model
                model_path = "/opt/spark/work-dir/models/salary_prediction_rf"
                
                try:
                    model = PipelineModel.load(model_path)
                    st.session_state.salary_model_loaded = model
                except Exception as e:
                    # Try alternative paths
                    import os
                    alt_paths = [
                        "./models/salary_prediction_rf",
                        os.path.join(os.path.dirname(__file__), "models", "salary_prediction_rf")
                    ]
                    model = None
                    for path in alt_paths:
                        if os.path.exists(path):
                            try:
                                model = PipelineModel.load(path)
                                break
                            except:
                                continue
                    
                    if model is None:
                        st.error(f"❌ Không thể load model! Vui lòng chạy train_salary_prediction.py trước.")
                        spark.stop()
                        st.stop()
                
                # Create input DataFrame with 16 features
                schema = StructType([
                    StructField("exp_final", DoubleType(), True),
                    StructField("is_hcm", DoubleType(), True),
                    StructField("is_hanoi", DoubleType(), True),
                    StructField("is_danang", DoubleType(), True),
                    StructField("is_it", DoubleType(), True),
                    StructField("is_sales", DoubleType(), True),
                    StructField("is_finance", DoubleType(), True),
                    StructField("is_education", DoubleType(), True),
                    StructField("is_engineering", DoubleType(), True),
                    StructField("is_intern", DoubleType(), True),
                    StructField("is_fresher", DoubleType(), True),
                    StructField("is_junior", DoubleType(), True),
                    StructField("is_staff", DoubleType(), True),
                    StructField("is_senior", DoubleType(), True),
                    StructField("is_team_lead", DoubleType(), True),
                    StructField("is_manager", DoubleType(), True),
                ])
                
                input_data = [(
                    exp_final,
                    is_hcm,
                    is_hanoi,
                    is_danang,
                    1.0 if is_it else 0.0,
                    1.0 if is_sales else 0.0,
                    1.0 if is_finance else 0.0,
                    1.0 if is_education else 0.0,
                    1.0 if is_engineering else 0.0,
                    is_intern,
                    is_fresher,
                    is_junior,
                    is_staff,
                    is_senior,
                    is_team_lead,
                    is_manager,
                )]
                
                input_df = spark.createDataFrame(input_data, schema)
                
                # Make prediction
                prediction_df = model.transform(input_df)
                raw_salary = prediction_df.select("prediction").collect()[0][0]
                
                # Điều chỉnh lương theo cấp bậc (do data thiếu cân bằng)
                # Hệ số điều chỉnh dựa trên mức lương thực tế thị trường VN
                SALARY_ADJUSTMENT = {
                    "🎓 Thực tập sinh": (2.0, 5.0),     # Floor, Ceiling (triệu)
                    "🌱 Fresher": (4.0, 10.0),
                    "📚 Junior": (7.0, 15.0),
                    "👤 Nhân viên/Chuyên viên": (10.0, 25.0),
                    "⭐ Senior": (18.0, 45.0),
                    "👥 Trưởng nhóm": (25.0, 60.0),
                    "👔 Quản lý/Trưởng phòng": (35.0, 100.0),
                }
                
                floor_salary, ceiling_salary = SALARY_ADJUSTMENT.get(pred_position, (5.0, 100.0))
                
                # Clamp predicted salary within reasonable range for position
                predicted_salary = max(floor_salary, min(raw_salary, ceiling_salary))
                
                # Bonus for IT field
                if is_it and predicted_salary < ceiling_salary:
                    predicted_salary = min(predicted_salary * 1.2, ceiling_salary)
                
                # Display result
                st.success(f"💰 **Lương dự đoán: {predicted_salary:.1f} Triệu VND/tháng**")
                
                # Debug info
                with st.expander("📊 Chi tiết tính toán"):
                    st.write(f"- Model raw prediction: **{raw_salary:.1f}** triệu")
                    st.write(f"- Điều chỉnh theo cấp bậc: **{floor_salary}-{ceiling_salary}** triệu")
                    st.write(f"- Kết quả sau điều chỉnh: **{predicted_salary:.1f}** triệu")
                
                # Feature summary
                features_selected = []
                if is_hcm: features_selected.append("📍 HCM")
                if is_hanoi: features_selected.append("📍 Hà Nội")
                if is_danang: features_selected.append("📍 Đà Nẵng")
                if is_it: features_selected.append("🖥️ IT")
                if is_sales: features_selected.append("📈 Sales")
                if is_finance: features_selected.append("💰 Tài chính")
                if is_education: features_selected.append("📚 Giáo dục")
                if is_engineering: features_selected.append("🔧 Kỹ thuật")
                
                st.info(f"""
                **Thông tin đầu vào:**
                - ⏱️ Kinh nghiệm: **{pred_experience} năm**
                - 📊 Cấp bậc: **{pred_position}**
                - 🏷️ Lĩnh vực: {', '.join(features_selected) if features_selected else 'Không xác định'}
                """)
                
                spark.stop()
                
        except Exception as e:
            st.error(f"Lỗi dự đoán: {str(e)}")
            import traceback
            with st.expander("Chi tiết lỗi"):
                st.code(traceback.format_exc())

