import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from src.data_manager import DataManager
from src.eda_utils import *
from src.recommender import Recommender
from src.resume_parser import ResumeParser
from src.analyzer import Analyzer

# Set Page Config
st.set_page_config(page_title="Campus Placement Predictor", page_icon="🎓", layout="wide")

# Custom CSS for Premium Linear Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #050506;
    }

    /* Background Blobs */
    .blob-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        overflow: hidden;
        pointer-events: none;
    }
    .blob {
        position: absolute;
        filter: blur(150px);
        opacity: 0.15;
        border-radius: 50%;
        animation: float 20s infinite ease-in-out;
    }
    .blob-1 {
        width: 800px;
        height: 800px;
        background: #5E6AD2;
        top: -20%;
        left: 30%;
    }
    .blob-2 {
        width: 600px;
        height: 600px;
        background: #6872D9;
        bottom: -10%;
        right: 10%;
        animation-duration: 25s;
    }
    @keyframes float {
        0%, 100% { transform: translate(0, 0); }
        50% { transform: translate(-30px, 40px); }
    }

    /* Typography */
    h1, h2, h3 {
        background: linear-gradient(to bottom, #FFFFFF 0%, #FFFFFF 70%, rgba(255,255,255,0.7) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        font-weight: 600;
    }
    
    /* Cards */
    [data-testid="stMetric"], .st-emotion-cache-12w0qpk, .prediction-card-new {
        background: linear-gradient(to bottom, rgba(255,255,255,0.08), rgba(255,255,255,0.02)) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
    }

    .prediction-card-v2 {
        padding: 40px;
        border-radius: 20px;
        background: radial-gradient(circle at top left, rgba(94,106,210,0.15), transparent), 
                    linear-gradient(to bottom, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
        border: 1px solid rgba(94,106,210,0.3);
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5), 0 0 80px rgba(94,106,210,0.1);
    }

    /* Buttons */
    .stButton>button {
        background: #5E6AD2 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 0 0 1px rgba(94,106,210,0.5), 0 4px 12px rgba(94,106,210,0.3) !important;
    }
    .stButton>button:hover {
        background: #6872D9 !important;
        transform: translateY(-2px);
        box-shadow: 0 0 0 1px rgba(94,106,210,0.6), 0 8px 20px rgba(94,106,210,0.4) !important;
    }

    /* Sidebar */
    /* Streamlit Alert Boxes */
    [data-testid="stNotification"] {
        background-color: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        color: #EDEDEF !important;
        border-radius: 12px !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #050506;
    }
    ::-webkit-scrollbar-thumb {
        background: #1a1a1e;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #5E6AD2;
    }
</style>

<div class="blob-container">
    <div class="blob blob-1"></div>
    <div class="blob blob-2"></div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🎓 Career Guide")
st.sidebar.markdown("---")

# Domain Toggle
domain_mode = st.sidebar.radio("⚡ Intelligence Mode", ["Engineering", "MBA"], index=1)
page = st.sidebar.selectbox("Navigate", ["Dashboard (EDA)", "Prediction & Recommendation"])

# Load Data and Models
@st.cache_resource
def load_resources(mode):
    dm = DataManager(mode=mode)
    df = dm.load_data()
    
    model_path = f'models/{mode.lower()}'
    placement_model = joblib.load(f'{model_path}/placement_model.pkl')
    try:
        salary_model = joblib.load(f'{model_path}/salary_model.pkl')
    except:
        salary_model = None
        
    features = joblib.load(f'{model_path}/features.pkl')
    encoders = joblib.load(f'{model_path}/label_encoders.pkl')
    recommender = Recommender()
    parser = ResumeParser()
    analyzer = Analyzer(df)
    return df, placement_model, salary_model, features, encoders, recommender, parser, analyzer

df, placement_model, salary_model, features, encoders, recommender, parser, analyzer = load_resources(domain_mode)

# Initialize session state for form fields
def init_session_state(mode):
    if mode == 'MBA':
        defaults = {
            'gender': 'M', 'ssc_p': 70.0, 'ssc_b': 'Central', 
            'hsc_p': 70.0, 'hsc_b': 'Central', 'hsc_s': 'Commerce', 
            'degree_p': 70.0, 'degree_t': 'Comm&Mgmt', 
            'workex': 'No', 'etest_p': 70.0, 
            'specialisation': 'Mkt&Fin', 'mba_p': 70.0
        }
    else:
        defaults = {
            'Gender': 'Male', '10th marks': 80.0, '10th board': 'CBSE',
            '12th marks': 80.0, '12th board': 'CBSE', 'Stream': 'Computer Science and Engineering',
            'Cgpa': 8.0, 'Internships(Y/N)': 'No', 'Training(Y/N)': 'No',
            'Backlog in 5th sem': 'No', 'Innovative Project(Y/N)': 'No',
            'Communication level': 3, 'Technical Course(Y/N)': 'No'
        }
    for key, val in defaults.items():
        # Reset if mode changed or not present
        if key not in st.session_state or 'current_mode' not in st.session_state or st.session_state['current_mode'] != mode:
            st.session_state[key] = val
    st.session_state['current_mode'] = mode

init_session_state(domain_mode)

# Helper function to update state from resume
def update_state_from_resume(extracted_features, mode):
    # Mapping extracted keys to mode-specific keys
    mapping = {
        'MBA': {
            'ssc_p': 'ssc_p', 'hsc_p': 'hsc_p', 'degree_p': 'degree_p', 'mba_p': 'mba_p',
            'gender': 'gender', 'workex': 'workex', 'specialisation': 'specialisation', 'degree_t': 'degree_t'
        },
        'Engineering': {
            'ssc_p': '10th marks', 'hsc_p': '12th marks', 'degree_p': 'Cgpa',
            'gender': 'Gender', 'workex': 'Internships(Y/N)', 'specialisation': 'Stream', 'degree_t': 'Stream'
        }
    }
    
    current_map = mapping.get(mode, {})
    for key, val in extracted_features.items():
        if key in current_map:
            # Special handling for CGPA (divide by 10 if looks like %)
            final_val = val
            target_key = current_map[key]
            if target_key == 'Cgpa' and val > 10:
                final_val = val / 10.0
            st.session_state[target_key] = final_val

# Domain Navigation logic already handled above
if page == "Dashboard (EDA)":
    st.title("📊 Placement Insights Dashboard")
    st.markdown("Explore the factors that influence campus placements.")
    
    set_style()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 📊 Performance Insights")
        fig5 = plot_correlation_matrix(df)
        st.pyplot(fig5)

    with col2:
        st.markdown("### 🧬 Placement Ratio")
        fig1 = plot_placement_distribution(df)
        st.pyplot(fig1)
        
    with col3:
        st.markdown("### 💼 Job Specialisation")
        cat_col = 'specialisation' if domain_mode == 'MBA' else 'Stream'
        fig4 = plot_categorical_impact(df, cat_col)
        st.pyplot(fig4)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 💰 Expected Package")
        if domain_mode == 'MBA':
            fig3 = plot_salary_distribution(df)
            st.pyplot(fig3)
        else:
            st.info("Salary data is not available for the Engineering dataset.")
    with col2:
        st.markdown("### ⚧ Gender Diversity")
        fig2 = plot_categorical_impact(df, 'Gender' if domain_mode == 'Engineering' else 'gender')
        st.pyplot(fig2)

elif page == "Prediction & Recommendation":
    st.title("🎯 Campus Placement Predictor")
    st.markdown("Fill in your details to get your placement probability and career roadmap.")

    # Resume Upload Section
    resume_file = st.file_uploader("📂 Upload your Resume (PDF) for specialized advice", type=["pdf"])
    resume_skills = []
    resume_domains = []
    
    if resume_file:
        with st.spinner("Analyzing resume..."):
            resume_text = parser.extract_text_from_pdf(resume_file)
            resume_skills, resume_domains = parser.identify_skills(resume_text)
            
            # Autofill features
            extracted_features = parser.extract_features(resume_text)
            if extracted_features:
                update_state_from_resume(extracted_features, domain_mode)
                st.toast("✅ Form autofilled from resume!", icon='🚀')
            
            if resume_skills:
                st.success(f"Detected Skills: {', '.join(resume_skills)}")
            else:
                st.info("No specific technical skills detected. The system will use academic data.")

    # Form inputs linked to session state
    with st.form("prediction_form"):
        if domain_mode == 'MBA':
            col1, col2, col3 = st.columns(3)
            with col1:
                gender = st.selectbox("Gender", ["M", "F"], index=["M", "F"].index(st.session_state['gender']))
                ssc_p = st.number_input("SSC Percentage (10th)", 0.0, 100.0, st.session_state['ssc_p'])
                ssc_b = st.selectbox("SSC Board", ["Central", "Others"], index=["Central", "Others"].index(st.session_state['ssc_b']))
                workex = st.selectbox("Work Experience", ["No", "Yes"], index=["No", "Yes"].index(st.session_state['workex']))
            with col2:
                hsc_p = st.number_input("HSC Percentage (12th)", 0.0, 100.0, st.session_state['hsc_p'])
                hsc_b = st.selectbox("HSC Board", ["Central", "Others"], index=["Central", "Others"].index(st.session_state['hsc_b']))
                hsc_s = st.selectbox("HSC Stream", ["Arts", "Commerce", "Science"], index=["Arts", "Commerce", "Science"].index(st.session_state['hsc_s']))
                specialisation = st.selectbox("MBA Specialisation", ["Mkt&HR", "Mkt&Fin"], index=["Mkt&HR", "Mkt&Fin"].index(st.session_state['specialisation']))
            with col3:
                degree_p = st.number_input("Degree Percentage", 0.0, 100.0, st.session_state['degree_p'])
                degree_t = st.selectbox("Degree type", ["Comm&Mgmt", "Sci&Tech", "Others"], index=["Comm&Mgmt", "Sci&Tech", "Others"].index(st.session_state['degree_t']))
                etest_p = st.number_input("Employability test %", 0.0, 100.0, st.session_state['etest_p'])
                mba_p = st.number_input("MBA Percentage", 0.0, 100.0, st.session_state['mba_p'])
            
            input_data = {
                'gender': gender, 'ssc_p': ssc_p, 'ssc_b': ssc_b, 'hsc_p': hsc_p,
                'hsc_b': hsc_b, 'hsc_s': hsc_s, 'degree_p': degree_p, 'degree_t': degree_t,
                'workex': workex, 'etest_p': etest_p, 'specialisation': specialisation, 'mba_p': mba_p
            }
        else:
            # Engineering Form
            col1, col2, col3 = st.columns(3)
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(st.session_state['Gender']))
                m10 = st.number_input("10th Marks (%)", 0.0, 100.0, st.session_state['10th marks'])
                b10 = st.selectbox("10th Board", df['10th board'].unique().tolist(), index=df['10th board'].unique().tolist().index(st.session_state['10th board']))
                intern = st.selectbox("Internships", ["No", "Yes"], index=["No", "Yes"].index(st.session_state['Internships(Y/N)']))
                backlog = st.selectbox("Backlog in 5th sem", ["No", "Yes"], index=["No", "Yes"].index(st.session_state['Backlog in 5th sem']))
            with col2:
                m12 = st.number_input("12th Marks (%)", 0.0, 100.0, st.session_state['12th marks'])
                b12 = st.selectbox("12th Board", df['12th board'].unique().tolist(), index=df['12th board'].unique().tolist().index(st.session_state['12th board']))
                stream = st.selectbox("Stream", df['Stream'].unique().tolist(), index=df['Stream'].unique().tolist().index(st.session_state['Stream']))
                training = st.selectbox("Training", ["No", "Yes"], index=["No", "Yes"].index(st.session_state['Training(Y/N)']))
            with col3:
                cgpa = st.number_input("CGPA", 0.0, 10.0, st.session_state['Cgpa'])
                project = st.selectbox("Innovative Project", ["No", "Yes"], index=["No", "Yes"].index(st.session_state['Innovative Project(Y/N)']))
                comm = st.slider("Communication Level", 1, 5, st.session_state['Communication level'])
                tech = st.selectbox("Technical Course", ["No", "Yes"], index=["No", "Yes"].index(st.session_state['Technical Course(Y/N)']))
            
            input_data = {
                'Gender': gender, '10th board': b10, '10th marks': m10, '12th board': b12,
                '12th marks': m12, 'Stream': stream, 'Cgpa': cgpa, 'Internships(Y/N)': intern,
                'Training(Y/N)': training, 'Backlog in 5th sem': backlog, 
                'Innovative Project(Y/N)': project, 'Communication level': comm, 'Technical Course(Y/N)': tech
            }

        submit = st.form_submit_button("Predict My Future")

    if submit:
        show_results(input_data, resume_domains, resume_skills)

# Helper function to show results & and interactive optimization
def show_results(input_data, resume_domains, resume_skills):
    # Encoding
    encoded_input = []
    for feature in features:
        val = input_data[feature]
        if feature in encoders:
            val = encoders[feature].transform([val])[0]
        encoded_input.append(val)
    
    # Prediction
    prob = placement_model.predict_proba([encoded_input])[0][1]
    place_status = placement_model.predict([encoded_input])[0]
    
    st.markdown("---")
    
    # Result Layout
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.markdown(f"""
        <div class="prediction-card-v2">
            <p style="color: #8A8F98; margin-bottom: 0;">Placement Probability</p>
            <h1 style="font-size: 3.5rem; margin: 10px 0;">{prob*100:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)
        
        if place_status == 1:
            st.success("🎉 You are likely to be PLACED!")
            salary_pred = salary_model.predict([encoded_input])[0]
            st.metric("Estimated Annual Package", f"₹ {salary_pred:,.0f}")
            
            # Company Tier
            tier = analyzer.get_company_tier(salary_pred)
            st.info(f"🏢 **Target Tier:** {tier}")
        else:
            st.warning("⚠️ You might need to work more on your skills.")
            st.metric("Estimated Package", "₹ 0")

    with res_col2:
        st.subheader("🎯 Career Roadmaps")
        recs = recommender.recommend_path(input_data, resume_domains=resume_domains)
        for r in recs:
            st.info(f"✨ {r}")
        
        # Skill Gap Analysis
        if recs and resume_skills:
            st.markdown("#### 🛠️ Skill Gap Analysis")
            matched, missing = recommender.get_skill_analysis(recs[0], resume_skills)
            
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.write("**Matched:**")
                for m in matched: st.write(f"✅ {m}")
            with m_col2:
                st.write("**Missing:**")
                for mi in missing: st.write(f"❌ {mi}")

    # Benchmarking Section
    st.markdown("---")
    st.subheader("📈 Peer Comparison Benchmarks")
    benchmarks = analyzer.get_benchmarks(input_data)
    
    b_cols = st.columns(len(benchmarks))
    for i, (metric, data) in enumerate(benchmarks.items()):
        with b_cols[i]:
            label = metric.replace('_p', '').upper()
            st.metric(label, f"{data['student']}%", f"{data['diff']:.1f}% vs Placed Avg")
            st.caption(f"Percentile: {data['percentile']:.1f}%")

    # Optimization Section
    st.markdown("---")
    with st.expander("🚀 Success Path Optimizer (What-If Analysis)"):
        st.write("Simulate how improving your scores affects your placement probability.")
        opt_col1, opt_col2 = st.columns([2, 1])
        
        with opt_col1:
            m_to_opt = st.select_slider("Select Metric to Optimize", options=['ssc_p', 'hsc_p', 'degree_p', 'mba_p', 'etest_p'])
            boost_val = st.slider(f"Improve {m_to_opt} by:", 0, 20, 5)
            
            # Recalculate
            opt_data = input_data.copy()
            opt_data[m_to_opt] = min(100, opt_data[m_to_opt] + boost_val)
            
            # Encoded opt
            opt_encoded = []
            for f in features:
                v = opt_data[f]
                if f in encoders: v = encoders[f].transform([v])[0]
                opt_encoded.append(v)
            
            new_prob = placement_model.predict_proba([opt_encoded])[0][1]
            gain = (new_prob - prob) * 100
            
        with opt_col2:
            st.metric("New Probability", f"{new_prob*100:.1f}%", f"+{gain:.1f}% Gain")
            st.write(f"Improving your **{m_to_opt}** by **{boost_val}%** significantly boosts your profile!")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Developed for ML Lab Project 🚀")
