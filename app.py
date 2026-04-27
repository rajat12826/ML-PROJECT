import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from src.data_manager import DataManager
from src.eda_utils import *
from src.recommender import Recommender
from src.analyzer import Analyzer

# Configuration
warnings.filterwarnings("ignore", category=UserWarning)
st.set_page_config(page_title="Campus Placement Predictor", page_icon="🎓", layout="wide")

# Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; background-color: #050506; }
    .blob-container { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; overflow: hidden; pointer-events: none; }
    .blob { position: absolute; filter: blur(150px); opacity: 0.15; border-radius: 50%; animation: float 20s infinite ease-in-out; }
    .blob-1 { width: 800px; height: 800px; background: #5E6AD2; top: -20%; left: 30%; }
    .blob-2 { width: 600px; height: 600px; background: #6872D9; bottom: -10%; right: 10%; animation-duration: 25s; }
    @keyframes float { 0%, 100% { transform: translate(0, 0); } 50% { transform: translate(-30px, 40px); } }
    h1, h2, h3 { background: linear-gradient(to bottom, #FFFFFF 0%, #FFFFFF 70%, rgba(255,255,255,0.7) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.02em; font-weight: 600; }
    [data-testid="stMetric"], .st-emotion-cache-12w0qpk { background: linear-gradient(to bottom, rgba(255,255,255,0.08), rgba(255,255,255,0.02)) !important; border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 16px !important; padding: 24px !important; box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important; }
    .prediction-card-v2 { padding: 40px; border-radius: 20px; background: radial-gradient(circle at top left, rgba(94,106,210,0.15), transparent), linear-gradient(to bottom, rgba(255,255,255,0.08), rgba(255,255,255,0.03)); border: 1px solid rgba(94,106,210,0.3); text-align: center; margin: 20px 0; box-shadow: 0 10px 40px rgba(0,0,0,0.5), 0 0 80px rgba(94,106,210,0.1); }
</style>
<div class="blob-container"><div class="blob blob-1"></div><div class="blob blob-2"></div></div>
""", unsafe_allow_html=True)

# Navigation
st.sidebar.title("🎓 Career Guide")
domain_mode = st.sidebar.radio("⚡ Intelligence Mode", ["Engineering", "MBA"], index=1)

# Model Selection Dropdown
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Brain Selection")
model_options = ["Gradient Boosting", "Random Forest", "Logistic Regression", "Extra Trees"]
selected_model_type = st.sidebar.selectbox("Choose AI Model", model_options, index=0)

page = st.sidebar.selectbox("Navigate", ["Dashboard (EDA)", "Prediction & Recommendation", "Model Forensics"])

if st.sidebar.button("🔄 Reset Cache & Reload"):
    st.cache_resource.clear()
    st.rerun()

# Smart Defaults Optimized for Rewritten Datasets
SMART_DEFAULTS = {
    'ssc_p': 85.0, 'hsc_p': 82.0, 'degree_p': 78.0, 'etest_p': 85.0, 'mba_p': 75.0,
    '10th marks': 85.0, '12th marks': 82.0, 'Cgpa': 8.5, 'Technical Score': 80.0,
    'Backlogs': 0.0, 'Projects': 3.0, 'workex': 'Yes', 'Internships(Y/N)': 'Yes'
}

# Loader
@st.cache_resource
def load_resources(mode, model_type):
    dm = DataManager(mode=mode)
    df = dm.load_data()
    model_path = f'models/{mode.lower()}'
    model_slug = model_type.lower().replace(" ", "_")
    
    try:
        placement_model = joblib.load(f'{model_path}/placement_model_{model_slug}.pkl')
    except:
        # Fallback to default if slug not found
        placement_model = joblib.load(f'{model_path}/placement_model.pkl')
        
    try:
        salary_model = joblib.load(f'{model_path}/salary_model_{model_slug}.pkl')
    except:
        try: salary_model = joblib.load(f'{model_path}/salary_model.pkl')
        except: salary_model = None
        
    features = joblib.load(f'{model_path}/features.pkl')
    encoders = joblib.load(f'{model_path}/label_encoders.pkl')
    
    try: 
        metrics = joblib.load(f'{model_path}/metrics_{model_slug}.pkl')
    except:
        try: metrics = joblib.load(f'{model_path}/metrics.pkl')
        except: metrics = None
    
    try: scaler = joblib.load(f'{model_path}/scaler.pkl')
    except: scaler = None
        
    recommender = Recommender()
    analyzer = Analyzer(df)
    return df, placement_model, salary_model, features, encoders, recommender, analyzer, metrics, scaler

df, placement_model, salary_model, features, encoders, recommender, analyzer, metrics, scaler = load_resources(domain_mode, selected_model_type)

def show_results(input_data, mode='MBA'):
    # AI DataFrame Integration
    encoded_input = []
    for f in features:
        val = input_data.get(f)
        if f in encoders:
            try: val_enc = encoders[f].transform([str(val).strip().title()])[0]
            except: val_enc = 0
            encoded_input.append(val_enc)
        else: encoded_input.append(float(val) if val is not None else 0.0)
    
    X_df = pd.DataFrame([encoded_input], columns=features)
    
    # Scaling if scaler exists
    if scaler:
        X_scaled = scaler.transform(X_df)
        prob = placement_model.predict_proba(X_scaled)[0][1]
    else:
        prob = placement_model.predict_proba(X_df)[0][1]
    
    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 1])
    with res_col1:
        prob_color = "#5E6AD2" if prob > 0.7 else "#f5a623" if prob > 0.4 else "#ff4b4b"
        st.markdown(f'<div class="prediction-card-v2" style="border-color: {prob_color}44;"><p style="color: #8A8F98; margin-bottom: 0;">AI Placement Confidence</p><h1 style="font-size: 3.5rem; margin: 10px 0; color: {prob_color};">{prob*100:.1f}%</h1></div>', unsafe_allow_html=True)
        
        # Logic Sensitivity Check
        critical_fail = False
        if mode == 'MBA' and (input_data.get('etest_p', 100) < 40 or input_data.get('degree_p', 100) < 45): critical_fail = True
        if mode == 'Engineering' and (input_data.get('Cgpa', 10) < 6.5 or input_data.get('Technical Score', 100) < 45 or input_data.get('Backlogs', 0) > 1): critical_fail = True
        
        if prob >= 0.5:
            if critical_fail:
                st.warning("⚠️ **Logical Sensitivity Alert**: High probability found but your profile hits critical 'Hard-Floor' failures (Backlogs/Low E-Test). Recruiter rejection is highly likely.")
            else:
                st.success("🎉 High Success Probability!")
            if salary_model:
                X_sal = scaler.transform(X_df) if scaler else X_df
                salary_pred = salary_model.predict(X_sal)[0]
                st.metric("Estimated Salary/CTC", f"₹ {salary_pred:,.2f}")
        else:
            st.error("📉 Low success probability. Focusing on core skills and clearing backlogs is essential.")

    with res_col2:
        st.subheader("🎯 Career Roadmaps")
        stream = next((v for k,v in input_data.items() if 'Stream' in k or 'specialisation' in k.lower()), "General")
        grade = next((v for k,v in input_data.items() if 'Cgpa' in k or 'ssc_p' in k.lower()), 0)
        recs = recommender.recommend_path({'Stream': stream, 'Cgpa': grade})
        for r in recs:
            with st.expander(f"✨ {r}", expanded=True): st.write("**Skills:**", ", ".join(recommender.get_requirements(r)))

if page == "Dashboard (EDA)":
    st.title("📊 Placement Insights")
    set_style()
    c1, c2, c3 = st.columns([2, 1, 1])
    impact_col = 'specialisation' if 'specialisation' in df.columns else 'Stream'
    with c1: st.pyplot(plot_correlation_matrix(df))
    with c2: st.pyplot(plot_placement_distribution(df))
    with c3: st.pyplot(plot_categorical_impact(df, impact_col))

elif page == "Prediction & Recommendation":
    st.title("🎯 Prediction Engine")
    st.info(f"AI Model optimized for real-world logic. Testing with {len(features)} parameters.")
    with st.form("dynamic_form"):
        cols = st.columns(3)
        input_data = {}
        for i, feat in enumerate(features):
            with cols[i % 3]:
                if feat in encoders:
                    options = df[feat].unique().tolist() if feat in df.columns else ["Unknown"]
                    def_val = SMART_DEFAULTS.get(feat)
                    idx = options.index(def_val) if def_val in options else options.index('Yes') if 'Yes' in options else 0
                    input_data[feat] = st.selectbox(feat, options, index=idx)
                else:
                    min_v, max_v = (0.0, 10.0) if 'Cgpa' in feat or 'CGPA' in feat else (0.0, 100.0)
                    default_v = float(SMART_DEFAULTS.get(feat, min_v))
                    input_data[feat] = st.number_input(feat, min_v, max_v, value=default_v)
        submit = st.form_submit_button("🚀 Simulate Outcome")
    if submit: show_results(input_data, mode=domain_mode)

elif page == "Model Forensics":
    st.title("🧬 Logic-Driven AI Status")
    if metrics:
        st.write(f"Architecture: {selected_model_type}")
        st.write(f"Test Accuracy: {metrics['classification']['accuracy']*100:.2f}% | Test F1: {metrics['classification']['f1_score']:.3f}")
        
        st.subheader(f"📊 Global Confusion Matrix (Total: {metrics['classification'].get('sample_count', '800')} samples)")
        # Use full matrix if available, otherwise fallback
        cm = metrics['classification'].get('confusion_matrix_full', metrics['classification']['confusion_matrix'])
        fig, ax = plt.subplots(); sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax); st.pyplot(fig)
        
        st.subheader("💡 Sensitivity Analysis")
        st.bar_chart(metrics['classification']['feature_importances'])
    else: st.warning("Metrics missing.")
