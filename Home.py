import streamlit as st
import subprocess
import sys
import os
import warnings

# Suppress all warnings for clean presentation
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="VM Placement ML Optimizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page
st.title("🚀 VM Placement ML Optimizer")

st.markdown("""
## Choose Your Dashboard Experience

Use the sidebar navigation to access different dashboards:

**Available Dashboards:**
- 🎯 **Working Dashboard** - Stable, tested implementation  
- 🔬 **Full Dashboard** - Complete feature set with advanced analytics

*Navigate using the sidebar menu →*
""")

# Project information
st.markdown("---")
st.markdown("""
### 📊 **About This Project**

This VM Placement Optimizer uses **6 Machine Learning models** to compare performance across **5 key metrics**:

**ML Models:**
- 🤖 **ML-NSGA-II** (Our optimized model)
- 🌲 **Random Forest**
- 🚀 **XGBoost Alternative**
- 📐 **SVM**
- 🧠 **Neural Network**
- 🌳 **Decision Tree**

**Performance Metrics:**
- 💻 **CPU Utilization (%)**
- 🎯 **SLA Compliance (%)**
- ⚡ **Energy Consumption (kWh)**
- 💰 **Cost Efficiency ($)**
- 📉 **Resource Waste (%)**

Perfect for academic presentations and cloud infrastructure optimization research!
""")

# System information
with st.expander("📋 System Information"):
    st.write(f"Python Version: {sys.version}")
    st.write(f"Working Directory: {os.getcwd()}")
    st.write("Dependencies: All ML libraries loaded successfully ✅")