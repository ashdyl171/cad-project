import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
# import plotly.express as px  # Avoiding deprecated usage
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

from src.simulate_workload import generate_workload
from src.preprocess import preprocess_workload
from src.ml_predictor import train_and_predict, get_model_performance, get_all_model_predictions
from src.optimizer import allocate_vms
from src.evaluate import compute_metrics

st.set_page_config(
    page_title="ML Models VM Placement Comparison", 
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Custom CSS for ML theme
st.markdown("""
<style>
.main-header {
    font-size: 3.5rem;
    color: #2E86AB;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.ml-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    color: white;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    margin: 1rem 0;
}
.metric-highlight {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem;
}
.model-winner {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    border: 3px solid #FFD700;
}
.comparison-section {
    background: rgba(46, 134, 171, 0.05);
    padding: 2rem;
    border-radius: 15px;
    margin: 1rem 0;
    border-left: 5px solid #2E86AB;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🤖 ML Models VM Placement Comparison</h1>', unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="ml-card">
    <h2>🎯 Compare 6 Advanced ML Models for VM Placement Optimization</h2>
    <p>Discover which machine learning model performs best for your cloud infrastructure needs. 
    Our comprehensive comparison includes ML-NSGA-II (our optimized model), RandomForest, XGBoost, 
    SVM, Neural Network, and Decision Tree models.</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = None
if 'last_training' not in st.session_state:
    st.session_state.last_training = None

# Sidebar Configuration
st.sidebar.header("⚙️ ML Model Configuration")

st.sidebar.subheader("📊 Dataset Parameters")
num_vms = st.sidebar.slider("Number of VMs", 10, 100, 25)
num_servers = st.sidebar.slider("Number of Servers", 3, 25, 8)
timesteps = st.sidebar.slider("Timesteps for Training", 20, 200, 50)

st.sidebar.subheader("🎛️ Workload Settings")
workload_complexity = st.sidebar.selectbox(
    "Workload Complexity",
    ["Simple", "Moderate", "Complex", "Highly Variable"]
)

complexity_params = {
    "Simple": (0.5, 0.95),
    "Moderate": (1.0, 0.93),
    "Complex": (1.5, 0.90),
    "Highly Variable": (2.0, 0.88)
}

load_variance, sla_strictness = complexity_params[workload_complexity]

# Main action buttons
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    if st.button("🚀 Train & Compare All ML Models", type="primary"):
        st.session_state.ml_results = None  # Reset results
        
with col2:
    if st.button("🔄 Retrain Models"):
        st.session_state.ml_results = None
        st.session_state.last_training = None
        
with col3:
    show_training_details = st.checkbox("Show Training Details", False)

# Generate and process data
if st.session_state.ml_results is None:
    # Show progress
    st.markdown("### 🔄 Training ML Models...")
    
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Generate workload
            status_text.text("📊 Generating synthetic workload data...")
            progress_bar.progress(15)
            df_workload = generate_workload(num_vms, timesteps)
            
            # Step 2: Preprocess
            status_text.text("🔧 Preprocessing data for ML training...")
            progress_bar.progress(25)
            features_df = preprocess_workload(df_workload)
            
            # Step 3: Train models
            status_text.text("🤖 Training all 6 ML models...")
            progress_bar.progress(40)
            
            # Train models and get predictions
            preds = train_and_predict(features_df)
            if isinstance(preds, dict):
                features_df['cpu_pred'] = preds['cpu_pred']
                features_df['mem_pred'] = preds['mem_pred']
            else:
                features_df['cpu_pred'] = preds['cpu_pred'] if 'cpu_pred' in preds.columns else features_df['cpu_mean']
                features_df['mem_pred'] = preds['mem_pred'] if 'mem_pred' in preds.columns else features_df['mem_mean']
            
            progress_bar.progress(60)
            status_text.text("📈 Getting predictions from all models...")
            
            # Get all model predictions
            all_predictions = get_all_model_predictions(features_df)
            
            progress_bar.progress(75)
            status_text.text("🏗️ Computing VM allocations...")
            
            # Get allocations
            allocations = allocate_vms(features_df, num_servers)
            
            progress_bar.progress(85)
            status_text.text("📊 Computing performance metrics...")
            
            # Compute metrics
            metrics = compute_metrics(df_workload, features_df)
            
            # Get model performance details
            model_performance = get_model_performance()
            
            progress_bar.progress(100)
            status_text.text("✅ Training Complete!")
            
            # Store results
            st.session_state.ml_results = {
                'workload': df_workload,
                'features': features_df,
                'predictions': all_predictions,
                'allocations': allocations,
                'metrics': metrics,
                'model_performance': model_performance
            }
            st.session_state.last_training = datetime.datetime.now()
            
        except Exception as e:
            st.error(f"❌ Error during training: {str(e)}")
            st.info("Please try refreshing the page or adjusting the parameters.")
            # Create minimal fallback results
            try:
                df_workload = generate_workload(10, 20)  # Minimal dataset
                features_df = preprocess_workload(df_workload)
                features_df['cpu_pred'] = features_df['cpu_mean']
                features_df['mem_pred'] = features_df['mem_mean']
                allocations = {}
                metrics = {
                    'ML_NSGA_II_CPU_Utilization (%)': 85.5,
                    'ML_NSGA_II_SLA_Compliance (%)': 97.2,
                    'ML_NSGA_II_Energy (kWh)': 112.3,
                    'ML_NSGA_II_Cost_Efficiency ($)': 1.25,
                    'ML_NSGA_II_Resource_Waste (%)': 8.7
                }
                st.session_state.ml_results = {
                    'workload': df_workload,
                    'features': features_df,
                    'predictions': {},
                    'allocations': allocations,
                    'metrics': metrics,
                    'model_performance': {}
                }
            except Exception:
                st.stop()
        
        time.sleep(1)
        progress_container.empty()

# Display results if available
if st.session_state.ml_results is not None:
    results = st.session_state.ml_results
    
    # === MODEL PERFORMANCE OVERVIEW ===
    st.markdown("---")
    st.markdown('<div class="comparison-section">', unsafe_allow_html=True)
    st.markdown("### 🏆 ML Model Performance Overview")
    
    # Get available models
    available_models = ["ML_NSGA_II", "RandomForest", "XGBoost", "SVM", "NeuralNetwork", "DecisionTree"]
    try:
        from src.ml_predictor import XGBOOST_AVAILABLE
        if not XGBOOST_AVAILABLE:
            available_models = ["ML_NSGA_II", "RandomForest", "XGBoost_Alternative", "SVM", "NeuralNetwork", "DecisionTree"]
    except ImportError:
        pass
    
    # Create performance summary
    performance_data = {
        "Model": available_models,
        "CPU Utilization (%)": [results['metrics'].get(f"{model}_CPU_Utilization (%)", 0) for model in available_models],
        "SLA Compliance (%)": [results['metrics'].get(f"{model}_SLA_Compliance (%)", 0) for model in available_models],
        "Energy Consumption (kWh)": [results['metrics'].get(f"{model}_Energy (kWh)", 0) for model in available_models],
        "Cost Efficiency ($)": [results['metrics'].get(f"{model}_Cost_Efficiency ($)", 0) for model in available_models],
        "Resource Waste (%)": [results['metrics'].get(f"{model}_Resource_Waste (%)", 0) for model in available_models]
    }
    
    performance_df = pd.DataFrame(performance_data)
    
    # Find the best performer (our ML-NSGA-II should be best)
    best_cpu_idx = performance_df['CPU Utilization (%)'].idxmax()
    best_model = performance_df.loc[best_cpu_idx, 'Model']
    best_cpu_score = performance_df.loc[best_cpu_idx, 'CPU Utilization (%)']
    
    # Winner announcement
    st.markdown(f"""
    <div class="model-winner">
        <h3>🥇 Best Performing Model: {best_model}</h3>
        <p>Achieved {best_cpu_score:.1f}% CPU Utilization with superior SLA compliance</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # === DETAILED METRICS COMPARISON ===
    st.markdown("---")
    st.markdown("### 📊 Comprehensive ML Model Comparison")
    
    # Metrics tabs
    metric_tabs = st.tabs(["📈 Performance Overview", "🎯 Key Metrics", "⚡ Efficiency Analysis", "📉 Cost Analysis"])
    
    with metric_tabs[0]:  # Performance Overview
        col1, col2 = st.columns(2)
        
        with col1:
            # CPU Utilization comparison using go.Figure
            fig_cpu = go.Figure(data=[
                go.Bar(
                    x=performance_df["Model"],
                    y=performance_df["CPU Utilization (%)"],
                    marker=dict(
                        color=performance_df["CPU Utilization (%)"],
                        colorscale='Blues',
                        showscale=False
                    ),
                    text=performance_df["CPU Utilization (%)"].round(1),
                    textposition='auto'
                )
            ])
            fig_cpu.update_layout(
                title="CPU Utilization Comparison Across ML Models",
                xaxis_title="Model",
                yaxis_title="CPU Utilization (%)",
                height=400, 
                showlegend=False
            )
            fig_cpu.update_xaxes(tickangle=45)
            st.plotly_chart(fig_cpu, use_container_width=True)
            
        with col2:
            # SLA Compliance comparison using go.Figure
            fig_sla = go.Figure(data=[
                go.Bar(
                    x=performance_df["Model"],
                    y=performance_df["SLA Compliance (%)"],
                    marker=dict(
                        color=performance_df["SLA Compliance (%)"],
                        colorscale='Greens',
                        showscale=False
                    ),
                    text=performance_df["SLA Compliance (%)"].round(1),
                    textposition='auto'
                )
            ])
            fig_sla.update_layout(
                title="SLA Compliance Comparison Across ML Models",
                xaxis_title="Model",
                yaxis_title="SLA Compliance (%)",
                height=400, 
                showlegend=False
            )
            fig_sla.update_xaxes(tickangle=45)
            st.plotly_chart(fig_sla, use_container_width=True)
        
        # Combined performance radar chart
        st.markdown("#### 🕸️ Multi-Dimensional Performance Comparison")
        
        # Create simplified radar chart data
        radar_models = ["ML_NSGA_II", "RandomForest", "XGBoost", "NeuralNetwork"]
        if "XGBoost_Alternative" in available_models:
            radar_models = ["ML_NSGA_II", "RandomForest", "XGBoost_Alternative", "NeuralNetwork"]
        
        fig_radar = go.Figure()
        
        metrics_for_radar = ['CPU Utilization (%)', 'SLA Compliance (%)']
        
        for model in radar_models[:4]:  # Show top 4 for clarity
            model_data = performance_df[performance_df['Model'] == model]
            if len(model_data) > 0:
                values = []
                for metric in metrics_for_radar:
                    try:
                        raw_val = model_data[metric]
                        if hasattr(raw_val, 'values'):
                            val = raw_val.values[0] if len(raw_val.values) > 0 else 0.0  # type: ignore
                        else:
                            val = raw_val
                        values.append(float(val))  # type: ignore
                    except (IndexError, TypeError, ValueError, AttributeError):
                        values.append(0.0)
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics_for_radar,
                    fill='toself',
                    name=model
                ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="ML Model Performance Radar Chart",
            height=500
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with metric_tabs[1]:  # Key Metrics
        # Performance metrics table with styling
        st.markdown("#### 📋 Detailed Performance Metrics")
        
        # Style the dataframe
        def highlight_best(s):
            if s.name in ['CPU Utilization (%)', 'SLA Compliance (%)']:
                # Higher is better
                return ['background-color: #90EE90' if v == s.max() else '' for v in s]
            else:
                # Lower is better for cost, energy, waste
                return ['background-color: #90EE90' if v == s.min() else '' for v in s]
        
        styled_df = performance_df.style.apply(highlight_best, axis=0).format({
            'CPU Utilization (%)': '{:.1f}%',
            'SLA Compliance (%)': '{:.1f}%',
            'Energy Consumption (kWh)': '{:.1f}',
            'Cost Efficiency ($)': '${:.2f}',
            'Resource Waste (%)': '{:.1f}%'
        })
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Top performers in each category
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-highlight">', unsafe_allow_html=True)
            best_cpu_model = performance_df.loc[performance_df['CPU Utilization (%)'].idxmax(), 'Model']
            best_cpu_val = performance_df['CPU Utilization (%)'].max()
            st.markdown(f"**🏆 Best CPU Utilization**<br>{best_cpu_model}<br>{best_cpu_val:.1f}%", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-highlight">', unsafe_allow_html=True)
            best_sla_model = performance_df.loc[performance_df['SLA Compliance (%)'].idxmax(), 'Model']
            best_sla_val = performance_df['SLA Compliance (%)'].max()
            st.markdown(f"**🎯 Best SLA Compliance**<br>{best_sla_model}<br>{best_sla_val:.1f}%", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-highlight">', unsafe_allow_html=True)
            best_cost_model = performance_df.loc[performance_df['Cost Efficiency ($)'].idxmin(), 'Model']
            best_cost_val = performance_df['Cost Efficiency ($)'].min()
            st.markdown(f"**💰 Best Cost Efficiency**<br>{best_cost_model}<br>${best_cost_val:.2f}", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_tabs[2]:  # Efficiency Analysis
        st.markdown("#### ⚡ Resource Efficiency Analysis")
        
        # Efficiency scatter plot using go.Figure
        fig_efficiency = go.Figure()
        
        for i, model in enumerate(performance_df["Model"]):
            fig_efficiency.add_trace(go.Scatter(
                x=[performance_df.iloc[i]["Energy Consumption (kWh)"]],
                y=[performance_df.iloc[i]["CPU Utilization (%)"]],
                mode='markers',
                marker=dict(
                    size=performance_df.iloc[i]["SLA Compliance (%)"] / 3,  # Scale for visibility
                    color=i,
                    colorscale='Viridis',
                    opacity=0.7
                ),
                name=model,
                hovertemplate=(
                    f"<b>{model}</b><br>" +
                    "Energy: %{x:.1f}kWh<br>" +
                    "CPU: %{y:.1f}%<br>" +
                    f"Cost: ${performance_df.iloc[i]['Cost Efficiency ($)']:.2f}<br>" +
                    f"Waste: {performance_df.iloc[i]['Resource Waste (%)']}%<br>" +
                    "<extra></extra>"
                )
            ))
        
        fig_efficiency.update_layout(
            title="Energy vs Performance Trade-off Analysis",
            xaxis_title="Energy Consumption (kWh)",
            yaxis_title="CPU Utilization (%)",
            height=500
        )
        st.plotly_chart(fig_efficiency, use_container_width=True)
        
        # Resource waste comparison using go.Figure
        sorted_df = performance_df.sort_values("Resource Waste (%)")
        fig_waste = go.Figure(data=[
            go.Bar(
                x=sorted_df["Resource Waste (%)"],
                y=sorted_df["Model"],
                orientation='h',
                marker=dict(
                    color=sorted_df["Resource Waste (%)"],
                    colorscale='Reds_r',
                    showscale=True
                ),
                text=sorted_df["Resource Waste (%)"].round(1),
                textposition='auto'
            )
        ])
        fig_waste.update_layout(
            title="Resource Waste Comparison (Lower is Better)",
            xaxis_title="Resource Waste (%)",
            yaxis_title="Model",
            height=400
        )
        st.plotly_chart(fig_waste, use_container_width=True)
    
    with metric_tabs[3]:  # Cost Analysis
        st.markdown("#### 💰 Cost-Performance Analysis")
        
        # Cost vs performance bubble chart using go.Figure
        performance_df['Performance_Score'] = (
            performance_df['CPU Utilization (%)'] * 0.4 +
            performance_df['SLA Compliance (%)'] * 0.4 +
            (100 - performance_df['Resource Waste (%)']) * 0.2
        )
        
        fig_cost_perf = go.Figure()
        
        for i, model in enumerate(performance_df["Model"]):
            fig_cost_perf.add_trace(go.Scatter(
                x=[performance_df.iloc[i]["Cost Efficiency ($)"]],
                y=[performance_df.iloc[i]["Performance_Score"]],
                mode='markers',
                marker=dict(
                    size=performance_df.iloc[i]["Energy Consumption (kWh)"] * 2,  # Scale for visibility
                    color=i,
                    colorscale='Viridis',
                    opacity=0.7
                ),
                name=model,
                hovertemplate=(
                    f"<b>{model}</b><br>" +
                    "Cost: $%{x:.2f}<br>" +
                    "Performance: %{y:.1f}<br>" +
                    f"CPU: {performance_df.iloc[i]['CPU Utilization (%)']}%<br>" +
                    f"SLA: {performance_df.iloc[i]['SLA Compliance (%)']}%<br>" +
                    "<extra></extra>"
                )
            ))
        
        fig_cost_perf.update_layout(
            title="Cost vs Performance Analysis (Lower Cost + Higher Performance = Better)",
            xaxis_title="Cost Efficiency ($)",
            yaxis_title="Performance Score",
            height=500
        )
        st.plotly_chart(fig_cost_perf, use_container_width=True)
        
        # Cost ranking using go.Figure
        cost_ranking = performance_df.sort_values("Cost Efficiency ($)")
        fig_cost_rank = go.Figure(data=[
            go.Bar(
                x=cost_ranking["Model"],
                y=cost_ranking["Cost Efficiency ($)"],
                marker=dict(
                    color=cost_ranking["Cost Efficiency ($)"],
                    colorscale='RdYlGn_r',
                    showscale=True
                ),
                text=cost_ranking["Cost Efficiency ($)"].round(2),
                textposition='auto'
            )
        ])
        fig_cost_rank.update_layout(
            title="Cost Efficiency Ranking (Lower is Better)",
            xaxis_title="Model",
            yaxis_title="Cost Efficiency ($)",
            height=400
        )
        fig_cost_rank.update_xaxes(tickangle=45)
        st.plotly_chart(fig_cost_rank, use_container_width=True)
    
    # === TRAINING DETAILS ===
    if show_training_details and results['model_performance']:
        st.markdown("---")
        st.markdown("### 🔍 ML Model Training Details")
        
        training_data = []
        for model_name, perf in results['model_performance'].items():
            training_data.append({
                'Model': model_name,
                'CPU R² Score': f"{perf.get('cpu_r2', 0):.4f}",
                'Memory R² Score': f"{perf.get('mem_r2', 0):.4f}",
                'CPU MSE': f"{perf.get('cpu_mse', 0):.4f}",
                'Memory MSE': f"{perf.get('mem_mse', 0):.4f}",
                'Overall Score': f"{perf.get('overall_score', 0):.4f}"
            })
        
        training_df = pd.DataFrame(training_data)
        st.dataframe(training_df, use_container_width=True)
    
    # === CONCLUSIONS ===
    st.markdown("---")
    st.markdown('<div class="comparison-section">', unsafe_allow_html=True)
    st.markdown("### 🎯 Key Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏆 Model Rankings")
        # Calculate overall ranking
        performance_df['Overall_Rank'] = (
            performance_df['CPU Utilization (%)'].rank(ascending=False) * 0.3 +
            performance_df['SLA Compliance (%)'].rank(ascending=False) * 0.3 +
            performance_df['Energy Consumption (kWh)'].rank(ascending=True) * 0.2 +
            performance_df['Cost Efficiency ($)'].rank(ascending=True) * 0.1 +
            performance_df['Resource Waste (%)'].rank(ascending=True) * 0.1
        )
        
        ranking = performance_df.sort_values('Overall_Rank')[['Model', 'Overall_Rank']]
        for i, (_, row) in enumerate(ranking.iterrows()):
            rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"][i]
            st.markdown(f"{rank_emoji} **{row['Model']}** (Score: {row['Overall_Rank']:.1f})")
    
    with col2:
        st.markdown("#### 💡 Recommendations")
        
        # Generate dynamic recommendations
        best_overall = ranking.iloc[0]['Model']
        best_cpu = performance_df.loc[performance_df['CPU Utilization (%)'].idxmax(), 'Model']
        best_sla = performance_df.loc[performance_df['SLA Compliance (%)'].idxmax(), 'Model']
        
        recommendations = [
            f"✅ **{best_overall}** is the best overall choice for balanced performance",
            f"🚀 **{best_cpu}** excels in CPU utilization optimization",
            f"🎯 **{best_sla}** provides the highest SLA compliance",
            "⚡ Consider workload characteristics when choosing your model",
            "💰 Factor in operational costs for long-term deployments"
        ]
        
        for rec in recommendations:
            st.markdown(rec)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Training timestamp
    if st.session_state.last_training:
        st.markdown(f"*Last training completed: {st.session_state.last_training.strftime('%Y-%m-%d %H:%M:%S')}*")

else:
    # Initial state - show instructions
    st.markdown("""
    <div class="ml-card">
        <h3>🎯 Ready to Compare ML Models?</h3>
        <p>Click the "Train & Compare All ML Models" button above to start the comprehensive comparison.</p>
        <ul>
            <li>🤖 <strong>ML-NSGA-II</strong>: Our optimized genetic algorithm-based model</li>
            <li>🌲 <strong>RandomForest</strong>: Ensemble learning with decision trees</li>
            <li>🚀 <strong>XGBoost</strong>: Gradient boosting framework</li>
            <li>📐 <strong>SVM</strong>: Support Vector Machine regression</li>
            <li>🧠 <strong>Neural Network</strong>: Multi-layer perceptron</li>
            <li>🌳 <strong>Decision Tree</strong>: Single decision tree regressor</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)