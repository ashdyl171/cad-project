#!/usr/bin/env python3

"""
Test script to verify the Streamlit VM Placement Optimizer app works correctly
"""

import os
import sys
import pandas as pd
import numpy as np

# Add the project root to Python path
sys.path.append(os.getcwd())

def test_imports():
    """Test all module imports"""
    print("🔍 Testing imports...")
    
    try:
        from src.simulate_workload import generate_workload
        from src.preprocess import preprocess_workload
        from src.ml_predictor import train_and_predict
        from src.optimizer import allocate_vms
        from src.evaluate import compute_metrics
        print("✅ All core module imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def test_basic_pipeline():
    """Test the basic ML pipeline"""
    print("\n🔄 Testing basic ML pipeline...")
    
    try:
        from src.simulate_workload import generate_workload
        from src.preprocess import preprocess_workload
        from src.ml_predictor import train_and_predict
        from src.evaluate import compute_metrics
        
        # Generate small test dataset
        df_workload = generate_workload(5, 10)
        print(f"✅ Generated workload: {len(df_workload)} rows")
        
        # Preprocess
        features_df = preprocess_workload(df_workload)
        print(f"✅ Preprocessed features: {len(features_df)} rows")
        
        # Train and predict
        preds = train_and_predict(features_df)
        print(f"✅ Training completed, prediction type: {type(preds)}")
        
        # Add predictions to features
        if isinstance(preds, dict):
            features_df['cpu_pred'] = preds['cpu_pred']
            features_df['mem_pred'] = preds['mem_pred']
        else:
            features_df['cpu_pred'] = preds['cpu_pred'] if 'cpu_pred' in preds.columns else features_df['cpu_mean']
            features_df['mem_pred'] = preds['mem_pred'] if 'mem_pred' in preds.columns else features_df['mem_mean']
        
        # Compute metrics
        metrics = compute_metrics(df_workload, features_df)
        print(f"✅ Metrics computed: {len(metrics)} metrics")
        
        return True
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_dataframe():
    """Test performance dataframe creation (similar to what dashboard does)"""
    print("\n📊 Testing performance dataframe creation...")
    
    try:
        # Simulate metrics data
        available_models = ["ML_NSGA_II", "RandomForest", "XGBoost_Alternative", "SVM", "NeuralNetwork", "DecisionTree"]
        model_display_names = ["ML-NSGA-II (Ours)", "Random Forest", "XGBoost Alternative", "SVM", "Neural Network", "Decision Tree"]
        
        # Create mock metrics
        metrics = {}
        for model in available_models:
            metrics[f"{model}_CPU_Utilization (%)"] = np.random.uniform(70, 95)
            metrics[f"{model}_SLA_Compliance (%)"] = np.random.uniform(85, 99)
            metrics[f"{model}_Energy (kWh)"] = np.random.uniform(100, 150)
            metrics[f"{model}_Cost_Efficiency ($)"] = np.random.uniform(1.0, 3.0)
            metrics[f"{model}_Resource_Waste (%)"] = np.random.uniform(5, 25)
        
        # Create performance dataframe
        performance_data = {
            "Model": model_display_names,
            "CPU Utilization (%)": [metrics.get(f"{model}_CPU_Utilization (%)", 0) for model in available_models],
            "SLA Compliance (%)": [metrics.get(f"{model}_SLA_Compliance (%)", 0) for model in available_models],
            "Energy Consumption (kWh)": [metrics.get(f"{model}_Energy (kWh)", 0) for model in available_models],
            "Cost Efficiency ($)": [metrics.get(f"{model}_Cost_Efficiency ($)", 0) for model in available_models],
            "Resource Waste (%)": [metrics.get(f"{model}_Resource_Waste (%)", 0) for model in available_models]
        }
        
        performance_df = pd.DataFrame(performance_data)
        print(f"✅ Performance dataframe created: {performance_df.shape}")
        
        # Test key operations
        best_cpu_idx = performance_df['CPU Utilization (%)'].idxmax()
        best_model = performance_df.loc[best_cpu_idx, 'Model']
        print(f"✅ Best model identified: {best_model}")
        
        return True
    except Exception as e:
        print(f"❌ DataFrame error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_components():
    """Test that streamlit components can be imported"""
    print("\n🎨 Testing Streamlit components...")
    
    try:
        import streamlit as st
        import plotly.graph_objects as go
        print("✅ Streamlit and Plotly imports successful")
        
        # Test basic plotly figure creation
        fig = go.Figure()
        fig.add_trace(go.Bar(x=['A', 'B', 'C'], y=[1, 2, 3]))
        print("✅ Plotly figure creation successful")
        
        return True
    except Exception as e:
        print(f"❌ Streamlit/Plotly error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting VM Placement Optimizer Tests")
    print("=" * 50)
    
    success = True
    
    success &= test_imports()
    success &= test_basic_pipeline()
    success &= test_performance_dataframe()
    success &= test_streamlit_components()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! The app should work correctly.")
        print("\n🌟 To run the app:")
        print("   streamlit run Home.py")
    else:
        print("❌ Some tests failed. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()