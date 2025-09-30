# VM Placement ML Optimizer ✨

**Status: ✅ FULLY WORKING - All errors fixed!**

🚀 A complete Machine Learning-powered VM placement optimization system with interactive dashboards.

A comprehensive machine learning dashboard for comparing VM placement optimization algorithms across multiple performance metrics.

## 🎯 Features

- **6 ML Models**: ML-NSGA-II, Random Forest, XGBoost Alternative, SVM, Neural Network, Decision Tree
- **5 Key Metrics**: CPU Utilization, SLA Compliance, Energy Consumption, Cost Efficiency, Resource Waste
- **Interactive Visualizations**: Plotly charts, radar plots, performance comparisons
- **Real-time Analysis**: Dynamic data generation and model training
- **Faculty-Ready**: Designed for academic presentations and research

## 🏃‍♂️ Quick Start

### Local Deployment

```bash
# Clone the repository
git clone <your-repo-url>
cd "vm placement optimizer"

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run Home.py
```

### Streamlit Cloud Deployment

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy using `Home.py` as the main file

## 📊 Usage

1. **Home Page**: Choose between Working or Full Feature dashboard
2. **Configure Parameters**: Adjust VMs, servers, and workload complexity
3. **Train Models**: Click "Train & Compare All ML Models"
4. **Analyze Results**: View comprehensive performance comparisons
5. **Present Findings**: Use faculty-ready visualizations and insights

## 🛠️ Technical Details

- **Framework**: Streamlit 1.28+
- **ML Libraries**: scikit-learn, pandas, numpy
- **Visualization**: Plotly, matplotlib, seaborn
- **Python Version**: 3.12.6

## 🎓 Academic Use

Perfect for:
- Cloud computing research
- Resource optimization studies
- ML algorithm comparisons
- Faculty presentations
- Graduate research projects

## 📁 Project Structure

```
vm placement optimizer/
├── Home.py                 # Main entry point
├── app_ml_working.py       # Stable dashboard
├── app_ml_comparison.py    # Full feature dashboard
├── src/
│   ├── ml_predictor.py     # ML model collection
│   ├── optimizer.py        # Placement algorithms
│   ├── evaluate.py         # Performance metrics
│   └── ...
├── requirements.txt        # Dependencies
└── .streamlit/
    └── config.toml         # Streamlit configuration
```

## 🚀 Deployment Options

### Option 1: Streamlit Community Cloud
- Free hosting
- Automatic updates from GitHub
- Perfect for demos and sharing

### Option 2: Local Development
- Full control over environment
- Ideal for development and testing
- Quick iteration cycles

### Option 3: Cloud Platforms
- Deploy to Heroku, AWS, or Google Cloud
- Production-ready scaling
- Custom domain support

## 🔧 Recent Fixes & Improvements

### Fixed Issues:
- ✅ **Plotly Chart Display Errors**: Fixed all `width="stretch"` parameters to `use_container_width=True`
- ✅ **DataFrame Styling Issues**: Added error handling for pandas styling operations
- ✅ **Graph Visualization Errors**: Fixed data type handling in 3D plots, radar charts, and complex visualizations
- ✅ **Import Errors**: Resolved all module import issues and dependencies
- ✅ **Session State Errors**: Added comprehensive error handling and fallback mechanisms
- ✅ **Syntax Errors**: Fixed all Python syntax and indentation issues

### New Features:
- 🎆 **Enhanced Error Handling**: Robust fallback mechanisms prevent crashes
- 📊 **Improved Visualizations**: 15+ interactive charts with real-time data
- ⚙️ **Better Performance**: Optimized ML pipeline for faster training
- 📈 **Comprehensive Metrics**: 5 key performance indicators across 6 ML models

### Test Suite:
- 🧪 **Full Test Coverage**: Run `python3 test_app.py` to verify functionality
- 🔍 **Automated Validation**: Tests imports, pipeline, dataframes, and visualizations
- ✅ **Pre-deployment Checks**: Ensures error-free operation

## 🚑 Troubleshooting

### Common Issues:

**1. Import Errors**
```bash
# Install missing dependencies
pip install -r requirements.txt

# If issues persist, create fresh virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Streamlit Won't Start**
```bash
# Check streamlit version
streamlit --version

# If outdated, upgrade
pip install --upgrade streamlit

# Run with specific port
streamlit run Home.py --server.port 8501
```

**3. Chart Display Issues**
```bash
# Verify plotly installation
python3 -c "import plotly; print(plotly.__version__)"

# Update if needed
pip install --upgrade plotly
```

**4. Memory Issues (Large Datasets)**
- Reduce number of VMs in sidebar (start with 10-25)
- Decrease timesteps (start with 20-50)
- Use "Simple" workload complexity initially

### Verification Steps:
1. Run the test suite: `python3 test_app.py`
2. Check all tests pass before using the app
3. Start with default parameters for first run
4. Gradually increase complexity as needed

## 🌐 Browser Compatibility

**Recommended Browsers:**
- ✅ Chrome 90+ (Best performance)
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

**Features Requiring Modern Browser:**
- Interactive 3D visualizations
- Real-time chart updates
- Advanced plotly interactions

---

✨ **Ready to use! All issues resolved.** ✨

Made with ❤️ for VM placement optimization research
