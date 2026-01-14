# Breast Cancer Classifier

A real-time ECG-based breast cancer classification application with enhanced performance, modern UI, and comprehensive data export capabilities.

## 🚀 Features

### Core Functionality
- **Real-time ECG Analysis**: Stream and analyze ECG data from Shimmer devices
- **File-based Analysis**: Load and process existing ECG recordings from CSV files
- **HRV Feature Extraction**: Compute heart rate variability features (LF, HF, Total Power)
- **Machine Learning Prediction**: Classify breast cancer risk using trained models
- **5-Minute Data Collection**: Real-time analysis requires 5 minutes of continuous ECG data for reliable results

### Enhanced Features
- **Smooth Real-time Processing**: Optimized DSP moved to background threads
- **Modern UI**: Clean, responsive interface with scrollable panels
- **Performance Optimized**: Reusable Matplotlib artists, cached filters, throttled updates
- **Comprehensive Export**: Export results, RR intervals, and plot images every 30 seconds
- **Robust Error Handling**: User-friendly error messages and graceful recovery

## 📋 Prerequisites

### Python Packages
```bash
pip install numpy pandas scipy matplotlib scikit-learn joblib
pip install pyserial pyshimmer
```

### Hardware Requirements
- Shimmer ECG device (tested with Shimmer3 ECG unit)
- USB connection for real-time streaming
- CSV files from Shimmer devices for file-based analysis

## 🔧 Installation

1. Clone or download the repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
   *(Create a `requirements.txt` file with the packages listed above)*

3. Ensure you have a trained model file (`best_xgb_pipeline.pkl`) in the same directory or load one via the UI

## 🎮 Usage

### Starting the Application
```bash
python BC_classifier.py
```

### Real-time ECG Analysis
1. **Connect Device**:
   - Click "Refresh" to scan for COM ports
   - Select your Shimmer device from the dropdown
   - Click "Connect"

2. **Start Streaming**:
   - Click "Start Streaming" to begin data collection
   - The application will collect 5 minutes of ECG data
   - Real-time visualization updates every 30 seconds

3. **View Results**:
   - After 5 minutes, the final analysis will be computed
   - View prediction, confidence, and HRV features
   - Export results for detailed review

### File-based Analysis
1. **Load CSV File**:
   - Click "Load & Analyze CSV File"
   - Select a Shimmer CSV recording
   - The application automatically detects timestamp and ECG columns

2. **Analyze Data**:
   - Automatic R-peak detection
   - HRV feature computation
   - Machine learning prediction

### Exporting Results
Click "Export Results" to create a comprehensive analysis package including:
- Main analysis results CSV
- RR interval analysis (30-second segments)
- Power spectral density plot
- ECG segment plots
- Summary report

## 📊 Output Files

When exporting results, the application creates a timestamped directory containing:

```
export_YYYYMMDD_HHMMSS/
├── analysis_results.csv          # Main results and features
├── rr_interval_analysis.csv      # Detailed RR interval data (30s segments)
├── power_spectral_density.png    # PSD visualization
├── analysis_summary.txt          # Comprehensive summary report
└── segment_plots/               # ECG plots for each 30-second segment
    ├── segment_01.png
    ├── segment_02.png
    └── ...
```

## ⚙️ Configuration

### Key Settings (in code)
- **Sampling Rate**: Default 128 Hz
- **LF Band**: 0.04-0.15 Hz
- **HF Band**: 0.15-0.40 Hz
- **Collection Duration**: 300 seconds (5 minutes) for real-time
- **Update Interval**: 200 ms for UI updates

### Model Requirements
The application expects a trained scikit-learn model with:
- `predict()` or `predict_proba()` method
- Feature scaling included (as `scaler_` attribute)
- Expects features in order: [LF_power, HF_power, Total_Power]

## 🐛 Troubleshooting

### Common Issues

1. **No COM Ports Found**:
   - Ensure Shimmer device is connected and powered on
   - Check Device Manager for COM port assignments
   - Try different USB ports

2. **Connection Timeouts**:
   - Restart the Shimmer device
   - Reconnect USB cable
   - Close other applications using the COM port

3. **Missing Model File**:
   - Place `best_xgb_pipeline.pkl` in the application directory
   - Or use "Load Model" button to select a model file

4. **Import Errors**:
   - Ensure all required packages are installed
   - For PyShimmer issues, check device compatibility

### Logs
- Application logs are saved to `app_enhanced.log`
- Detailed error information is available in the log file
- Enable debug logging by modifying the logging configuration in `setup_logging()`

## 🏗️ Architecture

### Key Components
1. **EnhancedApp**: Main application class with modern UI
2. **EnhancedShimmerRealtimeReader**: Real-time ECG data acquisition
3. **HRV Feature Extraction**: Signal processing and feature computation
4. **Model Predictor**: Machine learning classification
5. **Data Exporter**: Comprehensive result export system

### Performance Optimizations
- Background DSP processing to prevent UI blocking
- Cached bandpass filter coefficients
- Reusable Matplotlib artists (no per-frame recreation)
- Throttled PSD/HRV updates (~1.5 second intervals)

## 📈 HRV Features

The application computes and uses the following Heart Rate Variability features:

1. **LF Power (ms²)**: Low-frequency power (0.04-0.15 Hz)
2. **HF Power (ms²)**: High-frequency power (0.15-0.40 Hz)
3. **Total Power (ms²)**: Total power in the 0.0-0.4 Hz range

These features are used as input to the machine learning model for breast cancer classification.

## 🔒 Privacy & Data

- All ECG data is processed locally
- No data is transmitted externally
- Export functionality creates local files only
- Consider patient privacy when handling exported data

## 📝 License

This application is for research and educational purposes. Ensure proper ethical approvals for medical device usage.

## 🤝 Support

For issues, feature requests, or contributions:
1. Check the troubleshooting section
2. Review application logs
3. Ensure all dependencies are correctly installed

## 🎯 Quick Start Checklist

- [ ] Install required Python packages
- [ ] Connect Shimmer device (for real-time)
- [ ] Have CSV files ready (for file analysis)
- [ ] Place model file in application directory
- [ ] Run application: `python BC_classifier.py`
- [ ] For real-time: Connect → Start Streaming → Wait 5 minutes → View Results

---

*Note: This application is intended for research purposes and should be used under appropriate medical supervision. Always validate results with clinical assessments.*