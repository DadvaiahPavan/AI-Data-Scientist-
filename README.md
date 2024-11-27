# AI Data Analysis Tool

A powerful data analysis tool that leverages AI and machine learning to provide comprehensive insights into your datasets. This application combines the power of scikit-learn, pandas, and Streamlit to offer an intuitive interface for data analysis, visualization, and machine learning tasks.

## Getting Started

### Clone the Repository
```bash
git https://github.com/DadvaiahPavan/AI-Data-Scientist.git
cd AI_Data_Tool
```

## Features

### 1. Data Processing
- Support for multiple file formats (CSV, Excel, etc.)
- Automatic data type detection and preprocessing
- Missing value handling and visualization
- Memory-efficient processing for large datasets

### 2. AI Analysis Capabilities
- **Classification Analysis**: Using Random Forest Classifier for categorical predictions
- **Regression Analysis**: Random Forest Regressor for numerical predictions
- **Anomaly Detection**: Isolation Forest implementation for outlier detection
- **Text Analysis**: Sentiment analysis and text processing using TextBlob
- **Feature Importance**: Visualization of most influential features in your dataset

### 3. Visualization
- Interactive plots using Plotly
- Missing value analysis
- Feature correlation heatmaps
- Distribution analysis
- Custom visualization options

### 4. User Interface
- Modern, responsive Streamlit interface
- Intuitive navigation
- Real-time analysis updates
- Interactive parameter tuning
- Progress tracking for long operations

## Project Structure

```
AI_Data_Tool/
├── ai_analyzer.py         # Core AI analysis implementation
├── streamlit_app.py      # Main application entry point
├── requirements.txt      # Project dependencies
├── setup_dirs.py        # Directory setup utility
├── src/
│   └── ui/
│       └── streamlit_app.py  # UI implementation
├── data/                 # Data directory
└── logs/                 # Application logs
```

## Setup Instructions

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run streamlit_app.py
```

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt:
  - streamlit==1.24.0
  - pandas==2.0.3
  - numpy==1.25.0
  - plotly==5.15.0
  - scikit-learn==1.3.0
  - textblob==0.17.1
  - xmltodict==0.13.0
  - openpyxl==3.1.2
  - lxml==4.9.3
  - torch>=2.0.0
  - transformers>=4.30.0

## Usage

1. Launch the application using `streamlit run streamlit_app.py`
2. Upload your dataset through the web interface
3. Select the type of analysis you want to perform:
   - Data Overview
   - Classification Analysis
   - Regression Analysis
   - Anomaly Detection
   - Text Analysis
4. Configure analysis parameters if needed
5. View and interact with the results and visualizations

## Features in Detail

### Data Overview
- Basic statistics
- Data type information
- Missing value analysis
- Correlation analysis
- Distribution plots

### Classification Analysis
- Automatic feature preprocessing
- Model training and evaluation
- Performance metrics
- Confusion matrix visualization
- Feature importance analysis

### Regression Analysis
- Numerical feature preprocessing
- Model training and validation
- Error metrics (MSE, RMSE, etc.)
- Prediction visualization
- Feature importance ranking

### Anomaly Detection
- Outlier identification
- Anomaly scoring
- Visualization of anomalous data points
- Configurable contamination factor

### Text Analysis
- Sentiment analysis
- Text preprocessing
- Word frequency analysis
- Sentiment distribution visualization

## Performance Considerations

- The tool is optimized for memory efficiency
- Large datasets are processed in chunks
- Parallel processing is utilized where possible
- Model parameters are tuned for balance of accuracy and performance

## Contributing

Feel free to submit issues and enhancement requests!
