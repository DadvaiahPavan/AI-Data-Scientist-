import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.analyzers.ai_analyzer import AIAnalyzer
import io
import time

# Custom CSS for styling
def local_css():
    st.markdown("""
        <style>
        /* Reset default styles */
        .css-1d391kg, .css-1p1nwyz {
            padding: 0;
        }

        /* Global Styles */
        .main {
            padding: 1rem;
        }

        /* Sidebar Styles */
        [data-testid="stSidebar"] {
            background-color: #1a1a1a !important;
            width: 300px !important;
        }

        [data-testid="stSidebar"] .block-container {
            padding: 2rem 1rem !important;
        }

        [data-testid="stSidebar"] h1 {
            color: white !important;
            font-size: 1.2rem !important;
            margin-bottom: 1rem !important;
        }

        [data-testid="stSidebar"] .stButton > button {
            width: 100% !important;
            padding: 0.75rem 1rem !important;
            margin-bottom: 0.5rem !important;
            background: transparent !important;
            color: white !important;
            text-align: left !important;
            border: none !important;
            border-radius: 4px !important;
            transition: background-color 0.2s !important;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            background-color: rgba(255, 255, 255, 0.1) !important;
        }

        /* Main Content */
        .main .block-container {
            max-width: none !important;
            padding-right: 1rem !important;
            padding-left: 1rem !important;
        }

        @media (min-width: 992px) {
            .main .block-container {
                padding-left: 320px !important;
            }
        }

        /* Hide Streamlit Branding */
        #MainMenu, footer {
            visibility: hidden;
        }
        </style>
    """, unsafe_allow_html=True)

def plot_missing_values(data):
    missing_values = data.isnull().sum()
    missing_df = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Values': missing_values.values,
        'Percentage': (missing_values.values / len(data)) * 100
    })
    missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', ascending=True)
    
    if not missing_df.empty:
        fig = px.bar(missing_df, 
                    x='Missing Values', 
                    y='Column',
                    text='Percentage',
                    title='Missing Values Analysis',
                    labels={'Missing Values': 'Number of Missing Values', 'Column': 'Features'},
                    orientation='h')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        return fig
    return None

def main():
    # Page config
    st.set_page_config(
        page_title="AI Data Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    local_css()

    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = AIAnalyzer()
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "📤 Data Upload"
    if 'batch_size' not in st.session_state:
        st.session_state.batch_size = 32
    
    # Modern sidebar with improved navigation
    with st.sidebar:
        st.title("🔍 AI Data Assistant")
        st.markdown("---")
        
        # Navigation with icons and better styling
        nav_options = {
            "📤 Data Upload": "Upload and preview your data",
            "🧹 Data Cleaning": "Clean and preprocess your data",
            "🤖 AI Analysis": "Get AI-powered insights"
        }
        
        for option, description in nav_options.items():
            if st.button(
                option,
                key=f"nav_{option}",
                help=description,
                use_container_width=True
            ):
                st.session_state.current_page = option
        
        st.markdown("---")
        if st.session_state.data is not None:
            st.markdown("### Dataset Info")
            st.info(f"Rows: {len(st.session_state.data):,}")
            st.info(f"Columns: {len(st.session_state.data.columns):,}")
            
    # Main content area with modern card layout
    if st.session_state.current_page == "📤 Data Upload":
        st.title("📤 Data Upload & Preview")
        st.markdown("---")
        
        with st.container():
            # Modern file upload section
            upload_col1, upload_col2 = st.columns([2, 1])
            with upload_col1:
                uploaded_file = st.file_uploader(
                    "Drag and drop your data file here",
                    type=['csv', 'xlsx', 'json', 'xml'],
                    help="Supported formats: CSV, Excel, JSON, XML",
                    key="file_uploader"
                )
            with upload_col2:
                st.markdown("### 📁 Supported Formats")
                st.markdown("""
                - CSV files (.csv)
                - Excel files (.xlsx)
                - JSON files (.json)
                - XML files (.xml)
                """)
        
        if uploaded_file is not None:
            try:
                # Load data with progress bar
                with st.spinner("📊 Loading your data..."):
                    if uploaded_file.name.endswith('.csv'):
                        data = pd.read_csv(uploaded_file)
                        st.session_state.input_file_type = 'csv'
                    elif uploaded_file.name.endswith('.xlsx'):
                        data = pd.read_excel(uploaded_file)
                        st.session_state.input_file_type = 'excel'
                    elif uploaded_file.name.endswith('.json'):
                        data = pd.read_json(uploaded_file)
                        st.session_state.input_file_type = 'json'
                    elif uploaded_file.name.endswith('.xml'):
                        data = pd.read_xml(uploaded_file)
                        st.session_state.input_file_type = 'xml'
                
                st.session_state.data = data
                st.session_state.analyzer.set_data(data)
                st.session_state.processed_data = data.copy()
                
                # Success message with animation
                st.success("✅ Data loaded successfully!")
                
                # Data Statistics in modern cards
                st.markdown("### 📊 Data Statistics")
                
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    with st.container():
                        st.metric("📝 Rows", f"{len(data):,}")
                        st.metric("💾 Memory", f"{data.memory_usage().sum() / 1024 / 1024:.2f} MB")
                
                with metric_cols[1]:
                    with st.container():
                        st.metric("📊 Columns", f"{len(data.columns):,}")
                        st.metric("🔄 Duplicates", f"{data.duplicated().sum():,}")
                
                with metric_cols[2]:
                    with st.container():
                        st.metric("❓ Missing Values", f"{data.isna().sum().sum():,}")
                        st.metric("🏷️ Data Types", f"{len(data.dtypes.unique()):,}")
                
                # Data Preview with modern styling
                st.markdown("### 👀 Data Preview")
                with st.container():
                    st.dataframe(
                        data.head(),
                        use_container_width=True,
                        height=300
                    )
                
                # Missing Values Analysis
                st.markdown("### 📉 Missing Values Analysis")
                missing_fig = plot_missing_values(data)
                if missing_fig:
                    st.plotly_chart(missing_fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    elif st.session_state.current_page == "🧹 Data Cleaning":
        st.title("🧹 Data Cleaning")
        st.markdown("---")
        
        if st.session_state.data is None:
            st.warning("⚠️ Please upload data first in the Data Upload section!")
            return
        
        # Create two columns for cleaning options
        cleaning_col1, cleaning_col2 = st.columns(2)
        
        # Missing Values Section
        with cleaning_col1:
            with st.container():
                st.markdown("### 🔍 Missing Values")
                cols_with_missing = st.session_state.data.columns[
                    st.session_state.data.isnull().any()
                ].tolist()
                
                if not cols_with_missing:
                    st.info("✨ No missing values found in the dataset!")
                else:
                    # Use a form for missing values handling
                    with st.form(key='missing_values_form', clear_on_submit=False):
                        st.markdown(f"Found {len(cols_with_missing)} columns with missing values")
                        
                        # Column selection with search
                        selected_cols = st.multiselect(
                            "Select columns to clean:",
                            cols_with_missing,
                            help="Choose the columns you want to handle missing values for"
                        )
                        
                        # Method selection with descriptions
                        method_descriptions = {
                            "Drop rows": "Remove rows with missing values in selected columns",
                            "Fill with mean": "Replace missing values with column mean (numeric only)",
                            "Fill with median": "Replace missing values with column median (numeric only)",
                            "Fill with mode": "Replace missing values with most frequent value"
                        }
                        
                        method = st.selectbox(
                            "Choose method:",
                            list(method_descriptions.keys()),
                            help="Select how to handle missing values"
                        )
                        
                        # Show method description
                        if method:
                            st.info(method_descriptions[method])
                        
                        # Submit button
                        submitted = st.form_submit_button(
                            "Apply",
                            help="Click to apply the selected method to handle missing values"
                        )
                        
                        if submitted and selected_cols:
                            try:
                                with st.spinner("Processing..."):
                                    temp_data = st.session_state.processed_data.copy()
                                    initial_missing = temp_data[selected_cols].isnull().sum().sum()
                                    
                                    # Process data in batches
                                    total_rows = len(temp_data)
                                    progress_bar = st.progress(0)
                                    
                                    if method == "Drop rows":
                                        # Drop rows all at once since it's more efficient
                                        temp_data = temp_data.dropna(subset=selected_cols)
                                        progress_bar.progress(1.0)
                                    else:
                                        # Process other methods in batches
                                        for start_idx in range(0, total_rows, st.session_state.batch_size):
                                            end_idx = min(start_idx + st.session_state.batch_size, total_rows)
                                            
                                            # Fill missing values based on method
                                            for col in selected_cols:
                                                if method == "Fill with mean" and pd.api.types.is_numeric_dtype(temp_data[col]):
                                                    fill_value = temp_data[col].mean()
                                                    temp_data.loc[start_idx:end_idx-1, col] = temp_data.loc[start_idx:end_idx-1, col].fillna(fill_value)
                                                elif method == "Fill with median" and pd.api.types.is_numeric_dtype(temp_data[col]):
                                                    fill_value = temp_data[col].median()
                                                    temp_data.loc[start_idx:end_idx-1, col] = temp_data.loc[start_idx:end_idx-1, col].fillna(fill_value)
                                                elif method == "Fill with mode":
                                                    fill_value = temp_data[col].mode().iloc[0] if not temp_data[col].mode().empty else None
                                                    if fill_value is not None:
                                                        temp_data.loc[start_idx:end_idx-1, col] = temp_data.loc[start_idx:end_idx-1, col].fillna(fill_value)
                                            
                                            # Update progress
                                            progress = min((end_idx / total_rows), 1.0)
                                            progress_bar.progress(progress)
                                    
                                    final_missing = temp_data[selected_cols].isnull().sum().sum()
                                    st.session_state.processed_data = temp_data
                                    
                                    # Show success message with statistics
                                    st.success(f"""
                                    ✅ Successfully handled missing values!
                                    - Initial missing values: {initial_missing:,}
                                    - Remaining missing values: {final_missing:,}
                                    - Fixed values: {initial_missing - final_missing:,}
                                    """)
                            except Exception as e:
                                st.error(f"Error handling missing values: {str(e)}")
                                st.error("Details: Make sure the selected method is compatible with your data types")
        
        # Duplicate Rows Section
        with cleaning_col2:
            with st.container():
                st.markdown("### 🔄 Duplicate Rows")
                
                # Show duplicate statistics
                total_rows = len(st.session_state.processed_data)
                duplicate_rows = st.session_state.processed_data.duplicated().sum()
                
                if duplicate_rows == 0:
                    st.info("✨ No duplicate rows found in the dataset!")
                else:
                    st.warning(f"Found {duplicate_rows:,} duplicate rows ({(duplicate_rows/total_rows)*100:.1f}% of data)")
                    
                    # Use a form for duplicate handling
                    with st.form(key='duplicate_form', clear_on_submit=False):
                        st.markdown("Remove all duplicate rows while keeping the first occurrence")
                        
                        # Submit button
                        submitted = st.form_submit_button(
                            "Remove Duplicates",
                            help="Click to remove all duplicate rows"
                        )
                        
                        if submitted:
                            try:
                                with st.spinner("Removing duplicates..."):
                                    temp_data = st.session_state.processed_data.copy()
                                    initial_rows = len(temp_data)
                                    temp_data = temp_data.drop_duplicates()
                                    st.session_state.processed_data = temp_data
                                    removed_rows = initial_rows - len(temp_data)
                                    
                                    # Show success message with statistics
                                    st.success(f"""
                                    ✅ Successfully removed duplicates!
                                    - Initial rows: {initial_rows:,}
                                    - Rows after cleaning: {len(temp_data):,}
                                    - Removed rows: {removed_rows:,}
                                    """)
                            except Exception as e:
                                st.error(f"Error removing duplicates: {str(e)}")
        
        # Show current data statistics
        if st.session_state.processed_data is not None:
            st.markdown("---")
            st.markdown("### 📊 Current Data Statistics")
            
            # Create three columns for metrics
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                st.metric(
                    "Total Rows",
                    f"{len(st.session_state.processed_data):,}",
                    delta=f"{len(st.session_state.processed_data) - len(st.session_state.data):,}",
                    delta_color="inverse"
                )
            
            with stat_col2:
                st.metric(
                    "Missing Values",
                    f"{st.session_state.processed_data.isna().sum().sum():,}",
                    delta=f"{st.session_state.processed_data.isna().sum().sum() - st.session_state.data.isna().sum().sum():,}",
                    delta_color="inverse"
                )
            
            with stat_col3:
                st.metric(
                    "Duplicate Rows",
                    f"{st.session_state.processed_data.duplicated().sum():,}",
                    delta=f"{st.session_state.processed_data.duplicated().sum() - st.session_state.data.duplicated().sum():,}",
                    delta_color="inverse"
                )
            
            # Preview cleaned data
            st.markdown("### 👀 Preview Cleaned Data")
            with st.container():
                st.dataframe(
                    st.session_state.processed_data.head(),
                    use_container_width=True,
                    height=300
                )
            
            # Download options
            st.markdown("### 📥 Download Cleaned Data")
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                # Format selection
                download_format = st.selectbox(
                    "Select format:",
                    ["CSV", "Excel", "JSON", "XML"],
                    help="Choose the format for your cleaned data"
                )
            
            with download_col2:
                if download_format == "CSV":
                    csv_data = st.session_state.processed_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv_data,
                        file_name="cleaned_data.csv",
                        mime="text/csv"
                    )
                elif download_format == "Excel":
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        st.session_state.processed_data.to_excel(writer, index=False)
                    excel_data = excel_buffer.getvalue()
                    st.download_button(
                        label="📊 Download Excel",
                        data=excel_data,
                        file_name="cleaned_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                elif download_format == "JSON":
                    json_data = st.session_state.processed_data.to_json(orient='records', indent=2).encode('utf-8')
                    st.download_button(
                        label="🔧 Download JSON",
                        data=json_data,
                        file_name="cleaned_data.json",
                        mime="application/json"
                    )
                elif download_format == "XML":
                    xml_data = st.session_state.processed_data.to_xml(index=False).encode('utf-8')
                    st.download_button(
                        label="🌐 Download XML",
                        data=xml_data,
                        file_name="cleaned_data.xml",
                        mime="application/xml"
                    )
    
    elif st.session_state.current_page == "🤖 AI Analysis":
        st.title("🤖 AI Analysis")
        st.markdown("---")
        
        if st.session_state.data is None:
            st.warning("Please upload data first in the Data Upload section!")
            return
            
        # Create tabs for different analysis types
        analysis_tab, anomaly_tab, sentiment_tab, model_tab = st.tabs([
            "📊 Feature Analysis",
            "🔍 Anomaly Detection",
            "😊 Sentiment Analysis",
            "🎯 Predictive Modeling"
        ])
        
        # Feature Analysis Tab
        with analysis_tab:
            st.markdown("## 📈 Feature Analysis")
            if st.session_state.data is not None:
                try:
                    analysis_results = st.session_state.analyzer.analyze_features()
                    
                    # Display feature importance
                    if analysis_results['feature_importance'] is not None:
                        st.plotly_chart(
                            analysis_results['visualizations'][-1],
                            use_container_width=True,
                            config={'displayModeBar': False}
                        )
                    
                    # Display correlations
                    if analysis_results['correlations'] is not None:
                        st.subheader("Feature Correlations")
                        for viz in analysis_results['visualizations']:
                            if "Correlation" in viz.layout.title.text:
                                st.plotly_chart(viz, use_container_width=True)
                    
                    # Display feature distributions
                    st.subheader("Feature Distributions")
                    for viz in analysis_results['visualizations']:
                        if "Distribution of" in viz.layout.title.text:
                            st.plotly_chart(viz, use_container_width=True)
                    
                    # Display feature patterns
                    st.subheader("Feature Patterns")
                    patterns_df = pd.DataFrame.from_dict(analysis_results['patterns'], orient='index')
                    st.dataframe(patterns_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during feature analysis: {str(e)}")
        
        # Anomaly Detection Tab
        with anomaly_tab:
            st.markdown("### 🔍 Anomaly Detection")
            
            # Check if there are numeric columns
            numeric_cols = st.session_state.data.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) == 0:
                st.warning("No numeric columns found in the data. Anomaly detection requires numeric data.")
            else:
                if st.button("Detect Anomalies", key="detect_anomalies"):
                    try:
                        with st.spinner("Detecting anomalies..."):
                            anomaly_df, anomaly_results = st.session_state.analyzer.detect_anomalies()
                            
                            if anomaly_df is not None and anomaly_results is not None:
                                # Display anomaly statistics
                                st.subheader("Anomaly Statistics")
                                total_anomalies = anomaly_df['is_anomaly'].sum()
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Total Anomalies", total_anomalies)
                                with col2:
                                    st.metric("Anomaly Percentage", f"{(total_anomalies / len(anomaly_df)) * 100:.2f}%")
                                
                                # Display anomaly visualizations
                                st.subheader("Anomaly Visualizations")
                                for viz in anomaly_results['visualizations']:
                                    st.plotly_chart(viz, use_container_width=True)
                                
                                # Display anomalous records
                                st.subheader("Anomalous Records")
                                anomalous_records = st.session_state.data[anomaly_df['is_anomaly']]
                                st.dataframe(anomalous_records, use_container_width=True)
                            else:
                                st.warning("No anomalies detected or insufficient data for analysis.")
                                
                    except Exception as e:
                        st.error(f"Error during anomaly detection: {str(e)}")
        
        # Sentiment Analysis Tab
        with sentiment_tab:
            st.markdown("### 😊 Sentiment Analysis")
            
            # Create placeholder for sentiment plot
            sentiment_plot = st.empty()
            
            text_columns = st.session_state.data.select_dtypes(include=['object']).columns
            
            if len(text_columns) == 0:
                st.warning("No text columns found in the dataset")
            else:
                # Settings section
                with st.expander("⚙️ Analysis Settings", expanded=True):
                    st.session_state.batch_size = st.number_input(
                        "Batch Size",
                        min_value=16,
                        max_value=128,
                        value=st.session_state.batch_size,
                        step=16,
                        help="Number of texts to analyze at once. Lower values use less memory but may be slower. Default: 32"
                    )
                
                selected_column = st.selectbox(
                    "Select text column for sentiment analysis:",
                    text_columns,
                    help="Choose the column containing text to analyze"
                )
                
                if selected_column and st.button("Analyze Sentiment", key="analyze_sentiment"):
                    try:
                        with st.spinner("Performing sentiment analysis..."):
                            sentiment_results = st.session_state.analyzer.analyze_sentiment(selected_column)
                            
                            # Display sentiment plot in the placeholder
                            if sentiment_results['plot'] is not None:
                                with sentiment_plot:
                                    st.plotly_chart(
                                        sentiment_results['plot'],
                                        use_container_width=True,
                                        config={'displayModeBar': False}
                                    )
                            
                            # Display sentiment statistics
                            st.markdown("### 📊 Sentiment Statistics")
                            stats_cols = st.columns(3)
                            
                            total = sentiment_results['sentiment_counts'].sum()
                            for i, (sentiment, count) in enumerate(sentiment_results['sentiment_counts'].items()):
                                with stats_cols[i]:
                                    percentage = (count / total) * 100
                                    st.metric(
                                        sentiment,
                                        f"{percentage:.1f}%",
                                        f"{count:,} texts"
                                    )
                    
                    except Exception as e:
                        st.error(f"Error performing sentiment analysis: {str(e)}")
        
        # Predictive Modeling Tab
        with model_tab:
            st.markdown("### 🎯 Predictive Modeling")
            
            # Select target column
            target_col = st.selectbox(
                "Select target column for prediction:",
                st.session_state.data.columns.tolist()
            )
            
            # Select problem type
            problem_type = st.selectbox(
                "Select problem type:",
                ["classification", "regression"],
                help="Choose 'classification' for categorical target variables and 'regression' for numerical target variables."
            )
            
            if st.button("Train Model", key="train_model"):
                try:
                    # Create a container for status messages
                    status_container = st.container()
                    with status_container:
                        st.write("**Model Training Status:**")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        status_text.markdown("*Training model...*")
                    
                    # Update progress
                    status_text.markdown("*Preparing data...*")
                    progress_bar.progress(20)
                    
                    # Check if target column is appropriate for the problem type
                    is_numeric = pd.api.types.is_numeric_dtype(st.session_state.data[target_col])
                    if problem_type == "regression" and not is_numeric:
                        st.error("For regression, the target column must be numeric. Please choose a numeric column or change to classification.")
                        progress_bar.empty()
                        status_text.empty()
                    else:
                        # Update progress
                        status_text.markdown("*Training model...*")
                        progress_bar.progress(40)
                        
                        model_results = st.session_state.analyzer.train_model(
                            target_column=target_col,
                            problem_type=problem_type
                        )
                        
                        # Update progress
                        status_text.markdown("*Calculating metrics...*")
                        progress_bar.progress(60)
                        
                        # Display model performance metrics
                        st.subheader("📊 Model Performance")
                        
                        # Create metrics columns
                        metric_cols = st.columns(3)
                        
                        if problem_type == "classification":
                            with metric_cols[0]:
                                st.metric("Accuracy", f"{model_results['accuracy']:.2%}")
                            with metric_cols[1]:
                                st.metric("Precision", f"{model_results['precision']:.2%}")
                            with metric_cols[2]:
                                st.metric("F1 Score", f"{model_results['f1']:.2%}")
                        else:
                            with metric_cols[0]:
                                st.metric("Mean Squared Error", f"{model_results['mse']:.4f}")
                            with metric_cols[1]:
                                st.metric("Mean Absolute Error", f"{model_results['mae']:.4f}")
                            with metric_cols[2]:
                                st.metric("R² Score", f"{model_results['r2']:.4f}")
                        
                        # Update progress
                        status_text.markdown("*Generating visualizations...*")
                        progress_bar.progress(80)
                        
                        # Display visualizations
                        st.subheader("📈 Model Visualizations")
                        for viz in model_results['visualizations']:
                            st.plotly_chart(viz, use_container_width=True)
                        
                        # Complete progress
                        progress_bar.progress(100)
                        status_text.markdown("*✅ Model training complete!*")
                        
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")

if __name__ == "__main__":
    main()
