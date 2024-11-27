import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any, Optional
import gc

class AIAnalyzer:
    def __init__(self):
        self.data = None
        self.label_encoders = {}
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize model instances with memory-efficient parameters"""
        self.clf = RandomForestClassifier(
            n_estimators=50,  # Reduced from default 100
            max_depth=10,     # Limit tree depth
            n_jobs=-1
        )
        self.reg = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            n_jobs=-1
        )
        self.anomaly_detector = IsolationForest(
            n_estimators=50,
            max_samples='auto',
            contamination=0.1,
            n_jobs=-1
        )
        
    def set_data(self, data: pd.DataFrame):
        """Set the data for analysis"""
        self.data = data
        
    def _preprocess_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess numeric data in chunks to save memory"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        chunk_size = max(1, len(numeric_cols) // 4)  # Process in 4 chunks
        
        for i in range(0, len(numeric_cols), chunk_size):
            chunk_cols = numeric_cols[i:i + chunk_size]
            for col in chunk_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col].fillna(df[col].median(), inplace=True)
        
        return df
        
    def _encode_categorical(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Encode categorical variables efficiently"""
        for col in columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            # Fill NaN values with a placeholder before encoding
            df[col] = df[col].fillna('missing')
            df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        return df
        
    def analyze_features(self) -> Dict[str, Any]:
        """Analyze features with memory optimization"""
        if self.data is None or len(self.data.columns) == 0:
            return None
            
        results = {
            'correlations': None,
            'patterns': {},
            'feature_importance': None,
            'visualizations': []
        }
        
        # Process numeric columns for correlation
        numeric_data = self.data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            # Fill NaN values with median for correlation analysis
            numeric_data = numeric_data.fillna(numeric_data.median())
            correlations = numeric_data.corr()
            results['correlations'] = correlations
            
            # Create correlation heatmap
            fig = px.imshow(
                correlations,
                title="Feature Correlations",
                aspect="auto"
            )
            results['visualizations'].append(fig)
            
            # Calculate feature importance using a lightweight random forest
            rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1)
            target = numeric_data.columns[0]  # Use first numeric column as target
            features = numeric_data.columns[1:]
            
            if len(features) > 0:
                rf.fit(numeric_data[features], numeric_data[target])
                importance = dict(zip(features, rf.feature_importances_))
                results['feature_importance'] = importance
                
                # Create feature importance plot
                importance_df = pd.DataFrame(
                    sorted(importance.items(), key=lambda x: x[1], reverse=True),
                    columns=['Feature', 'Importance']
                )
                fig = px.bar(
                    importance_df,
                    x='Feature',
                    y='Importance',
                    title="Feature Importance"
                )
                results['visualizations'].append(fig)
        
        # Analyze patterns
        for col in self.data.columns:
            col_stats = {
                'dtype': str(self.data[col].dtype),
                'missing_values': self.data[col].isnull().sum(),
                'unique_values': len(self.data[col].unique())
            }
            
            if pd.api.types.is_numeric_dtype(self.data[col]):
                # Handle NaN values for statistics
                clean_col = self.data[col].dropna()
                if len(clean_col) > 0:
                    col_stats.update({
                        'mean': float(clean_col.mean()),
                        'std': float(clean_col.std()),
                        'skew': float(clean_col.skew())
                    })
                else:
                    col_stats.update({
                        'mean': 0.0,
                        'std': 0.0,
                        'skew': 0.0
                    })
            
            results['patterns'][col] = col_stats
        
        # Clean up memory
        gc.collect()
        
        return results
        
    def detect_anomalies(self) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """Detect anomalies with memory optimization"""
        if self.data is None or len(self.data.columns) == 0:
            return None, None
            
        # Select only numeric columns
        numeric_data = self.data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) == 0:
            return None, None
            
        # Fill missing values with median
        numeric_data = numeric_data.fillna(numeric_data.median())
        
        # Detect anomalies
        predictions = self.anomaly_detector.fit_predict(numeric_data)
        anomaly_mask = predictions == -1
        
        anomalies = self.data[anomaly_mask].copy()
        stats = {
            'anomalies_detected': int(anomaly_mask.sum()),
            'anomaly_percentage': float(100 * anomaly_mask.sum() / len(self.data)),
            'visualizations': []
        }
        
        # Create visualization for anomaly distribution
        for col in numeric_data.columns[:3]:  # Limit to first 3 columns
            fig = px.scatter(
                numeric_data,
                x=numeric_data.index,
                y=col,
                color=predictions == -1,
                title=f"Anomalies in {col}",
                labels={'color': 'Is Anomaly'}
            )
            stats['visualizations'].append(fig)
        
        # Clean up memory
        gc.collect()
        
        return anomalies, stats
        
    def analyze_sentiment(self, text_column: str) -> Optional[Dict[str, Any]]:
        """Analyze sentiment with memory optimization"""
        if self.data is None or text_column not in self.data.columns:
            return None
            
        # Process text in chunks
        chunk_size = 1000
        sentiments = []
        polarities = []
        
        for i in range(0, len(self.data), chunk_size):
            chunk = self.data[text_column].iloc[i:i + chunk_size]
            chunk_sentiments = []
            chunk_polarities = []
            
            for text in chunk:
                if pd.isna(text):
                    chunk_sentiments.append('neutral')
                    chunk_polarities.append(0)
                    continue
                    
                analysis = TextBlob(str(text))
                polarity = analysis.sentiment.polarity
                
                if polarity > 0:
                    sentiment = 'positive'
                elif polarity < 0:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                    
                chunk_sentiments.append(sentiment)
                chunk_polarities.append(polarity)
            
            sentiments.extend(chunk_sentiments)
            polarities.extend(chunk_polarities)
        
        # Create visualizations
        sentiment_counts = pd.Series(sentiments).value_counts()
        fig1 = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution"
        )
        
        fig2 = px.histogram(
            x=polarities,
            title="Polarity Distribution",
            labels={'x': 'Polarity', 'y': 'Count'}
        )
        
        results = {
            'sentiment_scores': sentiments,
            'polarity_scores': polarities,
            'visualizations': [fig1, fig2]
        }
        
        # Clean up memory
        gc.collect()
        
        return results
        
    def train_model(self, target_column: str, problem_type: str = 'classification',
                   test_size: float = 0.2) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """Train a model with memory optimization"""
        if self.data is None or target_column not in self.data.columns:
            return None, None
            
        # Prepare features and target
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        # Remove rows where target variable is NaN
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(y) == 0:
            return None, None
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        X = self._encode_categorical(X, categorical_cols)
        
        # Handle numeric variables
        X = self._preprocess_numeric(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train model
        if problem_type == 'classification':
            model = self.clf
            if y.dtype == 'object':
                y = y.fillna('missing')  # Handle NaN in target variable
                y_train = self.label_encoders[target_column].fit_transform(y_train.astype(str))
                y_test = self.label_encoders[target_column].transform(y_test.astype(str))
        else:
            model = self.reg
            # For regression, remove any remaining NaN values
            y_train = pd.to_numeric(y_train, errors='coerce')
            y_test = pd.to_numeric(y_test, errors='coerce')
            valid_train = ~np.isnan(y_train)
            valid_test = ~np.isnan(y_test)
            X_train = X_train[valid_train]
            y_train = y_train[valid_train]
            X_test = X_test[valid_test]
            y_test = y_test[valid_test]
        
        if len(y_train) == 0 or len(y_test) == 0:
            return None, None
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {}
        if problem_type == 'classification':
            metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
            conf_matrix = confusion_matrix(y_test, y_pred)
        else:
            metrics['mse'] = float(mean_squared_error(y_test, y_pred))
            metrics['rmse'] = float(np.sqrt(metrics['mse']))
            conf_matrix = None
        
        # Get feature importance
        feature_importance = dict(zip(X.columns, model.feature_importances_))
        
        results = {
            'metrics': metrics,
            'confusion_matrix': conf_matrix,
            'feature_importance': feature_importance
        }
        
        # Clean up memory
        gc.collect()
        
        return model, results
