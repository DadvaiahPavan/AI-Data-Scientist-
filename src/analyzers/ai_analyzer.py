import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from typing import Dict, List, Any, Tuple, Optional
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
import re
import warnings
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import emoji

class AIAnalyzer:
    def __init__(self):
        self.data = None
        self.numeric_columns = None
        self.categorical_columns = None
        self.pipeline = None
        self.sentiment_analyzer = None
        self.sentiment_results = None
        self._cached_feature_importance = None
        self._cached_sentiment_plot = None
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Filter torch warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', message='.*torch.classes.*')

    def set_data(self, data: pd.DataFrame):
        """Set the data for analysis"""
        self.data = data.copy()
        self.numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        self.logger.info(f"Data set with {len(data)} rows, {len(self.numeric_columns)} numeric columns, {len(self.categorical_columns)} categorical columns")

    def analyze_features(self) -> Dict[str, Any]:
        """Analyze features and return insights"""
        if self.data is None:
            raise ValueError("No data available for analysis")

        results = {
            'correlations': None,
            'patterns': {},
            'feature_importance': None,
            'visualizations': []
        }

        # Analyze correlations for numeric columns
        if len(self.numeric_columns) > 1:
            corr_matrix = self.data[self.numeric_columns].corr()
            results['correlations'] = corr_matrix
            
            # Create correlation heatmap
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                title="Feature Correlations"
            )
            results['visualizations'].append(fig)

        # Analyze patterns for all columns
        for col in self.data.columns:
            col_stats = {
                'missing_percentage': (self.data[col].isna().sum() / len(self.data)) * 100
            }
            
            if col in self.numeric_columns:
                col_stats.update({
                    'mean': self.data[col].mean(),
                    'std': self.data[col].std(),
                    'min': self.data[col].min(),
                    'max': self.data[col].max()
                })
            elif col in self.categorical_columns:
                col_stats.update({
                    'unique_values': self.data[col].nunique(),
                    'most_common': self.data[col].mode().iloc[0] if not self.data[col].mode().empty else None
                })
            
            results['patterns'][col] = col_stats

        # Feature importance using random forest (cached)
        if self._cached_feature_importance is None and len(self.numeric_columns) > 0:
            X = self.data[self.numeric_columns].fillna(self.data[self.numeric_columns].mean())
            y = X[X.columns[0]]  # Use first column as target for demonstration
            X = X.drop(columns=[X.columns[0]])
            
            if len(X.columns) > 0:
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)
                
                importance_scores = pd.Series(
                    rf.feature_importances_,
                    index=X.columns
                ).sort_values(ascending=True)
                
                self._cached_feature_importance = importance_scores
                results['feature_importance'] = importance_scores
                results['visualizations'].append(self._create_feature_importance_plot(importance_scores))
        else:
            results['feature_importance'] = self._cached_feature_importance
            results['visualizations'].append(self._create_feature_importance_plot(self._cached_feature_importance))

        return results

    def detect_anomalies(self) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """Detect anomalies in the data using Isolation Forest with improved preprocessing"""
        try:
            if self.data is None:
                raise ValueError("No data available for analysis")

            numeric_data = self.data[self.numeric_columns].copy()
            if numeric_data.empty:
                raise ValueError("No numeric columns available for analysis")

            # Step 1: Preprocess the data
            # Handle missing values
            imputer = SimpleImputer(strategy='mean')
            numeric_data_imputed = pd.DataFrame(
                imputer.fit_transform(numeric_data),
                columns=numeric_data.columns,
                index=numeric_data.index
            )

            # Step 2: Scale the features
            scaler = StandardScaler()
            numeric_data_scaled = pd.DataFrame(
                scaler.fit_transform(numeric_data_imputed),
                columns=numeric_data_imputed.columns,
                index=numeric_data_imputed.index
            )

            # Step 3: Calculate dynamic contamination based on data distribution
            # Use IQR method to estimate the proportion of outliers
            q1 = numeric_data_scaled.quantile(0.25)
            q3 = numeric_data_scaled.quantile(0.75)
            iqr = q3 - q1
            outlier_mask = ((numeric_data_scaled < (q1 - 1.5 * iqr)) | 
                           (numeric_data_scaled > (q3 + 1.5 * iqr))).any(axis=1)
            estimated_contamination = max(0.01, min(0.1, outlier_mask.mean()))

            # Step 4: Apply Isolation Forest with optimized parameters
            iso_forest = IsolationForest(
                contamination=estimated_contamination,
                n_estimators=100,
                max_samples='auto',
                random_state=42,
                bootstrap=True
            )
            
            predictions = iso_forest.fit_predict(numeric_data_scaled)
            anomaly_scores = iso_forest.score_samples(numeric_data_scaled)
            
            # Normalize anomaly scores to [0,1] range for better interpretability
            anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
            
            anomaly_df = pd.DataFrame({
                'is_anomaly': predictions == -1,
                'anomaly_score': anomaly_scores
            }, index=numeric_data.index)

            # Create visualizations
            visualizations = []
            
            # Overall anomaly score distribution
            fig_dist = px.histogram(
                anomaly_scores,
                title="Distribution of Anomaly Scores",
                labels={'value': 'Anomaly Score', 'count': 'Count'},
                template='plotly_white'
            )
            visualizations.append(fig_dist)

            # Scatter plots for each feature vs anomaly score
            for col in numeric_data.columns:
                fig = px.scatter(
                    x=numeric_data[col],
                    y=anomaly_scores,
                    color=predictions == -1,
                    title=f"Anomaly Scores vs {col}",
                    labels={'x': col, 'y': 'Anomaly Score', 'color': 'Is Anomaly'},
                    template='plotly_white'
                )
                visualizations.append(fig)

            results = {
                'num_anomalies': (predictions == -1).sum(),
                'anomaly_percentage': ((predictions == -1).sum() / len(predictions)) * 100,
                'estimated_contamination': estimated_contamination,
                'visualizations': visualizations
            }

            return anomaly_df, results

        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {str(e)}")
            return None, None

    def preprocess_target(self, target_column: str) -> Tuple[pd.Series, Dict[str, Any]]:
        """Preprocess the target column"""
        if target_column not in self.data.columns:
            raise ValueError(f"Target column {target_column} not found in data")

        target = self.data[target_column].copy()
        target_info = {'type': None, 'final_classes': None}

        # Check if numeric
        if pd.api.types.is_numeric_dtype(target):
            target_info['type'] = 'numeric'
            # For regression, we'll keep the original numeric values
            target = pd.to_numeric(target, errors='coerce')
            # Remove any NaN values
            target = target.dropna()
            target_info['final_classes'] = None
        else:
            # For non-numeric columns, check if they contain interval strings like '(6.1, 6.8]'
            if target.dtype == 'object' and target.str.contains(r'^\([\d.]+,\s*[\d.]+\]$').all():
                # Convert interval strings to numeric values using the midpoint
                def extract_midpoint(interval_str):
                    try:
                        # Extract numbers from string like '(6.1, 6.8]'
                        numbers = re.findall(r'[\d.]+', str(interval_str))
                        if len(numbers) == 2:
                            return (float(numbers[0]) + float(numbers[1])) / 2
                        return None
                    except (ValueError, TypeError):
                        return None

                target = target.apply(extract_midpoint)
                # Remove any NaN values that resulted from conversion
                target = target.dropna()
                target_info['type'] = 'numeric'
                target_info['final_classes'] = None

                self.logger.info(f"Converted interval strings to numeric values. Sample: {target.head()}")
                self.logger.info(f"Number of valid samples after conversion: {len(target)}")
            else:
                # Handle categorical data
                target_info['type'] = 'categorical'
                # Remove any NaN values
                target = target.dropna()
                # Get unique classes
                unique_classes = target.unique()
                if len(unique_classes) > 10:  # If too many classes, group by frequency
                    value_counts = target.value_counts()
                    top_classes = value_counts.nlargest(10).index
                    target = target.apply(lambda x: x if x in top_classes else 'Other')
                    unique_classes = target.unique()
                
                target_info['final_classes'] = sorted(unique_classes)

        # Final check for NaN values
        if target.isna().any():
            self.logger.warning(f"Found {target.isna().sum()} NaN values in target after preprocessing")
            target = target.dropna()
            self.logger.info(f"Removed NaN values. Remaining samples: {len(target)}")

        # Ensure we have enough samples
        if len(target) < 10:
            raise ValueError("Not enough valid samples in target column after preprocessing")

        self.logger.info(f"Target type determined: {target_info['type']}")
        self.logger.info(f"Final number of samples: {len(target)}")
        if target_info['final_classes'] is not None:
            self.logger.info(f"Final classes: {target_info['final_classes']}")

        return target, target_info

    def get_target_info(self, target_column: str) -> Dict[str, Any]:
        """Get detailed information about the target column"""
        if self.data is None or target_column not in self.data.columns:
            return {}
            
        target = self.data[target_column]
        value_counts = target.value_counts()
        
        info = {
            'column_type': str(target.dtype),
            'unique_values': len(value_counts),
            'sample_distribution': {}
        }
        
        if target.dtype in ['int64', 'float64']:
            info.update({
                'min': float(target.min()),
                'max': float(target.max()),
                'mean': float(target.mean()),
                'median': float(target.median())
            })
        else:
            # For categorical data like TV shows
            high_pop = value_counts[value_counts >= 40].index
            medium_pop = value_counts[(value_counts >= 20) & (value_counts < 40)].index
            low_pop = value_counts[(value_counts < 20)].index
            
            info['sample_distribution'] = {
                'High Popularity (40+ episodes)': len(high_pop),
                'Medium Popularity (20-39 episodes)': len(medium_pop),
                'Low Popularity (<20 episodes)': len(low_pop)
            }
            
            # Sample of shows in each category
            info['category_examples'] = {
                'High Popularity Shows': list(high_pop)[:5],
                'Medium Popularity Shows': list(medium_pop)[:5],
                'Low Popularity Shows': list(low_pop)[:5]
            }
            
        return info

    def train_model(self, target_column: str, problem_type: str = None) -> Dict[str, Any]:
        """Train a model and return performance metrics"""
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data available for training")

        self.logger.info(f"Starting model training for target: {target_column}")
        
        try:
            # Preprocess target
            processed_target, target_info = self.preprocess_target(target_column)
            self.logger.info(f"Processed target column. Final class distribution: {processed_target.value_counts().to_dict()}")
            
            # Get valid indices (where target is not null after preprocessing)
            valid_indices = processed_target.index
            
            # Separate features and filter to match target indices
            feature_columns = [col for col in self.data.columns if col != target_column]
            X = self.data.loc[valid_indices, feature_columns]
            y = processed_target
            
            # Verify data alignment
            if len(X) != len(y):
                raise ValueError(f"Feature and target dimensions don't match. Features: {len(X)}, Target: {len(y)}")
            
            self.logger.info(f"Final dataset size after preprocessing: {len(X)} samples")
            
            # Split data
            if problem_type == 'classification':
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                # For regression, don't use stratification
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            
            # Create preprocessing pipeline
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns
            
            # Log feature types
            self.logger.info(f"Numeric features: {len(numeric_features)}")
            self.logger.info(f"Categorical features: {len(categorical_features)}")
            
            transformers = []
            
            if len(numeric_features) > 0:
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
                transformers.append(('num', numeric_transformer, numeric_features))
            
            if len(categorical_features) > 0:
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])
                transformers.append(('cat', categorical_transformer, categorical_features))
            
            # Create preprocessor only if we have transformers
            if transformers:
                preprocessor = ColumnTransformer(transformers=transformers)
            else:
                raise ValueError("No valid features found for preprocessing")
            
            # Determine if classification or regression
            if problem_type is None:
                if target_info['type'] == 'categorical' or len(target_info['final_classes']) <= 10:
                    problem_type = 'classification'
                else:
                    problem_type = 'regression'
            
            self.logger.info(f"Problem type determined: {problem_type}")
            
            # Create full pipeline
            if problem_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
            
            self.pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Train model with error handling
            try:
                self.logger.info("Training model...")
                self.pipeline.fit(X_train, y_train)
                self.logger.info("Model training completed")
                
                # Get predictions
                y_pred = self.pipeline.predict(X_test)
                
                # Calculate metrics
                results = {
                    'target_info': target_info,
                    'feature_importance': self._get_feature_importance(feature_columns),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'visualizations': []
                }
                
                if problem_type == 'classification':
                    results.update({
                        'accuracy': float(accuracy_score(y_test, y_pred)),
                        'precision': float(precision_score(y_test, y_pred, average='weighted')),
                        'f1': float(f1_score(y_test, y_pred, average='weighted'))
                    })
                    
                    # Create confusion matrix visualization
                    cm = confusion_matrix(y_test, y_pred)
                    labels = sorted(list(set(y_test.unique()) | set(y_pred)))
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=labels,
                        y=labels,
                        colorscale='Viridis'
                    ))
                    
                    fig.update_layout(
                        title='Confusion Matrix',
                        xaxis_title='Predicted',
                        yaxis_title='Actual',
                        width=800,
                        height=800
                    )
                    
                    results['visualizations'].append(fig)
                else:
                    results.update({
                        'mse': float(mean_squared_error(y_test, y_pred)),
                        'mae': float(mean_absolute_error(y_test, y_pred)),
                        'r2': float(r2_score(y_test, y_pred))
                    })
                    
                    # Create scatter plot
                    fig = px.scatter(
                        x=y_test,
                        y=y_pred,
                        title='Predicted vs Actual Values',
                        labels={'x': 'Actual Values', 'y': 'Predicted Values'}
                    )
                    
                    fig.update_layout(
                        width=800,
                        height=600
                    )
                    
                    results['visualizations'].append(fig)
                
                # Add feature importance visualization
                if results['feature_importance']:
                    importance_df = pd.DataFrame(
                        sorted(results['feature_importance'].items(), key=lambda x: x[1], reverse=True),
                        columns=['Feature', 'Importance']
                    )
                    
                    fig = px.bar(
                        importance_df.head(20),  # Show top 20 features
                        x='Feature',
                        y='Importance',
                        title='Top 20 Feature Importance',
                        labels={'Feature': 'Features', 'Importance': 'Importance Score'}
                    )
                    
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        width=800,
                        height=400
                    )
                    
                    results['visualizations'].append(fig)
                
                return results
                
            except Exception as e:
                self.logger.error(f"Error during model training: {str(e)}")
                raise ValueError(f"Model training failed: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Error in train_model: {str(e)}")
            raise

    def _get_feature_importance(self, feature_columns: List[str]) -> Dict[str, float]:
        """Extract feature importance from the trained model"""
        if not hasattr(self.pipeline, 'named_steps') or 'model' not in self.pipeline.named_steps:
            return {}
            
        model = self.pipeline.named_steps['model']
        if not hasattr(model, 'feature_importances_'):
            return {}
            
        # Get feature names after preprocessing
        preprocessor = self.pipeline.named_steps['preprocessor']
        feature_names = preprocessor.get_feature_names_out()
        
        # Get feature importance
        importance = model.feature_importances_
        return dict(zip(feature_names, importance))

    def _initialize_sentiment_analyzer(self):
        """Initialize a fast sentiment analyzer with optimized settings"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            
            # Use small, fast model
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            
            # Load tokenizer and model with optimized settings
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Optimize model for inference
            model.eval()
            torch.set_grad_enabled(False)
            
            # Create pipeline with optimized settings
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=-1,  # CPU
                batch_size=32,
                max_length=128,
                truncation=True,
                framework="pt"
            )
            self.logger.info("Fast sentiment analyzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing sentiment analyzer: {str(e)}")
            raise

    def analyze_sentiment(self, text_column: str) -> Dict[str, Any]:
        """Fast sentiment analysis with optimized settings"""
        if self.data is None or text_column not in self.data.columns:
            raise ValueError("Invalid text column for sentiment analysis")

        if self.sentiment_analyzer is None:
            self._initialize_sentiment_analyzer()

        # Use cached results if available
        if self.sentiment_results is None:
            self.logger.info("Performing sentiment analysis...")
            
            # Get texts and remove nulls
            texts = self.data[text_column].fillna("").astype(str).tolist()
            
            # Process in optimized batches
            results = self.sentiment_analyzer(texts)
            
            # Convert results to sentiment labels
            sentiments = []
            for result in results:
                # Fast label mapping
                sentiments.append('Positive' if result['label'] == 'POSITIVE' else 'Negative')
            
            sentiment_counts = pd.Series(sentiments).value_counts()
            self.sentiment_results = {
                'sentiment_counts': sentiment_counts,
                'plot': None
            }
            
            # Create plot if needed
            if self._cached_sentiment_plot is None:
                self._cached_sentiment_plot = self._create_sentiment_plot(sentiment_counts)
            self.sentiment_results['plot'] = self._cached_sentiment_plot
        
        return self.sentiment_results

    def get_sentiment_examples(self, text_column: str, n_examples: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get example texts for each sentiment category
        
        Args:
            text_column: Name of the column containing text data
            n_examples: Number of examples to return for each category
            
        Returns:
            Dictionary with examples for each sentiment category
        """
        try:
            if self.sentiment_results is None or self.sentiment_results['text_column'] != text_column:
                raise ValueError("Sentiment analysis must be run first")

            result_df = self.sentiment_results['result_df']
            examples = {}
            
            for sentiment in result_df[f'{text_column}_sentiment'].unique():
                sentiment_texts = result_df[
                    result_df[f'{text_column}_sentiment'] == sentiment
                ].sort_values(f'{text_column}_confidence', ascending=False)
                
                examples[sentiment] = [
                    {
                        'text': row[text_column],
                        'confidence': row[f'{text_column}_confidence']
                    }
                    for _, row in sentiment_texts.head(n_examples).iterrows()
                ]

            return examples

        except Exception as e:
            self.logger.error(f"Error getting sentiment examples: {str(e)}")
            raise

    def _create_feature_importance_plot(self, importance_scores: pd.Series) -> go.Figure:
        """Create a cached feature importance plot"""
        if importance_scores is None:
            return None
            
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=importance_scores.index,
            x=importance_scores.values,
            orientation='h',
            marker_color='#1f77b4'
        ))
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            showlegend=False,
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        
        return fig

    def _create_sentiment_plot(self, sentiment_counts: pd.Series) -> go.Figure:
        """Create a cached sentiment plot"""
        if sentiment_counts is None:
            return None
            
        colors = {'Positive': '#2ecc71', 'Neutral': '#3498db', 'Negative': '#e74c3c'}
        
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            marker=dict(colors=[colors[label] for label in sentiment_counts.index]),
            textinfo='percent+label',
            hole=0.3
        ))
        
        fig.update_layout(
            title="Sentiment Distribution",
            showlegend=True,
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        
        return fig
