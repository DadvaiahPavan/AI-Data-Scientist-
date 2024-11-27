import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging

class DataCleaner:
    def __init__(self):
        self.data = None
        self.original_data = None
        self.cleaning_history = []
    
    def set_data(self, data: pd.DataFrame):
        """Set the data and keep a copy of the original"""
        self.data = data.copy()
        self.original_data = data.copy()
        self.cleaning_history = []
    
    def handle_missing_values(self, method: str, columns: list = None, custom_value: Any = None) -> pd.DataFrame:
        """Handle missing values in the data
        
        Args:
            method: One of ['drop_rows', 'drop_columns', 'fill_mean', 'fill_median', 'fill_mode', 'fill_custom']
            columns: List of columns to apply the method to. If None, applies to all columns
            custom_value: Custom value to use when method is 'fill_custom'
        """
        if self.data is None:
            raise ValueError("No data available. Please set data first.")
        
        try:
            # Make a copy of the current data
            df = self.data.copy()
            
            # If no columns specified, use all columns
            if columns is None:
                columns = df.columns
            
            # Apply the selected method
            if method == "drop_rows":
                df = df.dropna(subset=columns)
            
            elif method == "drop_columns":
                df = df.drop(columns=[col for col in columns if df[col].isna().any()])
            
            elif method == "fill_mean":
                for col in columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].mean())
            
            elif method == "fill_median":
                for col in columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].median())
            
            elif method == "fill_mode":
                for col in columns:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
            
            elif method == "fill_custom":
                if custom_value is None:
                    raise ValueError("Custom value must be provided when using 'fill_custom' method")
                df[columns] = df[columns].fillna(custom_value)
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Record the changes
            self.cleaning_history.append({
                'operation': 'handle_missing_values',
                'method': method,
                'columns': columns,
                'custom_value': custom_value
            })
            
            # Update the data
            self.data = df
            return self.data
            
        except Exception as e:
            logging.error(f"Error handling missing values: {str(e)}")
            raise
    
    def remove_duplicates(self, subset: list = None) -> pd.DataFrame:
        """Remove duplicate rows from the data
        
        Args:
            subset: List of columns to consider for identifying duplicates
        """
        if self.data is None:
            raise ValueError("No data available. Please set data first.")
        
        try:
            # Make a copy of the current data
            df = self.data.copy()
            
            # Remove duplicates
            df = df.drop_duplicates(subset=subset)
            
            # Record the changes
            self.cleaning_history.append({
                'operation': 'remove_duplicates',
                'subset': subset
            })
            
            # Update the data
            self.data = df
            return self.data
            
        except Exception as e:
            logging.error(f"Error removing duplicates: {str(e)}")
            raise
    
    def convert_data_types(self, conversions: Dict[str, str]) -> pd.DataFrame:
        """Convert data types of specified columns
        
        Args:
            conversions: Dictionary mapping column names to desired data types
                       e.g., {'col1': 'int64', 'col2': 'float64', 'col3': 'string', 'col4': 'datetime64'}
        """
        if self.data is None:
            raise ValueError("No data available. Please set data first.")
        
        try:
            # Make a copy of the current data
            df = self.data.copy()
            
            # Apply conversions
            for column, dtype in conversions.items():
                if column not in df.columns:
                    raise ValueError(f"Column {column} not found in data")
                
                try:
                    if dtype == 'datetime64':
                        df[column] = pd.to_datetime(df[column])
                    else:
                        df[column] = df[column].astype(dtype)
                except Exception as e:
                    raise ValueError(f"Error converting {column} to {dtype}: {str(e)}")
            
            # Record the changes
            self.cleaning_history.append({
                'operation': 'convert_data_types',
                'conversions': conversions
            })
            
            # Update the data
            self.data = df
            return self.data
            
        except Exception as e:
            logging.error(f"Error converting data types: {str(e)}")
            raise
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """Get a summary of the cleaning operations performed"""
        if self.data is None or self.original_data is None:
            return {}
        
        return {
            'original_rows': len(self.original_data),
            'current_rows': len(self.data),
            'rows_removed': len(self.original_data) - len(self.data),
            'missing_values_before': self.original_data.isnull().sum().sum(),
            'missing_values_after': self.data.isnull().sum().sum(),
            'duplicates_before': len(self.original_data) - len(self.original_data.drop_duplicates()),
            'duplicates_after': len(self.data) - len(self.data.drop_duplicates()),
            'cleaning_history': self.cleaning_history
        }
