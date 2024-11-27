import pandas as pd
import json
import xml.etree.ElementTree as ET
import xmltodict
import logging
from typing import Optional

def read_csv(file) -> Optional[pd.DataFrame]:
    """Read CSV file and return DataFrame"""
    try:
        return pd.read_csv(file)
    except Exception as e:
        logging.error(f"Error reading CSV file: {str(e)}")
        return None

def read_excel(file) -> Optional[pd.DataFrame]:
    """Read Excel file and return DataFrame"""
    try:
        return pd.read_excel(file)
    except Exception as e:
        logging.error(f"Error reading Excel file: {str(e)}")
        return None

def read_json(file) -> Optional[pd.DataFrame]:
    """Read JSON file and return DataFrame"""
    try:
        with open(file, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        logging.error(f"Error reading JSON file: {str(e)}")
        return None

def read_xml(file) -> Optional[pd.DataFrame]:
    """Read XML file and return DataFrame"""
    try:
        tree = ET.parse(file)
        xml_data = tree.getroot()
        dict_data = xmltodict.parse(ET.tostring(xml_data))
        return pd.DataFrame(dict_data)
    except Exception as e:
        logging.error(f"Error reading XML file: {str(e)}")
        return None

def save_dataframe(df: pd.DataFrame, format: str) -> bytes:
    """Save DataFrame to specified format"""
    try:
        if format == 'csv':
            return df.to_csv(index=False).encode('utf-8')
        elif format == 'json':
            return df.to_json(orient='records').encode('utf-8')
        elif format == 'excel':
            return df.to_excel(index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    except Exception as e:
        logging.error(f"Error saving DataFrame: {str(e)}")
        return None
