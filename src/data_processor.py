import pandas as pd
import numpy as np
from typing import Dict, List, Any
import requests

class DataProcessor:
    """
    Handles data processing for medical research and clinical trial data
    """
    def __init__(self):
        self.data = None
    
    def load_clinical_data(self, file_path: str) -> pd.DataFrame:
        """
        Load clinical trial data from various sources
        
        Args:
            file_path (str): Path to the clinical data file
        
        Returns:
            pd.DataFrame: Processed clinical data
        """
        try:
            # Support multiple file formats
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.data = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                self.data = pd.read_json(file_path)
            else:
                raise ValueError("Unsupported file format")
            
            # Basic data cleaning
            self.data.dropna(inplace=True)
            return self.data
        except Exception as e:
            print(f"Error loading clinical data: {e}")
            return pd.DataFrame()
    
    def preprocess_data(self, columns_to_encode: List[str] = None) -> pd.DataFrame:
        """
        Preprocess the clinical data for AI analysis
        
        Args:
            columns_to_encode (List[str], optional): Columns to one-hot encode
        
        Returns:
            pd.DataFrame: Preprocessed data
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data loaded or data is empty. Use load_clinical_data() first.")
        
        # One-hot encoding for categorical variables
        if columns_to_encode:
            self.data = pd.get_dummies(self.data, columns=columns_to_encode)
        
        # Normalize numerical columns
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        self.data[numerical_cols] = (self.data[numerical_cols] - self.data[numerical_cols].mean()) / self.data[numerical_cols].std()
        
        return self.data
    
    def fetch_pubmed_data(self, search_term: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch research papers from PubMed API
        
        Args:
            search_term (str): Medical research topic
            max_results (int): Maximum number of results to fetch
        
        Returns:
            List of research paper metadata
        """
        # Validate input
        if not search_term or not isinstance(search_term, str):
            print("Warning: Invalid search term. Returning empty list.")
            return []
        
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": search_term,
            "retmax": max_results,
            "retmode": "json"
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            # Validate response
            if not response.json():
                print(f"No results found for search term: {search_term}")
                return []
            
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching PubMed data: {e}")
            return []
    
    def anonymize_patient_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Anonymize patient data by removing or hashing identifiable information
        
        Args:
            data (pd.DataFrame): Patient data
        
        Returns:
            pd.DataFrame: Anonymized patient data
        """
        try:
            # Validate input DataFrame
            if data is None or data.empty:
                print("Warning: Input data is None or empty. Returning empty DataFrame.")
                return pd.DataFrame()
            
            # Create a copy of the dataframe to avoid modifying original
            anonymized_data = data.copy()
            
            # Remove direct identifiers
            identifier_columns = ['name', 'patient_id', 'social_security_number']
            for col in identifier_columns:
                if col in anonymized_data.columns:
                    anonymized_data.drop(columns=[col], inplace=True)
            
            # Hash remaining potentially identifiable columns
            hash_columns = ['date_of_birth', 'address']
            for col in hash_columns:
                if col in anonymized_data.columns:
                    anonymized_data[col] = anonymized_data[col].apply(lambda x: hash(str(x)))
            
            return anonymized_data
        except Exception as e:
            print(f"Error anonymizing patient data: {e}")
            return pd.DataFrame()
