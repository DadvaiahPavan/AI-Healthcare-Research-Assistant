import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.drug_discovery import DrugDiscoveryAnalyzer
from src.llama_model import LlamaResearchAssistant
import pandas as pd

@pytest.fixture
def llama_assistant():
    """Fixture to create a Llama Research Assistant"""
    api_key = os.getenv('GROQ_API_KEY')
    return LlamaResearchAssistant(api_key)

@pytest.fixture
def drug_analyzer(llama_assistant):
    """Fixture to create a Drug Discovery Analyzer"""
    return DrugDiscoveryAnalyzer(llama_assistant)

def test_find_drug_candidates(drug_analyzer):
    """Test finding drug candidates for a disease"""
    candidates = drug_analyzer.find_drug_candidates('Alzheimer\'s Disease')
    
    assert isinstance(candidates, list), "Should return a list of candidates"
    assert len(candidates) > 0, "Should find at least one drug candidate"
    assert all('description' in candidate for candidate in candidates), "Each candidate should have a description"

def test_predict_disease_outcomes(drug_analyzer, tmp_path):
    """Test predicting disease outcomes"""
    # Create a sample patient data CSV
    sample_data = pd.DataFrame({
        'patient_id': [1, 2, 3],
        'age': [65, 45, 55],
        'gender': ['M', 'F', 'M'],
        'symptoms': ['memory_loss', 'headache', 'fatigue']
    })
    
    sample_file_path = tmp_path / "sample_patient_data.csv"
    sample_data.to_csv(sample_file_path, index=False)
    
    with open(sample_file_path, 'rb') as f:
        predictions = drug_analyzer.predict_disease_outcomes(f)
    
    assert isinstance(predictions, pd.DataFrame), "Should return a DataFrame of predictions"
    assert len(predictions) > 0, "Should generate predictions for patients"
    assert 'patient_id' in predictions.columns, "Predictions should include patient ID"
    assert 'predicted_outcome' in predictions.columns, "Predictions should include outcome"

def test_analyze_molecular_interactions(drug_analyzer):
    """Test molecular interaction analysis"""
    sample_molecular_data = pd.DataFrame({
        'molecule_id': [1, 2],
        'structure': ['SMILES1', 'SMILES2'],
        'interaction_score': [0.7, 0.5]
    })
    
    interactions = drug_analyzer.analyze_molecular_interactions(sample_molecular_data)
    
    assert isinstance(interactions, dict), "Should return a dictionary of interactions"
    assert 'error' not in interactions, "Should not contain error"
