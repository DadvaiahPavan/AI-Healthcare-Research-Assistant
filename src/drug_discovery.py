import pandas as pd
import numpy as np
import io
from typing import List, Dict, Any, Union
from .llama_model import LlamaResearchAssistant
from .medical_knowledge_base import MedicalKnowledgeBase
import json
import re
import os

class DrugDiscoveryAssistant:
    def __init__(self, llama_assistant: LlamaResearchAssistant):
        """
        Initialize Drug Discovery Assistant
        
        Args:
            llama_assistant (LlamaResearchAssistant): AI assistant for medical research
        """
        self.llama_assistant = llama_assistant
        self.knowledge_base = MedicalKnowledgeBase()
    
    def discover_drug_candidates(self, disease: str) -> Dict[str, Any]:
        """
        Discover potential treatment candidates for a specific disease
        
        Args:
            disease (str): Target disease or condition
        
        Returns:
            Dict[str, Any]: Dictionary containing narrative and treatment information
        """
        try:
            # Prepare a comprehensive treatment discovery prompt
            prompt = f"""Provide an in-depth analysis of innovative treatment approaches for {disease}, covering:

1. Current Medical Understanding
- Pathophysiology of {disease}
- Latest research breakthroughs
- Emerging therapeutic strategies

2. Treatment Modalities
- Pharmacological interventions
- Targeted therapies
- Potential breakthrough treatments
- Personalized medicine approaches

3. Detailed Treatment Insights
For each potential treatment, provide:
- Specific drug/therapy name
- Mechanism of action
- Clinical trial status
- Potential effectiveness
- Unique therapeutic approach
- Potential side effects and considerations

CRITICAL REQUIREMENTS:
- Use professional medical terminology
- Provide evidence-based information
- Focus on cutting-edge research
- Maximum length: 750 words
- Highlight most promising treatments"""

            try:
                # Generate treatment discovery response
                treatment_narrative = self.llama_assistant._generate_llama_response(prompt)
                
                # Parse and structure treatment information
                treatments = self._parse_treatment_details(treatment_narrative, disease)
                
                # Prepare comprehensive response
                return {
                    'narrative': treatment_narrative,
                    'treatments': treatments
                }
            
            except Exception as ai_error:
                print(f"AI treatment discovery failed: {ai_error}")
                # Fallback to knowledge base
                kb_response = self.knowledge_base.get_treatments(disease)
                return {
                    'narrative': kb_response.get('narrative', f"Treatment research for {disease} is ongoing."),
                    'treatments': kb_response.get('treatments', [])
                }
        
        except Exception as e:
            print(f"Error discovering treatment candidates: {e}")
            return {
                'narrative': f"""Treatment Research Status: {disease}

Current medical understanding of {disease} treatments continues to evolve. While specific treatment details require professional medical evaluation, ongoing research shows promising avenues for developing more effective therapeutic approaches.

Key Research Directions:
- Personalized treatment strategies
- Advanced molecular targeting
- Minimally invasive interventions
- Comprehensive patient management

Note: This information is generated for research purposes and should not replace professional medical advice.""",
                'treatments': [{
                    'name': f"Research-Based Treatment for {disease}",
                    'drug_name': f"Personalized Treatment for {disease}",
                    'mechanism': "Comprehensive medical evaluation required",
                    'effectiveness': "Based on individual patient profile",
                    'research_status': "Ongoing Investigation",
                    'side_effects': "Varies by individual patient characteristics"
                }]
            }
    
    def _parse_treatment_details(self, narrative: str, disease: str) -> List[Dict[str, Any]]:
        """
        Parse treatment details from generated narrative
        
        Args:
            narrative (str): Generated treatment narrative
            disease (str): Target disease
        
        Returns:
            List[Dict[str, Any]]: Parsed treatment candidates
        """
        try:
            # Define key treatment information extractors
            treatment_patterns = [
                r'(?:Drug|Treatment|Therapy):\s*([^\n]+)',
                r'Mechanism:\s*([^\n]+)',
                r'Effectiveness:\s*([^\n]+)',
                r'Clinical\s*Trial\s*Status:\s*([^\n]+)',
                r'Side\s*Effects:\s*([^\n]+)'
            ]
            
            # Extract treatment information
            treatments = []
            for i in range(min(3, len(treatment_patterns))):  # Limit to 3 treatments
                treatment = {
                    'name': f"Innovative {disease} Treatment {i+1}",
                    'drug_name': f"Personalized {disease} Therapy",
                    'mechanism': "Advanced molecular targeting",
                    'effectiveness': "Promising research stage",
                    'research_status': "Active Investigation",
                    'side_effects': "Individual assessment required"
                }
                
                # Try to extract specific details from narrative
                for pattern in treatment_patterns:
                    match = re.search(pattern, narrative, re.IGNORECASE)
                    if match:
                        detail = match.group(1).strip()
                        if 'drug' in pattern.lower() or 'treatment' in pattern.lower():
                            treatment['name'] = detail
                        elif 'mechanism' in pattern.lower():
                            treatment['mechanism'] = detail
                        elif 'effectiveness' in pattern.lower():
                            treatment['effectiveness'] = detail
                        elif 'status' in pattern.lower():
                            treatment['research_status'] = detail
                        elif 'side effects' in pattern.lower():
                            treatment['side_effects'] = detail
                
                treatments.append(treatment)
            
            return treatments
        
        except Exception as e:
            print(f"Error parsing treatment details: {e}")
            return [{
                'name': f"Research-Based Treatment for {disease}",
                'drug_name': f"Personalized Treatment for {disease}",
                'mechanism': "Comprehensive medical evaluation required",
                'effectiveness': "Based on individual patient profile",
                'research_status': "Ongoing Investigation",
                'side_effects': "Varies by individual patient characteristics"
            }]
    
    def find_drug_candidates(self, disease: str) -> List[Dict[str, Any]]:
        """
        Find potential drug candidates for a specific disease
        
        Args:
            disease (str): Target disease for drug discovery
        
        Returns:
            List[Dict[str, Any]]: Potential drug candidates
        """
        prompt = f"""
        Analyze potential drug candidates for {disease}. 
        For each candidate, provide:
        1. Drug name or molecular structure
        2. Mechanism of action
        3. Potential effectiveness
        4. Current research status
        5. Potential side effects
        
        Prioritize novel approaches and potential repurposing of existing drugs.
        """
        
        drug_candidates_text = self.llama_assistant._generate_llama_response(prompt)
        
        # Basic parsing of drug candidates
        candidates = []
        for line in drug_candidates_text.split('\n'):
            if line.strip():
                candidates.append({
                    'description': line.strip()
                })
        
        return candidates
    
    def analyze_molecular_interactions(self, molecular_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze molecular interactions and drug compatibility
        
        Args:
            molecular_data (pd.DataFrame): Molecular structure and interaction data
        
        Returns:
            Dict[str, Any]: Analysis of molecular interactions
        """
        try:
            # Convert molecular data to dictionary for Llama processing
            molecular_dict = molecular_data.to_dict(orient='records')
            
            prompt = f"""
            Analyze the following molecular interaction data:
            {molecular_dict}
            
            Provide insights on:
            1. Potential drug-target interactions
            2. Molecular compatibility
            3. Predicted binding affinity
            4. Potential side effects or conflicts
            """
            
            return self.llama_assistant._generate_llama_response(prompt)
        except Exception as e:
            print(f"Error analyzing molecular interactions: {e}")
            return {"error": str(e)}

    def predict_disease_outcomes(self, patient_data_file: Union[str, bytes, pd.DataFrame], max_patients: int = None) -> pd.DataFrame:
        """
        Predict disease outcomes based on patient data with optimized processing
        
        Args:
            patient_data_file (Union[str, bytes, pd.DataFrame]): File or DataFrame containing patient data
            max_patients (int, optional): Maximum number of patients to process. If None, process all patients.
        
        Returns:
            pd.DataFrame: Predictions for each patient
        """
        try:
            # Load patient data
            if isinstance(patient_data_file, pd.DataFrame):
                patient_data = patient_data_file
            elif isinstance(patient_data_file, bytes):
                patient_data = pd.read_csv(io.BytesIO(patient_data_file))
            elif isinstance(patient_data_file, str):
                patient_data = pd.read_csv(patient_data_file)
            else:
                raise ValueError("Invalid input type for patient data")
            
            # Ensure required columns exist
            required_columns = ['patient_id', 'age', 'gender', 'symptoms']
            missing_columns = [col for col in required_columns if col not in patient_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Limit number of patients if specified
            if max_patients is not None:
                patient_data = patient_data.head(max_patients)
            
            # Prepare predictions list with default values
            predictions = []
            
            # Generate predictions with a fallback mechanism
            for _, patient in patient_data.iterrows():
                try:
                    # Calculate risk score first (quick operation)
                    risk_score = self._calculate_risk_score(patient)
                    
                    # Prepare a quick, template-based prediction
                    predicted_outcome = self._generate_quick_prediction(patient, risk_score)
                    
                    # Prepare prediction entry
                    prediction_entry = {
                        'patient_id': patient['patient_id'],
                        'age': patient['age'],
                        'gender': patient['gender'],
                        'symptoms': patient['symptoms'],
                        'risk_score': risk_score,
                        'predicted_outcome': predicted_outcome
                    }
                    
                    predictions.append(prediction_entry)
                
                except Exception as pred_error:
                    print(f"Error generating prediction for patient {patient.get('patient_id', 'Unknown')}: {pred_error}")
                    # Fallback prediction
                    predictions.append({
                        'patient_id': patient.get('patient_id', 'N/A'),
                        'age': patient.get('age', 0),
                        'gender': patient.get('gender', 'N/A'),
                        'symptoms': patient.get('symptoms', 'No symptoms recorded'),
                        'risk_score': 50,  # Default risk
                        'predicted_outcome': "Unable to generate detailed prediction. Recommend comprehensive medical evaluation."
                    })
            
            # Convert predictions to DataFrame
            predictions_df = pd.DataFrame(predictions)
            
            return predictions_df
        
        except Exception as e:
            print(f"Error in predicting disease outcomes: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def _generate_quick_prediction(self, patient: pd.Series, risk_score: int) -> str:
        """
        Generate a detailed, personalized prediction
        
        Args:
            patient (pd.Series): Patient data
            risk_score (int): Calculated risk score
        
        Returns:
            str: Prediction narrative
        """
        try:
            # Basic patient information
            symptoms = str(patient['symptoms']).lower()
            age = patient['age']
            gender = patient['gender']
            
            # Risk level categorization
            if risk_score <= 30:
                risk_level = "Low"
                risk_color = "green"
                risk_description = "minimal health concerns requiring routine monitoring"
            elif 30 < risk_score <= 60:
                risk_level = "Moderate"
                risk_color = "yellow"
                risk_description = "potential health risks that require proactive management"
            else:
                risk_level = "High"
                risk_color = "red"
                risk_description = "significant health risks requiring immediate medical intervention"
            
            # Comprehensive symptom insights
            symptom_insights = {
                'chest pain': {
                    'concern': "Cardiovascular Health",
                    'detailed_insight': "Potential indicators of heart-related issues, such as coronary artery disease or cardiac stress",
                    'recommendations': [
                        "Immediate cardiac evaluation",
                        "ECG and stress test",
                        "Cholesterol and blood pressure monitoring"
                    ]
                },
                'fatigue': {
                    'concern': "Metabolic and Hormonal Balance",
                    'detailed_insight': "Possible signs of thyroid dysfunction, chronic fatigue syndrome, or nutritional deficiencies",
                    'recommendations': [
                        "Comprehensive metabolic panel",
                        "Thyroid function tests",
                        "Vitamin and mineral level assessment"
                    ]
                },
                'joint pain': {
                    'concern': "Inflammatory Conditions",
                    'detailed_insight': "Potential markers of autoimmune disorders, arthritis, or systemic inflammation",
                    'recommendations': [
                        "Rheumatology consultation",
                        "Inflammatory marker tests",
                        "Physical therapy evaluation"
                    ]
                },
                'shortness of breath': {
                    'concern': "Respiratory and Cardiac Function",
                    'detailed_insight': "Possible indications of pulmonary issues, heart conditions, or respiratory infections",
                    'recommendations': [
                        "Pulmonary function tests",
                        "Chest X-ray",
                        "Cardiovascular screening"
                    ]
                },
                'headache': {
                    'concern': "Neurological and Stress Indicators",
                    'detailed_insight': "Potential signs of tension, migraines, or underlying neurological conditions",
                    'recommendations': [
                        "Neurological consultation",
                        "Stress management assessment",
                        "Sleep pattern evaluation"
                    ]
                },
                'dizziness': {
                    'concern': "Neurological and Inner Ear Health",
                    'detailed_insight': "Possible vestibular disorders, blood pressure irregularities, or neurological imbalances",
                    'recommendations': [
                        "Vestibular function test",
                        "Blood pressure monitoring",
                        "Neurological screening"
                    ]
                }
            }
            
            # Find most relevant symptom insight
            primary_symptom_insight = None
            for symptom, insight in symptom_insights.items():
                if symptom in symptoms:
                    primary_symptom_insight = insight
                    break
            
            # Default insight if no specific symptom match
            if primary_symptom_insight is None:
                primary_symptom_insight = {
                    'concern': "General Health Assessment",
                    'detailed_insight': "Comprehensive health evaluation recommended to identify potential underlying conditions",
                    'recommendations': [
                        "Full medical check-up",
                        "Comprehensive blood panel",
                        "Lifestyle and nutrition consultation"
                    ]
                }
            
            # Construct detailed prediction
            prediction_summary = (
                f"Risk Profile: {risk_level} Risk ({risk_score}/100) | "
                f"Primary Concern: {primary_symptom_insight['concern']} | "
                f"Detailed Insight: {primary_symptom_insight['detailed_insight']} | "
                f"Key Recommendations: {' | '.join(primary_symptom_insight['recommendations'])}"
            )
            
            return prediction_summary
        
        except Exception as e:
            print(f"Error generating quick prediction: {e}")
            return "Comprehensive health assessment recommended. Consult healthcare professional for personalized insights."
    
    def _calculate_risk_score(self, patient: pd.Series) -> int:
        """
        Calculate a risk score based on patient characteristics
        
        Args:
            patient (pd.Series): Patient data row
        
        Returns:
            int: Calculated risk score (0-100)
        """
        try:
            # Base risk calculation logic
            base_risk = 50  # Starting at medium risk
            
            # Age risk factor
            age = patient['age']
            if age < 30:
                base_risk -= 20
            elif 30 <= age < 50:
                base_risk += 10
            elif 50 <= age < 70:
                base_risk += 30
            else:
                base_risk += 40
            
            # Symptom risk factors
            symptoms = str(patient['symptoms']).lower()
            risk_symptoms = {
                'chest pain': 25,
                'irregular heartbeat': 30,
                'fatigue': 15,
                'joint pain': 20,
                'shortness of breath': 25,
                'dizziness': 15,
                'headache': 10,
                'swelling': 20,
                'weakness': 25
            }
            
            # Calculate additional risk from symptoms
            for symptom, risk_value in risk_symptoms.items():
                if symptom in symptoms:
                    base_risk += risk_value
            
            # Ensure risk is between 0 and 100
            return max(0, min(100, base_risk))
        
        except Exception as e:
            print(f"Error calculating risk score: {e}")
            return 50  # Default risk score

    def search_patients(self, patient_data: pd.DataFrame, search_criteria: Dict[str, Any]) -> pd.DataFrame:
        """
        Search and filter patients based on specific criteria
        
        Args:
            patient_data (pd.DataFrame): DataFrame containing patient predictions
            search_criteria (Dict[str, Any]): Search parameters
        
        Returns:
            pd.DataFrame: Filtered patient data
        """
        try:
            # Validate input
            if patient_data is None or patient_data.empty:
                return pd.DataFrame()
            
            # Create a copy to avoid modifying original data
            filtered_data = patient_data.copy()
            
            # Apply filters based on search criteria
            if search_criteria:
                # Patient ID filter (handle both string and integer inputs)
                if search_criteria.get('patient_id') is not None:
                    patient_id = search_criteria['patient_id']
                    filtered_data = filtered_data[
                        filtered_data['patient_id'].astype(str) == str(patient_id)
                    ]
                
                # Age range filter
                if search_criteria.get('min_age') is not None:
                    filtered_data = filtered_data[filtered_data['age'] >= search_criteria['min_age']]
                if search_criteria.get('max_age') is not None:
                    filtered_data = filtered_data[filtered_data['age'] <= search_criteria['max_age']]
                
                # Gender filter (case-insensitive)
                if search_criteria.get('gender'):
                    filtered_data = filtered_data[
                        filtered_data['gender'].str.upper() == search_criteria['gender'].upper()
                    ]
                
                # Symptoms filter (case-insensitive, partial match)
                if search_criteria.get('symptoms'):
                    filtered_data = filtered_data[
                        filtered_data['symptoms'].str.contains(
                            search_criteria['symptoms'], 
                            case=False, 
                            na=False
                        )
                    ]
                
                # Risk score filter
                if search_criteria.get('min_risk_score') is not None:
                    filtered_data = filtered_data[
                        filtered_data['risk_score'] >= search_criteria['min_risk_score']
                    ]
                if search_criteria.get('max_risk_score') is not None:
                    filtered_data = filtered_data[
                        filtered_data['risk_score'] <= search_criteria['max_risk_score']
                    ]
            
            return filtered_data
        
        except Exception as e:
            print(f"Error searching patients: {e}")
            return pd.DataFrame()

    def _generate_generic_treatment(self, disease: str) -> Dict[str, Any]:
        """
        Generate a generic treatment response when no specific data is available
        
        Args:
            disease (str): Disease name
            
        Returns:
            Dict[str, Any]: Treatment information with narrative and structured data
        """
        try:
            # Get treatment innovation from Llama
            treatment_info = self.llama_assistant.generate_treatment_innovation(disease)
            if isinstance(treatment_info, dict) and 'narrative' in treatment_info:
                return treatment_info
            
            # If we got a string instead of a dict (legacy format), wrap it
            if isinstance(treatment_info, str):
                return {
                    'narrative': treatment_info,
                    'treatments': [{
                        "Treatment Name": f"Research Overview for {disease}",
                        "Mechanism of Action": "Based on current medical research",
                        "Potential Effectiveness": "Varies by approach",
                        "Research Status": "Active investigation",
                        "Potential Side Effects": "Treatment-specific",
                        "Patient Populations": f"Individuals with {disease}",
                        "Recommendation": "Consult healthcare professionals"
                    }]
                }
            
            # Fallback response if something went wrong
            return {
                'narrative': f"""Treatment Research Status: {disease}

The current medical understanding of {disease} treatments continues to evolve. While specific treatment details require professional medical evaluation, ongoing research in this field shows promise for developing more effective therapeutic approaches.

Medical researchers are actively investigating various treatment strategies, though all potential treatments must undergo rigorous clinical validation before becoming standard practice. For the most appropriate treatment guidance, please consult with healthcare providers who can evaluate your specific medical situation.""",
                'treatments': [{
                    "Treatment Name": f"Personalized Treatment for {disease}",
                    "Mechanism of Action": "Requires medical evaluation",
                    "Potential Effectiveness": "Individual assessment needed",
                    "Research Status": "Ongoing investigation",
                    "Potential Side Effects": "Treatment-specific",
                    "Patient Populations": f"Individuals with {disease}",
                    "Recommendation": "Consult healthcare professionals"
                }]
            }
            
        except Exception as e:
            print(f"Error generating generic treatment: {e}")
            return {
                'narrative': f"""Treatment Research Status: {disease}

We apologize, but we encountered an issue while generating treatment information. While we continue to research treatments for {disease}, we recommend consulting with healthcare professionals for the most current and accurate treatment guidance.

This temporary technical limitation does not reflect on the availability or quality of actual medical treatments. Please speak with qualified medical providers who can provide personalized treatment recommendations.""",
                'treatments': [{
                    "Treatment Name": "Error in Treatment Generation",
                    "Mechanism of Action": "Technical error occurred",
                    "Potential Effectiveness": "Cannot be determined",
                    "Research Status": "Error in processing",
                    "Potential Side Effects": "Not available",
                    "Patient Populations": "Not available",
                    "Recommendation": "Please try again or consult healthcare professionals"
                }]
            }

    def download_literature_review(self, disease: str) -> str:
        """
        Generate and download a comprehensive literature review for a specific disease in .txt format
        
        Args:
            disease (str): Target disease for literature review
        
        Returns:
            str: Path to the downloaded literature review file
        """
        # Generate comprehensive literature review
        literature_content = self.discover_drug_candidates(disease)
        
        # Prepare file path
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'literature_reviews')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with .txt extension
        filename = f"{disease.lower().replace(' ', '_')}_literature_review.txt"
        file_path = os.path.join(output_dir, filename)
        
        # Write content to file in plain text format
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"LITERATURE REVIEW: {disease.upper()}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. CURRENT MEDICAL UNDERSTANDING\n")
            f.write("-" * 40 + "\n")
            for key, value in literature_content.get('medical_understanding', {}).items():
                f.write(f"{key.upper()}: {value}\n")
            
            f.write("\n2. TREATMENT MODALITIES\n")
            f.write("-" * 40 + "\n")
            for key, value in literature_content.get('treatment_modalities', {}).items():
                f.write(f"{key.upper()}: {value}\n")
            
            f.write("\n3. DETAILED TREATMENT INSIGHTS\n")
            f.write("-" * 40 + "\n")
            for key, value in literature_content.get('treatment_insights', {}).items():
                f.write(f"{key.upper()}: {value}\n")
        
        return file_path
