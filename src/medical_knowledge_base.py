import json
import os
from typing import Dict, List, Any, Union
import requests
import logging

class MedicalKnowledgeBase:
    """
    Dynamic medical knowledge base with expandable treatment insights
    """
    
    def __init__(self, knowledge_file: str = None):
        """
        Initialize Medical Knowledge Base
        
        Args:
            knowledge_file (str, optional): Path to custom knowledge file
        """
        self.DISEASE_TREATMENTS = {
            "chronic kidney disease": [
                {
                    "Treatment Name": "ACE Inhibitors",
                    "Mechanism of Action": "Reduce blood pressure and slow kidney damage",
                    "Potential Effectiveness": "60-70% in slowing disease progression",
                    "Research Status": "Well-established, FDA-approved treatments",
                    "Potential Side Effects": "Dry cough, dizziness, increased potassium levels",
                    "Patient Populations": "Patients with hypertension and early-stage kidney disease"
                }
            ],
            "pneumonia": [
                {
                    "Treatment Name": "Antibiotics",
                    "Mechanism of Action": "Target and eliminate bacterial infection in lungs",
                    "Potential Effectiveness": "80-90% for bacterial pneumonia",
                    "Research Status": "Standard of care, well-established treatment protocols",
                    "Potential Side Effects": "Gastrointestinal issues, potential allergic reactions",
                    "Patient Populations": "Patients with confirmed bacterial pneumonia"
                }
            ]
        }
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Load custom knowledge if file provided
        if knowledge_file and os.path.exists(knowledge_file):
            self._load_custom_knowledge(knowledge_file)
    
    def _load_custom_knowledge(self, knowledge_file: str):
        """
        Load custom medical knowledge from a file
        
        Args:
            knowledge_file (str): Path to knowledge JSON file
        """
        try:
            with open(knowledge_file, 'r') as f:
                custom_knowledge = json.load(f)
                self.DISEASE_TREATMENTS.update(custom_knowledge)
        except Exception as e:
            print(f"Error loading custom knowledge: {e}")
    
    def _fetch_medical_insights(self, disease: str) -> List[Dict[str, Any]]:
        """
        Dynamically fetch medical insights from external medical databases
        
        Args:
            disease (str): Target disease
        
        Returns:
            List[Dict[str, Any]]: Medical treatment insights
        """
        try:
            # Example: Using a hypothetical medical research API
            api_url = "https://medical-research-api.example.com/treatments"
            response = requests.get(api_url, params={"disease": disease})
            
            if response.status_code == 200:
                return response.json().get('treatments', [])
            return []
        
        except Exception as e:
            print(f"Error fetching medical insights: {e}")
            return []
    
    def get_treatments(self, disease: str) -> Union[str, List[Dict[str, Any]]]:
        """
        Get treatment information for a specific disease
        
        Args:
            disease (str): Name of the disease
            
        Returns:
            Union[str, List[Dict[str, Any]]]: Narrative description or list of treatment dictionaries
        """
        try:
            # First try to get real medical data
            treatments = self.query_medical_database(disease)
            if treatments:
                # If treatments are found, return both narrative and structured data
                return {
                    'narrative': self._format_treatment_narrative(disease, treatments),
                    'treatments': treatments
                }
            
            # If no specific data found, generate a research-based narrative
            return {
                'narrative': f"""Innovative Treatments for {disease}: Current Research Landscape

{disease} remains an active area of medical research, with scientists and healthcare professionals exploring multiple therapeutic approaches. The medical community continues to investigate both traditional and innovative treatment strategies, focusing on improving patient outcomes and quality of life.

Current research directions include targeted therapies, novel drug delivery systems, and personalized treatment approaches. These investigations aim to develop more effective and patient-specific interventions, though all potential treatments must undergo rigorous clinical validation before becoming standard practice.

For the most accurate and up-to-date treatment information specific to your case, we strongly recommend consulting with qualified healthcare professionals who can provide personalized medical guidance based on your individual health profile and the latest clinical evidence.

Note: This information is generated based on current medical research trends. Treatment decisions should always be made in consultation with healthcare providers.""",
                'treatments': [{
                    "Treatment Name": f"Personalized Treatment for {disease}",
                    "Mechanism of Action": "Requires comprehensive medical evaluation",
                    "Potential Effectiveness": "Cannot be determined without specific medical assessment",
                    "Research Status": "Needs further investigation",
                    "Potential Side Effects": "Varies based on individual patient characteristics",
                    "Patient Populations": "Individuals diagnosed with specific condition",
                    "Recommendation": "Consult a healthcare professional for personalized medical advice"
                }]
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving treatments: {e}")
            return {
                'narrative': f"""Treatment Research Status: {disease}

The current medical understanding of {disease} treatments continues to evolve. While specific treatment details require professional medical evaluation, ongoing research in this field shows promise for developing more effective therapeutic approaches.

Please consult healthcare professionals for personalized medical advice and treatment options.""",
                'treatments': [{
                    "Treatment Name": f"Personalized Treatment for {disease}",
                    "Mechanism of Action": "Requires comprehensive medical evaluation",
                    "Potential Effectiveness": "Cannot be determined without specific medical assessment",
                    "Research Status": "Needs further investigation",
                    "Potential Side Effects": "Varies based on individual patient characteristics",
                    "Patient Populations": "Individuals diagnosed with specific condition",
                    "Recommendation": "Consult a healthcare professional for personalized medical advice"
                }]
            }

    def _format_treatment_narrative(self, disease: str, treatments: list) -> str:
        """
        Format treatment data into a flowing narrative
        
        Args:
            disease (str): Name of the disease
            treatments (list): List of treatment information
            
        Returns:
            str: Formatted narrative
        """
        narrative = f"""Innovative Treatments for {disease}: Evidence-Based Approaches

Medical research has identified several promising treatment approaches for {disease}, each supported by clinical evidence and ongoing investigation. """

        # Integrate treatment information into a flowing narrative
        for treatment in treatments:
            # Safely access treatment details with default values
            description = treatment.get('description', 'Research continues')
            effectiveness = treatment.get('effectiveness', 'under evaluation')
            status = treatment.get('status', 'being investigated')
            
            narrative += f"""

{description} This therapeutic approach has shown {effectiveness} effectiveness and is currently {status}. """

        narrative += """

As medical science advances, these treatment options continue to be refined and improved. For personalized medical advice and treatment recommendations, please consult with qualified healthcare professionals who can evaluate your specific situation."""

        return narrative

    def add_treatment(self, disease: str, treatment: Dict[str, Any]):
        """
        Add a new treatment to the knowledge base
        
        Args:
            disease (str): Target disease
            treatment (Dict[str, Any]): Treatment details
        """
        disease = disease.lower()
        if disease not in self.DISEASE_TREATMENTS:
            self.DISEASE_TREATMENTS[disease] = []
        
        self.DISEASE_TREATMENTS[disease].append(treatment)

    def query_medical_database(self, disease: str) -> List[Dict[str, Any]]:
        """
        Query medical database for treatment information
        
        Args:
            disease (str): Target disease
        
        Returns:
            List[Dict[str, Any]]: Treatment information
        """
        # Normalize disease name
        disease = disease.lower()
        
        # Check predefined knowledge
        for key, treatments in self.DISEASE_TREATMENTS.items():
            if disease in key or key in disease:
                return treatments
        
        # Attempt to fetch from external sources
        external_treatments = self._fetch_medical_insights(disease)
        if external_treatments:
            return external_treatments
        
        # Return empty list if no information found
        return []
