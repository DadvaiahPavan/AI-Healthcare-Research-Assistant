import os
from typing import Dict, List, Any
import json
import requests
from dotenv import load_dotenv
import logging
from llama_model import LlamaResearchAssistant

class LlamaAssistant:
    def __init__(self, api_key: str = None):
        """
        Initialize Llama Assistant with optional API configuration
        
        Args:
            api_key (str, optional): API key for external medical knowledge service
        """
        load_dotenv()
        self.api_key = api_key or os.getenv('MEDICAL_API_KEY')
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Fallback medical knowledge base
        self.medical_knowledge = {
            "pneumonia": {
                "standard_treatments": [
                    {
                        "name": "Antibiotics (Bacterial Pneumonia)",
                        "mechanism": "Target and eliminate bacterial infection in lungs",
                        "first_line_drugs": ["Azithromycin", "Amoxicillin", "Doxycycline"],
                        "effectiveness": "80-90% for bacterial pneumonia"
                    },
                    {
                        "name": "Antiviral Medications (Viral Pneumonia)",
                        "mechanism": "Inhibit viral replication in respiratory tract",
                        "first_line_drugs": ["Oseltamivir", "Zanamivir"],
                        "effectiveness": "70-80% for influenza-related pneumonia"
                    }
                ],
                "research_status": "Well-established treatment protocols",
                "patient_considerations": [
                    "Age",
                    "Underlying health conditions",
                    "Immunocompromised status"
                ]
            }
        }
    
    def generate_medical_insights(self, prompt: str) -> str:
        """
        Generate medical insights using a combination of AI and predefined knowledge
        
        Args:
            prompt (str): Input medical query
        
        Returns:
            str: Detailed medical insights
        """
        try:
            # Extract key disease from prompt
            disease = self._extract_disease(prompt)
            
            # Check predefined knowledge base first
            if disease.lower() in self.medical_knowledge:
                return self._format_medical_knowledge(self.medical_knowledge[disease.lower()])
            
            # If no predefined knowledge, use external API or fallback
            insights = self._call_medical_api(prompt)
            
            return insights or self._generate_fallback_response(disease)
        
        except Exception as e:
            return f"Error generating medical insights: {str(e)}"
    
    def _extract_disease(self, prompt: str) -> str:
        """
        Extract disease name from prompt
        
        Args:
            prompt (str): Input medical query
        
        Returns:
            str: Extracted disease name
        """
        # Simple extraction logic
        words = prompt.split()
        for word in words:
            if len(word) > 3 and word.lower() not in ['the', 'and', 'for']:
                return word
        return "Unknown Disease"
    
    def _call_medical_api(self, prompt: str) -> str:
        """
        Call external medical knowledge API
        
        Args:
            prompt (str): Input medical query
        
        Returns:
            str: API response or None
        """
        if not self.api_key:
            return None
        
        try:
            response = requests.post(
                "https://medical-insights-api.example.com/generate",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={"prompt": prompt}
            )
            
            if response.status_code == 200:
                return response.json().get('insights', '')
            return None
        
        except Exception:
            return None
    
    def _format_medical_knowledge(self, knowledge: Dict[str, Any]) -> str:
        """
        Format predefined medical knowledge into a readable string
        
        Args:
            knowledge (Dict[str, Any]): Predefined medical knowledge
        
        Returns:
            str: Formatted medical insights
        """
        formatted_text = "Treatment Options:\n\n"
        
        for treatment in knowledge.get('standard_treatments', []):
            formatted_text += f"Drug/Treatment Name: {treatment['name']}\n"
            formatted_text += f"Mechanism of Action: {treatment['mechanism']}\n"
            formatted_text += f"First-Line Drugs: {', '.join(treatment['first_line_drugs'])}\n"
            formatted_text += f"Effectiveness: {treatment['effectiveness']}\n\n"
        
        formatted_text += f"Research Status: {knowledge.get('research_status', 'Ongoing')}\n"
        formatted_text += "Patient Considerations:\n"
        for consideration in knowledge.get('patient_considerations', []):
            formatted_text += f"- {consideration}\n"
        
        return formatted_text
    
    def _generate_fallback_response(self, disease: str) -> str:
        """
        Generate a generic fallback response when no specific information is available
        
        Args:
            disease (str): Disease name
        
        Returns:
            str: Fallback medical response
        """
        return f"""
        Drug/Treatment Name: Personalized Treatment Required
        Mechanism of Action: Comprehensive medical evaluation needed
        Clinical Evidence: Limited information available for {disease}
        Effectiveness: Cannot be determined without specific medical assessment
        Side Effects: N/A
        Patient Populations: Requires individual medical consultation
        
        Recommendation: Consult a healthcare professional for personalized treatment options.
        """

    def predict_drug_candidates(self, prompt: str) -> str:
        """
        Legacy method for backward compatibility
        
        Args:
            prompt (str): Input medical query
        
        Returns:
            str: Generated drug candidates information
        """
        return self.generate_medical_insights(prompt)

    def generate_treatment_innovation(self, disease: str) -> Dict[str, Any]:
        """
        Generate innovative treatment narratives for a specific disease.
        
        Args:
            disease (str): The disease to generate treatments for
        
        Returns:
            Dict[str, Any]: Dictionary containing narrative and treatments
        """
        try:
            # Initialize the research assistant
            research_assistant = LlamaResearchAssistant(
                section='treatment_innovation',
                model_name='google/gemini-2.0-flash-exp'
            )

            # Construct a narrative-focused prompt
            prompt = f"""Generate a comprehensive medical narrative about innovative treatments for {disease}.

REQUIRED FORMAT:
Title: "Innovative Treatments for {disease}: A Glimpse into the Future"

1. Begin with an introduction about {disease}, its impact on patients, and current challenges
2. Describe 4-6 innovative treatment approaches, focusing on:
   - Recent medical breakthroughs
   - Novel therapeutic strategies
   - Emerging technologies
   - Promising research directions
3. End with future outlook and medical guidance

STRICT REQUIREMENTS:
- Write as a FLOWING NARRATIVE
- NO bullet points or numbered lists
- NO sections labeled "Treatment" or "Solution"
- NO technical headers or labels
- Use professional but accessible medical language
- Focus on real medical innovations and research
- Maximum length: 500 words

Example style:
"{disease} treatment has entered an exciting era of innovation, with several groundbreaking approaches showing promise. At the forefront is [describe first innovation], which represents a significant advance in... This development is complemented by [describe second innovation], offering new hope for patients... Another promising direction involves [describe third innovation]..."

Generate a cohesive, hopeful narrative about treating {disease}."""

            # Generate the narrative
            treatment_narrative = research_assistant._generate_llama_response(prompt)

            # Post-process to ensure narrative format
            if "Solution" in treatment_narrative or "Treatment Name:" in treatment_narrative:
                # If we got a solution-style response, try one more time with an even stricter prompt
                strict_prompt = f"""Create a SINGLE FLOWING NARRATIVE about innovative treatments for {disease}.

ABSOLUTELY NO:
- Numbered solutions
- Treatment names as headers
- Bullet points
- Technical sections
- Lists of any kind

Instead, write a cohesive story about how new medical innovations are transforming the treatment of {disease}. Begin with the current state of treatment, flow through the most promising innovations, and end with a message of hope for the future.

Remember: This must be ONE CONTINUOUS NARRATIVE with no sections or breaks."""

                treatment_narrative = research_assistant._generate_llama_response(strict_prompt)

            # Clean up the text
            import re
            # Remove any remaining headers or labels
            treatment_narrative = re.sub(r'^[A-Za-z\s]+:', '', treatment_narrative, flags=re.MULTILINE)
            treatment_narrative = re.sub(r'^\d+\.', '', treatment_narrative, flags=re.MULTILINE)
            
            # Ensure proper title
            if not treatment_narrative.startswith(f"Innovative Treatments for {disease}"):
                treatment_narrative = f"Innovative Treatments for {disease}: A Glimpse into the Future\n\n{treatment_narrative}"

            # Return both narrative and treatments
            return {
                'narrative': treatment_narrative,
                'treatments': [{
                    "Treatment Name": f"AI-Generated Research Overview for {disease}",
                    "Mechanism of Action": "Based on latest medical research and innovations",
                    "Potential Effectiveness": "Varies by treatment approach",
                    "Research Status": "Active investigation",
                    "Potential Side Effects": "Treatment-specific considerations required",
                    "Patient Populations": f"Individuals diagnosed with {disease}",
                    "Recommendation": "Consult healthcare professionals for personalized guidance"
                }]
            }

        except Exception as e:
            self.logger.error(f"Treatment innovation generation failed: {e}")
            return {
                'narrative': f"""Innovative Treatments for {disease}: Research Insights

Medical research into {disease} continues to evolve, with ongoing investigations into innovative treatment approaches. While specific details about emerging treatments require careful medical validation, the scientific community remains committed to advancing our understanding and treatment options for this condition.

The current research landscape shows promise in several areas, though it's important to note that all potential treatments must undergo rigorous clinical evaluation before becoming available to patients. Medical professionals are actively studying various therapeutic approaches, each tailored to address different aspects of {disease}.

For the most current and personalized treatment information, we strongly recommend consulting with healthcare professionals who can provide guidance based on your specific medical situation and the latest clinical research.

This field continues to advance, and new developments emerge regularly. Stay informed through reputable medical sources and maintain open communication with your healthcare team.

Disclaimer: This information is generated by an AI research assistant and should not be considered medical advice. Always consult qualified healthcare professionals for personalized medical guidance.""",
                'treatments': [{
                    "Treatment Name": f"Research Overview for {disease}",
                    "Mechanism of Action": "Requires comprehensive medical evaluation",
                    "Potential Effectiveness": "Varies by treatment approach",
                    "Research Status": "Active investigation",
                    "Potential Side Effects": "Treatment-specific considerations required",
                    "Patient Populations": f"Individuals diagnosed with {disease}",
                    "Recommendation": "Consult healthcare professionals for personalized guidance"
                }]
            }

    def generate_innovative_treatments(self, disease_name):
        """
        Generate a comprehensive narrative of innovative treatments for a specific disease
        
        Args:
            disease_name (str): Name of the disease to research
        
        Returns:
            str: Detailed narrative of innovative treatments
        """
        try:
            # Initialize Gemini-based research assistant for treatment innovation
            gemini_assistant = LlamaResearchAssistant(
                section='treatment_innovation', 
                model_name='google/gemini-2.0-flash-exp'
            )
            
            # Comprehensive prompt for innovative treatments narrative
            prompt = f"""
            Provide a precise, comprehensive overview of {disease_name} in a single, 
            coherent paragraph. Focus on delivering clear, actionable medical information.

            Your response MUST:
            - Be exactly ONE paragraph
            - Cover definition, causes, symptoms, and treatment
            - Use clear, accessible medical language
            - Avoid technical jargon
            - Provide practical, informative insights

            Strictly limit your response to 250-350 words.
            Do NOT generate multiple solutions or treatment lists.
            """
            
            # Generate response using Gemini model
            response = gemini_assistant._generate_llama_response(prompt)
            
            return response
        
        except Exception as e:
            self.logger.error(f"Disease overview generation failed for {disease_name}: {e}")
            return f"Unable to generate overview for {disease_name}. Please consult a medical professional."

    def generate_disease_overview(self, disease_name):
        """
        Generate a comprehensive, concise overview of a specific disease
        
        Args:
            disease_name (str): Name of the disease to research
        
        Returns:
            str: Detailed yet concise disease overview
        """
        try:
            # Initialize Gemini-based research assistant 
            gemini_assistant = LlamaResearchAssistant(
                section='treatment_innovation', 
                model_name='google/gemini-2.0-flash-exp'
            )
            
            # Predefined comprehensive disease overview prompt
            prompt = f"""
            Provide a precise, comprehensive overview of {disease_name} in a single, 
            coherent paragraph. Focus on delivering clear, actionable medical information.

            Your response MUST:
            - Be exactly ONE paragraph
            - Cover definition, causes, symptoms, and treatment
            - Use clear, accessible medical language
            - Avoid technical jargon
            - Provide practical, informative insights

            Strictly limit your response to 250-350 words.
            Do NOT generate multiple solutions or treatment lists.
            """
            
            # Generate response using Gemini model
            response = gemini_assistant._generate_llama_response(prompt)
            
            return response
        
        except Exception as e:
            self.logger.error(f"Disease overview generation failed for {disease_name}: {e}")
            return f"Unable to generate overview for {disease_name}. Please consult a medical professional."
