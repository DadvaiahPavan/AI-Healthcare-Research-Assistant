import os
import json
import logging
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import datetime
import uuid
import time

class LlamaResearchAssistant:
    def __init__(self, 
                 section: Optional[str] = None,
                 model_name: Optional[str] = None, 
                 api_key: Optional[str] = None,
                 api_provider: str = "openrouter"):
        """
        Initialize Research Assistant with section-specific model configuration
        
        Args:
            section (str, optional): Specific research section 
            model_name (str, optional): Explicit model override
            api_key (str, optional): Explicit API key if not in environment
            api_provider (str): API provider for model access
        """
        # Load environment variables
        load_dotenv()
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Section-specific model and API key mapping
        SECTION_CONFIGS = {
            "clinical_trial": {
                "model": os.getenv('CLINICAL_TRIAL_MODEL', 'meta-llama/llama-3.1-8b-instruct'),
                "api_key": os.getenv('OPENROUTER_API_KEY')
            },
            "disease_prediction": {
                "model": os.getenv('DISEASE_PREDICTION_MODEL', 'meta-llama/llama-3.1-8b-instruct'),
                "api_key": os.getenv('OPENROUTER_API_KEY')
            },
            "literature_review": {
                "model": os.getenv('LITERATURE_REVIEW_MODEL', 'meta-llama/llama-3.1-8b-instruct'),
                "api_key": os.getenv('OPENROUTER_API_KEY')
            },
            "treatment_innovation": {
                "model": os.getenv('TREATMENT_INNOVATION_MODEL', 'google/gemini-2.0-flash-exp:free'),
                "api_key": os.getenv('OPENROUTER_GEMINI_KEY')
            }
        }
        
        # Determine model and API key
        if section and section.lower() in SECTION_CONFIGS:
            # Use section-specific configuration
            section_config = SECTION_CONFIGS[section.lower()]
            self.current_model = model_name or section_config['model']
            self.api_key = api_key or section_config['api_key']
        else:
            # Fallback to default configuration
            self.current_model = model_name or os.getenv('DEFAULT_MODEL', 'meta-llama/llama-3.1-8b-instruct')
            self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        
        # Validate API key
        if not self.api_key:
            raise ValueError(f"No API key found for section: {section}")
        
        # API Configuration
        self.api_base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Model-specific configurations
        self.model_config = {
            "meta-llama/llama-3.1-8b-instruct": {
                "medical_system_prompt": """You are an advanced medical research AI assistant specializing in comprehensive medical analysis. 
Provide scientifically rigorous, evidence-based insights with precision and clarity.""",
                "temperature": 0.3,
                "max_tokens": 2048,
                "top_p": 0.7,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.1
            },
            "google/gemini-2.0-flash-exp:free": {
                "medical_system_prompt": """You are an advanced medical research AI specializing in Treatment Innovation Tracking. 
Your mission is to provide cutting-edge, scientifically rigorous insights into medical treatment innovations.""",
                "temperature": 0.3,
                "max_tokens": 2048,
                "top_p": 0.7,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.1
            }
        }
        
        # Ensure model is configured
        if self.current_model not in self.model_config:
            # Dynamically add basic configuration for unknown models
            self.model_config[self.current_model] = {
                "medical_system_prompt": "You are an advanced medical research AI assistant.",
                "temperature": 0.3,
                "max_tokens": 2048,
                "top_p": 0.7,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.1
            }
        
        # Logging configuration details
        self.logger.info(f"Initialized with Model: {self.current_model}")
        self.logger.info(f"Section: {section or 'General'}")
        
        # Initialize error tracking
        self.error_log = []

    def _select_fallback_model(self, current_model: str) -> str:
        """
        Intelligently select a fallback model based on current model's failure
        
        Args:
            current_model (str): Model that failed
        
        Returns:
            str: Alternative model to use
        """
        try:
            # Find index of current model
            current_index = list(self.model_config.keys()).index(current_model) if current_model in self.model_config else -1
            
            # Select next available model
            for i in range(current_index + 1, len(self.model_config)):
                fallback_model = list(self.model_config.keys())[i]
                self.logger.warning(f"Switching from {current_model} to fallback model: {fallback_model}")
                return fallback_model
            
            # If no fallback found, return original model
            return current_model
        
        except Exception as e:
            self.logger.error(f"Fallback model selection failed: {e}")
            return current_model

    def _handle_api_failure(self, 
                           method_name: str, 
                           error: Exception, 
                           context: Optional[Dict[str, Any]] = None) -> str:
        """
        Comprehensive error handling for API and model failures
        
        Args:
            method_name (str): Name of the method that failed
            error (Exception): Original error encountered
            context (Dict, optional): Additional context about the failure
        
        Returns:
            str: Detailed error report and recommendations
        """
        # Log the detailed error
        self.logger.error(f"API Failure in {method_name}: {str(error)}")
        
        # Detailed error analysis
        error_type = type(error).__name__
        error_details = str(error)
        
        # Construct comprehensive error report
        error_report = f"""COMPREHENSIVE MEDICAL RESEARCH ANALYSIS FAILURE

Error Methodology: {method_name}
Error Classification: {error_type}
Specific Error Details: {error_details}

DIAGNOSTIC BREAKDOWN:
1. Potential API Connectivity Issues
   - Network interruption detected
   - API endpoint unreachable
   - Authentication failure

2. Model Availability Assessment
   - Verify current model status
   - Check OpenRouter service health
   - Validate API credentials

3. Research Query Evaluation
   - Analyze query complexity
   - Simplify research parameters
   - Reduce contextual depth

RECOMMENDED MITIGATION STRATEGIES:
- Verify internet connectivity
- Regenerate API authentication token
- Switch to alternative research model
- Implement exponential backoff retry mechanism
- Consult multiple medical research databases

CRITICAL ADVISORY:
 THIS IS AN AUTOMATED FALLBACK RESPONSE
- Do NOT rely solely on this generated output
- Seek professional medical consultation
- Validate findings through authoritative sources

SYSTEM RECOMMENDATIONS:
- Retry research query with reduced complexity
- Check system logs for detailed error trace
- Contact technical support if issue persists

Timestamp: {datetime.datetime.now().isoformat()}
Error Tracking ID: {uuid.uuid4()}"""

        # Optional context logging if provided
        if context:
            self.logger.info(f"Additional Context: {json.dumps(context, indent=2)}")
        
        return error_report

    def _generate_llama_response(self, prompt: str) -> str:
        """
        Generate a response using the current model with enhanced error handling
        
        Args:
            prompt (str): Input prompt for the model
        
        Returns:
            str: Generated response text
        """
        try:
            # Prepare headers for API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/Codeium/ai-healthcare-research",
                "X-Title": "AI Healthcare Research Assistant"
            }
            
            # Construct payload with extremely strict instructions
            payload = {
                "model": self.current_model,
                "messages": [
                    {
                        "role": "system", 
                        "content": """You are an advanced medical research AI assistant.
                        
ABSOLUTE REQUIREMENTS:
- Generate ONLY single, flowing narratives
- NO numbered solutions or lists
- NO sections starting with "Treatment Name:", "Mechanism:", etc.
- Professional medical language
- Integrated treatment descriptions
- Maximum 500 words"""
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.1,  # Extremely low for consistency
                "max_tokens": 2000,
                "top_p": 0.7,
                "frequency_penalty": 0.9,  # Extremely high to prevent repetition
                "presence_penalty": 0.9,
                "stop": ["Solution", "Solution:", "Treatment Name:", "Mechanism of Action:"]
            }
            
            # Make API request
            response = requests.post(
                self.api_base_url, 
                headers=headers, 
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                generated_text = response_data['choices'][0]['message']['content'].strip()
                
                # Remove any remaining solution-like patterns
                import re
                patterns_to_remove = [
                    r'^Solution \d+:.*$',
                    r'^Solution:.*$',
                    r'^Treatment Name:.*$',
                    r'^Mechanism of Action:.*$',
                    r'^Potential Effectiveness:.*$',
                    r'^Research Status:.*$',
                    r'^Potential Side Effects:.*$',
                    r'^\d+\.\s*',
                    r'^[A-Za-z]+\s*\d+:.*$'
                ]
                
                for pattern in patterns_to_remove:
                    generated_text = re.sub(pattern, '', generated_text, flags=re.MULTILINE)
                
                # Clean up the text
                lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
                generated_text = '\n\n'.join(lines)
                
                return generated_text
            
            raise Exception(f"API request failed with status code {response.status_code}")
            
        except Exception as e:
            self.logger.error(f"Error in generating response: {e}")
            raise

    def validate_medical_response(self, response: str) -> Dict[str, Any]:
        """
        Advanced validation and structuring of medical treatment response
        
        Args:
            response (str): Raw model-generated response
        
        Returns:
            Dict[str, Any]: Structured and validated medical treatment information
        """
        try:
            # Use another Llama call for advanced validation and structuring
            validation_prompt = f"""ADVANCED MEDICAL RESPONSE VALIDATION

Analyze the following medical treatment description and STRICTLY STRUCTURE it:

{response}

MANDATORY VALIDATION CRITERIA:
1. Verify scientific accuracy
2. Confirm research-based claims
3. Ensure comprehensive medical insights
4. Validate statistical claims
5. Check for recent research references

REQUIRED STRUCTURED OUTPUT FORMAT (JSON):
{{
    "treatment_name": "Precise Medical Intervention Name",
    "mechanism_of_action": "Detailed Molecular/Physiological Explanation",
    "research_status": {{
        "clinical_phase": "Exact Trial Phase",
        "publication_details": "Journal Name, Year, DOI",
        "current_research_stage": "Ongoing/Completed/Approved"
    }},
    "potential_effectiveness": {{
        "statistical_evidence": "Percentage Improvement",
        "comparative_analysis": "Comparison with Standard Treatments",
        "patient_response_rate": "Quantitative Success Rate"
    }},
    "patient_populations": {{
        "target_demographics": "Specific Age, Gender, Condition Criteria",
        "inclusion_criteria": "Detailed Patient Selection Parameters",
        "exclusion_criteria": "Conditions Preventing Treatment"
    }},
    "clinical_evidence": [
        {{
            "research_paper": "Full Citation",
            "key_findings": "Summarized Research Outcomes"
        }}
    ],
    "emerging_innovations": "Cutting-Edge Technological Advancements",
    "safety_profile": {{
        "common_side_effects": "Documented Adverse Reactions",
        "rare_side_effects": "Uncommon but Potential Risks",
        "long_term_implications": "Projected Health Impacts"
    }}
}}

Ensure MAXIMUM scientific rigor and precision!"""
            
            # Generate structured response
            structured_response = self._generate_llama_response(validation_prompt)
            
            # Parse the structured response
            try:
                # Attempt to parse as JSON
                parsed_response = json.loads(structured_response)
                
                # Additional validation
                if not parsed_response.get('treatment_name'):
                    raise ValueError("Invalid or incomplete medical response")
                
                return parsed_response
            
            except (json.JSONDecodeError, ValueError) as parse_error:
                self.logger.warning(f"Response parsing error: {parse_error}")
                
                # Fallback structured response
                return {
                    "treatment_name": "Comprehensive Medical Analysis",
                    "mechanism_of_action": "Advanced research-based medical evaluation",
                    "research_status": {
                        "clinical_phase": "Preliminary",
                        "publication_details": "Ongoing research",
                        "current_research_stage": "Investigation"
                    },
                    "potential_effectiveness": {
                        "statistical_evidence": "Requires further investigation",
                        "comparative_analysis": "Insufficient current data",
                        "patient_response_rate": "Not yet determined"
                    },
                    "patient_populations": {
                        "target_demographics": "Broad medical research context",
                        "inclusion_criteria": "Comprehensive medical assessment needed",
                        "exclusion_criteria": "To be defined through further research"
                    },
                    "clinical_evidence": [],
                    "emerging_innovations": "Continuous medical research exploration",
                    "safety_profile": {
                        "common_side_effects": "Not yet comprehensively documented",
                        "rare_side_effects": "Requires extensive clinical trials",
                        "long_term_implications": "Ongoing medical investigation"
                    }
                }
        
        except Exception as e:
            self.logger.error(f"Comprehensive medical response validation failed: {e}")
            
            # Ultimate fallback response
            return {
                "error": "Advanced Medical Research Validation Unsuccessful",
                "details": str(e),
                "recommendation": "Consult multiple medical research sources and healthcare professionals"
            }
    
    def track_treatment_innovations(self, disease: str) -> List[Dict[str, Any]]:
        """
        Advanced Treatment Innovation Tracking using Gemini model
        
        Args:
            disease (str): Target medical condition for innovation analysis
        
        Returns:
            List[Dict[str, Any]]: Comprehensive treatment innovation insights
        """
        try:
            # Comprehensive treatment innovation tracking prompt
            innovation_tracking_prompt = f"""ADVANCED TREATMENT INNOVATION TRACKER: {disease}

COMPREHENSIVE INNOVATION ANALYSIS FRAMEWORK:

1. EMERGING TREATMENT TECHNOLOGIES
- Identify breakthrough medical interventions
- Analyze cutting-edge technological approaches
- Assess potential paradigm-shifting methodologies

2. RESEARCH LANDSCAPE
- Map current research ecosystem
- Highlight leading research institutions
- Identify key research methodologies

3. TECHNOLOGICAL INNOVATIONS
- Breakthrough medical technologies
- Advanced diagnostic techniques
- Precision medicine approaches

4. CLINICAL IMPACT ASSESSMENT
- Potential patient outcome improvements
- Comparative effectiveness analysis
- Risk-benefit evaluation

5. FUTURE TREND PREDICTIONS
- Anticipated medical technology developments
- Potential long-term clinical implications
- Emerging research directions

SPECIFIC FOCUS: {disease}

MANDATORY REQUIREMENTS:
- Provide scientifically validated information
- Reference recent clinical research (last 3-5 years)
- Quantify potential medical advancements
- Maintain highest standards of medical research integrity"""
            
            # Generate comprehensive treatment innovation insights
            innovation_text = self._generate_llama_response(innovation_tracking_prompt)
            
            # Validate and structure the response
            structured_innovations = self.validate_medical_response(innovation_text)
            
            return [structured_innovations]
        
        except Exception as e:
            self.logger.error(f"Treatment innovation tracking failed: {e}")
            return [{
                "disease": disease,
                "status": "Advanced Innovation Analysis Unsuccessful",
                "error": str(e),
                "recommendation": "Conduct comprehensive manual medical research review"
            }]
    
    def generate_literature_review(self, research_topic: str) -> str:
        """
        Generate a comprehensive literature review with advanced medical insights
        
        Args:
            research_topic (str): Topic for in-depth medical literature review
        
        Returns:
            str: Comprehensive literature review as plain text
        """
        try:
            # Comprehensive literature review prompt
            literature_review_prompt = f"""ADVANCED MEDICAL LITERATURE REVIEW

Conduct an exhaustive literature review on the research topic: {research_topic}

MANDATORY REVIEW COMPONENTS:
1. Current State of Research
2. Key Breakthrough Findings
3. Methodological Approaches
4. Conflicting Research Perspectives
5. Emerging Research Trends
6. Future Research Recommendations

REVIEW GUIDELINES:
- Reference minimum 5 peer-reviewed sources
- Cover research from last 5-7 years
- Provide critical analysis
- Highlight scientific significance
- Identify research gaps

DETAILED OUTPUT REQUIREMENTS:
- Comprehensive summary of existing research
- Critical evaluation of methodologies
- Identification of potential future research directions
- Quantitative analysis of research trends"""
            
            # Generate comprehensive literature review
            literature_review_text = self._generate_llama_response(literature_review_prompt)
            
            return literature_review_text
        
        except Exception as e:
            self.logger.error(f"Literature review generation failed: {e}")
            return f"""Literature Review Generation Error

Research Topic: {research_topic}

Unable to generate a comprehensive literature review due to the following error:
{str(e)}

Recommendations:
- Verify the research topic specificity
- Try a more focused or narrower research query
- Check system connectivity and API availability
- Consult with a research professional for manual review"""

    def summarize_clinical_report(self, report_file: Any) -> str:
        """
        Concise clinical report summarization with key insights
        
        Args:
            report_file (Any): Uploaded clinical report file
        
        Returns:
            str: Streamlined clinical report summary
        """
        try:
            # Import PDF extraction library
            import PyPDF2
            import io
            
            # Check if it's a PDF file
            if report_file.name.lower().endswith('.pdf'):
                # Create a PDF reader object
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(report_file.read()))
                
                # Extract text from all pages
                full_text = ""
                for page in pdf_reader.pages:
                    full_text += page.extract_text() + "\n"
                
                # Truncate text if too long
                full_text = full_text[:15000]  # Limit to first 15000 characters
            else:
                # For non-PDF files, try to decode
                full_text = report_file.read().decode('utf-8', errors='ignore')
            
            # Concise clinical report summarization prompt
            summarization_prompt = f"""CLINICAL REPORT SUMMARY

Report Context: {full_text[:2000]}

SUMMARY REQUIREMENTS:
- Extract core medical findings
- Highlight key patient insights
- Provide actionable medical recommendations
- Use clear, concise language

OUTPUT FORMAT:
1. Key Findings
2. Critical Observations
3. Recommended Actions
4. Potential Implications"""
            
            # Generate clinical report summary
            summary_text = self._generate_llama_response(summarization_prompt)
            
            return summary_text
        
        except Exception as e:
            self.logger.error(f"Clinical report summarization failed: {e}")
            return f"""Clinical Report Summarization Error
Source: {getattr(report_file, 'name', 'Unknown')}
Details: Unable to process the clinical report
Error Message: {str(e)}

Recommendation:
- Verify the report file format
- Ensure the document is readable
- Check for any file corruption"""

    def predict_medical_outcomes(self, patient_data: Dict[str, Any]) -> str:
        """
        Advanced medical outcome prediction with actionable insights
        
        Args:
            patient_data (Dict[str, Any]): Detailed patient medical information
        
        Returns:
            str: Medical outcome prediction as plain text
        """
        try:
            # Comprehensive medical outcome prediction prompt
            outcome_prediction_prompt = f"""PATIENT OUTCOME PREDICTION

Patient Data Overview:
{json.dumps(patient_data, indent=2)}

PREDICTION FRAMEWORK:
1. Risk Stratification
2. Outcome Probability
3. Intervention Recommendations
4. Personalized Care Strategy

ANALYSIS GUIDELINES:
- Provide precise, data-driven predictions
- Focus on actionable medical insights
- Prioritize patient-specific risk factors
- Recommend targeted interventions

REQUIRED OUTPUT:
- Comprehensive risk assessment
- Probability of different outcomes
- Specific intervention strategies
- Personalized care recommendations"""
            
            # Generate medical outcome prediction
            prediction_text = self._generate_llama_response(outcome_prediction_prompt)
            
            return prediction_text
        
        except Exception as e:
            self.logger.error(f"Medical outcome prediction failed: {e}")
            return f"""Medical Outcome Prediction Error
Patient ID: {patient_data.get('patient_id', 'Unknown')}
Status: Prediction Unsuccessful

Error Details: {str(e)}

Recommendations:
- Review patient data completeness
- Consult medical professional
- Conduct additional diagnostic tests"""

    def _parse_medical_summary(self, summary_text: str) -> Dict[str, Any]:
        """
        Parse and structure medical summary text
        
        Args:
            summary_text (str): Raw summary text from model
        
        Returns:
            Dict[str, Any]: Structured medical summary
        """
        try:
            # Basic parsing logic
            key_sections = {
                "key_findings": [],
                "critical_observations": [],
                "recommended_actions": [],
                "potential_implications": []
            }
            
            # Simple parsing based on section headers
            sections = summary_text.split('\n\n')
            for section in sections:
                section = section.strip()
                if section.lower().startswith('key findings'):
                    key_sections['key_findings'] = [
                        finding.strip() 
                        for finding in section.split('\n')[1:] 
                        if finding.strip()
                    ]
                elif section.lower().startswith('critical observations'):
                    key_sections['critical_observations'] = [
                        obs.strip() 
                        for obs in section.split('\n')[1:] 
                        if obs.strip()
                    ]
                elif section.lower().startswith('recommended actions'):
                    key_sections['recommended_actions'] = [
                        action.strip() 
                        for action in section.split('\n')[1:] 
                        if action.strip()
                    ]
                elif section.lower().startswith('potential implications'):
                    key_sections['potential_implications'] = [
                        implication.strip() 
                        for implication in section.split('\n')[1:] 
                        if implication.strip()
                    ]
            
            return key_sections
        
        except Exception as e:
            self.logger.warning(f"Summary parsing failed: {e}")
            return {
                "parsing_error": "Could not parse medical summary",
                "raw_text": summary_text
            }

    def _parse_outcome_prediction(self, prediction_text: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and structure medical outcome prediction
        
        Args:
            prediction_text (str): Raw prediction text from model
            patient_data (Dict[str, Any]): Original patient data
        
        Returns:
            Dict[str, Any]: Structured outcome prediction
        """
        try:
            # Structured prediction parsing
            prediction_sections = {
                "risk_profile": {
                    "overall_risk": "Not Assessed",
                    "key_risk_factors": []
                },
                "outcome_probabilities": {
                    "favorable_outcome": 0,
                    "adverse_outcome": 0
                },
                "recommended_interventions": [],
                "personalized_care_strategy": []
            }
            
            # Basic parsing logic
            sections = prediction_text.split('\n\n')
            for section in sections:
                section = section.strip()
                
                # Parse Risk Profile
                if section.lower().startswith('risk profile'):
                    lines = section.split('\n')[1:]
                    prediction_sections['risk_profile']['overall_risk'] = lines[0].strip() if lines else "Not Assessed"
                    prediction_sections['risk_profile']['key_risk_factors'] = [
                        factor.strip() for factor in lines[1:] if factor.strip()
                    ]
                
                # Parse Outcome Probabilities
                elif section.lower().startswith('outcome probabilities'):
                    lines = section.split('\n')[1:]
                    try:
                        prediction_sections['outcome_probabilities'] = {
                            "favorable_outcome": float(lines[0].split(':')[1].strip().rstrip('%')) / 100,
                            "adverse_outcome": float(lines[1].split(':')[1].strip().rstrip('%')) / 100
                        }
                    except (IndexError, ValueError):
                        pass
                
                # Parse Recommended Interventions
                elif section.lower().startswith('recommended interventions'):
                    prediction_sections['recommended_interventions'] = [
                        intervention.strip() 
                        for intervention in section.split('\n')[1:] 
                        if intervention.strip()
                    ]
                
                # Parse Personalized Care Strategy
                elif section.lower().startswith('personalized care strategy'):
                    prediction_sections['personalized_care_strategy'] = [
                        strategy.strip() 
                        for strategy in section.split('\n')[1:] 
                        if strategy.strip()
                    ]
            
            return prediction_sections
        
        except Exception as e:
            self.logger.warning(f"Prediction parsing failed: {e}")
            return {
                "parsing_error": "Could not parse medical outcome prediction",
                "raw_text": prediction_text
            }
