import streamlit as st
from streamlit_navigation_bar import st_navbar
import os
from dotenv import load_dotenv
from src.data_processor import DataProcessor
from src.drug_discovery import DrugDiscoveryAssistant
from src.llama_model import LlamaResearchAssistant
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import io
from datetime import datetime

# Load environment variables
load_dotenv()

def initialize_session_state():
    """
    Initialize and manage session state variables
    Ensures data persistence across different app sections
    """
    # Initialize session state if not already set
    default_session_state = {
        # Clinical Trials Section
        'clinical_trials': {
            'selected_disease': None,
            'trials_data': None,
            'analysis_results': None
        },
        
        # Disease Prediction Section
        'disease_prediction': {
            'uploaded_file': None,
            'patient_data': None,
            'predictions': None,
            'risk_analysis': None
        },
        
        # Treatment Innovation Section
        'treatment_innovation': {
            'selected_disease': None,
            'treatment_discovery': None,
            'literature_summary': None
        },
        
        # Literature Review Section
        'literature_review': {
            'research_topic': None,
            'generated_summary': None,
            'download_content': None
        },
        
        # Global application settings
        'app_settings': {
            'theme': 'light',
            'language': 'en',
            'last_accessed_section': None
        }
    }
    
    # Initialize each key if not already in session state
    for section, defaults in default_session_state.items():
        if section not in st.session_state:
            st.session_state[section] = defaults

def main():
    # Page Configuration
    st.set_page_config(
        page_title="AI Healthcare Research Assistant",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Initialize session state
    initialize_session_state()

    # Custom CSS for better styling
    st.markdown("""
        <style>
        /* Custom styling for the app */
        .stApp {
            background-color: #0E1117;
            color: #ffffff;
        }
        /* Sidebar styling */
        .css-1aumxhk {
            background-color: #161b22;
        }
        .css-1aumxhk .stMarkdown {
            color: #ffffff;
        }
        .css-1aumxhk .stButton {
            width: 100%;
        }
        /* Navigation buttons styling */
        .stButton>button {
            background-color: #21262d;
            color: #ffffff;
            border: 1px solid #30363d;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #30363d;
            color: #58a6ff;
        }
        .stButton>button:focus {
            background-color: #388bfd;
            color: #ffffff;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar Navigation
    with st.sidebar:
        st.title("🩺 Healthcare AI")
        st.markdown("---")
        
        # Navigation buttons in sidebar
        if st.button("🏠 Home", key="sidebar_home", use_container_width=True):
            st.session_state.current_page = "Home"
        
        if st.button("🔬 Clinical Trials", key="sidebar_trials", use_container_width=True):
            st.session_state.current_page = "Clinical Trials"
        
        if st.button("🩺 Disease Prediction", key="sidebar_prediction", use_container_width=True):
            st.session_state.current_page = "Disease Prediction"
        
        if st.button("💊 Treatment Innovation", key="sidebar_treatment", use_container_width=True):
            st.session_state.current_page = "Treatment Innovation"
        
        if st.button("📚 Literature Review", key="sidebar_literature", use_container_width=True):
            st.session_state.current_page = "Literature Review"
        
        if st.button("👨‍💻 About", key="sidebar_about", use_container_width=True):
            st.session_state.current_page = "About"
        
        # Additional sidebar information
        st.markdown("---")
        st.markdown("""
        **AI Healthcare Research Assistant**
        *Version 1.0.0*
        
        Powered by advanced machine learning
        algorithms and medical research insights.
        """)

    # Load necessary assistants
    load_dotenv()
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    llama_assistant = LlamaResearchAssistant(openrouter_api_key)
    drug_analyzer = DrugDiscoveryAssistant(llama_assistant)

    # Ensure Home is the default page on first load
    if 'current_page' not in st.session_state or not st.session_state.current_page:
        st.session_state.current_page = "Home"

    # Render selected page based on current_page
    if st.session_state.current_page == "Home":
        render_home_page()
    elif st.session_state.current_page == "Clinical Trials":
        render_clinical_trials_page(llama_assistant)
    elif st.session_state.current_page == "Disease Prediction":
        render_disease_prediction_page(drug_analyzer)
    elif st.session_state.current_page == "Treatment Innovation":
        render_treatment_innovation_page(drug_analyzer)
    elif st.session_state.current_page == "Literature Review":
        render_literature_review_page(llama_assistant)
    elif st.session_state.current_page == "About":
        render_about_page()

def render_home_page():
    """
    Render a fully responsive home page for AI Healthcare Research Assistant
    """
    # Enhanced Responsive CSS
    st.markdown("""
    <style>
    /* Global Responsive Styles */
    @media (max-width: 768px) {
        .stApp {
            padding: 10px !important;
        }
    }

    /* Hero Section Responsive */
    .hero-section {
        background: linear-gradient(135deg, #0E1117 0%, #21262d 100%);
        color: #ffffff;
        padding: 2rem 1rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        margin-bottom: 1rem;
    }
    .hero-section h1 {
        font-size: clamp(1.5rem, 5vw, 3rem);
        margin-bottom: 1rem;
    }
    .hero-section p {
        font-size: clamp(0.9rem, 3vw, 1.2rem);
    }

    /* Responsive Feature Section */
    .feature-section {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 1rem;
        margin-top: 1rem;
    }
    .feature-card {
        flex: 1;
        min-width: 250px;
        max-width: 350px;
        background-color: #161b22;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.3s ease;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
    }
    .feature-card:hover {
        transform: scale(1.05);
    }
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        color: #58a6ff;
    }

    /* Responsive Columns */
    .stColumn {
        padding: 0.5rem;
    }

    /* Responsive Technology Section */
    .tech-section {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 1rem;
        margin-top: 1rem;
    }
    .tech-item {
        flex: 1;
        min-width: 200px;
        text-align: center;
        padding: 1rem;
        background-color: #161b22;
        border-radius: 10px;
    }

    /* Responsive CTA Button */
    .cta-button {
        display: inline-block;
        background-color: #2ea44f;
        color: white !important;
        text-decoration: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        margin-top: 1rem;
        transition: background-color 0.3s ease;
    }
    .cta-button:hover {
        background-color: #2c974b;
    }

    /* Responsive Image */
    .stImage {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1>AI Healthcare Research Assistant</h1>
        <p>Revolutionizing Medical Research with Artificial Intelligence</p>
        <p>Discover, Analyze, and Innovate in Healthcare</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature Sections
    st.markdown("""
    <div class="feature-section">
        <div class="feature-card">
            <div class="feature-icon">🔬</div>
            <h3>Treatment Innovation</h3>
            <p>Advanced AI-powered drug candidate identification</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3>Clinical Trials</h3>
            <p>Comprehensive medical research data analysis</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📝</div>
            <h3>Literature Review</h3>
            <p>AI-generated comprehensive research summaries</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main Content with Responsive Columns
    st.markdown("## Welcome to the Future of Medical Research")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Our AI Healthcare Research Assistant is designed to:
        - Accelerate treatment innovation processes
        - Analyze complex medical datasets
        - Generate comprehensive literature reviews
        - Provide insights into clinical trials
        
        Powered by cutting-edge AI technologies, we aim to support medical researchers, 
        pharmaceutical companies, and healthcare professionals in making data-driven decisions.
        """)
        
    
    with col2:
        # Medical research themed image with responsive sizing
        st.image("https://i.ibb.co/MSSK63X/th.jpg", 
                 caption="AI-Powered Medical Research and Innovation", 
                 use_container_width=True)
    
    # Responsive Technology Section
    st.markdown("## Our Key Technologies")
    
    st.markdown("""
    <div class="tech-section">
        <div class="tech-item">
            <h4>🤖 Machine Learning</h4>
            <p>Advanced data analysis algorithms</p>
        </div>
        <div class="tech-item">
            <h4>📡 Natural Language Processing</h4>
            <p>Insights from medical literature</p>
        </div>
        <div class="tech-item">
            <h4>🧬 Predictive Modeling</h4>
            <p>Identifying treatment strategies</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer-like section
    st.markdown("""
    ---
    **Disclaimer**: This is an AI-assisted research tool. Always consult professional 
    medical advice for clinical decisions.
    """)

def render_disease_prediction_page(drug_analyzer):
    st.header("Disease Outcome Prediction")
    
    # Initialize session state for tracking prediction state
    if 'predictions_generated' not in st.session_state:
        st.session_state.predictions_generated = False
    
    # File uploader for patient data
    uploaded_file = st.file_uploader("Upload Patient Data", type=['csv', 'xlsx'])
    
    # Slider to control number of patients to process
    max_patients = st.slider(
        "Select Maximum Number of Patients to Process", 
        min_value=10, 
        max_value=500, 
        value=50, 
        step=10
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                patient_data = pd.read_csv(uploaded_file)
            else:
                patient_data = pd.read_excel(uploaded_file)
            
            # Validate DataFrame
            if patient_data is None or patient_data.empty:
                st.error("The uploaded file contains no data. Please upload a valid file.")
                return
            
            # Validate required columns
            required_columns = ['patient_id', 'age', 'gender', 'symptoms']
            missing_columns = [col for col in required_columns if col not in patient_data.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                return
            
            # Store patient data in session state
            st.session_state.disease_prediction['patient_data'] = patient_data
            
            # Prediction button
            if st.button("Predict Disease Outcomes") or st.session_state.predictions_generated:
                # Ensure we have patient data
                if st.session_state.disease_prediction['patient_data'] is not None and not st.session_state.disease_prediction['patient_data'].empty:
                    with st.spinner('Generating Predictions...'):
                        try:
                            # Debug: Print column names and first few rows
                            st.write("Columns in uploaded data:", list(patient_data.columns))
                            st.write("First few rows:", patient_data.head())
                            
                            # Validate column names
                            column_mapping = {
                                'patient_id': ['patient_id', 'id', 'Patient ID', 'patient_ID'],
                                'age': ['age', 'patient_age', 'Age', 'Patient Age'],
                                'gender': ['gender', 'patient_gender', 'Gender', 'Patient Gender'],
                                'symptoms': ['symptoms', 'patient_symptoms', 'medical_symptoms', 'Symptoms', 'Patient Symptoms']
                            }
                            
                            # Find matching columns
                            mapped_columns = {}
                            for target_col, possible_cols in column_mapping.items():
                                found_col = next((col for col in possible_cols if col in patient_data.columns), None)
                                if found_col:
                                    mapped_columns[target_col] = found_col
                                else:
                                    st.error(f"Could not find column for {target_col}")
                                    return
                            
                            # Rename columns to standard names
                            patient_data = patient_data.rename(columns={
                                mapped_columns['patient_id']: 'patient_id',
                                mapped_columns['age']: 'age',
                                mapped_columns['gender']: 'gender',
                                mapped_columns['symptoms']: 'symptoms'
                            })
                            
                            # Use max_patients from slider
                            predictions = drug_analyzer.predict_disease_outcomes(
                                patient_data, 
                                max_patients=max_patients
                            )
                            
                            # Validate predictions
                            if predictions is not None and not predictions.empty:
                                # Mark predictions as generated
                                st.session_state.predictions_generated = True
                                st.session_state.disease_prediction['predictions'] = predictions
                                
                                # Tabs for different views
                                tab1, tab2 = st.tabs([
                                    "Patient Predictions", 
                                    "Risk Distribution"
                                ])
                                
                                with tab1:
                                    st.write(f"### Detailed Predictions for {len(predictions)} Patients")
                                    
                                    # Columns for patient selection and details
                                    col1, col2 = st.columns([1, 3])
                                    
                                    with col1:
                                        # Select patient for detailed view
                                        selected_patient_id = st.selectbox(
                                            "Select Patient ID", 
                                            predictions['patient_id'].tolist(),
                                            key='patient_id_selection_unique'
                                        )
                                    
                                    with col2:
                                        if selected_patient_id:
                                            # Find the selected patient's data
                                            patient_prediction = predictions[predictions['patient_id'] == selected_patient_id]
                                            
                                            # Detailed patient information card
                                            st.info(f"### Patient ID: {selected_patient_id}")
                                            
                                            # Create columns for better layout
                                            info_col1, info_col2 = st.columns(2)
                                            
                                            with info_col1:
                                                st.markdown("""
                                                #### Patient Profile
                                                - **Age:** {}
                                                - **Gender:** {}
                                                - **Symptoms:** {}
                                                """.format(
                                                    patient_prediction['age'].values[0],
                                                    patient_prediction['gender'].values[0],
                                                    patient_prediction['symptoms'].values[0]
                                                ))
                                            
                                            with info_col2:
                                                st.markdown("""
                                                #### Risk Assessment
                                                - **Risk Score:** {}/100
                                                """.format(
                                                    patient_prediction['risk_score'].values[0]
                                                ))
                                            
                                            # Prediction Details
                                            st.markdown("### Detailed Health Prediction")
                                            specific_patient_prediction = patient_prediction['predicted_outcome'].values[0]
                                            st.write(specific_patient_prediction)
                                            
                                            # Recommendations Section
                                            st.markdown("### Recommendations")
                                            st.warning("""
                                            - Consult with a healthcare professional for personalized medical advice
                                            - Consider additional diagnostic tests based on the prediction
                                            - Maintain a healthy lifestyle and follow preventive measures
                                            """)
                                
                                with tab2:
                                    st.write("### Risk Score Distribution")
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        # Risk Score Histogram
                                        fig_risk = px.histogram(
                                            predictions, 
                                            x='risk_score', 
                                            title='Distribution of Patient Risk Scores',
                                            labels={'risk_score': 'Risk Score', 'count': 'Number of Patients'},
                                            color_discrete_sequence=['#FF6B6B']
                                        )
                                        st.plotly_chart(fig_risk, use_container_width=True)
                                    
                                    with col2:
                                        # Risk Score Box Plot
                                        fig_risk_box = px.box(
                                            predictions, 
                                            y='risk_score', 
                                            title='Risk Score Box Plot',
                                            labels={'risk_score': 'Risk Score'}
                                        )
                                        st.plotly_chart(fig_risk_box, use_container_width=True)
                                
                                # Optional: Provide download option for predictions
                                export_predictions = predictions.copy()
                                export_predictions['predicted_outcome'] = export_predictions['predicted_outcome'].str.replace('\n', ' | ')
                                
                                csv_buffer = io.StringIO()
                                export_predictions.to_csv(csv_buffer, index=False)
                                csv_buffer.seek(0)
                                
                                st.download_button(
                                    label="Download Full Predictions",
                                    data=csv_buffer.getvalue(),
                                    file_name='disease_predictions.csv',
                                    mime='text/csv',
                                    key='download_predictions_unique'
                                )
                            else:
                                st.warning("No predictions could be generated.")
                        
                        except Exception as pred_error:
                            # Detailed error logging
                            st.error(f"Error generating predictions: {pred_error}")
                            st.error(f"Error Type: {type(pred_error)}")
                            import traceback
                            st.error(traceback.format_exc())
                else:
                    st.error("Please upload patient data before predicting.")
        
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            st.warning("Please check your file format and contents.")

def render_clinical_trials_page(llama_assistant):
    st.header("Clinical Trial Summarization")
    
    # Check for previous results
    if st.session_state.clinical_trials['analysis_results']:
        st.sidebar.success("Previous Clinical Trial Summary Available")
        if st.sidebar.button("Restore Previous Summary"):
            st.session_state.clinical_trials['analysis_results'] = st.session_state.clinical_trials['analysis_results']
    
    uploaded_file = st.file_uploader("Upload Clinical Trial Report", type=['pdf', 'txt'])
    
    if uploaded_file is not None:
        # Save the uploaded PDF to session state
        st.session_state.clinical_trials['uploaded_file'] = uploaded_file
        
        with st.spinner('Analyzing Clinical Trial Report...'):
            try:
                # Summarize the uploaded PDF
                summary = llama_assistant.summarize_clinical_report(uploaded_file)
                
                # Save summary to session state and previous results
                st.session_state.clinical_trials['analysis_results'] = summary
                
                # Display summary
                st.write(summary)
            
            except Exception as e:
                st.error(f"Error summarizing clinical trial report: {e}")
    
    # Display previously uploaded PDF summary if exists
    elif st.session_state.clinical_trials['analysis_results']:
        st.subheader("Previous Summary")
        st.write(st.session_state.clinical_trials['analysis_results'])

def render_treatment_innovation_page(drug_analyzer):
    st.header("Treatment Innovation Tracker")
    
    # Check for previous results
    if st.session_state.treatment_innovation['treatment_discovery']:
        st.sidebar.success("Previous Treatment Candidates Available")
        if st.sidebar.button("Restore Previous Candidates"):
            st.session_state.treatment_innovation['treatment_discovery'] = st.session_state.treatment_innovation['treatment_discovery']
    
    # Section-specific disclaimer
    st.warning("""
    **Disclaimer**: 
    These treatment insights are AI-generated research perspectives. 
    They are NOT medical recommendations. Always consult healthcare 
    professionals for treatment decisions.
    """)
    
    selected_disease = st.text_input("Enter Disease or Medical Condition",
                                   placeholder="e.g., Alzheimer's Disease, Rare Genetic Disorder, etc.",
                                   key="treatment_innovation_input")
    
    # Save selected disease to session state
    st.session_state.treatment_innovation['selected_disease'] = selected_disease
    
    if st.button("Explore Therapeutic Solutions"):
        try:
            with st.spinner(f'Exploring Innovative Treatments for {selected_disease}...'):
                # Discover treatment candidates
                response = drug_analyzer.discover_drug_candidates(selected_disease)
                
                # Save response to session state and previous results
                st.session_state.treatment_innovation['treatment_discovery'] = response
                
                if response and 'narrative' in response:
                    # Display the narrative first
                    st.markdown(response['narrative'])
                    
                    # Then display structured treatment information if available
                    if 'treatments' in response and response['treatments']:
                        st.subheader("Detailed Treatment Analysis")
                        for treatment in response['treatments']:
                            with st.expander(f"Treatment Details: {treatment.get('name', 'Research-Based Treatment')}"):
                                st.write("**Treatment Name:**", treatment.get('drug_name', 'Not specified'))
                                st.write("**Mechanism:**", treatment.get('mechanism', 'Not specified'))
                                st.write("**Effectiveness:**", treatment.get('effectiveness', 'Not specified'))
                                st.write("**Research Status:**", treatment.get('research_status', 'Not specified'))
                                st.write("**Potential Side Effects:**", treatment.get('side_effects', 'Not specified'))
                else:
                    st.error("No treatment information available.")
        
        except Exception as e:
            st.error(f"Error exploring treatments: {str(e)}")
            st.info("Please try again with a different medical condition or contact support if the issue persists.")
    
    # Display previous treatment information if exists
    elif st.session_state.treatment_innovation['treatment_discovery']:
        response = st.session_state.treatment_innovation['treatment_discovery']
        if 'narrative' in response:
            st.markdown(response['narrative'])
            
            if 'treatments' in response and response['treatments']:
                st.subheader("Detailed Treatment Analysis")
                for treatment in response['treatments']:
                    with st.expander(f"Treatment Details: {treatment.get('name', 'Research-Based Treatment')}"):
                        st.write("**Treatment Name:**", treatment.get('drug_name', 'Not specified'))
                        st.write("**Mechanism:**", treatment.get('mechanism', 'Not specified'))
                        st.write("**Effectiveness:**", treatment.get('effectiveness', 'Not specified'))
                        st.write("**Research Status:**", treatment.get('research_status', 'Not specified'))
                        st.write("**Potential Side Effects:**", treatment.get('side_effects', 'Not specified'))

def render_literature_review_page(llama_assistant):
    st.header("Automated Literature Review")
    
    # Check for previous results
    if st.session_state.literature_review['generated_summary']:
        st.sidebar.success("Previous Literature Review Available")
        if st.sidebar.button("Restore Previous Review"):
            st.session_state.literature_review['generated_summary'] = st.session_state.literature_review['generated_summary']
    
    # Section-specific disclaimer
    st.warning("""
    **Disclaimer**: 
    This literature review is an AI-generated summary. 
    It should NOT be considered comprehensive medical research. 
    Always verify information with peer-reviewed sources and experts.
    """)
    
    research_topic = st.text_input("Enter Research Topic")
    
    # Save research topic to session state
    st.session_state.literature_review['research_topic'] = research_topic
    
    if st.button("Generate Review"):
        with st.spinner('Analyzing Research Literature...'):
            literature_summary = llama_assistant.generate_literature_review(research_topic)
            
            # Save literature review to session state and previous results
            st.session_state.literature_review['generated_summary'] = literature_summary
            
            st.markdown(literature_summary)
            
            # Generate downloadable literature review
            literature_download = generate_downloadable_literature_review(research_topic, literature_summary)
            st.session_state.literature_review['download_content'] = literature_download
            
            # Download button
            st.download_button(
                label="Download Full Literature Review",
                data=literature_download['content'],
                file_name=literature_download['filename'],
                mime='text/plain',
                key='download_full_literature_review'
            )

def generate_downloadable_literature_review(research_topic: str, literature_summary: str) -> dict:
    """
    Generate a comprehensive, downloadable literature review
    
    Args:
        research_topic (str): Topic of the literature review
        literature_summary (str): Generated literature summary
    
    Returns:
        Dict containing formatted literature review content and filename
    """
    try:
        # Prepare comprehensive literature review content
        literature_content = f"""Comprehensive Literature Review: {research_topic}
{'=' * 50}

Research Overview
-----------------
{literature_summary}

Additional Context
------------------
- Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Research Topic: {research_topic}

Disclaimer
----------
This literature review is generated using advanced AI research techniques. 
It should be used as a preliminary research tool and not as a substitute for 
comprehensive academic or professional medical research.

Recommended Citation
--------------------
AI-Generated Literature Review, {datetime.now().year}
Research Topic: {research_topic}
Generated using Advanced Medical Research Assistant
"""
        
        # Generate a unique, descriptive filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{research_topic.lower().replace(' ', '_')}_literature_review_{timestamp}.txt"
        
        return {
            'content': literature_content,
            'filename': filename
        }
    
    except Exception as e:
        print(f"Error generating literature review download: {e}")
        return {
            'content': f"Unable to generate literature review for {research_topic}",
            'filename': f"literature_review_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        }


def render_about_page():
    # Title of the About Page
    st.title('About the AI Healthcare Research Assistant')

    # Short Project Introduction
    st.write("""
    The **AI Healthcare Research Assistant** is a revolutionary platform aimed at transforming healthcare research by providing intelligent insights. 
    Leveraging advanced machine learning algorithms and AI, the platform accelerates data analysis and aids critical decision-making, ultimately improving patient outcomes and healthcare practices.
    """)

    # Capabilities Section with Bullet Points and Emphasis
    st.write('### Capabilities:')
    st.write('- **Clinical Trials Explorer**: Analyzes ongoing and recent clinical trials, presenting findings through intuitive data visualizations.')
    st.write('- **Disease Prediction System**: Uses AI-driven predictive modeling to forecast disease progression and deliver personalized health insights.')
    st.write('- **Treatment Innovation Tracker**: Helps researchers and healthcare professionals discover cutting-edge treatment strategies and medical interventions.')
    st.write('- **Medical Literature Review**: Provides automated summaries and extracts key insights from vast amounts of medical literature.')

    # Author Section with a Bio and Call to Action
    st.write('### Author:')
    st.write("""
        This platform was built by **Dadvaiah Pavan**, a passionate developer and AI enthusiast with a specialization in **Artificial Intelligence** and **Machine Learning**. 
        With experience in **Data Science**, **Artificial Intelligence**, and **Machine Learning**, Iam dedicated to leveraging advanced technologies to improve healthcare research and outcomes. 
        Through this project, I aim to bridge the gap between cutting-edge AI research and practical healthcare solutions, providing intelligent insights to accelerate medical discoveries and decision-making.
        """)


    # Call to Action (Personal Website / GitHub)
    st.write('Explore more of my work on my [GitHub](https://github.com/DadvaiahPavan) or visit my [portfolio](https://pavandadvaiah.netlify.app/) for further details.')

    # Optionally, add an image or icon related to healthcare or AI
    st.image('https://i.ibb.co/KsTCN62/th-1.jpg', caption='AI in Healthcare', use_container_width=True)


if __name__ == "__main__":
    main()
