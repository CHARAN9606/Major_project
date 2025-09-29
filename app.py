# Add this at the very top of your app.py file, before any other imports

import os
import sys
import warnings

# Fix for PyTorch/Streamlit compatibility issue
def fix_torch_streamlit_compatibility():
    """Fix PyTorch and Streamlit compatibility issues"""
    try:
        # Suppress specific warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="torch")
        warnings.filterwarnings("ignore", message=".*torch._classes.*")
        
        # Set environment variables to prevent torch from interfering with Streamlit
        os.environ['TORCH_DISABLE_EXTENSIONS'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Monkey patch torch classes if needed
        try:
            import torch
            # Prevent torch from trying to access __path__._path
            if hasattr(torch, '_classes'):
                torch._classes.__path__ = None
        except ImportError:
            pass  # torch not installed, no need to fix
        except Exception:
            pass  # ignore torch patching errors
            
    except Exception as e:
        print(f"Warning: Could not apply torch compatibility fix: {e}")

# Apply the fix immediately
fix_torch_streamlit_compatibility()

# Check and import dependencies with better error handling
missing_deps = []
dependency_errors = {}

# Core dependencies
try:
    import streamlit as st
except ImportError as e:
    missing_deps.append("streamlit")
    dependency_errors["streamlit"] = str(e)

try:
    import pandas as pd
except ImportError as e:
    missing_deps.append("pandas")
    dependency_errors["pandas"] = str(e)

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as e:
    missing_deps.append("plotly")
    dependency_errors["plotly"] = str(e)

from pathlib import Path
import json
import ast
import shutil
from typing import Any, Dict

# Check if we have critical dependencies
if missing_deps:
    print("=" * 60)
    print("MISSING DEPENDENCIES DETECTED!")
    print("=" * 60)
    for dep in missing_deps:
        print(f"❌ {dep}: {dependency_errors.get(dep, 'Module not found')}")
    
    print("\n📦 INSTALLATION INSTRUCTIONS:")
    print("Run the following command to install missing dependencies:")
    print(f"   pip install {' '.join(missing_deps)}")
    
    if 'plotly' in missing_deps:
        print("\n🔧 Quick fix for plotly:")
        print("   pip install plotly>=5.15.0")
    
    print("\n📋 Or install all dependencies:")
    print("   pip install -r requirements.txt")
    print("=" * 60)
    
    # Exit if critical dependencies are missing
    sys.exit(1)

# Import your modules with better error handling
try:
    from parser import parse_folder
    parser_available = True
except ImportError as e:
    st.error(f"Parser module error: {e}")
    st.error("Please make sure parser.py is in the same directory and all dependencies are installed")
    parser_available = False

try:
    from scoring import score_dataframe, summarize
    scoring_available = True
except ImportError as e:
    st.error(f"Scoring module error: {e}")
    st.error("Please make sure scoring.py is in the same directory and all dependencies are installed")
    scoring_available = False

# Only proceed if both modules are available
if not (parser_available and scoring_available):
    st.stop()

# ------------------- HELPER FUNCTIONS -------------------
def safe_filter_errors(df):
    """Safely filter out error rows from dataframe"""
    if df.empty:
        return df, df.copy()
    
    # Check if 'error' column exists
    if 'error' not in df.columns:
        return df, pd.DataFrame()
    
    # Create boolean mask safely
    error_mask = df['error'].notna()
    success_df = df[~error_mask].copy()
    error_df = df[error_mask].copy()
    
    return success_df, error_df

def get_success_count(df):
    """Get count of successfully parsed resumes"""
    if df.empty:
        return 0
    
    if 'error' not in df.columns:
        return len(df)
    
    return len(df[df['error'].isna()])

def get_error_count(df):
    """Get count of resumes with parsing errors"""
    if df.empty:
        return 0
    
    if 'error' not in df.columns:
        return 0
    
    return len(df[df['error'].notna()])

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Enhanced Resume Parser", layout="wide")

# Test if NLP model is working
@st.cache_resource
def test_nlp_setup():
    """Test NLP setup and show results"""
    try:
        from parser import nlp  # Import the nlp model from parser
        model_info = {
            'name': nlp.meta.get('name', 'Unknown'),
            'version': nlp.meta.get('version', 'Unknown'),
            'vectors': getattr(nlp.vocab, 'vectors_length', 0),
            'status': 'Working'
        }
        return model_info
    except Exception as e:
        return {
            'name': 'Error',
            'version': 'N/A',
            'vectors': 0,
            'status': f'Error: {str(e)}',
            'error': True
        }

# Show NLP model status
model_info = test_nlp_setup()

# ------------------- ENHANCED THEME -------------------
st.markdown("""
    <style>
    .stApp { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff; 
        font-family: 'Inter', 'Segoe UI', sans-serif; 
    }
    .main-content {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        color: #2c3e50;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .nlp-status {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: white;
    }
    .status-good { border-left: 4px solid #4CAF50; }
    .status-error { border-left: 4px solid #f44336; }
    
    section[data-testid="stSidebar"] { 
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] p { 
        color: #ffffff; 
    }
    div.stButton > button { 
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white; 
        border-radius: 25px; 
        padding: 0.75rem 2rem;
        font-weight: 600; 
        border: none; 
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    div.stButton > button:hover { 
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    h1, h2, h3 { 
        color: #2c3e50 !important; 
        font-weight: 700; 
    }
    .dependency-warning {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    .installation-code {
        background: #f8f9fa;
        border-radius: 4px;
        padding: 0.5rem;
        font-family: monospace;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- HEADER WITH MODEL STATUS -------------------
st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem; color: white; margin-bottom: 0.5rem;'>🚀 Enhanced Resume Parser</h1>
        <p style='font-size: 1.2rem; color: rgba(255,255,255,0.8);'>AI-powered resume analysis with advanced NLP</p>
    </div>
""", unsafe_allow_html=True)

# Show NLP Model Status
status_class = "status-good" if not model_info.get('error') else "status-error"
st.markdown(f"""
    <div class="nlp-status {status_class}">
        <strong>🤖 NLP Model Status:</strong> {model_info['status']}<br>
        <strong>Model:</strong> {model_info['name']} v{model_info['version']}<br>
        <strong>Word Vectors:</strong> {model_info['vectors']:,}
        {f"<br><strong>⚠️ Issue:</strong> {model_info.get('error', '')}" if model_info.get('error') else ""}
    </div>
""", unsafe_allow_html=True)

# Paths
skills_path = Path("skills.json")
dataset_path = Path("dataset")
uploads_dir = Path("uploads")

# Create skills.json if it doesn't exist
if not skills_path.exists():
    default_skills = [
        "Python", "Java", "JavaScript", "React", "Angular", "Vue.js", "Node.js",
        "Django", "Flask", "Spring", "HTML", "CSS", "SQL", "MySQL", "PostgreSQL",
        "MongoDB", "AWS", "Azure", "Docker", "Kubernetes", "Git", "Jenkins",
        "Machine Learning", "Data Science", "TensorFlow", "PyTorch", "Pandas",
        "NumPy", "Scikit-learn", "REST API", "GraphQL", "Microservices"
    ]
    skills_path.write_text(json.dumps(default_skills, indent=2))
    st.info("Created default skills.json file")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    use_dataset = st.checkbox("🔍 Use included dataset", value=True)
    
    st.markdown("### 💼 Job Description")
    jd = st.text_area(
        "Enter job requirements:", 
        height=200,
        placeholder="""Looking for a Senior Python Developer with 4+ years experience.
Required: Python, Django, React, AWS, Docker, PostgreSQL."""
    )
    
    st.markdown("### 📤 Upload Resumes")
    uploaded = st.file_uploader(
        "Choose resume files", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        parse_btn = st.button("🔍 Parse", use_container_width=True)
    with col2:
        score_btn = st.button("📊 Score", use_container_width=True)

# Session state
if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame()
if "scored_df" not in st.session_state:
    st.session_state["scored_df"] = pd.DataFrame()

# Main content
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Parsing section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🧾 Parsing Results")
    
    if parse_btn:
        if model_info.get('error'):
            st.error("⚠️ NLP model has issues. Parsing may not work correctly.")
            st.error(f"Error details: {model_info.get('error', '')}")
        
        # Save uploaded files
        if uploaded:
            uploads_dir.mkdir(parents=True, exist_ok=True)
            for f in uploaded:
                (uploads_dir / f.name).write_bytes(f.read())
            st.success(f"✅ Uploaded {len(uploaded)} file(s)")

        # Parse resumes
        with st.spinner("🔍 Parsing resumes..."):
            frames = []
            
            if use_dataset and dataset_path.exists():
                try:
                    dataset_df = parse_folder(dataset_path, skills_path)
                    if not dataset_df.empty:
                        frames.append(dataset_df)
                        st.info(f"📂 Processed dataset: {len(dataset_df)} files")
                except Exception as e:
                    st.error(f"Error parsing dataset: {e}")
            
            if uploads_dir.exists() and any(uploads_dir.iterdir()):
                try:
                    uploads_df = parse_folder(uploads_dir, skills_path)
                    if not uploads_df.empty:
                        frames.append(uploads_df)
                        st.info(f"📤 Processed uploads: {len(uploads_df)} files")
                except Exception as e:
                    st.error(f"Error parsing uploads: {e}")

            if frames:
                new_df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
                st.session_state["df"] = new_df
                
                # Show parsing results using safe functions
                success_count = get_success_count(new_df)
                error_count = get_error_count(new_df)
                
                if success_count > 0:
                    st.success(f"🎉 Successfully parsed {success_count} resumes!")
                
                if error_count > 0:
                    st.warning(f"⚠️ {error_count} files had parsing errors")
                    
                    with st.expander("Show Parsing Errors"):
                        _, error_df = safe_filter_errors(new_df)
                        for _, row in error_df.iterrows():
                            st.write(f"❌ **{row['file']}**: {row.get('error', 'Unknown error')}")
                
                # Display successfully parsed data
                success_df, _ = safe_filter_errors(new_df)
                if not success_df.empty:
                    display_cols = ['file', 'name', 'language']
                    available_cols = [col for col in display_cols if col in success_df.columns]
                    st.dataframe(success_df[available_cols], use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ No resumes found or processed.")

with col2:
    st.markdown("### 📋 Resume Preview")
    if not st.session_state["df"].empty:
        success_df, _ = safe_filter_errors(st.session_state["df"])
        
        if not success_df.empty:
            selected_idx = st.selectbox(
                "Select resume to preview:",
                range(len(success_df)),
                format_func=lambda x: success_df.iloc[x]['file']
            )
            
            if selected_idx is not None:
                row = success_df.iloc[selected_idx]
                
                st.markdown(f"**📄 {row['file']}**")
                st.markdown(f"**👤 Name:** {row.get('name', 'N/A')}")
                st.markdown(f"**🌍 Language:** {row.get('language', 'N/A')}")
                
                # Show confidence scores
                if 'confidence' in row and row['confidence']:
                    conf = row['confidence']
                    if isinstance(conf, dict):
                        st.markdown("**📊 Confidence Scores:**")
                        for key, score in conf.items():
                            if isinstance(score, (int, float)):
                                st.progress(score/100, text=f"{key.title()}: {score}%")
                
                # Show skills if available
                if 'skills' in row and row['skills']:
                    skills = row['skills']
                    if isinstance(skills, list) and len(skills) > 0:
                        st.markdown("**🛠️ Skills Found:**")
                        skills_text = ", ".join(skills[:8])
                        if len(skills) > 8:
                            skills_text += f" (+{len(skills)-8} more)"
                        st.write(skills_text)

# Scoring Section
st.divider()
st.markdown("### 🎯 Resume Scoring & Analysis")

if score_btn:
    if not jd.strip():
        st.error("❌ Please provide a job description!")
    elif st.session_state["df"].empty:
        st.error("❌ Please parse resumes first!")
    else:
        # Filter out error rows for scoring using safe function
        success_df, _ = safe_filter_errors(st.session_state["df"])
        
        if success_df.empty:
            st.error("❌ No successfully parsed resumes available for scoring!")
        else:
            with st.spinner("🤖 AI is analyzing resumes..."):
                try:
                    # Score resumes using the enhanced scoring
                    scored_df = score_dataframe(success_df, jd)
                    st.session_state["scored_df"] = scored_df
                    
                    st.success("🎉 Scoring completed successfully!")
                    
                    # Display top candidates
                    if not scored_df.empty:
                        st.markdown("#### 🏆 Top Candidates")
                        
                        top_candidates = scored_df.head(3)
                        
                        for idx, (_, candidate) in enumerate(top_candidates.iterrows()):
                            with st.container():
                                st.markdown(f"""
                                <div style='border-left: 4px solid #667eea; padding: 1rem; margin: 1rem 0; background: #f8f9ff; border-radius: 0 10px 10px 0;'>
                                    <h4 style='color: #667eea; margin: 0;'>#{idx+1} {candidate.get('file', 'Unknown')} - {candidate.get('name', 'Unknown')}</h4>
                                    <div style='display: flex; gap: 2rem; margin: 1rem 0;'>
                                        <div><strong>Overall Score:</strong> {candidate.get('score', 0):.1f}%</div>
                                        <div><strong>Skill Match:</strong> {candidate.get('skill_match', 0):.1f}%</div>
                                        <div><strong>Skill Gaps:</strong> {candidate.get('skill_gaps', 0)}</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**✅ Matched Skills:**")
                                    matched = candidate.get('matched', [])
                                    if isinstance(matched, list) and matched:
                                        matched_text = ", ".join(str(item) for item in matched[:8])
                                        if len(matched) > 8:
                                            matched_text += f" (+{len(matched)-8} more)"
                                        st.write(matched_text)
                                    else:
                                        st.write("None found")
                                
                                with col2:
                                    st.markdown("**❌ Missing Skills:**")
                                    missing = candidate.get('missing', [])
                                    if isinstance(missing, list) and missing:
                                        missing_text = ", ".join(str(item) for item in missing[:8])
                                        if len(missing) > 8:
                                            missing_text += f" (+{len(missing)-8} more)"
                                        st.write(missing_text)
                                    else:
                                        st.write("None")
                        
                        # Detailed scoring table
                        st.markdown("#### 📊 Detailed Results")
                        
                        display_cols = ['file', 'name', 'score', 'skill_match', 'skill_gaps']
                        available_cols = [col for col in display_cols if col in scored_df.columns]
                        
                        if available_cols:
                            display_scored = scored_df[available_cols].copy()
                            
                            # Rename columns for better display
                            column_names = {
                                'file': 'File',
                                'name': 'Name', 
                                'score': 'Overall Score (%)',
                                'skill_match': 'Skill Match (%)',
                                'skill_gaps': 'Skill Gaps'
                            }
                            
                            display_scored = display_scored.rename(columns=column_names)
                            
                            st.dataframe(
                                display_scored,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Overall Score (%)": st.column_config.ProgressColumn(
                                        "Overall Score (%)",
                                        help="Overall compatibility score",
                                        min_value=0,
                                        max_value=100,
                                    ),
                                    "Skill Match (%)": st.column_config.ProgressColumn(
                                        "Skill Match (%)", 
                                        help="Percentage of required skills matched",
                                        min_value=0,
                                        max_value=100,
                                    ),
                                }
                            )
                        
                        # Simple analytics
                        st.markdown("#### 📈 Quick Analytics")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            avg_score = scored_df['score'].mean() if 'score' in scored_df.columns else 0
                            st.metric("Average Score", f"{avg_score:.1f}%")
                        
                        with col2:
                            top_score = scored_df['score'].max() if 'score' in scored_df.columns else 0
                            st.metric("Highest Score", f"{top_score:.1f}%")
                        
                        with col3:
                            total_candidates = len(scored_df)
                            st.metric("Total Candidates", total_candidates)
                        
                        # Export functionality
                        st.markdown("#### 💾 Export Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            csv_data = scored_df.to_csv(index=False)
                            st.download_button(
                                label="📄 Download CSV",
                                data=csv_data,
                                file_name="resume_scores.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            json_data = scored_df.to_json(orient='records', indent=2)
                            st.download_button(
                                label="📋 Download JSON",
                                data=json_data, 
                                file_name="resume_scores.json",
                                mime="application/json"
                            )
                    
                except Exception as e:
                    st.error(f"❌ Error during scoring: {str(e)}")
                    st.error("Please check your resume data and job description.")
                    
                    # Show debug info
                    with st.expander("Debug Information"):
                        st.write("**DataFrame columns:**", list(success_df.columns))
                        st.write("**DataFrame shape:**", success_df.shape)
                        st.write("**Job Description length:**", len(jd))
                        st.write("**Error details:**", str(e))

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; padding: 2rem; color: rgba(255,255,255,0.7);'>
        <p>🚀 <strong>Enhanced Resume Parser</strong> | Advanced NLP • Skill Gap Analysis • Multi-language Support</p>
        <p>Built with Streamlit, SpaCy, and Plotly | Fully Offline Processing</p>
    </div>
""", unsafe_allow_html=True)

# Error handling for common issues
if __name__ == "__main__":
    # Check if all required files exist
    required_files = ["parser.py", "scoring.py"]
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        st.error(f"Missing required files: {', '.join(missing_files)}")
        st.info("Please ensure all required files are in the same directory as app.py")
    
    # Test imports and show status
    try:
        import spacy
        try:
            nlp_test = spacy.load("en_core_web_lg")
            st.success("✅ NLP model loaded successfully!")
        except OSError:
            try:
                nlp_test = spacy.load("en_core_web_sm")
                st.warning("⚠️ Using basic NLP model. Install en_core_web_lg for better accuracy")
            except OSError:
                st.error("❌ No spaCy language models found!")
                st.info("Install a language model: python -m spacy download en_core_web_sm")
    except Exception as e:
        st.warning(f"⚠️ NLP model issue: {e}")
        st.info("Try running: python -m spacy download en_core_web_lg")