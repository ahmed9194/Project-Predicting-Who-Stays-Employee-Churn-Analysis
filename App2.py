import streamlit as st
from streamlit_option_menu import option_menu
from nbconvert import HTMLExporter
import nbformat
import os
from pathlib import Path

# ================ PATH CONFIGURATION ================
# Set your base directory (where the app and notebooks are stored)
BASE_DIR = Path(__file__).parent

# Configure notebook paths - THREE OPTIONS (choose one):

# OPTION 1: Relative paths (recommended if notebooks are in same folder as app)
NOTEBOOK_PATHS = {
    "📊 EDA": BASE_DIR / "EDA.ipynb",
    "🧹 Preprocessing": BASE_DIR / "preprocessing.ipynb",
    "⚙️ Feature Engineering": BASE_DIR / "Feature Engineering & Selection.ipynb",
    "🎛️ Hyperparameter Tuning": BASE_DIR / "Hyperparameter_Tuning.ipynb",
    "🏆 Model Evaluation": BASE_DIR / "Model Building and Evaluation.ipynb"
}

# ================ STYLING ================
def inject_custom_css():
    st.markdown(f"""
    <style>
        /* Main container */
        .main {{
            background-color: #0E1117;
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%) !important;
            border-right: 1px solid #4e4376;
        }}
        
        /* Notebook container */
        .notebook-container {{
            background-color: #1E1E1E !important;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            margin-top: 20px;
        }}
        
        /* Menu styling */
        .st-b7 {{
            color: white !important;
        }}
        
        /* Titles */
        h1 {{
            color: #FF4B4B !important;
            text-align: center;
            margin-bottom: 30px;
            font-family: 'Arial', sans-serif;
        }}
        
        /* Loading spinner */
        .stSpinner > div {{
            border-top-color: #FF4B4B !important;
        }}
        
        /* Code cells */
        .code_cell {{
            background-color: #2E2E2E !important;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }}
    </style>
    """, unsafe_allow_html=True)

# ================ NOTEBOOK RENDERING ================
@st.cache_data
def convert_notebook_to_html(notebook_path):
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'classic'
        (body, _) = html_exporter.from_notebook_node(notebook)
        
        # Add custom styling to notebook HTML
        styled_html = f"""
        <div class="notebook-container">
            {body}
        </div>
        """
        return styled_html
    except Exception as e:
        st.error(f"Error converting notebook: {str(e)}")
        return None

def render_notebook(notebook_path):
    if not os.path.exists(notebook_path):
        st.error(f"Notebook not found at: {notebook_path}")
        st.info(f"Current working directory: {os.getcwd()}")
        return
    
    with st.spinner(f"Loading {Path(notebook_path).name}..."):
        notebook_html = convert_notebook_to_html(notebook_path)
        if notebook_html:
            st.components.v1.html(notebook_html, height=1200, scrolling=True)

# ================ MAIN APP ================
def main():
    # Configure page
    st.set_page_config(
        page_title="🔥 Project Predicting Who Stays: Employee Churn Analysis",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject custom CSS
    inject_custom_css()
    
    # ===== SIDEBAR =====
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
        st.title("🔥 DS Dashboard")
        
        selected = option_menu(
            menu_title=None,
            options=["🏠 Home"] + list(NOTEBOOK_PATHS.keys()),
            icons=["house"] * (len(NOTEBOOK_PATHS)+1),
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "orange", "font-size": "18px"}, 
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "5px 0",
                    "border-radius": "5px",
                    "padding": "10px",
                    "--hover-color": "#434343",
                },
                "nav-link-selected": {
                    "background-color": "#FF4B4B",
                    "font-weight": "bold"
                },
            }
        )

    # ===== MAIN CONTENT =====
    if selected == "🏠 Home":
        st.title("🔥 Project Predicting Who Stays: Employee Churn Analysis")
        st.markdown("---")
        
        cols = st.columns(2)
        with cols[0]:
            st.markdown("""
            ### 🚀 Project Overview
            This interactive dashboard showcases the complete data science workflow:
            
            - **📊 EDA**: Exploratory Data Analysis
            - **🧹 Preprocessing**: Data cleaning and transformation
            - **⚙️ Feature Engineering**: Feature creation and selection
            - **🎛️ Hyperparameter Tuning**: Model optimization
            - **🏆 Model Evaluation**: Final performance assessment
            """)
            
        with cols[1]:
            st.image("https://miro.medium.com/max/1400/1*V-Jp13LvtVc2IiY2fp4qYw.gif", 
                    caption="Data Science Workflow")
        
        st.markdown("---")
        st.success("💡 Select a notebook from the sidebar to explore each phase in detail!")
    
    else:
        notebook_path = NOTEBOOK_PATHS[selected]
        st.title(f"📋 {selected}")
        render_notebook(str(notebook_path))  # Convert Path object to string

if __name__ == "__main__":
    main()