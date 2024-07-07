import streamlit as st
from st_pages import Page, Section, add_page_title, show_pages
from pathlib import Path

def load_css():
    with open('static/style.css') as f:
        css_code = f.read()
    st.markdown(f'<style>{css_code}</style>', unsafe_allow_html=True)

st.set_page_config(page_title="InsightPilot", page_icon="🧠")




# Initialize session state attributes
if 'show_data_types' not in st.session_state:
    st.session_state.show_data_types = False
if 'show_distinct_values' not in st.session_state:
    st.session_state.show_distinct_values = False
if 'show_basic_statistics' not in st.session_state:
    st.session_state.show_basic_statistics = False
if 'show_profile_report' not in st.session_state:
    st.session_state.show_profile_report = False



# Adding a logo to the sidebar with specified width
st.sidebar.image("static/logo.png", width=200)  # Set the desired width in pixels




# Define pages
pages = [
    st.Page("pages/dataset_upload.py", title="Dataset Upload", icon="⬆️"),
    st.Page("pages/search_dataset.py", title="Search Dataset", icon="🔍"),
    st.Page("pages/dataset_summary.py", title="Dataset Summary", icon="📊"),
    st.Page("pages/data_preprocessing.py", title="Data Preprocessing", icon="🔧"),
    st.Page("pages/data_visualization.py", title="Data Visualization", icon="📈"),
    st.Page("pages/chatbot.py", title="Chatbot", icon="🤖")

]

 

pg = st.navigation(pages)
pg.run()