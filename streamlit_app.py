"""DKT Research Lab — multi-page Streamlit dashboard.

Run with:  python -m streamlit run streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="DKT Research Lab", layout="wide")

pages = st.navigation([
    st.Page("app_pages/simulator.py", title="Student Simulator",
            icon=":material/play_arrow:", default=True),
    st.Page("app_pages/mastery_heatmap.py", title="Mastery Heatmap",
            icon=":material/grid_on:"),
    st.Page("app_pages/prediction_autopsy.py", title="Prediction Autopsy",
            icon=":material/monitoring:"),
    st.Page("app_pages/model_comparison.py", title="Model Comparison",
            icon=":material/analytics:"),
    st.Page("app_pages/overview.py", title="Overview",
            icon=":material/home:"),
    st.Page("app_pages/reality_check_page.py", title="Mastery Reality Check",
            icon=":material/task_alt:"),
    st.Page("app_pages/settings.py", title="Dataset & Experiment Settings",
            icon=":material/settings:"),
])
pages.run()
