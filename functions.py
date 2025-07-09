%pip install opik

st.set_page_config(layout="wide", initial_sidebar_state='collapsed')

import comet_ml
import opik

api_ml = comet_ml.API()

with st.sidebar:
    comet_ml_api_key = st.text_input("Comet ML API key: ", value=api_ml._client.api_key)
    opik_api_key = st.text_input("Opik API key: ")

if comet_ml_api_key:
    api_ml = comet_ml.API(api_key=comet_ml_api_key)
if opik_api_key:
    opik.configure(api_key=opik_api_key, use_local=False)
    opik_client = opik.Opik()

def banner():
    st.image(
        "https://cdn.vectorstock.com/i/2000v/20/87/black-avatar-generic-person-symbol-profile-vector-54382087.avif",
        width=100,
    )
    
def activities():
    st.html("<b>Activities</b>")

def opik_summary():
    st.html("<b>Opik Summary</b>")

def em_summary():
    st.html("<b>Experiment Management<b>")
    if comet_ml_api_key:
        #workspaces = api_ml.get_workspaces()
        experiments = api_ml.get_panel_experiments()
        st.html(f"* {len(experiments)} Experiments")
    else:
        st.html("* Unknown experiments (set Comet ML API key)")
