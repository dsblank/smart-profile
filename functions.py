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
    columns_1 = st.columns([0.22, 0.40, 0.02, 0.11, 0.11, 0.13])
    
    with columns_1[0]:
        image_container = st.container(border=False)
        
    with columns_1[1]:
        name_container = st.container(border=False)
        email_container = st.container(border=False)
        
    with columns_1[3]:
        upper_left_badge = st.container(border=False)
        lower_left_badge = st.container(border=False)
        
    with columns_1[4]:
        upper_right_badge = st.container(border=False)
        lower_right_badge = st.container(border=False)

    with image_container:
        st.image(
            "https://static.vecteezy.com/system/resources/thumbnails/024/983/914/small/simple-user-default-icon-free-png.png", 
            use_container_width=True
        )

    with name_container:
        st.markdown(
            body="Name: John Doe"
        )

    with email_container:
        st.markdown(
            body="Email: john.doe@gmail.com"
        )

    with upper_left_badge:
        st.image(
            "https://static.vecteezy.com/system/resources/thumbnails/047/309/918/small_2x/verified-badge-profile-icon-png.png", 
            use_container_width=True
        )

    with lower_left_badge:
        st.image(
            "https://static.vecteezy.com/system/resources/thumbnails/047/309/918/small_2x/verified-badge-profile-icon-png.png", 
            use_container_width=True
        )

    with upper_right_badge:
        st.image(
            "https://static.vecteezy.com/system/resources/thumbnails/047/309/918/small_2x/verified-badge-profile-icon-png.png", 
            use_container_width=True
        )

    with lower_right_badge:
        st.image(
            "https://static.vecteezy.com/system/resources/thumbnails/047/309/918/small_2x/verified-badge-profile-icon-png.png", 
            use_container_width=True
        )

    columns_2 = st.columns([0.7, 0.3])
    
    with columns_2[0]:
        AI_summary = st.container(border=False)
        AI_advice = st.container(border=False)

    with columns_2[1]:
        view_more = st.container(border=False)

    with AI_summary:
        st.markdown(
            body=":chart_with_upwards_trend: Worked mostly on EM, registered new model"
        )

    with AI_advice:
        st.markdown(
            body=":bulb: Try Opik today!"
        )

    with view_more:
        st.markdown(
            body="[view more]()"
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
