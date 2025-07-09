%pip install opik

st.set_page_config(layout="wide", initial_sidebar_state='collapsed')

import opik
from opik.rest_api.client import OpikApi
from datetime import datetime, timezone, timedelta

api_ml = comet_ml.API()

with st.sidebar:
    github_name = st.text_input("Github name: ", value="user")
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
            f"https://github.com/{github_name}?size=200",
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
    st.markdown("### 🤖 Opik Summary")
    st.markdown("*Over the past 3 days*")
    
    # Get data
    data = get_opik_data()
    
    # Create container with border styling
    with st.container():
        st.markdown("""
        <style>
        .opik-summary-box {
            border: 2px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            background-color: #f9f9f9;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="opik-summary-box">', unsafe_allow_html=True)
        
        # Core metrics - using actual Opik data fields
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Traces", data["metrics"]["total_traces"])
            st.metric("Avg Duration", f"{data['metrics']['avg_duration']:.2f}s")
        
        with col2:
            st.metric("Total Cost", f"${data['metrics']['total_cost']:.2f}")
            st.metric("Total Spans", data["metrics"]["span_count"])
        
        # Show errors if any
        if data["metrics"]["error_count"] > 0:
            st.metric("Errors", data["metrics"]["error_count"], delta=f"⚠️ {data['metrics']['error_count']} traces with errors")
        
        st.markdown("---")
        
        # Quick links section
        st.markdown("**Quick Links:**")
        
        # Last trace
        if data["recent_traces"]:
            last_trace = data["recent_traces"][0]
            if isinstance(last_trace["start_time"], datetime):
                time_ago = datetime.now(timezone.utc) - last_trace["start_time"]
                hours_ago = max(1, int(time_ago.total_seconds() / 3600))
                cost_info = f" (${last_trace['cost']:.3f})" if last_trace.get('cost') else ""
                st.markdown(f"• **Last trace:** {last_trace['name']}{cost_info} ({hours_ago}h ago)")
            else:
                st.markdown(f"• **Last trace:** {last_trace['name']}")
        
        # Last dataset
        if data["recent_datasets"]:
            last_dataset = data["recent_datasets"][0]
            if isinstance(last_dataset["created_at"], datetime):
                time_ago = datetime.now(timezone.utc) - last_dataset["created_at"]
                days_ago = int(time_ago.total_seconds() / 86400)
                st.markdown(f"• **Last dataset:** {last_dataset['name']} ({days_ago}d ago)")
            else:
                st.markdown(f"• **Last dataset:** {last_dataset['name']}")
        
        # Last experiment (placeholder for future implementation)
        if data["recent_experiments"]:
            last_experiment = data["recent_experiments"][0]
            time_ago = datetime.now(timezone.utc) - last_experiment["created_at"]
            hours_ago = max(1, int(time_ago.total_seconds() / 3600))
            st.markdown(f"• **Last experiment:** {last_experiment['name']} ({hours_ago}h ago)")
        else:
            st.markdown("• **Experiments:** Coming soon...")
        
        st.markdown('</div>', unsafe_allow_html=True)

def em_summary():
    st.html("<b>Experiment Management<b>")
    if comet_ml_api_key:
        #workspaces = api_ml.get_workspaces()
        experiments = api_ml.get_panel_experiments()
        st.html(f"* {len(experiments)} Experiments")
    else:
        st.html("* Unknown experiments (set Comet ML API key)")
