%pip install opik

st.set_page_config(layout="wide", initial_sidebar_state='collapsed')

# Need new magic %download:
import requests
response = requests.get("https://raw.githubusercontent.com/dsblank/smart-profile/refs/heads/main/helpers.py")
with open("helpers.py", "wb") as fp:
    fp.write(response.content)

import opik
from opik.rest_api.client import OpikApi
from datetime import datetime, timezone, timedelta
from helpers import get_opik_data, generate_ai_summary
import comet_ml

api_ml = comet_ml.API()

import os

with st.sidebar:
    github_name = st.text_input("Github name: ", value=os.environ.get("GITHUB_NAME", "user"))
    os.environ["GITHUB_NAME"] = github_name
    comet_ml_api_key = st.text_input("Comet ML API key: ", value=os.environ.get("COMET_API_KEY", api_ml._client.api_key))
    os.environ["COMET_API_KEY"] = comet_ml_api_key
    opik_api_key = st.text_input("Opik API key: ", value=os.environ.get("OPIK_API_KEY", ""))
    os.environ["OPIK_API_KEY"] = opik_api_key

if comet_ml_api_key:
    api_ml = comet_ml.API(api_key=comet_ml_api_key)
if opik_api_key:
    opik.configure(api_key=opik_api_key, url='https://www.comet.com/opik/api', workspace=github_name, force=True)
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
            f"https://github.com/{github_name}.png?size=200",
            use_container_width=True
        )

    with name_container:
        response = requests.get(f"https://api.github.com/users/{github_name}")
        data = response.json()
        
        st.markdown(f"""
        * **Name**: {data["name"]}
        * **Company**: {data["company"]}
        * **Location**: {data["location"]}
        * **Biography**: {data["bio"]}
        """)

    with upper_left_badge:
        st.image(
            "https://img.shields.io/badge/Framework-tensorflow-red", 
        )

    with lower_left_badge:
        st.image(
            "https://img.shields.io/badge/%E2%9C%A8%20Opik_user-green", 
        )

    with upper_right_badge:
        st.image(
            "https://img.shields.io/badge/Framework-ADK_Agent-violet", 
        )

    with lower_right_badge:
        st.image(
            "https://img.shields.io/badge/%F0%9F%8C%A0%20100_Experiments-darkblue", 
        )

    columns_2 = st.columns([0.7, 0.3])
    
    with columns_2[0]:
        AI_summary = st.container(border=False)
        AI_advice = st.container(border=False)

    with columns_2[1]:
        view_more = st.container(border=False)

    with AI_summary:
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        if st.button("Generate AI Summary"):
            result = generate_ai_summary(
                openai_api_key=openai_api_key,
                opik_api_key=opik_api_key,
                comet_ml_api_key=comet_ml_api_key,
                workspace_name=github_name
            )
            
            if result.get("success"):
                st.markdown(result["ai_summary"])
                st.json(result["data_sources"])
            else:
                st.error(result.get("error"))

    with AI_advice:
        st.markdown(
            body=":bulb: Try Opik today!"
        )

    with view_more:
        st.markdown(
            body="[view more]()"
        )
    
def activities():
    st.markdown("### 📝 Activities")

def opik_summary():
    st.markdown("### 🤖 Opik Summary")
    st.markdown("*Over the past 3 days*")
    
    # Get data
    data = get_opik_data(api_key=opik_api_key, workspace_name=github_name)
    
    # Create container with border styling
    with st.container(border=True):
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
                trace_url = f"https://www.comet.com/opik/{github_name}/projects/{last_trace['project_id']}/traces?size=100&height=small&traces_filters=%5B%5D&trace={last_trace['id']}&span="
                st.markdown(f"• **Last trace:** [{last_trace['name']}]({trace_url}){cost_info} ({hours_ago}h ago)")
            else:
                trace_url = f"https://www.comet.com/opik/{github_name}/projects/{last_trace['project_id']}/traces?size=100&height=small&traces_filters=%5B%5D&trace={last_trace['id']}&span="
                st.markdown(f"• **Last trace:** [{last_trace['name']}]({trace_url})")
        
        # Last dataset
        if data["recent_datasets"]:
            last_dataset = data["recent_datasets"][0]
            if isinstance(last_dataset["created_at"], datetime):
                time_ago = datetime.now(timezone.utc) - last_dataset["created_at"]
                days_ago = int(time_ago.total_seconds() / 86400)
                dataset_url = f"https://www.comet.com/opik/{github_name}/datasets/{last_dataset['id']}"
                st.markdown(f"• **Last dataset:** [{last_dataset['name']}]({dataset_url}) ({days_ago}d ago)")
            else:
                dataset_url = f"https://www.comet.com/opik/{github_name}/datasets/{last_dataset['id']}"
                st.markdown(f"• **Last dataset:** [{last_dataset['name']}]({dataset_url})")
        
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
    st.markdown("### 🔬 EM Summary")
    with st.container(border=True):
        if comet_ml_api_key:
            #workspaces = api_ml.get_workspaces()
            experiments = api_ml.get_panel_experiments()
            st.html(f"* {len(experiments)} Experiments")
        else:
            st.html("* Unknown experiments (set Comet ML API key)")
        projects = api_ml.get(workspace=github_name)[:3]
        if (projects):
            project_links = [f'https://www.comet.com/{github_name}/{projects[0]}', 
                             f'https://www.comet.com/{github_name}/{projects[1]}', 
                             f'https://www.comet.com/{github_name}/{projects[2]}']
            st.markdown(f"• **New experiments in projects:** [{projects[0]}]({project_links[0]}), [{projects[1]}]({project_links[1]}), [{projects[2]}]({project_links[2]})")
        else:
            st.markdown(f"• **New experiments in projects:** nothing new")
            
    
        models = api_ml.get_registry_model_names(workspace=github_name)[:3]
        if (models):
            model_links = [f'https://www.comet.com/{github_name}/model-registry/{models[0]}',
                          f'https://www.comet.com/{github_name}/model-registry/{models[1]}',
                          f'https://www.comet.com/{github_name}/model-registry/{models[2]}']
            st.markdown(f"• **Changes in models:** [{models[0]}]({model_links[0]}), [{models[1]}]({model_links[1]}), [{models[2]}]({model_links[2]})")
        else:
            st.markdown(f"• **Changes in models:** nothing new")
    
        artifacts = api_ml.get_artifact_list(workspace=github_name)['artifacts'][:3]
        if (artifacts):
            artifact_names = []
            for artifact in artifacts:
                artifact_names.append(artifact["name"])
            artifact_links = [f'https://www.comet.com/{github_name}/artifacts/{artifact_names[0]}',
                             f'https://www.comet.com/{github_name}/artifacts/{artifact_names[1]}',
                             f'https://www.comet.com/{github_name}/artifacts/{artifact_names[2]}']
            st.markdown(f"• **Updates in artifacts:** [{artifact_names[0]}]({artifact_links[0]}), [{artifact_names[1]}]({artifact_links[1]}), [{artifact_names[2]}]({artifact_links[2]})")
        else:
            st.markdown(f"• **Updates in artifacts:** nothing new")
