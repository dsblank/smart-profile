%pip install opik

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Need new magic %download:
import requests

response = requests.get(
    "https://raw.githubusercontent.com/dsblank/smart-profile/refs/heads/main/helpers.py"
)
with open("helpers.py", "wb") as fp:
    fp.write(response.content)

import opik
from opik.rest_api.client import OpikApi
from datetime import datetime, timezone, timedelta
from helpers import get_opik_data, generate_ai_summary, generate_user_badges
import comet_ml
import json

import os

with st.sidebar:
    github_name = st.text_input(
        "Github name:", value=os.environ.get("GITHUB_NAME", "user")
    )
    os.environ["GITHUB_NAME"] = github_name
    comet_api_key = st.text_input(
        "Comet/Opik API key:",
        type="password",
        value=os.environ.get("COMET_API_KEY", ""),
    )
    os.environ["COMET_API_KEY"] = comet_api_key
    openai_api_key = st.text_input(
        "OpenAI API Key:", type="password", value=os.environ.get("OPENAI_API_KEY", "")
    )
    os.environ["OPENAI_API_KEY"] = openai_api_key

if comet_api_key:
    comet_api = comet_ml.API(api_key=comet_api_key)
    opik.configure(
        api_key=comet_api_key,
        url="https://www.comet.com/opik/api",
        workspace=github_name,
        force=True,
    )
    os.environ["OPIK_PROJECT_NAME"] = "smart-profile"
    opik_client = opik.Opik()
    opik_api = OpikApi(
        base_url="https://www.comet.com/opik/api",
        api_key=comet_api_key,
        workspace_name=github_name,
    )


def banner(badges=None):
    """
    Render the user banner with profile info and badges.
    
    Args:
        badges: Optional list of badge dictionaries with 'label' and 'color' keys.
                If None, shows loading placeholders.
    """
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
        st.html(f"""
<div class="circle-image">
    <img src="https://github.com/{github_name}.png?size=200" alt="My Rectangular Image">
</div>
""")

        st.markdown("""
<style>
    .circle-image {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        overflow: hidden;
        box-shadow: 0 0 5px rgba(0, 0, 0, 0.3);
    }
          
    .circle-image img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
</style>
""", unsafe_allow_html=True)

    with name_container:
        response = requests.get(f"https://api.github.com/users/{github_name}")
        data = response.json()

        st.markdown(
            f"""
        * **Name**: {data["name"]}
        * **Company**: {data["company"]}
        * **Location**: {data["location"]}
        * **Biography**: {data["bio"]}
        """
        )

    # Badge containers
    badge_containers = [upper_left_badge, lower_left_badge, upper_right_badge, lower_right_badge]
    
    if badges and len(badges) == 4:
        # Display AI-generated badges
        for i, (container, badge) in enumerate(zip(badge_containers, badges)):
            with container:
                # URL encode the badge label for shields.io
                import urllib.parse
                encoded_label = urllib.parse.quote(badge["label"])
                badge_url = f"https://img.shields.io/badge/{encoded_label}-{badge['color']}"
                st.image(badge_url, use_container_width=True)
    else:
        # Display loading placeholders or default badges
        default_badges = [
            "https://img.shields.io/badge/⏳%20Loading...-lightgrey",
            "https://img.shields.io/badge/⏳%20Analyzing...-lightgrey", 
            "https://img.shields.io/badge/⏳%20Computing...-lightgrey",
            "https://img.shields.io/badge/⏳%20Processing...-lightgrey"
        ]
        
        for container, badge_url in zip(badge_containers, default_badges):
            with container:
                st.image(badge_url, use_container_width=True)

def activities():
    st.markdown("### 📝 AI Overview")
    if comet_api_key == "":
        print("Define OpenAI API key in sidebar")
        return

    result = generate_ai_summary(
        openai_api_key=openai_api_key,
        _comet_api=comet_api,
        _opik_api=opik_api,
        _opik_client=opik_client,
        workspace_name=github_name,
    )

    if result.get("success"):
        st.markdown(result["ai_summary"])
        st.json(result["data_sources"])
        return result["ai_summary"]  # Return summary for badge generation
    else:
        st.error(result.get("error"))
        return None

def get_ai_generated_badges(ai_summary_text):
    """
    Helper function to generate badges from AI summary text.
    Returns list of badge dictionaries or None if generation fails.
    """
    try:
        
        # Check if we have the required API key
        if not openai_api_key:
            st.error("❌ OpenAI API key not found - badges cannot be generated")
            return None
        
        result = generate_user_badges(
            openai_api_key=openai_api_key,
            ai_summary=ai_summary_text
        )
        
        if result.get("success"):
            return result["badges"]
        else:
            st.warning(f"Badge generation failed: {result.get('error')}")
            return None
            
    except Exception as e:
        st.error(f"Error in badge generation: {e}")
        st.exception(e)  # Show full traceback
        return None

def opik_summary():
    st.markdown("### 🤖 Opik Summary")
    if comet_api_key == "":
        print("Define Comet API key in sidebar")
        return

    st.markdown("*Over the past 3 days*")

    # Get data
    data = get_opik_data(
        _opik_client=opik_client, _opik_api=opik_api, workspace_name=github_name
    )

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
            st.metric(
                "Errors",
                data["metrics"]["error_count"],
                delta=f"⚠️ {data['metrics']['error_count']} traces with errors",
            )

        st.markdown("---")

        # Quick links section
        st.markdown("**Quick Links:**")

        # Last trace
        if data["recent_traces"]:
            last_trace = data["recent_traces"][0]
            if isinstance(last_trace["start_time"], datetime):
                time_ago = datetime.now(timezone.utc) - last_trace["start_time"]
                hours_ago = max(1, int(time_ago.total_seconds() / 3600))
                cost_info = (
                    f" (${last_trace['cost']:.3f})" if last_trace.get("cost") else ""
                )
                trace_url = f"https://www.comet.com/opik/{github_name}/projects/{last_trace['project_id']}/traces?size=100&height=small&traces_filters=%5B%5D&trace={last_trace['id']}&span="
                st.markdown(
                    f"• **Last trace:** [{last_trace['name']}]({trace_url}){cost_info} ({hours_ago}h ago)"
                )
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
                st.markdown(
                    f"• **Last dataset:** [{last_dataset['name']}]({dataset_url}) ({days_ago}d ago)"
                )
            else:
                dataset_url = f"https://www.comet.com/opik/{github_name}/datasets/{last_dataset['id']}"
                st.markdown(
                    f"• **Last dataset:** [{last_dataset['name']}]({dataset_url})"
                )

        # Last experiment (placeholder for future implementation)
        if data["recent_experiments"]:
            last_experiment = data["recent_experiments"][0]
            time_ago = datetime.now(timezone.utc) - last_experiment["created_at"]
            hours_ago = max(1, int(time_ago.total_seconds() / 3600))
            st.markdown(
                f"• **Last experiment:** {last_experiment['name']} ({hours_ago}h ago)"
            )
        else:
            st.markdown("• **Experiments:** Coming soon...")

        st.markdown("</div>", unsafe_allow_html=True)


def em_summary():
    st.markdown("### 🔬 EM Summary")
    if comet_api_key == "":
        print("Define Comet API key in sidebar")
        return

    with st.container(border=True):
        if comet_api_key:
            # workspaces = comet_api.get_workspaces()
            experiments = comet_api.get_panel_experiments()
            st.metric("Experiments in current project", len(experiments))
        else:
            st.html("* Unknown experiments (set Comet ML API key)")
            
        projects = comet_api.get(workspace=github_name)[:3]
        if projects:
            project_links = [
                f"https://www.comet.com/{github_name}/{project}" for project in projects
            ]
            links_proj = [f'[{project}]({link})' for project, link in zip(projects, project_links)]
            ready_str = ', '.join(links_proj)
            st.markdown(
                f"• **New experiments in projects:** {ready_str}"
            )
        else:
            st.markdown(f"• **New experiments in projects:** nothing new")

        models = comet_api.get_registry_model_names(workspace=github_name)[:3]
        if models:
            model_links = [
                f"https://www.comet.com/{github_name}/model-registry/{model}" for model in models
            ]
            links_mod = [f'[{model}]({link})' for model, link in zip(models, model_links)]
            ready_str = ', '.join(links_mod)
            st.markdown(
                f"• **Changes in models:** {ready_str}"
            )
        else:
            st.markdown(f"• **Changes in models:** nothing new")

        artifacts = comet_api.get_artifact_list(workspace=github_name)["artifacts"][:3]
        if artifacts:
            artifact_names = []
            for artifact in artifacts:
                artifact_names.append(artifact["name"])
            artifact_links = [
                f"https://www.comet.com/{github_name}/artifacts/{artifact_name}" for artifact_name in artifact_names
            ]
            links_art = [f'[{artifact_name}]({link})' for artifact_name, link in zip(artifact_names, artifact_links)]
            ready_str = ', '.join(links_art)
            st.markdown(
                f"• **Updates in artifacts:** {ready_str}"
            )
        else:
            st.markdown(f"• **Updates in artifacts:** nothing new")
