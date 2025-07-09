import streamlit as st
import os
from datetime import datetime, timedelta, timezone

def get_mock_data():
    """Fallback mock data if Opik SDK is not available"""
    return {
        "recent_traces": [
            {"id": "trace_1", "name": "customer_query", "start_time": datetime.now(timezone.utc) - timedelta(hours=1), "duration": 850, "cost": 0.05},
            {"id": "trace_2", "name": "product_search", "start_time": datetime.now(timezone.utc) - timedelta(hours=3), "duration": 1200, "cost": 0.08},
            {"id": "trace_3", "name": "support_chat", "start_time": datetime.now(timezone.utc) - timedelta(days=1), "duration": 600, "cost": 0.03},
        ],
        "recent_datasets": [
            {"name": "customer_queries_v2", "created_at": datetime.now(timezone.utc) - timedelta(days=2)},
            {"name": "product_catalog", "created_at": datetime.now(timezone.utc) - timedelta(days=5)},
        ],
        "recent_experiments": [
            {"name": "prompt_optimization_v3", "created_at": datetime.now(timezone.utc) - timedelta(hours=6)},
            {"name": "response_quality_test", "created_at": datetime.now(timezone.utc) - timedelta(days=1)},
        ],
        "metrics": {
            "total_traces": 156,
            "total_cost": 23.45,
            "avg_duration": 0.85,
            "error_count": 3,
            "span_count": 234
        }
    }

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_opik_data(api_key="", workspace_name=""):
    """Get actual data from Opik SDK"""
    try:
        import opik
        from opik.rest_api.client import OpikApi
        
        # Initialize API client
        client_api = OpikApi(base_url="https://www.comet.com/opik/api", api_key=api_key, workspace_name=workspace_name)
        
        # Initialize regular client for search_traces
        opik.configure(api_key=api_key, workspace=workspace_name, url='https://www.comet.com/opik/api', force=True)
        client = opik.Opik(api_key=api_key)
        
        # Get all projects
        projects_page = client_api.projects.find_projects()
        all_projects = projects_page.content
        
        # Filter projects that have had traces within the past 3 days
        three_days_ago = datetime.now(timezone.utc) - timedelta(days=3)
        active_projects = [project for project in all_projects if project.last_updated_trace_at and project.last_updated_trace_at >= three_days_ago]
        
        # Get traces from all active projects
        all_traces = []
        for project in active_projects:
            project_traces = client.search_traces(max_results=1000, project_name=project.name, truncate=True)
            all_traces.extend(project_traces)
        
        # Filter traces to only include those from the past 3 days
        traces = [trace for trace in all_traces if trace.last_updated_at and trace.last_updated_at >= three_days_ago]
        
        # Get datasets using OpikApi
        datasets_page = client_api.datasets.find_datasets()
        all_datasets = datasets_page.content
        
        # Calculate metrics from filtered data (past 3 days only)
        total_traces = len(traces)
        total_cost = sum(trace.total_estimated_cost or 0 for trace in traces)
        
        # Calculate average duration in seconds
        durations = [trace.duration for trace in traces if trace.duration]
        avg_duration = sum(durations) / len(durations) / 1000 if durations else 0  # Convert ms to seconds
        
        # Count spans with errors
        error_traces = [trace for trace in traces if trace.error_info]
        error_count = len(error_traces)
        
        # Get recent traces for quick links
        recent_traces = []
        for trace in traces[:5]:  # Get last 5 traces
            recent_traces.append({
                "id": trace.id,
                "project_id": trace.project_id,
                "name": trace.name or "Unnamed trace",
                "start_time": trace.last_updated_at,
                "duration": trace.duration,
                "cost": trace.total_estimated_cost
            })
        
        # Get recent datasets for quick links - sort by last_updated_at
        sorted_datasets = sorted(all_datasets, key=lambda d: d.last_updated_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        recent_datasets = []
        for dataset in sorted_datasets[:5]:  # Get last 5 datasets
            recent_datasets.append({
                "id": dataset.id,
                "name": dataset.name,
                "created_at": dataset.last_updated_at
            })
        
        return {
            "recent_traces": recent_traces,
            "recent_datasets": recent_datasets,
            "recent_experiments": [],  # Will implement later
            "metrics": {
                "total_traces": total_traces,
                "total_cost": total_cost,
                "avg_duration": avg_duration,
                "error_count": error_count,
                "span_count": sum(trace.span_count or 0 for trace in traces)
            }
        }
    

        
    except ImportError:
        st.warning("Opik SDK not installed. Install with: pip install opik")
        return get_mock_data()
    except Exception as e:
        st.error(f"Error connecting to Opik: {str(e)}")
        return get_mock_data()