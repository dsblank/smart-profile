import streamlit as st
import os
from datetime import datetime, timedelta, timezone
from collections import Counter
import json
from opik import track
from opik.integrations.openai import track_openai

import openai


SYSTEM_PROMPT = """
You are an expert AI/ML development analyst who specializes in understanding developer workflows, productivity patterns, and technical insights from development platform data.

Your task is to analyze comprehensive user activity data from two major AI/ML platforms:
1. **Opik** - An LLM evaluation and observability platform for AI application development
2. **Comet ML** - An ML experiment tracking and model management platform

Based on the provided data, generate a professional, insightful summary that reads like it was written by a senior AI/ML engineer who has been observing this developer's work patterns.

## Analysis Focus Areas:

### 1. **Development Activity & Patterns**
- What types of AI/ML projects are they working on?
- Recent development activity and productivity trends
- Project focus areas and technical domains
- Development velocity and consistency

### 2. **Technical Stack & Tool Preferences**
- Preferred frameworks (PyTorch, TensorFlow, etc.)
- Libraries and tools being used
- Model types and architectures
- Development environment preferences

### 3. **AI/LLM Development Insights**
- LLM application patterns and use cases
- Prompt engineering approaches
- Model evaluation strategies
- Cost optimization and efficiency

### 4. **Experimentation & Methodology**
- Experiment design patterns
- Dataset management approaches
- Evaluation metrics and success criteria
- Reproducibility practices

### 5. **Professional Recommendations**
- Areas for improvement or optimization
- Suggested tools or techniques
- Best practices alignment
- Growth opportunities

## Response Format:
Structure your response as a professional development summary with:

**🔍 Recent Activity Overview**
- Brief summary of what they've been working on
- Key projects and focus areas

**🛠️ Technical Stack & Preferences**
- Frameworks, libraries, and tools
- Development patterns observed

**🤖 AI/LLM Development Insights**
- LLM usage patterns and applications
- Evaluation and optimization approaches

**📊 Experimentation & Data Management**
- Experiment design and tracking
- Dataset and model management

**💡 Professional Recommendations**
- 2-3 specific, actionable suggestions
- Growth opportunities and optimizations

## Writing Style:
- Professional yet conversational tone
- Focus on insights rather than just data summary
- Include specific technical details when relevant
- Provide actionable recommendations
- Avoid overly technical jargon
- Make it engaging and valuable for the developer

## Important Notes:
- If data is missing from either platform, acknowledge it gracefully
- Focus on patterns and trends rather than absolute numbers
- Emphasize practical insights over raw statistics
- Consider the developer's apparent skill level and experience
- Provide forward-looking recommendations"""


def get_mock_data():
    """Fallback mock data if Opik SDK is not available"""
    return {
        "recent_traces": [
            {
                "id": "trace_1",
                "name": "customer_query",
                "start_time": datetime.now(timezone.utc) - timedelta(hours=1),
                "duration": 850,
                "cost": 0.05,
            },
            {
                "id": "trace_2",
                "name": "product_search",
                "start_time": datetime.now(timezone.utc) - timedelta(hours=3),
                "duration": 1200,
                "cost": 0.08,
            },
            {
                "id": "trace_3",
                "name": "support_chat",
                "start_time": datetime.now(timezone.utc) - timedelta(days=1),
                "duration": 600,
                "cost": 0.03,
            },
        ],
        "recent_datasets": [
            {
                "name": "customer_queries_v2",
                "created_at": datetime.now(timezone.utc) - timedelta(days=2),
            },
            {
                "name": "product_catalog",
                "created_at": datetime.now(timezone.utc) - timedelta(days=5),
            },
        ],
        "recent_experiments": [
            {
                "name": "prompt_optimization_v3",
                "created_at": datetime.now(timezone.utc) - timedelta(hours=6),
            },
            {
                "name": "response_quality_test",
                "created_at": datetime.now(timezone.utc) - timedelta(days=1),
            },
        ],
        "metrics": {
            "total_traces": 156,
            "total_cost": 23.45,
            "avg_duration": 0.85,
            "error_count": 3,
            "span_count": 234,
        },
    }


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_opik_data(_opik_client, _opik_api, workspace_name):
    """Get actual data from Opik SDK"""
    client = _opik_client
    client_api = _opik_api
    try:
        # Get all projects
        projects_page = client_api.projects.find_projects()
        all_projects = projects_page.content

        # Filter projects that have had traces within the past 3 days
        three_days_ago = datetime.now(timezone.utc) - timedelta(days=3)
        active_projects = [
            project
            for project in all_projects
            if project.last_updated_trace_at
            and project.last_updated_trace_at >= three_days_ago
        ]

        # Get traces from all active projects
        all_traces = []
        for project in active_projects:
            project_traces = client.search_traces(
                max_results=1000, project_name=project.name, truncate=True
            )
            all_traces.extend(project_traces)

        # Filter traces to only include those from the past 3 days
        traces = [
            trace
            for trace in all_traces
            if trace.last_updated_at and trace.last_updated_at >= three_days_ago
        ]

        # Get datasets using OpikApi
        datasets_page = client_api.datasets.find_datasets()
        all_datasets = datasets_page.content

        # Calculate metrics from filtered data (past 3 days only)
        total_traces = len(traces)
        total_cost = sum(trace.total_estimated_cost or 0 for trace in traces)

        # Calculate average duration in seconds
        durations = [trace.duration for trace in traces if trace.duration]
        avg_duration = (
            sum(durations) / len(durations) / 1000 if durations else 0
        )  # Convert ms to seconds

        # Count spans with errors
        error_traces = [trace for trace in traces if trace.error_info]
        error_count = len(error_traces)

        # Get recent traces for quick links
        recent_traces = []
        for trace in traces[:5]:  # Get last 5 traces
            recent_traces.append(
                {
                    "id": trace.id,
                    "project_id": trace.project_id,
                    "name": trace.name or "Unnamed trace",
                    "start_time": trace.last_updated_at,
                    "duration": trace.duration,
                    "cost": trace.total_estimated_cost,
                }
            )

        # Get recent datasets for quick links - sort by last_updated_at
        sorted_datasets = sorted(
            all_datasets,
            key=lambda d: d.last_updated_at
            or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        recent_datasets = []
        for dataset in sorted_datasets[:5]:  # Get last 5 datasets
            recent_datasets.append(
                {
                    "id": dataset.id,
                    "name": dataset.name,
                    "created_at": dataset.last_updated_at,
                }
            )

        return {
            "active_projects": active_projects,
            "recent_traces": recent_traces,
            "recent_datasets": recent_datasets,
            "recent_experiments": [],  # Will implement later
            "metrics": {
                "total_traces": total_traces,
                "total_cost": total_cost,
                "avg_duration": avg_duration,
                "error_count": error_count,
                "span_count": sum(trace.span_count or 0 for trace in traces),
            },
        }

    except Exception as e:
        st.error(f"Error connecting to Opik: {str(e)}")
        return get_mock_data()

@track
@st.cache_data(ttl=600)  # Cache for 10 minutes
def compile_ai_summary_data(_comet_api, _opik_api, _opik_client, workspace_name):
    """Compile comprehensive user activity data for AI summary generation"""
    client_api = _opik_api
    client = _opik_client
    try:
        # Time ranges for analysis
        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
        three_days_ago = datetime.now(timezone.utc) - timedelta(days=3)

        # === PROJECT ANALYSIS ===
        projects_page = client_api.projects.find_projects()
        all_projects = projects_page.content

        # Sort projects by most recent activity and take top 3
        active_projects = sorted(
            [project for project in all_projects if project.last_updated_trace_at],
            key=lambda p: p.last_updated_trace_at,
            reverse=True,
        )[:3]

        project_analysis = {
            "total_projects": len(all_projects),
            "active_projects_3d": [],
            "active_projects_7d": [],
            "project_creation_pattern": {},
            "project_names": [p.name for p in all_projects],
        }

        for project in active_projects:
            # All selected projects are recent (since we sorted by activity)
            project_analysis["active_projects_3d"].append(
                {
                    "name": project.name,
                    "last_updated": project.last_updated_trace_at.isoformat(),
                    "description": project.description or "No description",
                }
            )

        # === TRACE & SPAN ANALYSIS ===
        trace_analysis = {
            "total_traces_3d": 0,
            "total_spans_3d": 0,
            "trace_patterns": {},
            "model_usage": Counter(),
            "provider_usage": Counter(),
            "error_patterns": [],
            "cost_breakdown": {"total_cost": 0, "by_model": Counter()},
            "trace_names": Counter(),
            "input_output_examples": [],
        }

        # Get traces and spans from active projects
        for project in active_projects:
            try:
                # Get traces
                traces = client_api.traces.get_traces_by_project(
                    project_id=project.id, size=100
                )
                recent_traces = [
                    t
                    for t in traces.content
                    if t.last_updated_at and t.last_updated_at >= three_days_ago
                ]

                trace_analysis["total_traces_3d"] += len(recent_traces)

                for trace in recent_traces[:10]:  # Analyze up to 10 traces per project
                    # Trace name patterns
                    trace_analysis["trace_names"][trace.name or "unnamed"] += 1

                    # Error tracking
                    if trace.error_info:
                        trace_analysis["error_patterns"].append(
                            {
                                "trace_name": trace.name,
                                "error": str(trace.error_info)[
                                    :200
                                ],  # Truncate long errors
                            }
                        )

                    # Input/output examples (first few)
                    if (
                        len(trace_analysis["input_output_examples"]) < 5
                        and trace.input
                        and trace.output
                    ):
                        trace_analysis["input_output_examples"].append(
                            {
                                "trace_name": trace.name,
                                "input_summary": (
                                    str(trace.input)[:200] + "..."
                                    if len(str(trace.input)) > 200
                                    else str(trace.input)
                                ),
                                "output_summary": (
                                    str(trace.output)[:200] + "..."
                                    if len(str(trace.output)) > 200
                                    else str(trace.output)
                                ),
                            }
                        )

                # Get spans for this project
                spans = client_api.spans.get_spans_by_project(
                    project_id=project.id, size=100
                )
                recent_spans = [
                    s
                    for s in spans.content
                    if s.last_updated_at and s.last_updated_at >= three_days_ago
                ]

                trace_analysis["total_spans_3d"] += len(recent_spans)

                for span in recent_spans:
                    # Model and provider usage
                    if span.model:
                        trace_analysis["model_usage"][span.model] += 1
                    if span.provider:
                        trace_analysis["provider_usage"][span.provider] += 1

                    # Cost analysis
                    if span.usage and "total_tokens" in span.usage:
                        # Estimate cost (rough calculation)
                        estimated_cost = (
                            span.usage.get("total_tokens", 0) * 0.00001
                        )  # Rough estimate
                        trace_analysis["cost_breakdown"]["total_cost"] += estimated_cost
                        if span.model:
                            trace_analysis["cost_breakdown"]["by_model"][
                                span.model
                            ] += estimated_cost

            except Exception as e:
                st.warning(f"Error analyzing project {project.name}: {e}")
                continue

        # === DATASET ANALYSIS ===
        datasets_page = client_api.datasets.find_datasets()
        all_datasets = datasets_page.content

        dataset_analysis = {
            "total_datasets": len(all_datasets),
            "recent_datasets": [],
            "dataset_names": [d.name for d in all_datasets],
            "dataset_creation_pattern": {},
        }

        # Sort datasets by last update
        sorted_datasets = sorted(
            all_datasets,
            key=lambda d: d.last_updated_at
            or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        for dataset in sorted_datasets[:5]:  # Analyze top 5 recent datasets
            dataset_info = {
                "name": dataset.name,
                "description": getattr(dataset, "description", "No description"),
                "last_updated": (
                    dataset.last_updated_at.isoformat()
                    if dataset.last_updated_at
                    else None
                ),
                "created_at": (
                    dataset.created_at.isoformat() if dataset.created_at else None
                ),
            }
            dataset_analysis["recent_datasets"].append(dataset_info)

        # === EXPERIMENT ANALYSIS ===
        experiment_analysis = {
            "total_experiments": 0,
            "recent_experiments": [],
            "experiment_patterns": {},
        }

        # Try to get experiments (note: might need different API call structure)
        try:
            for project in all_projects[:3]:  # Check first 3 projects for experiments
                experiments = client_api.experiments.find_experiments(size=10)
                if experiments.content:
                    experiment_analysis["total_experiments"] += len(experiments.content)
                    for exp in experiments.content[:3]:  # Top 3 experiments
                        experiment_analysis["recent_experiments"].append(
                            {
                                "name": exp.name,
                                "dataset_name": getattr(
                                    exp, "dataset_name", "Unknown dataset"
                                ),
                                "created_at": (
                                    exp.created_at.isoformat()
                                    if exp.created_at
                                    else None
                                ),
                            }
                        )
        except Exception as e:
            st.warning(f"Could not fetch experiments: {e}")

        # === PROMPT ANALYSIS ===
        prompt_analysis = {"total_prompts": 0, "recent_prompts": []}

        try:
            prompts = client_api.prompts.get_prompts(size=10)
            if prompts.content:
                prompt_analysis["total_prompts"] = len(prompts.content)
                for prompt in prompts.content[:5]:
                    prompt_analysis["recent_prompts"].append(
                        {
                            "name": prompt.name,
                            "description": getattr(
                                prompt, "description", "No description"
                            ),
                            "created_at": (
                                prompt.created_at.isoformat()
                                if prompt.created_at
                                else None
                            ),
                        }
                    )
        except Exception as e:
            st.warning(f"Could not fetch prompts: {e}")

        # === COMPILE FINAL DATA FOR AI ===
        ai_summary_data = {
            "workspace_info": {
                "workspace_name": workspace_name,
                "analysis_period": "Past 3-7 days",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            "project_activity": project_analysis,
            "trace_and_span_activity": trace_analysis,
            "dataset_activity": dataset_analysis,
            "experiment_activity": experiment_analysis,
            "prompt_activity": prompt_analysis,
            "opik_platform_info": {
                "description": "Opik is an LLM evaluation and observability platform that helps developers trace, evaluate, and optimize their AI applications. Users can track traces, manage datasets, run experiments, create prompts, and monitor AI model performance.",
                "key_features": [
                    "LLM tracing",
                    "Dataset management",
                    "Experiment tracking",
                    "Prompt versioning",
                    "Model evaluation",
                    "Cost monitoring",
                ],
            },
        }

        return ai_summary_data

    except Exception as e:
        st.error(f"Error compiling AI summary data: {e}")
        return {"error": str(e)}


@track
@st.cache_data(ttl=600)  # Cache for 10 minutes
def compile_comet_ml_summary_data(_comet_api, workspace_name):
    """Compile comprehensive user activity data from Comet ML for AI summary generation"""
    api = _comet_api
    try:
        # Time ranges for analysis
        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
        three_days_ago = datetime.now(timezone.utc) - timedelta(days=3)

        # === WORKSPACE ANALYSIS ===
        workspaces = api.get_workspaces()
        workspace_analysis = {
            "total_workspaces": len(workspaces) if workspaces else 0,
            "workspace_names": workspaces or [],
            "target_workspace": workspace_name,
        }

        # === PROJECT ANALYSIS ===
        projects = api.get_projects(workspace_name)
        project_analysis = {
            "total_projects": len(projects),
            "project_names": projects[:20],  # Limit to first 20 for AI processing
            "active_projects": [],
            "project_details": [],
        }

        # === EXPERIMENT ANALYSIS ===
        experiment_analysis = {
            "total_experiments": 0,
            "recent_experiments": [],
            "experiment_patterns": {
                "duration_stats": [],
                "file_patterns": Counter(),
                "tag_usage": Counter(),
                "parameter_patterns": Counter(),
                "metric_patterns": Counter(),
            },
            "framework_usage": Counter(),
            "library_usage": Counter(),
            "system_info": {
                "python_versions": Counter(),
                "os_types": Counter(),
                "hostnames": Counter(),
            },
        }

        # Find projects with most recent activity
        project_last_activity = []
        for project_name in projects:
            try:
                experiments = api.get_experiments(workspace_name, project_name)
                if not experiments:
                    continue

                # Find the most recent experiment in this project
                latest_experiment_time = max(
                    exp.end_server_timestamp or exp.start_server_timestamp or 0
                    for exp in experiments
                )

                project_last_activity.append(
                    {
                        "name": project_name,
                        "experiment_count": len(experiments),
                        "last_activity": latest_experiment_time,
                        "experiments": experiments,
                    }
                )

            except Exception as e:
                continue

        # Sort by most recent activity and take top 3
        project_last_activity.sort(key=lambda x: x["last_activity"], reverse=True)
        most_active_projects = project_last_activity[:3]

        # Analyze experiments from most recently active projects
        for project_data in most_active_projects:
            project_name = project_data["name"]
            experiments = project_data["experiments"]

            project_analysis["active_projects"].append(
                {
                    "name": project_name,
                    "experiment_count": len(experiments),
                    "last_activity": project_data["last_activity"],
                }
            )

            experiment_analysis["total_experiments"] += len(experiments)

            # Analyze recent experiments from this active project
            for exp in experiments[:10]:
                # Basic experiment info
                experiment_info = {
                    "id": exp.id,
                    "name": exp.name,
                    "project": project_name,
                    "duration_ms": exp.duration_millis,
                    "start_time": exp.start_server_timestamp,
                    "end_time": exp.end_server_timestamp,
                    "file_name": exp.file_name,
                    "file_path": exp.file_path,
                }

                # File pattern analysis
                if exp.file_name:
                    file_ext = (
                        exp.file_name.split(".")[-1]
                        if "." in exp.file_name
                        else "no_ext"
                    )
                    experiment_analysis["experiment_patterns"]["file_patterns"][
                        file_ext
                    ] += 1

                # Duration statistics
                if exp.duration_millis:
                    experiment_analysis["experiment_patterns"]["duration_stats"].append(
                        exp.duration_millis
                    )

                # Get detailed experiment data (first 5 experiments total for performance)
                if len(experiment_analysis["recent_experiments"]) < 5:
                    try:
                        # Get metadata
                        metadata = exp.get_metadata()

                        # Get tags
                        tags = exp.get_tags()
                        if tags:
                            for tag in tags:
                                experiment_analysis["experiment_patterns"]["tag_usage"][
                                    tag
                                ] += 1

                        # Get parameters
                        params = exp.get_parameters_summary()
                        if params:
                            for param in params:
                                experiment_analysis["experiment_patterns"][
                                    "parameter_patterns"
                                ][param["name"]] += 1

                        # Get metrics
                        metrics = exp.get_metrics_summary()
                        if metrics:
                            for metric in metrics:
                                experiment_analysis["experiment_patterns"][
                                    "metric_patterns"
                                ][metric["name"]] += 1

                        # Get installed packages to infer framework usage
                        packages = exp.get_installed_packages()
                        if packages:
                            for package in packages:
                                pkg_name = package.split("==")[0].lower()
                                # Identify ML frameworks
                                if any(
                                    framework in pkg_name
                                    for framework in ["torch", "pytorch"]
                                ):
                                    experiment_analysis["framework_usage"][
                                        "PyTorch"
                                    ] += 1
                                elif any(
                                    framework in pkg_name
                                    for framework in ["tensorflow", "tf"]
                                ):
                                    experiment_analysis["framework_usage"][
                                        "TensorFlow"
                                    ] += 1
                                elif "keras" in pkg_name:
                                    experiment_analysis["framework_usage"]["Keras"] += 1
                                elif "sklearn" in pkg_name:
                                    experiment_analysis["framework_usage"][
                                        "scikit-learn"
                                    ] += 1
                                elif "xgboost" in pkg_name:
                                    experiment_analysis["framework_usage"][
                                        "XGBoost"
                                    ] += 1
                                elif "lightgbm" in pkg_name:
                                    experiment_analysis["framework_usage"][
                                        "LightGBM"
                                    ] += 1
                                elif "pandas" in pkg_name:
                                    experiment_analysis["library_usage"]["pandas"] += 1
                                elif "numpy" in pkg_name:
                                    experiment_analysis["library_usage"]["numpy"] += 1
                                elif "matplotlib" in pkg_name:
                                    experiment_analysis["library_usage"][
                                        "matplotlib"
                                    ] += 1
                                elif "seaborn" in pkg_name:
                                    experiment_analysis["library_usage"]["seaborn"] += 1

                        # Get system info
                        try:
                            python_version = exp.get_python_version()
                            if python_version:
                                experiment_analysis["system_info"]["python_versions"][
                                    python_version
                                ] += 1
                        except:
                            pass

                        try:
                            os_type = exp.get_os_type()
                            if os_type:
                                experiment_analysis["system_info"]["os_types"][
                                    os_type
                                ] += 1
                        except:
                            pass

                        try:
                            hostname = exp.get_hostname()
                            if hostname:
                                experiment_analysis["system_info"]["hostnames"][
                                    hostname
                                ] += 1
                        except:
                            pass

                        # Get asset information
                        try:
                            assets = exp.get_asset_list()
                            asset_info = {
                                "total_assets": len(assets) if assets else 0,
                                "asset_types": Counter(),
                                "asset_sizes": [],
                            }

                            if assets:
                                for asset in assets:
                                    asset_info["asset_types"][
                                        asset.get("type", "unknown")
                                    ] += 1
                                    if asset.get("fileSize"):
                                        asset_info["asset_sizes"].append(
                                            asset["fileSize"]
                                        )

                            experiment_info["assets"] = asset_info
                        except:
                            pass

                        experiment_analysis["recent_experiments"].append(
                            experiment_info
                        )

                    except Exception as e:
                        # Skip experiments that fail to load details
                        continue

        # === MODEL REGISTRY ANALYSIS ===
        model_analysis = {"total_models": 0, "model_names": [], "model_details": []}

        # Try to get model information (sample common model names)
        common_model_names = [
            "test-model",
            "model",
            "pytorch-model",
            "tensorflow-model",
            "production-model",
        ]
        for model_name in common_model_names:
            try:
                model = api.get_model(workspace_name, model_name)
                if model:
                    model_analysis["total_models"] += 1
                    model_analysis["model_names"].append(model_name)

                    # Get model details if available
                    try:
                        model_details = {
                            "name": model_name,
                            "workspace": workspace_name,
                        }
                        model_analysis["model_details"].append(model_details)
                    except:
                        pass
            except:
                continue

        # === CALCULATE SUMMARY STATISTICS ===
        # Duration statistics
        if experiment_analysis["experiment_patterns"]["duration_stats"]:
            durations = experiment_analysis["experiment_patterns"]["duration_stats"]
            duration_summary = {
                "avg_duration_ms": sum(durations) / len(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "total_experiments_with_duration": len(durations),
            }
        else:
            duration_summary = {
                "avg_duration_ms": 0,
                "min_duration_ms": 0,
                "max_duration_ms": 0,
                "total_experiments_with_duration": 0,
            }

        # === COMPILE FINAL DATA FOR AI ===
        comet_ml_summary_data = {
            "workspace_info": {
                "workspace_name": workspace_name,
                "analysis_period": "All available data",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            "workspace_activity": workspace_analysis,
            "project_activity": project_analysis,
            "experiment_activity": experiment_analysis,
            "model_activity": model_analysis,
            "summary_statistics": {
                "duration_summary": duration_summary,
                "top_frameworks": dict(
                    experiment_analysis["framework_usage"].most_common(5)
                ),
                "top_libraries": dict(
                    experiment_analysis["library_usage"].most_common(5)
                ),
                "top_tags": dict(
                    experiment_analysis["experiment_patterns"]["tag_usage"].most_common(
                        5
                    )
                ),
                "top_parameters": dict(
                    experiment_analysis["experiment_patterns"][
                        "parameter_patterns"
                    ].most_common(5)
                ),
                "top_metrics": dict(
                    experiment_analysis["experiment_patterns"][
                        "metric_patterns"
                    ].most_common(5)
                ),
                "file_extensions": dict(
                    experiment_analysis["experiment_patterns"][
                        "file_patterns"
                    ].most_common(5)
                ),
                "system_summary": {
                    "python_versions": dict(
                        experiment_analysis["system_info"]["python_versions"]
                    ),
                    "os_types": dict(experiment_analysis["system_info"]["os_types"]),
                    "unique_hostnames": len(
                        experiment_analysis["system_info"]["hostnames"]
                    ),
                },
            },
            "comet_ml_platform_info": {
                "description": "Comet ML is a comprehensive ML experiment tracking and model management platform that helps data scientists and ML engineers track experiments, manage models, collaborate on projects, and deploy ML models to production.",
                "key_features": [
                    "Experiment tracking",
                    "Model versioning",
                    "Hyperparameter optimization",
                    "Collaborative workspaces",
                    "Model deployment",
                    "Asset management",
                    "System monitoring",
                ],
            },
        }

        return comet_ml_summary_data

    except Exception as e:
        st.error(f"Error compiling Comet ML summary data: {e}")
        return {"error": str(e)}

@track
def get_summary_from_openai(client, system_prompt, user_prompt):
    """
    Get a summary from OpenAI using the provided system and user prompts.

    Args:
        client: OpenAI client instance
        system_prompt: System prompt for the AI model
        user_prompt: User prompt containing the data to analyze

    Returns:
        str: AI generated summary
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=5000,  # Reasonable length for summary
            temperature=0.3,
            presence_penalty=0.1,
            frequency_penalty=0.1,
        )
        return response
    except Exception as e:
        st.error(f"Error generating summary from OpenAI: {e}")
        return str(e)

@track
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def generate_ai_summary(
    openai_api_key, _comet_api, _opik_api, _opik_client, workspace_name
):
    """
    Generate an AI summary using OpenAI based on combined data from Opik and Comet ML platforms.

    Args:
        openai_api_key: OpenAI API key
        opik_api_key: Opik API key
        comet_ml_api_key: Comet ML API key
        workspace_name: Workspace name for both platforms

    Returns:
        dict: AI generated summary with insights and recommendations
    """
    try:
        if not openai_api_key:
            return {"error": "OpenAI API key is required"}

        # Initialize OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)
        client = track_openai(client)

        # Gather data from both platforms
        # st.info("🤖 Gathering data from Opik platform...")
        opik_data = compile_ai_summary_data(
            _comet_api, _opik_api, _opik_client, workspace_name=workspace_name
        )

        # st.info("🧪 Gathering data from Comet ML platform...")
        comet_ml_data = compile_comet_ml_summary_data(
            _comet_api=_comet_api, workspace_name=workspace_name
        )

        # Check for errors in data collection
        if opik_data.get("error"):
            st.warning(f"Opik data collection issue: {opik_data['error']}")
        if comet_ml_data.get("error"):
            st.warning(f"Comet ML data collection issue: {comet_ml_data['error']}")

        # Create combined data structure
        combined_data = {
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "workspace_name": workspace_name,
            "opik_data": opik_data,
            "comet_ml_data": comet_ml_data,
        }

        # Create the user prompt with the actual data
        USER_PROMPT = f"""
Please analyze the following comprehensive AI/ML development data and provide insights:

**Workspace:** {workspace_name}
**Analysis Period:** Recent activity (past 3-7 days focus)

**Platform Data:**
```json
{json.dumps(combined_data, indent=2, default=str)}
```
Generate a professional development summary following the format and guidelines provided in your system prompt.
"""
        # Extract the generated summary
        response = get_summary_from_openai(client, SYSTEM_PROMPT, USER_PROMPT)
        ai_summary = response.choices[0].message.content

        # Return structured result
        return {
            "success": True,
            "ai_summary": ai_summary,
            "data_sources": {
                "opik_projects": opik_data.get("project_activity", {}).get(
                    "total_projects", 0
                ),
                "opik_traces": opik_data.get("trace_and_span_activity", {}).get(
                    "total_traces_3d", 0
                ),
                "comet_ml_projects": comet_ml_data.get("project_activity", {}).get(
                    "total_projects", 0
                ),
                "comet_ml_experiments": comet_ml_data.get(
                    "experiment_activity", {}
                ).get("total_experiments", 0),
            },
            "model_used": "gpt-4o",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "token_usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

    except Exception as e:
        st.error(f"Error generating AI summary: {e}")
        return {"error": str(e)}

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def generate_user_badges(openai_api_key="", ai_summary=""):
    """
    Generate 4 user badges based on AI summary analysis.
    
    Args:
        openai_api_key: OpenAI API key
        ai_summary: The AI-generated user summary
    
    Returns:
        dict: List of 4 badges with labels and colors
    """
    try:
        if not openai_api_key:
            return {"error": "OpenAI API key is required"}
            
        if not ai_summary:
            return {"error": "AI summary is required"}
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)
        
        # Create the system prompt for badge generation
        badge_system_prompt = """You are an expert at creating concise, meaningful user achievement badges based on developer activity analysis.

Your task is to analyze an AI-generated developer summary and create exactly 4 badges that capture the most important aspects of their development profile.

## Badge Guidelines:

### Badge Categories (choose 4 from these types):
1. **Technical Stack Badges** - Primary frameworks/tools (e.g., "🔥 PyTorch Pro", "⚡ TensorFlow Expert", "🚀 LLM Builder")
2. **Activity Level Badges** - Development intensity (e.g., "📈 Active Builder", "⚡ Rapid Prototyper", "🔄 Daily Coder")
3. **Specialty Badges** - Domain expertise (e.g., "🤖 AI Engineer", "📊 ML Researcher", "💬 Chatbot Specialist")
4. **Achievement Badges** - Notable accomplishments (e.g., "🎯 100+ Experiments", "💰 Cost Optimizer", "🔍 Debug Master")
5. **Methodology Badges** - Approach patterns (e.g., "📋 Systematic Tester", "🔬 Data Scientist", "🎨 Prompt Artist")

### Badge Format Requirements:
- Each badge must be 2-4 words maximum
- Include one relevant emoji at the start
- Use dynamic, engaging language
- Focus on standout characteristics
- Avoid generic terms like "Developer" or "User"

### Badge Colors (choose appropriate shield.io colors):
- red, orange, yellow, green, blue, purple, pink, brown, grey, black
- brightgreen, lightgrey, etc.

## Response Format:
Return exactly 4 badges in this JSON format:
```json
{
  "badges": [
    {"label": "🔥 PyTorch Pro", "color": "red"},
    {"label": "🤖 LLM Builder", "color": "blue"},
    {"label": "⚡ Active Coder", "color": "green"},
    {"label": "💰 Cost Master", "color": "orange"}
  ]
}
```

## Important Notes:
- Create badges that feel authentic to this specific developer
- Prioritize their most distinctive characteristics
- Make badges feel earned rather than generic
- Focus on recent activity patterns when possible
- Ensure variety across the 4 badge types"""

        # Create the user prompt with the AI summary
        badge_user_prompt = f"""Based on the following AI-generated developer summary, create exactly 4 meaningful user badges that capture this developer's key characteristics, technical strengths, and achievements:

**Developer Summary:**
{ai_summary}

Generate 4 badges that authentically represent this developer's profile. Focus on their standout characteristics, technical expertise, activity patterns, and notable achievements."""

        # Make the API call to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cost-effective model for simple task
            messages=[
                {"role": "system", "content": badge_system_prompt},
                {"role": "user", "content": badge_user_prompt}
            ],
            max_tokens=300,  # Short response needed
            temperature=0.7,  # Some creativity for badge variety
            response_format={"type": "json_object"}  # Ensure JSON response
        )
        
        # Parse the JSON response
        import json
        badge_data = json.loads(response.choices[0].message.content)
        
        # Validate response structure
        if "badges" not in badge_data or len(badge_data["badges"]) != 4:
            return {"error": "Invalid badge response format"}
        
        # Return structured result
        return {
            "success": True,
            "badges": badge_data["badges"],
            "model_used": "gpt-4o-mini",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "token_usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        
    except ImportError as e:
        st.error(f"OpenAI library not available: {e}")
        return {"error": f"OpenAI library not available: {e}"}
    except Exception as e:
        st.error(f"Error generating badges: {e}")
        return {"error": str(e)}
