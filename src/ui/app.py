"""
Main Streamlit application for HIPO - Multi-Cloud Kubernetes Infrastructure for ML/LLM
"""
import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
from pathlib import Path

# Import secure weights UI
from src.ui.secure_weights import render_secure_weights_ui

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.config.config import load_config
from src.cloud.factory import CloudProviderFactory
from src.kubernetes.orchestrator import KubernetesOrchestrator

# Set page config
st.set_page_config(
    page_title="HIPO - Multi-Cloud K8s ML Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit app configuration
def local_css():
    # CSS is now handled through pure Streamlit components
    # This function remains in case we need to add minor custom styling later
    pass

# Apply the CSS
local_css()

# Title
st.title("HIPO - Multi-Cloud Kubernetes ML Platform")
st.subheader("Manage and deploy ML/LLM models across multiple cloud providers with Kubernetes")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Model Deployment", "Model Inference", "Configuration", "Monitoring", "Logs", "Secure Weights"],
)


# Mock functions for demo purposes
def get_cluster_status():
    """Get status of Kubernetes clusters across cloud providers"""
    return {
        "aws": {"status": "Running", "nodes": 4, "pods": 12, "region": "us-west-2", "cost_per_hour": 2.14},
        "gcp": {"status": "Running", "nodes": 3, "pods": 10, "region": "us-central1", "cost_per_hour": 1.87},
    }


def get_deployed_models():
    """Get list of deployed models"""
    return [
        {"name": "llama2-7b", "version": "1.0", "provider": "aws", "endpoints": 2, "status": "Running"},
        {"name": "bert-base", "version": "2.1", "provider": "gcp", "endpoints": 1, "status": "Running"},
        {"name": "gpt-j-6b", "version": "1.2", "provider": "aws", "endpoints": 3, "status": "Scaling"},
    ]


def get_resource_usage():
    """Get resource usage metrics"""
    return {
        "cpu_usage": [65, 58, 72, 59, 63, 75, 68],
        "memory_usage": [72, 75, 70, 68, 75, 80, 78],
        "gpu_usage": [45, 52, 58, 62, 48, 55, 60],
        "network": [120, 132, 145, 89, 97, 105, 110],
        "timestamps": ["12:00", "12:10", "12:20", "12:30", "12:40", "12:50", "13:00"],
    }


def get_request_metrics():
    """Get API request metrics"""
    return {
        "total_requests": 15243,
        "success_rate": 99.2,
        "avg_latency": 213,  # in ms
        "p95_latency": 350,  # in ms
        "requests_per_model": {"llama2-7b": 8731, "bert-base": 3245, "gpt-j-6b": 3267},
    }


def get_cost_metrics():
    """Get cost metrics"""
    return {
        "total_cost_today": 156.32,
        "total_cost_month": 3245.87,
        "cost_by_provider": {"aws": 1876.43, "gcp": 1369.44},
        "cost_by_model": {"llama2-7b": 1542.31, "bert-base": 723.45, "gpt-j-6b": 980.11},
    }


def load_model_configs():
    """Load available model configurations"""
    # In a real implementation, this would load from the config directory
    return [
        {
            "name": "llama2-7b",
            "type": "LLM",
            "parameters": "7B",
            "requirements": {"gpu_memory": "16GB", "cpu": "4", "memory": "32GB"},
        },
        {
            "name": "bert-base",
            "type": "Embedding",
            "parameters": "110M",
            "requirements": {"gpu_memory": "8GB", "cpu": "2", "memory": "16GB"},
        },
        {
            "name": "gpt-j-6b",
            "type": "LLM",
            "parameters": "6B",
            "requirements": {"gpu_memory": "16GB", "cpu": "4", "memory": "32GB"},
        },
    ]


# Dashboard page
if page == "Dashboard":
    # Layout with columns
    col1, col2 = st.columns(2)

    with col1:
        st.header("Cloud Infrastructure")

        cluster_status = get_cluster_status()
        for provider, status in cluster_status.items():
            # Create a card-like container with Streamlit components
            with st.container():
                st.markdown(f"### {provider.upper()} Cluster - {status['region']}")
                
                # Display status with colored indicator
                status_color = ":green[Running]" if status['status'] == "Running" else ":red[Error]"
                st.markdown(f"**Status:** {status_color}")
                
                # Create metrics using Streamlit's metric component
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric(label="Nodes", value=status['nodes'])
                with metric_cols[1]:
                    st.metric(label="Pods", value=status['pods'])
                with metric_cols[2]:
                    st.metric(label="Cost/Hour", value=f"${status['cost_per_hour']:.2f}")
                
                # Add separator between providers
                st.divider()

    with col2:
        st.header("Model Deployment Status")

        models = get_deployed_models()
        for model in models:
            # Create a card-like container for each model
            with st.container():
                st.markdown(f"### {model['name']} v{model['version']}")
                st.markdown(f"**Provider:** {model['provider'].upper()}")
                
                # Display status with colored indicator
                status_color = ":green[Running]" if model["status"] == "Running" else ":orange[Scaling]"
                st.markdown(f"**Status:** {status_color}")
                
                # Show endpoints metric
                st.metric(label="Endpoints", value=model['endpoints'])
                
                # Add separator between models
                st.divider()

    # Resource usage metrics
    st.header("Resource Usage")
    metrics = get_resource_usage()

    # Create tabs for different metrics
    tab1, tab2, tab3, tab4 = st.tabs(["CPU", "Memory", "GPU", "Network"])

    with tab1:
        fig = px.line(
            x=metrics["timestamps"],
            y=metrics["cpu_usage"],
            labels={"x": "Time", "y": "CPU Usage (%)"},
            title="CPU Usage",
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.line(
            x=metrics["timestamps"],
            y=metrics["memory_usage"],
            labels={"x": "Time", "y": "Memory Usage (%)"},
            title="Memory Usage",
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.line(
            x=metrics["timestamps"],
            y=metrics["gpu_usage"],
            labels={"x": "Time", "y": "GPU Usage (%)"},
            title="GPU Usage",
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = px.line(
            x=metrics["timestamps"],
            y=metrics["network"],
            labels={"x": "Time", "y": "Network Traffic (MB/s)"},
            title="Network Traffic",
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Request metrics
    col1, col2 = st.columns(2)

    with col1:
        st.header("Request Metrics")
        request_metrics = get_request_metrics()

        # Create a container for request metrics
        with st.container():
            # Display metrics in a grid
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric(label="Total Requests", value=f"{request_metrics['total_requests']:,}")
            with metric_cols[1]:
                st.metric(label="Success Rate", value=f"{request_metrics['success_rate']}%")
            with metric_cols[2]:
                st.metric(label="Avg Latency", value=f"{request_metrics['avg_latency']} ms")
            with metric_cols[3]:
                st.metric(label="P95 Latency", value=f"{request_metrics['p95_latency']} ms")

        fig = px.pie(
            names=list(request_metrics["requests_per_model"].keys()),
            values=list(request_metrics["requests_per_model"].values()),
            title="Requests by Model",
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.header("Cost Metrics")
        cost_metrics = get_cost_metrics()

        # Create a container for cost metrics
        with st.container():
            # Display metrics in a grid
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric(label="Today's Cost", value=f"${cost_metrics['total_cost_today']:.2f}")
            with metric_cols[1]:
                st.metric(label="Monthly Cost", value=f"${cost_metrics['total_cost_month']:.2f}")

        # Cost by provider
        fig = px.bar(
            x=list(cost_metrics["cost_by_provider"].keys()),
            y=list(cost_metrics["cost_by_provider"].values()),
            labels={"x": "Provider", "y": "Cost ($)"},
            title="Cost by Provider",
            text_auto=".2f",
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Cost by model
        fig = px.bar(
            x=list(cost_metrics["cost_by_model"].keys()),
            y=list(cost_metrics["cost_by_model"].values()),
            labels={"x": "Model", "y": "Cost ($)"},
            title="Cost by Model",
            text_auto=".2f",
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# Model Deployment page
elif page == "Model Deployment":
    st.header("Deploy ML/LLM Models")

    col1, col2 = st.columns([1, 2])

    with col1:
        with st.container():
            st.subheader("Model Configuration")

        # Model selection
        model_configs = load_model_configs()
        model_names = [model["name"] for model in model_configs]
        selected_model = st.selectbox("Select a model", model_names)

        # Get the selected model config
        model_config = next((model for model in model_configs if model["name"] == selected_model), None)

        if model_config:
            st.write(f"Type: {model_config['type']}")
            st.write(f"Parameters: {model_config['parameters']}")

            st.write("### Hardware Requirements")
            for resource, value in model_config["requirements"].items():
                st.write(f"- {resource}: {value}")

        # Deployment options
        st.write("### Deployment Options")

        provider = st.radio("Cloud Provider", ["AWS", "GCP", "Multi-Cloud"])

        if provider == "AWS":
            region = st.selectbox("AWS Region", ["us-east-1", "us-west-2", "eu-west-1"])
        elif provider == "GCP":
            region = st.selectbox("GCP Region", ["us-central1", "us-east1", "europe-west1"])
        else:
            st.write("Using optimal regions for each provider")
            region = "Multiple"

        # Scaling options
        st.write("### Scaling Configuration")
        min_replicas = st.slider("Minimum Replicas", 1, 10, 2)
        max_replicas = st.slider("Maximum Replicas", min_replicas, 20, 5)
        target_cpu = st.slider("Target CPU Utilization (%)", 50, 90, 70)

        # Deploy button
        if st.button("Deploy Model"):
            st.success(f"Deploying {selected_model} to {provider} in {region} region...")
            st.info("This would trigger the actual deployment in a real implementation")

    with col2:
        with st.container():
            st.subheader("Deployed Models")

        models = get_deployed_models()
        df_models = pd.DataFrame(models)
        st.dataframe(df_models, use_container_width=True)

        st.write("### Model Details")

        if models:
            for i, model in enumerate(models):
                with st.expander(f"{model['name']} v{model['version']}"):
                    st.write(f"Provider: {model['provider'].upper()}")
                    st.write(f"Status: {model['status']}")
                    st.write(f"Endpoints: {model['endpoints']}")

                    # Mock data for model metrics
                    st.write("### Performance Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Avg. Latency", f"{100 + i * 20} ms")
                    col2.metric("Success Rate", f"{99.5 - i * 0.2}%")
                    col3.metric("Requests/min", f"{120 + i * 50}")

                    # Actions
                    st.write("### Actions")
                    col1, col2, col3 = st.columns(3)
                    restart = col1.button("Restart", key=f"restart_{i}")
                    scale = col2.button("Scale", key=f"scale_{i}")
                    delete = col3.button("Delete", key=f"delete_{i}")

                    if restart:
                        st.info(f"Restarting {model['name']}...")
                    if scale:
                        st.info(f"Opening scaling options for {model['name']}...")
                    if delete:
                        st.warning(f"Delete {model['name']}? This action cannot be undone.")
                        confirm = st.button("Confirm Delete", key=f"confirm_delete_{i}")
                        if confirm:
                            st.success(f"{model['name']} deletion initiated...")

        with st.container():
            st.subheader("Deployment Logs")

        # Mock deployment logs
        log_messages = [
            "[INFO] 2025-05-06 12:34:56 - Initializing deployment for llama2-7b",
            "[INFO] 2025-05-06 12:35:01 - Creating Kubernetes deployment configuration",
            "[INFO] 2025-05-06 12:35:12 - Applying Kubernetes configuration",
            "[INFO] 2025-05-06 12:36:05 - Pods scheduled on AWS cluster",
            "[INFO] 2025-05-06 12:38:41 - Model container pulling image",
            "[INFO] 2025-05-06 12:40:23 - Model container started",
            "[INFO] 2025-05-06 12:41:15 - Health check passed",
            "[INFO] 2025-05-06 12:41:30 - API gateway routes configured",
            "[SUCCESS] 2025-05-06 12:41:45 - Model deployment completed successfully",
        ]

        for log in log_messages:
            if "[SUCCESS]" in log:
                st.success(log)
            elif "[WARNING]" in log:
                st.warning(log)
            elif "[ERROR]" in log:
                st.error(log)
            else:
                st.info(log)

        # End of deployment logs section

# Model Inference page
elif page == "Model Inference":
    st.header("Model Inference")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Create a container for the Test LLM Model section
        with st.container():
            st.subheader("Test LLM Model")

        # Get deployed models
        models = get_deployed_models()
        llm_models = [model["name"] for model in models if "llama" in model["name"] or "gpt" in model["name"]]

        # Parameters
        selected_model = st.selectbox("Select LLM Model", llm_models)

        st.text_area("Prompt", height=150, value="Explain the benefits of using Kubernetes for ML model deployment")

        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.slider("Max tokens", 50, 1000, 250, 50)

        # Generate button
        if st.button("Generate Text"):
            with st.spinner("Generating response..."):
                # In a real app, this would call the model API
                st.info("This would send a request to the deployed model in a real implementation")

                # Mock response
                st.write("### Model Response")
                st.markdown(
                    """
                Kubernetes offers several benefits for ML model deployment:
                
                1. **Scalability**: Automatically scales based on demand
                2. **Resource Efficiency**: Optimal resource allocation
                3. **High Availability**: No single point of failure
                4. **Declarative Configuration**: Infrastructure as code
                5. **Portability**: Deploy on any cloud or on-premises
                6. **Cost Optimization**: Dynamic resource allocation
                
                For ML specifically, Kubernetes helps manage GPU resources efficiently and enables 
                seamless model updates with minimal downtime.
                """
                )

                # Mock metrics
                st.write("### Request Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Latency", "378 ms")
                col2.metric("Tokens Generated", "127")
                col3.metric("Cost", "$0.0023")

        # End of Test LLM Model section

    with col2:
        # Create a container for the Batch Processing section
        with st.container():
            st.subheader("Batch Processing")

        # Setup
        st.write("Upload file(s) for batch processing:")
        uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False)

        # Models
        all_models = [model["name"] for model in models]
        selected_model = st.selectbox("Select Model", all_models, key="batch_model")

        # Configuration
        st.write("### Batch Configuration")
        batch_size = st.slider("Batch Size", 1, 100, 20)
        priority = st.radio("Processing Priority", ["Low", "Medium", "High"])

        # Process button
        if st.button("Start Batch Processing"):
            if uploaded_file:
                st.success(f"Batch processing started for {uploaded_file.name}")

                # Progress simulation
                import time

                progress_bar = st.progress(0)
                status_text = st.empty()

                for i in range(101):
                    progress_bar.progress(i)
                    status_text.text(f"Processing: {i}% complete")
                    time.sleep(0.02)

                status_text.text("Processing complete!")
                st.success("Batch results are available for download")

                st.download_button(
                    label="Download Results",
                    data=b"This would be the actual results in a real implementation",
                    file_name="batch_results.csv",
                    mime="text/csv",
                )
            else:
                st.error("Please upload a file for processing")

        # End of Batch Processing section

        # Create a container for the API Integration section
        with st.container():
            st.subheader("API Integration")

            st.code(
                """
# Python example
import requests

API_URL = "https://api.example.com/v1/models/llama2-7b/generate"
API_KEY = "your_api_key_here"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "prompt": "Explain the benefits of Kubernetes",
    "max_tokens": 250,
    "temperature": 0.7
}

response = requests.post(API_URL, headers=headers, json=data)
result = response.json()
print(result["generated_text"])
            """,
                language="python",
            )

            st.markdown("See the API documentation for more endpoints and options.")

        # End of API Integration section

# Configuration page
elif page == "Configuration":
    st.header("Platform Configuration")

    # Tab navigation
    tab1, tab2, tab3 = st.tabs(["Cloud Providers", "Kubernetes", "Models"])

    with tab1:
        # Create a container for Cloud Provider Configuration
        with st.container():
            st.subheader("Cloud Provider Configuration")

        provider_tab1, provider_tab2 = st.tabs(["AWS", "GCP"])

        with provider_tab1:
            # AWS Configuration
            st.write("### AWS Configuration")

            aws_region = st.selectbox(
                "AWS Region", ["us-east-1", "us-east-2", "us-west-1", "us-west-2", "eu-west-1", "eu-central-1"]
            )

            aws_instance_type = st.selectbox(
                "Instance Type", ["g4dn.xlarge", "g4dn.2xlarge", "g5.xlarge", "g5.2xlarge", "p3.2xlarge"]
            )

            aws_min_nodes = st.slider("Minimum Nodes", 1, 10, 2)
            aws_max_nodes = st.slider("Maximum Nodes", aws_min_nodes, 50, 5)

            # Advanced options
            with st.expander("Advanced AWS Options"):
                st.checkbox("Enable EBS Optimization", value=True)
                st.checkbox("Use Spot Instances", value=False)
                st.selectbox("AMI ID", ["ami-default", "ami-custom", "Custom..."])
                st.text_input("VPC ID")
                st.text_input("Subnet IDs")
                st.text_input("Security Group IDs")

        with provider_tab2:
            # GCP Configuration
            st.write("### GCP Configuration")

            gcp_region = st.selectbox(
                "GCP Region", ["us-central1", "us-east1", "us-west1", "europe-west1", "europe-west4"]
            )

            gcp_machine_type = st.selectbox(
                "Machine Type", ["n1-standard-4", "n1-standard-8", "n1-highcpu-8", "n1-highmem-8", "a2-highgpu-1g"]
            )

            gcp_min_nodes = st.slider("Minimum Nodes", 1, 10, 2, key="gcp_min")
            gcp_max_nodes = st.slider("Maximum Nodes", gcp_min_nodes, 50, 5, key="gcp_max")

            # Advanced options
            with st.expander("Advanced GCP Options"):
                st.checkbox("Enable Preemptible VMs", value=False)
                st.text_input("Network Name")
                st.text_input("Subnetwork Name")
                st.selectbox("Disk Type", ["pd-standard", "pd-ssd"])
                st.slider("Disk Size (GB)", 100, 1000, 200, 50)

        if st.button("Save Cloud Configuration"):
            st.success("Cloud provider configuration saved successfully")

        # End of Cloud Provider Configuration

    with tab2:
        # Create a container for Kubernetes Configuration
        with st.container():
            st.subheader("Kubernetes Configuration")

        # Load a sample Kubernetes config
        try:
            with open(project_root / "config" / "kubernetes_config.yaml", "r") as f:
                k8s_config = f.read()
        except Exception:
            # Provide a sample config if file doesn't exist
            k8s_config = """apiVersion: v1
kind: ConfigMap
metadata:
  name: hipo-config
  namespace: ml-models
data:
  log_level: "INFO"
  monitoring_enabled: "true"
  auto_scaling_enabled: "true"
  default_replicas: "2"
  gpu_resource_limit: "1"
---
apiVersion: v1
kind: Secret
metadata:
  name: hipo-secrets
  namespace: ml-models
type: Opaque
data:
  # Base64 encoded values
  aws_access_key: <base64_encoded_key>
  aws_secret_key: <base64_encoded_secret>
  gcp_service_account: <base64_encoded_json>
"""

        # Configuration editor
        st.write("### Kubernetes ConfigMap and Secrets")
        edited_config = st.text_area("Edit Kubernetes Configuration", k8s_config, height=400)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Validate Configuration"):
                try:
                    yaml.safe_load(edited_config)
                    st.success("Configuration is valid YAML")
                except Exception as e:
                    st.error(f"Invalid YAML: {str(e)}")

        with col2:
            if st.button("Apply Configuration"):
                st.success("Configuration applied to cluster")
                st.info("In a real implementation, this would update the Kubernetes resources")

        st.write("### Resource Quotas")

        quota_col1, quota_col2 = st.columns(2)

        with quota_col1:
            st.number_input("CPU Limit", min_value=1, max_value=100, value=20)
            st.number_input("Memory Limit (GB)", min_value=1, max_value=1000, value=64)

        with quota_col2:
            st.number_input("GPU Limit", min_value=0, max_value=16, value=4)
            st.number_input("Storage Limit (GB)", min_value=10, max_value=10000, value=500)

        if st.button("Update Resource Quotas"):
            st.success("Resource quotas updated successfully")

        # End of Kubernetes Configuration

    with tab3:
        # Create a container for Model Configuration
        with st.container():
            st.subheader("Model Configuration")

        # Sample model config
        try:
            with open(project_root / "config" / "models" / "llama2-7b.yaml", "r") as f:
                model_config = f.read()
        except Exception:
            # Provide a sample config if file doesn't exist
            model_config = """apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: llama2-7b
  namespace: ml-models
spec:
  name: llama2
  predictors:
  - name: default
    graph:
      name: llama2-7b-container
      implementation: CUSTOM_CONTAINER
      endpoint:
        type: REST
      type: MODEL
      children: []
    replicas: 2
    annotations:
      seldon.io/svc-name: llama2-7b-api
    componentSpecs:
    - spec:
        containers:
        - name: llama2-7b-container
          image: hipo/llama2-7b:latest
          resources:
            limits:
              cpu: "4"
              memory: 32Gi
              nvidia.com/gpu: 1
            requests:
              cpu: "2"
              memory: 16Gi
              nvidia.com/gpu: 1
          env:
          - name: MODEL_PATH
            value: "/models/llama2-7b"
          - name: MAX_BATCH_SIZE
            value: "8"
          - name: MAX_SEQUENCE_LENGTH
            value: "2048"
"""

        # Model configuration editor
        st.write("### Edit Model Configuration")
        edited_model_config = st.text_area("Model YAML Configuration", model_config, height=400)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Validate Model Config"):
                try:
                    yaml.safe_load(edited_model_config)
                    st.success("Model configuration is valid YAML")
                except Exception as e:
                    st.error(f"Invalid YAML: {str(e)}")

        with col2:
            if st.button("Apply Model Config"):
                st.success("Model configuration applied")
                st.info("In a real implementation, this would update the model deployment")

        st.write("### Model Library")

        # Available models
        model_library = [
            {"name": "llama2-7b", "type": "LLM", "parameters": "7B", "status": "Available"},
            {"name": "llama2-13b", "type": "LLM", "parameters": "13B", "status": "Available"},
            {"name": "mistral-7b", "type": "LLM", "parameters": "7B", "status": "Available"},
            {"name": "bert-base", "type": "Embedding", "parameters": "110M", "status": "Available"},
            {"name": "gpt-j-6b", "type": "LLM", "parameters": "6B", "status": "Available"},
            {"name": "clip", "type": "Multimodal", "parameters": "400M", "status": "Available"},
        ]

        df_models = pd.DataFrame(model_library)
        st.dataframe(df_models, use_container_width=True)

        st.write("### Import New Model")
        new_model_name = st.text_input("Model Name")
        new_model_type = st.selectbox("Model Type", ["LLM", "Embedding", "Classification", "Regression", "Multimodal"])
        new_model_source = st.selectbox("Model Source", ["Hugging Face", "Custom Registry", "Local Upload"])

        if new_model_source == "Hugging Face":
            st.text_input("Hugging Face Model ID", "meta-llama/Llama-2-7b")
        elif new_model_source == "Custom Registry":
            st.text_input("Registry URL")
            st.text_input("Model Path")

        if st.button("Import Model"):
            if new_model_name:
                st.success(f"Model {new_model_name} import initiated")
                st.info("This would download and register the model in a real implementation")
            else:
                st.error("Please provide a model name")

        # End of Model Configuration

# Monitoring page
elif page == "Monitoring":
    st.header("System Monitoring")

    # Time range selection
    time_range = st.selectbox(
        "Time Range", ["Last 1 hour", "Last 6 hours", "Last 24 hours", "Last 7 days", "Last 30 days"]
    )

    # Generate mock time series data
    import numpy as np
    import datetime as dt

    now = dt.datetime.now()
    if time_range == "Last 1 hour":
        timestamps = [now - dt.timedelta(minutes=i) for i in range(60, 0, -1)]
    elif time_range == "Last 6 hours":
        timestamps = [now - dt.timedelta(minutes=i * 6) for i in range(60, 0, -1)]
    elif time_range == "Last 24 hours":
        timestamps = [now - dt.timedelta(minutes=i * 24) for i in range(60, 0, -1)]
    else:
        timestamps = [now - dt.timedelta(hours=i * 24) for i in range(30, 0, -1)]

    timestamp_strs = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps]

    # Generate some mock data with trends
    cpu_usage = np.clip(
        50 + 15 * np.sin(np.linspace(0, 4 * np.pi, len(timestamps))) + np.random.normal(0, 5, len(timestamps)), 0, 100
    )
    memory_usage = np.clip(
        60 + 10 * np.sin(np.linspace(0, 2 * np.pi, len(timestamps))) + np.random.normal(0, 8, len(timestamps)), 0, 100
    )
    gpu_usage = np.clip(
        40 + 20 * np.sin(np.linspace(np.pi / 4, 3 * np.pi, len(timestamps))) + np.random.normal(0, 10, len(timestamps)),
        0,
        100,
    )

    # Request rate has spikes
    base_rate = 100 + 50 * np.sin(np.linspace(0, 2 * np.pi, len(timestamps)))
    spikes = np.zeros(len(timestamps))
    spike_positions = np.random.choice(range(len(timestamps)), size=5, replace=False)
    spikes[spike_positions] = np.random.uniform(100, 200, size=5)
    request_rate = base_rate + spikes + np.random.normal(0, 10, len(timestamps))

    # Error rate is generally low with occasional spikes
    error_rate = np.random.normal(0.5, 0.3, len(timestamps))
    error_spikes = np.zeros(len(timestamps))
    error_spike_positions = np.random.choice(range(len(timestamps)), size=3, replace=False)
    error_spikes[error_spike_positions] = np.random.uniform(3, 8, size=3)
    error_rate = np.clip(error_rate + error_spikes, 0, 10)

    # Tabs for different metric categories
    tab1, tab2, tab3, tab4 = st.tabs(["System Resources", "API Performance", "Cost", "Alerts"])

    with tab1:
        # Create a container for System Resource Monitoring
        with st.container():
            st.subheader("System Resource Monitoring")

        # Resource usage charts
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=timestamp_strs, y=cpu_usage, mode="lines", name="CPU Usage (%)", line=dict(color="#4b7bec"))
        )

        fig.add_trace(
            go.Scatter(
                x=timestamp_strs, y=memory_usage, mode="lines", name="Memory Usage (%)", line=dict(color="#3867d6")
            )
        )

        fig.add_trace(
            go.Scatter(x=timestamp_strs, y=gpu_usage, mode="lines", name="GPU Usage (%)", line=dict(color="#fed330"))
        )

        fig.update_layout(
            title="Resource Usage Over Time",
            xaxis_title="Time",
            yaxis_title="Utilization (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Node status
        st.subheader("Node Status")

        # Mock node data
        nodes = [
            {"name": "aws-node-1", "status": "Running", "cpu": "65%", "memory": "72%", "gpu": "82%"},
            {"name": "aws-node-2", "status": "Running", "cpu": "48%", "memory": "63%", "gpu": "75%"},
            {"name": "gcp-node-1", "status": "Running", "cpu": "71%", "memory": "56%", "gpu": "90%"},
            {"name": "gcp-node-2", "status": "Warning", "cpu": "92%", "memory": "87%", "gpu": "95%"},
        ]

        node_df = pd.DataFrame(nodes)

        # Color the status column
        def color_status(val):
            color = "green" if val == "Running" else "orange" if val == "Warning" else "red"
            return f"background-color: {color}; color: white"

        styled_df = node_df.style.applymap(color_status, subset=["status"])
        st.dataframe(styled_df, use_container_width=True)

        # End of System Resource Monitoring

    with tab2:
        # Create a container for API Performance Monitoring
        with st.container():
            st.subheader("API Performance Monitoring")

        # Request rate chart
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=timestamp_strs, y=request_rate, mode="lines", name="Requests per Minute", line=dict(color="#4b7bec")
            )
        )

        fig.update_layout(
            title="Request Rate Over Time", xaxis_title="Time", yaxis_title="Requests per Minute", height=300
        )

        st.plotly_chart(fig, use_container_width=True)

        # Error rate chart
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=timestamp_strs, y=error_rate, mode="lines", name="Error Rate (%)", line=dict(color="#fc5c65"))
        )

        fig.update_layout(title="Error Rate Over Time", xaxis_title="Time", yaxis_title="Error Rate (%)", height=300)

        st.plotly_chart(fig, use_container_width=True)

        # API Latency by endpoint
        st.subheader("API Latency by Endpoint")

        # Mock latency data
        endpoints = [
            {
                "endpoint": "/api/v1/models/llama2-7b/generate",
                "avg_latency": 320,
                "p50_latency": 310,
                "p95_latency": 450,
                "p99_latency": 600,
            },
            {
                "endpoint": "/api/v1/models/gpt-j-6b/generate",
                "avg_latency": 280,
                "p50_latency": 270,
                "p95_latency": 390,
                "p99_latency": 520,
            },
            {
                "endpoint": "/api/v1/models/bert-base/embed",
                "avg_latency": 65,
                "p50_latency": 60,
                "p95_latency": 110,
                "p99_latency": 150,
            },
        ]

        endpoint_df = pd.DataFrame(endpoints)
        st.dataframe(endpoint_df, use_container_width=True)

        # API Status
        st.subheader("API Status")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Availability", "99.98%", "0.01%")

        with col2:
            st.metric("Success Rate", "99.5%", "-0.2%")

        with col3:
            st.metric("Avg. Response Time", "215ms", "-12ms")

        # End of API Performance Monitoring

    with tab3:
        # Create a container for Cost Monitoring
        with st.container():
            st.subheader("Cost Monitoring")

        # Daily cost trend
        days = 30
        daily_timestamps = [now - dt.timedelta(days=i) for i in range(days, 0, -1)]
        daily_timestamp_strs = [ts.strftime("%Y-%m-%d") for ts in daily_timestamps]

        # Generate mock cost data with a weekly pattern
        base_cost = 120 + 20 * np.sin(np.linspace(0, 4 * np.pi, days))
        weekend_effect = np.array([20 if i % 7 >= 5 else 0 for i in range(days)])
        daily_cost = base_cost - weekend_effect + np.random.normal(0, 15, days)

        # Split by provider
        aws_ratio = 0.6 + 0.1 * np.sin(np.linspace(0, 2 * np.pi, days))
        aws_cost = daily_cost * aws_ratio
        gcp_cost = daily_cost * (1 - aws_ratio)

        # Cost by provider chart
        fig = go.Figure()

        fig.add_trace(go.Bar(x=daily_timestamp_strs, y=aws_cost, name="AWS Cost", marker_color="#4b7bec"))

        fig.add_trace(go.Bar(x=daily_timestamp_strs, y=gcp_cost, name="GCP Cost", marker_color="#3867d6"))

        fig.update_layout(
            title="Daily Cost by Cloud Provider",
            xaxis_title="Date",
            yaxis_title="Cost ($)",
            barmode="stack",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Cost by component
        st.subheader("Cost Breakdown")

        # Mock cost breakdown
        cost_components = {
            "Compute (GPU)": 2451.32,
            "Compute (CPU)": 856.78,
            "Storage": 423.45,
            "Network": 289.76,
            "Managed Services": 175.32,
        }

        fig = px.pie(
            names=list(cost_components.keys()),
            values=list(cost_components.values()),
            title="Cost Components This Month",
        )

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Cost metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Today's Estimate", "$168.45", "12.5%")

        with col2:
            st.metric("Month to Date", "$4,196.63", "-3.2%")

        with col3:
            st.metric("Projected Monthly", "$5,235.87", "8.3%")

        # Budget tracking
        st.subheader("Budget Tracking")

        budget = 6000.00
        spent = 4196.63
        remaining = budget - spent
        percent_used = (spent / budget) * 100

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=percent_used,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Budget Utilized"},
                delta={"reference": 75, "increasing": {"color": "red"}, "decreasing": {"color": "green"}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#4b7bec"},
                    "steps": [
                        {"range": [0, 50], "color": "#43d787"},
                        {"range": [50, 80], "color": "#f7b731"},
                        {"range": [80, 100], "color": "#fc5c65"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 90},
                },
            )
        )

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Monthly Budget", f"${budget:.2f}")

        with col2:
            st.metric("Remaining", f"${remaining:.2f}", f"{(budget - spent) / budget:.1%}")

        # End of Cost Monitoring

    with tab4:
        # Create a container for System Alerts
        with st.container():
            st.subheader("System Alerts")

        # Mock alerts
        alerts = [
            {
                "timestamp": "2025-05-06 09:23:15",
                "severity": "Critical",
                "resource": "gcp-node-2",
                "message": "High GPU utilization (95%) exceeded threshold (90%) for >15 minutes",
                "status": "Active",
            },
            {
                "timestamp": "2025-05-06 08:45:32",
                "severity": "Warning",
                "resource": "llama2-7b-deployment",
                "message": "High API latency (450ms) exceeded threshold (400ms)",
                "status": "Active",
            },
            {
                "timestamp": "2025-05-05 23:12:08",
                "severity": "Warning",
                "resource": "aws-node-1",
                "message": "Memory utilization (85%) approaching threshold (90%)",
                "status": "Resolved",
            },
            {
                "timestamp": "2025-05-05 15:36:51",
                "severity": "Critical",
                "resource": "api-gateway",
                "message": "Error rate (5.2%) exceeded threshold (2%)",
                "status": "Resolved",
            },
        ]

        alerts_df = pd.DataFrame(alerts)

        # Style the dataframe
        def color_severity(val):
            color = "red" if val == "Critical" else "orange" if val == "Warning" else "green"
            return f"color: {color}; font-weight: bold"

        def color_status(val):
            color = "red" if val == "Active" else "green"
            return f"color: {color}; font-weight: bold"

        styled_alerts = alerts_df.style.applymap(color_severity, subset=["severity"]).applymap(
            color_status, subset=["status"]
        )

        st.dataframe(styled_alerts, use_container_width=True)

        # Alert settings
        st.subheader("Alert Configuration")

        with st.expander("Resource Utilization Alerts"):
            st.slider("CPU Utilization Threshold (%)", 50, 100, 80)
            st.slider("Memory Utilization Threshold (%)", 50, 100, 85)
            st.slider("GPU Utilization Threshold (%)", 50, 100, 90)
            st.number_input("Alert Duration (minutes)", 1, 60, 15)

        with st.expander("Performance Alerts"):
            st.slider("API Latency Threshold (ms)", 100, 1000, 400)
            st.slider("Error Rate Threshold (%)", 0.1, 10.0, 2.0, 0.1)
            st.slider("Request Rate Change Threshold (%)", 10, 100, 50)

        with st.expander("Cost Alerts"):
            st.slider("Daily Budget Threshold (%)", 50, 150, 120)
            st.number_input("Cost Spike Threshold ($)", 50, 1000, 200)

        with st.expander("Notification Settings"):
            st.checkbox("Email Notifications", value=True)
            st.text_input("Email Recipients", "admin@example.com, alerts@example.com")
            st.checkbox("Slack Notifications", value=True)
            st.text_input("Slack Channel", "#ml-platform-alerts")
            st.checkbox("PagerDuty Integration", value=False)

        if st.button("Save Alert Configuration"):
            st.success("Alert configuration saved successfully")

        # End of System Alerts

# Logs page
elif page == "Logs":
    st.header("System Logs")

    # Log filters
    col1, col2, col3 = st.columns(3)

    with col1:
        log_level = st.multiselect(
            "Log Level", ["INFO", "WARNING", "ERROR", "DEBUG"], default=["INFO", "WARNING", "ERROR"]
        )

    with col2:
        component = st.multiselect(
            "Component",
            ["kubernetes", "api", "model", "gateway", "autoscaler", "observability"],
            default=["kubernetes", "api", "model"],
        )

    with col3:
        timeframe = st.selectbox("Time Range", ["Last 15 minutes", "Last hour", "Last 6 hours", "Last 24 hours"])

    # Add search filter
    search_term = st.text_input("Search in logs", "")

# Secure Weights page
elif page == "Secure Weights":
    # Render secure weights UI from imported module
    render_secure_weights_ui()

    # Create a container for logs display
    with st.container():
        # Generate mock logs
        import random

        log_levels = ["INFO", "WARNING", "ERROR", "DEBUG"]
        components = ["kubernetes", "api", "model", "gateway", "autoscaler", "observability"]

        log_messages = [
            "[INFO] Kubernetes pod created: llama2-7b-deployment-5d4f8b9c7-xjlm2",
            "[INFO] API request received: /api/v1/models/llama2-7b/generate",
            "[INFO] Model loaded: llama2-7b v1.0",
            "[INFO] Gateway routing request to AWS cluster",
            "[WARNING] High GPU utilization detected: 92%",
            "[INFO] Autoscaling triggered: increasing replicas to 3",
            "[INFO] Kubernetes service started: llama2-7b-api",
            "[ERROR] Failed to load model weights: /models/gpt-j-6b/weights.bin not found",
            "[WARNING] API latency above threshold: 450ms",
            "[INFO] Observability metrics collected",
            "[DEBUG] Kubernetes config validated",
            "[INFO] Gateway health check successful",
            "[ERROR] API request failed: invalid input parameters",
            "[WARNING] Memory utilization high: 87%",
            "[INFO] Model inference completed in 320ms",
            "[INFO] Kubernetes deployment scaled: replicas=3",
            "[DEBUG] Cache hit ratio: 78%",
            "[INFO] Autoscaler computed optimal replicas: 3",
            "[ERROR] Database connection failed: timeout",
            "[INFO] API rate limiting applied: 429 response",
        ]

        # Generate 100 random log entries
        logs = []
        now = dt.datetime.now()

        for i in range(100):
            timestamp = now - dt.timedelta(minutes=random.randint(0, 360))
            level = random.choice(log_levels)
            comp = random.choice(components)
            message = random.choice(log_messages)

            # Make sure the log level in the message matches the selected level
            message = (
                message.replace("[INFO]", f"[{level}]")
                .replace("[WARNING]", f"[{level}]")
                .replace("[ERROR]", f"[{level}]")
                .replace("[DEBUG]", f"[{level}]")
            )

            logs.append(
                {
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "level": level,
                    "component": comp,
                    "message": message,
                }
            )

        # Sort logs by timestamp (newest first)
        logs = sorted(logs, key=lambda x: x["timestamp"], reverse=True)

        # Filter logs based on user selection
        filtered_logs = [log for log in logs if log["level"] in log_level and log["component"] in component]

        # Filter by search term if provided
        if search_term:
            filtered_logs = [log for log in filtered_logs if search_term.lower() in log["message"].lower()]

        # Display log count
        st.write(f"Displaying {len(filtered_logs)} logs")

        # Display logs with Streamlit native components
        for log in filtered_logs:
            # Create a formatted log message
            log_message = f"{log['timestamp']} [{log['component']}] {log['message']}"
            
            # Use appropriate Streamlit components based on log level
            if log["level"] == "INFO":
                st.info(log_message)
            elif log["level"] == "WARNING":
                st.warning(log_message)
            elif log["level"] == "ERROR":
                st.error(log_message)
            elif log["level"] == "DEBUG":
                st.text(log_message)

    # End of log display

    # Log download options
    with st.container():
        st.subheader("Export Logs")

    col1, col2 = st.columns(2)

    with col1:
        st.selectbox("Format", ["JSON", "CSV", "Text"])

    with col2:
        st.selectbox("Time Range", ["Last 15 minutes", "Last hour", "Last 6 hours", "Last 24 hours", "Custom..."])

    if st.button("Download Logs"):
        st.success("Log download prepared")
        st.download_button(
            label="Click to Download",
            data="This would be the actual logs in a real implementation",
            file_name="hipo_logs.json",
            mime="application/json",
        )

    # End of export logs section
