"""
Kubernetes Orchestrator for multi-cloud Kubernetes infrastructure.
"""
import logging
import yaml
import json
import os
import time
import threading
from typing import Dict, List, Any, Optional, Tuple

from src.cloud.provider import CloudProvider
from src.cloud.factory import CloudProviderFactory
from src.gateway.api_gateway import APIGateway
from src.autoscaling.gpu_autoscaler import GPUAutoscaler
from src.secrets.secret_manager import SecretManager
from src.kubernetes.cluster_api_manager import ClusterAPIManager

logger = logging.getLogger(__name__)


class KubernetesOrchestrator:
    """Kubernetes Orchestrator for multi-cloud infrastructure."""

    def __init__(self, config_path: str):
        """Initialize Kubernetes Orchestrator.

        Args:
            config_path: Path to configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(f"{__name__}.KubernetesOrchestrator")

        # Initialize cloud providers
        self.cloud_providers = CloudProviderFactory.create_providers(self.config)

        # Initialize components
        self.api_gateway = None
        self.gpu_autoscaler = None
        self.secret_manager = None
        
        # Initialize Cluster API Manager if enabled
        self.cluster_api_enabled = self.config.get("cluster_api", {}).get("enabled", False)
        self.cluster_api_manager = None
        if self.cluster_api_enabled:
            self.logger.info("Cluster API support is enabled")
            self.cluster_api_manager = ClusterAPIManager(self.config.get("cluster_api", {}))

        # State
        self.running = False
        self.status_thread = None
        self.status = {"cloud_providers": {}, "api_gateway": {}, "autoscaling": {}, "secret_manager": {}, "cluster_api": {}}

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file.

        Args:
            config_path: Path to configuration file.

        Returns:
            Configuration dictionary.
        """
        with open(config_path, "r") as f:
            if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                return yaml.safe_load(f)
            elif config_path.endswith(".json"):
                return json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path}")

    def initialize(self) -> bool:
        """Initialize the Kubernetes Orchestrator.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Initialize API Gateway
            api_gateway_config = self.config.get("api_gateway", {})
            self.api_gateway = APIGateway(api_gateway_config, self.cloud_providers)

            # Initialize GPU Autoscaler
            autoscaling_config = self.config.get("autoscaling", {})
            self.gpu_autoscaler = GPUAutoscaler(autoscaling_config, self.cloud_providers)

            # Initialize Secret Manager
            secrets_config = self.config.get("secrets", {})
            self.secret_manager = SecretManager(secrets_config, self.cloud_providers)

            # Initialize Cluster API Manager if enabled
            if self.cluster_api_enabled and self.cluster_api_manager:
                self.logger.info("Initializing Cluster API Manager")
                if not self.cluster_api_manager.initialize():
                    self.logger.error("Failed to initialize Cluster API Manager")
                    return False
                self.logger.info("Cluster API Manager initialized successfully")

            # Update status
            self.status["cloud_providers"] = {
                name: {"enabled": provider.is_enabled(), "region": provider.region}
                for name, provider in self.cloud_providers.items()
            }
            
            # Update Cluster API status if enabled
            if self.cluster_api_enabled and self.cluster_api_manager:
                self.status["cluster_api"] = {
                    "enabled": True,
                    "initialized": self.cluster_api_manager.initialized
                }

            return True
        except Exception as e:
            self.logger.error(f"Error initializing Kubernetes Orchestrator: {e}")
            return False

    def start(self) -> bool:
        """Start the Kubernetes Orchestrator.

        Returns:
            True if successful, False otherwise.
        """
        if self.running:
            self.logger.warning("Kubernetes Orchestrator is already running")
            return True

        try:
            # Start API Gateway
            if self.api_gateway:
                self.api_gateway.setup_global_load_balancer()
                self.api_gateway.configure_security()
                self.api_gateway.deploy()

            # Start GPU Autoscaler
            if self.gpu_autoscaler:
                self.gpu_autoscaler.start()

            # Start Secret Manager
            if self.secret_manager:
                self.secret_manager.start()

            # Start status thread
            self.running = True
            self.status_thread = threading.Thread(target=self._status_loop)
            self.status_thread.daemon = True
            self.status_thread.start()

            self.logger.info("Kubernetes Orchestrator started")
            return True
        except Exception as e:
            self.logger.error(f"Error starting Kubernetes Orchestrator: {e}")
            return False

    def stop(self) -> bool:
        """Stop the Kubernetes Orchestrator.

        Returns:
            True if successful, False otherwise.
        """
        if not self.running:
            self.logger.warning("Kubernetes Orchestrator is not running")
            return True

        try:
            self.running = False

            # Stop status thread
            if self.status_thread:
                self.status_thread.join(timeout=5.0)
                self.status_thread = None

            # Stop GPU Autoscaler
            if self.gpu_autoscaler:
                self.gpu_autoscaler.stop()

            # Stop Secret Manager
            if self.secret_manager:
                self.secret_manager.stop()

            self.logger.info("Kubernetes Orchestrator stopped")
            return True
        except Exception as e:
            self.logger.error(f"Error stopping Kubernetes Orchestrator: {e}")
            return False

    def _status_loop(self) -> None:
        """Status monitoring loop."""
        while self.running:
            try:
                self._update_status()
            except Exception as e:
                self.logger.error(f"Error in status loop: {e}")

            time.sleep(30)  # Update status every 30 seconds

    def _update_status(self) -> None:
        """Update orchestrator status."""
        # Update API Gateway status
        if self.api_gateway:
            self.status["api_gateway"] = {"routing": self.api_gateway.get_routing_info()}

        # Update GPU Autoscaler status
        if self.gpu_autoscaler:
            self.status["autoscaling"] = {
                "metrics": self.gpu_autoscaler.get_current_metrics(),
                "scaling_history": self.gpu_autoscaler.get_scaling_history(),
            }

        # Update Secret Manager status
        if self.secret_manager:
            self.status["secret_manager"] = {"rotation_status": self.secret_manager.get_rotation_status()}
            
        # Update Cluster API status
        if self.cluster_api_enabled and self.cluster_api_manager and self.cluster_api_manager.initialized:
            clusters = self.cluster_api_manager.get_workload_clusters()
            self.status["cluster_api"] = {
                "enabled": True,
                "initialized": self.cluster_api_manager.initialized,
                "clusters": {
                    cluster["name"]: {
                        "provider": cluster["provider"],
                        "region": cluster["region"],
                        "status": cluster["status"],
                        "control_plane_ready": cluster["control_plane_ready"],
                        "infrastructure_ready": cluster["infrastructure_ready"]
                    }
                    for cluster in clusters
                }
            }

    def deploy_llm_model(self, model_config: Dict[str, Any]) -> bool:
        """Deploy an LLM model to the Kubernetes clusters.

        Args:
            model_config: LLM model configuration.

        Returns:
            True if successful, False otherwise.
        """
        model_name = model_config.get("name")
        if not model_name:
            self.logger.error("Model name is required")
            return False

        self.logger.info(f"Deploying LLM model {model_name}")

        # Generate Kubernetes manifests for the model
        manifests = self._generate_model_manifests(model_config)

        # Check if we should deploy to Cluster API-managed clusters
        deploy_to_capi_clusters = model_config.get("cluster_api", {}).get("enabled", False)
        cluster_names = model_config.get("cluster_api", {}).get("clusters", [])
        
        # Deploy manifests to all enabled cloud providers (directly managed clusters)
        success = True

        for provider_name, provider in self.cloud_providers.items():
            if not provider.is_enabled():
                continue

            try:
                # In a real implementation, this would use the Kubernetes client
                # to deploy the manifests to the cluster
                self.logger.info(f"Deploying LLM model {model_name} to {provider_name}")

                # Simulated success for demonstration
                model_endpoint = f"https://{model_name}.{provider_name}.example.com"

                # Register model endpoint in API Gateway
                if self.api_gateway:
                    self.api_gateway.register_route(
                        path=f"/api/models/{model_name}",
                        method="POST",
                        service_name=f"{model_name}-service",
                        service_port=8000,
                    )

                self.logger.info(f"Successfully deployed LLM model {model_name} to {provider_name}")
            except Exception as e:
                self.logger.error(f"Error deploying LLM model {model_name} to {provider_name}: {e}")
                success = False
        
        # Deploy to Cluster API-managed clusters if requested
        if deploy_to_capi_clusters and self.cluster_api_enabled and self.cluster_api_manager and self.cluster_api_manager.initialized:
            # Get available clusters if no specific clusters specified
            if not cluster_names:
                all_clusters = self.cluster_api_manager.get_workload_clusters()
                cluster_names = [cluster["name"] for cluster in all_clusters]
            
            for cluster_name in cluster_names:
                try:
                    self.logger.info(f"Deploying LLM model {model_name} to CAPI-managed cluster {cluster_name}")
                    
                    # In a real implementation, this would:
                    # 1. Get the kubeconfig for the cluster
                    # 2. Apply the manifests using that kubeconfig
                    
                    # For simplicity, we'll just log the action
                    self.logger.info(f"Successfully deployed LLM model {model_name} to CAPI-managed cluster {cluster_name}")
                    
                    # Register model endpoint in API Gateway (if needed)
                    if self.api_gateway:
                        self.api_gateway.register_route(
                            path=f"/api/capi/{cluster_name}/models/{model_name}",
                            method="POST",
                            service_name=f"{model_name}-service",
                            service_port=8000,
                        )
                except Exception as e:
                    self.logger.error(f"Error deploying LLM model {model_name} to CAPI-managed cluster {cluster_name}: {e}")
                    success = False

        return success

    def _generate_model_manifests(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Kubernetes manifests for an LLM model.

        Args:
            model_config: LLM model configuration.

        Returns:
            Dictionary of Kubernetes manifests.
        """
        model_name = model_config.get("name")
        version = model_config.get("version", "1.0")
        resource_requirements = model_config.get("resource_requirements", {})
        container_config = model_config.get("container", {})

        # Generate individual manifests
        deployment = self._generate_deployment_manifest(model_name, version, resource_requirements, container_config)
        service = self._generate_service_manifest(model_name, container_config)

        # If Istio is enabled, add VirtualService
        if self.config.get("kubernetes", {}).get("istio", {}).get("enabled", False):
            virtual_service = self._generate_virtual_service_manifest(model_name)
            return {"deployment": deployment, "service": service, "virtual_service": virtual_service}
        else:
            return {"deployment": deployment, "service": service}

    def _generate_deployment_manifest(
        self, model_name: str, version: str, resource_requirements: Dict[str, Any], container_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a Kubernetes Deployment manifest for an LLM model.

        Args:
            model_name: Name of the model.
            version: Model version.
            resource_requirements: Resource requirements for the model.
            container_config: Container configuration for the model.

        Returns:
            Kubernetes Deployment manifest.
        """
        namespace = self.config.get("kubernetes", {}).get("namespace", "llm-serving")
        
        # Create container specification
        container = self._create_container_spec(model_name, resource_requirements, container_config)
        
        # Create node selector based on GPU requirements
        node_selector = {"node-type": "gpu" if resource_requirements.get("gpu", 0) > 0 else "cpu"}
        
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{model_name}-deployment",
                "namespace": namespace,
                "labels": {"app": model_name, "version": version},
            },
            "spec": {
                "replicas": 1,  # Initial replica count, will be scaled by autoscaler
                "selector": {"matchLabels": {"app": model_name}},
                "template": {
                    "metadata": {"labels": {"app": model_name, "version": version}},
                    "spec": {
                        "containers": [container],
                        "nodeSelector": node_selector,
                        "volumes": [{"name": "models-volume", "persistentVolumeClaim": {"claimName": "models-pvc"}}],
                    },
                },
            },
        }

    def _create_container_spec(
        self, model_name: str, resource_requirements: Dict[str, Any], container_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a container specification for an LLM model.

        Args:
            model_name: Name of the model.
            resource_requirements: Resource requirements for the model.
            container_config: Container configuration for the model.

        Returns:
            Container specification.
        """
        # Create environment variables
        env = [
            {"name": name, "value": value}
            for name, value in container_config.get("environment", {}).items()
        ]
        
        # Create volume mounts
        volume_mounts = [
            {"name": mount.get("name"), "mountPath": mount.get("mount_path")}
            for mount in container_config.get("volume_mounts", [])
        ]
        
        return {
            "name": model_name,
            "image": container_config.get("image"),
            "ports": [{"containerPort": container_config.get("port", 8000)}],
            "resources": {
                "requests": {
                    "cpu": f"{resource_requirements.get('cpu', 1)}",
                    "memory": resource_requirements.get("memory", "1Gi"),
                    "nvidia.com/gpu": resource_requirements.get("gpu", 0),
                },
                "limits": {
                    "cpu": f"{resource_requirements.get('cpu', 1) * 2}",
                    "memory": resource_requirements.get("memory", "1Gi"),
                    "nvidia.com/gpu": resource_requirements.get("gpu", 0),
                },
            },
            "env": env,
            "volumeMounts": volume_mounts,
        }

    def _generate_service_manifest(self, model_name: str, container_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a Kubernetes Service manifest for an LLM model.

        Args:
            model_name: Name of the model.
            container_config: Container configuration for the model.

        Returns:
            Kubernetes Service manifest.
        """
        namespace = self.config.get("kubernetes", {}).get("namespace", "llm-serving")
        
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{model_name}-service",
                "namespace": namespace,
            },
            "spec": {
                "selector": {"app": model_name},
                "ports": [{"port": 80, "targetPort": container_config.get("port", 8000)}],
            },
        }

    def _generate_virtual_service_manifest(self, model_name: str) -> Dict[str, Any]:
        """Generate an Istio VirtualService manifest for an LLM model.

        Args:
            model_name: Name of the model.

        Returns:
            Istio VirtualService manifest.
        """
        namespace = self.config.get("kubernetes", {}).get("namespace", "llm-serving")
        gateway_name = self.config.get("kubernetes", {}).get("istio", {}).get("gateway", {}).get("name", "llm-gateway")
        
        return {
            "apiVersion": "networking.istio.io/v1alpha3",
            "kind": "VirtualService",
            "metadata": {
                "name": f"{model_name}-vs",
                "namespace": namespace,
            },
            "spec": {
                "hosts": [f"{model_name}.example.com"],
                "gateways": [gateway_name],
                "http": [{"route": [{"destination": {"host": f"{model_name}-service", "port": {"number": 80}}}]}],
            },
        }

    def handle_cloud_failure(self, failed_provider: str) -> bool:
        """Handle failure of a cloud provider.

        Args:
            failed_provider: Failed cloud provider name.

        Returns:
            True if successfully handled, False otherwise.
        """
        self.logger.warning(f"Handling failure of cloud provider {failed_provider}")

        try:
            # Update API Gateway routing to avoid the failed provider
            if self.api_gateway:
                # In a real implementation, this would update the global load balancer
                # to avoid routing traffic to the failed provider
                self.logger.info(f"Updating API Gateway routing to avoid {failed_provider}")

            # Scale up resources in other providers to handle the load
            for provider_name, provider in self.cloud_providers.items():
                if provider_name == failed_provider or not provider.is_enabled():
                    continue

                self.logger.info(f"Scaling up resources in {provider_name} to handle load from {failed_provider}")

                # In a real implementation, this would scale up resources in the other providers

            self.logger.info(f"Successfully handled failure of cloud provider {failed_provider}")
            return True
        except Exception as e:
            self.logger.error(f"Error handling failure of cloud provider {failed_provider}: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status.

        Returns:
            Dictionary of orchestrator status.
        """
        # Return a copy of the status
        return self.status.copy()

    def monitor_costs(self) -> Dict[str, float]:
        """Monitor costs across all cloud providers.

        Returns:
            Dictionary of costs per provider.
        """
        costs = {}

        for provider_name, provider in self.cloud_providers.items():
            if not provider.is_enabled():
                continue

            try:
                cost_metrics = provider.get_cost_metrics(timeframe="daily")
                costs[provider_name] = cost_metrics.get("total_cost", 0.0)
            except Exception as e:
                self.logger.error(f"Error getting cost metrics for {provider_name}: {e}")
                costs[provider_name] = 0.0

        return costs

    def optimize_costs(self) -> bool:
        """Optimize costs across all cloud providers.

        Returns:
            True if successful, False otherwise.
        """
        cost_optimization_config = self.config.get("cost_management", {}).get("optimization", {})
        enable_spot_instances = cost_optimization_config.get("enable_spot_instances", True)

        self.logger.info(f"Optimizing costs with spot instances {'enabled' if enable_spot_instances else 'disabled'}")

        # In a real implementation, this would optimize costs across cloud providers
        # For example, by using spot/preemptible instances, right-sizing resources, etc.

        # For simplicity, we'll just return success
        return True

    def sync_configurations(self) -> bool:
        """Synchronize configurations across all cloud providers.

        Returns:
            True if successful, False otherwise.
        """
        self.logger.info("Synchronizing configurations across all cloud providers")

        # In a real implementation, this would ensure that configurations are synchronized
        # across all cloud providers

        # For simplicity, we'll just return success
        return True

    def deploy_monitoring(self) -> bool:
        """Deploy monitoring infrastructure across all cloud providers.

        Returns:
            True if successful, False otherwise.
        """
        monitoring_config = self.config.get("monitoring", {})
        prometheus_enabled = monitoring_config.get("prometheus", {}).get("enabled", True)
        grafana_enabled = monitoring_config.get("grafana", {}).get("enabled", True)

        prom_status = 'enabled' if prometheus_enabled else 'disabled'
        grafana_status = 'enabled' if grafana_enabled else 'disabled'
        self.logger.info(
            f"Deploying monitoring infrastructure with Prometheus {prom_status} and Grafana {grafana_status}"
        )

        # In a real implementation, this would deploy monitoring infrastructure
        # such as Prometheus, Grafana, and alerts

        # For simplicity, we'll just return success
        return True

    def get_deployment_details(self, model_name: str) -> Dict[str, Any]:
        """Get deployment details for a specific model.

        Args:
            model_name: Model name.

        Returns:
            Dictionary of deployment details.
        """
        details = {}

        for provider_name, provider in self.cloud_providers.items():
            if not provider.is_enabled():
                continue

            # In a real implementation, this would get deployment details
            # from the Kubernetes cluster

            # Simulated details for demonstration
            details[provider_name] = {
                "status": "Running",
                "replicas": 3,
                "endpoint": f"https://{model_name}.{provider_name}.example.com",
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-01T00:00:00Z",
            }

        return details
        
    # Cluster API methods
    
    def create_cluster(self, 
                      name: str,
                      provider: str,
                      region: str,
                      flavor: str = "ml-llm",
                      control_plane_machine_count: int = 3,
                      worker_machine_count: int = 3,
                      gpu_machine_count: int = 0,
                      gpu_instance_type: str = None) -> Dict[str, Any]:
        """Create a new Kubernetes cluster using Cluster API.
        
        Args:
            name: Name of the cluster.
            provider: Cloud provider (aws, gcp, azure).
            region: Region to deploy the cluster.
            flavor: Cluster flavor (ml-llm, base, etc.).
            control_plane_machine_count: Number of control plane machines.
            worker_machine_count: Number of worker machines.
            gpu_machine_count: Number of GPU machines.
            gpu_instance_type: Instance type for GPU machines.
            
        Returns:
            Dictionary with cluster details or error information.
        """
        if not self.cluster_api_enabled:
            self.logger.error("Cluster API is not enabled")
            return {"error": "Cluster API is not enabled"}
            
        if not self.cluster_api_manager or not self.cluster_api_manager.initialized:
            self.logger.error("Cluster API Manager is not initialized")
            return {"error": "Cluster API Manager is not initialized"}
            
        self.logger.info(f"Creating cluster {name} on {provider} in {region}")
        
        # Create the cluster using Cluster API Manager
        result = self.cluster_api_manager.create_workload_cluster(
            name=name,
            provider=provider,
            region=region,
            flavor=flavor,
            control_plane_machine_count=control_plane_machine_count,
            worker_machine_count=worker_machine_count,
            gpu_machine_count=gpu_machine_count,
            gpu_instance_type=gpu_instance_type
        )
        
        # Update status
        if "error" not in result:
            self._update_status()
            
        return result
        
    def delete_cluster(self, name: str) -> bool:
        """Delete a Kubernetes cluster using Cluster API.
        
        Args:
            name: Name of the cluster to delete.
            
        Returns:
            True if deletion was initiated successfully, False otherwise.
        """
        if not self.cluster_api_enabled:
            self.logger.error("Cluster API is not enabled")
            return False
            
        if not self.cluster_api_manager or not self.cluster_api_manager.initialized:
            self.logger.error("Cluster API Manager is not initialized")
            return False
            
        self.logger.info(f"Deleting cluster {name}")
        
        # Delete the cluster using Cluster API Manager
        success = self.cluster_api_manager.delete_workload_cluster(name)
        
        # Update status
        if success:
            self._update_status()
            
        return success
        
    def get_clusters(self) -> List[Dict[str, Any]]:
        """Get all Kubernetes clusters managed by Cluster API.
        
        Returns:
            List of cluster details.
        """
        if not self.cluster_api_enabled:
            self.logger.warning("Cluster API is not enabled")
            return []
            
        if not self.cluster_api_manager or not self.cluster_api_manager.initialized:
            self.logger.warning("Cluster API Manager is not initialized")
            return []
            
        # Get clusters using Cluster API Manager
        return self.cluster_api_manager.get_workload_clusters()
        
    def get_cluster_kubeconfig(self, cluster_name: str) -> str:
        """Get kubeconfig for a specific cluster.
        
        Args:
            cluster_name: Name of the cluster.
            
        Returns:
            Kubeconfig content as string if successful, empty string otherwise.
        """
        if not self.cluster_api_enabled:
            self.logger.error("Cluster API is not enabled")
            return ""
            
        if not self.cluster_api_manager or not self.cluster_api_manager.initialized:
            self.logger.error("Cluster API Manager is not initialized")
            return ""
            
        # Get kubeconfig using Cluster API Manager
        return self.cluster_api_manager.get_kubeconfig(cluster_name)
        
    def scale_cluster_node_group(self, cluster_name: str, node_group_name: str, desired_size: int) -> bool:
        """Scale a node group in a cluster.
        
        Args:
            cluster_name: Name of the cluster.
            node_group_name: Name of the node group.
            desired_size: Desired number of nodes.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.cluster_api_enabled:
            self.logger.error("Cluster API is not enabled")
            return False
            
        if not self.cluster_api_manager or not self.cluster_api_manager.initialized:
            self.logger.error("Cluster API Manager is not initialized")
            return False
            
        self.logger.info(f"Scaling node group {node_group_name} in cluster {cluster_name} to {desired_size} nodes")
        
        # Scale node group using Cluster API Manager
        success = self.cluster_api_manager.scale_node_group(cluster_name, node_group_name, desired_size)
        
        # Update status
        if success:
            self._update_status()
            
        return success
        
    def get_cluster_node_groups(self, cluster_name: str) -> List[Dict[str, Any]]:
        """Get node groups for a specific cluster.
        
        Args:
            cluster_name: Name of the cluster.
            
        Returns:
            List of node group details.
        """
        if not self.cluster_api_enabled:
            self.logger.warning("Cluster API is not enabled")
            return []
            
        if not self.cluster_api_manager or not self.cluster_api_manager.initialized:
            self.logger.warning("Cluster API Manager is not initialized")
            return []
            
        # Get node groups using Cluster API Manager
        return self.cluster_api_manager.get_node_groups(cluster_name)
        
    def handle_cluster_failure(self, cluster_name: str) -> bool:
        """Handle failure of a Cluster API-managed cluster.
        
        Args:
            cluster_name: Name of the failed cluster.
            
        Returns:
            True if successfully handled, False otherwise.
        """
        if not self.cluster_api_enabled:
            self.logger.error("Cluster API is not enabled")
            return False
            
        if not self.cluster_api_manager or not self.cluster_api_manager.initialized:
            self.logger.error("Cluster API Manager is not initialized")
            return False
            
        self.logger.warning(f"Handling failure of CAPI-managed cluster {cluster_name}")
        
        try:
            # Get cluster details to determine provider, region, etc.
            clusters = self.cluster_api_manager.get_workload_clusters()
            failed_cluster = next((c for c in clusters if c["name"] == cluster_name), None)
            
            if not failed_cluster:
                self.logger.error(f"Cluster {cluster_name} not found")
                return False
                
            provider = failed_cluster.get("provider")
            region = failed_cluster.get("region")
            
            # Get all deployments from the failed cluster
            # In a real implementation, this would extract all the deployments 
            # and stateful workloads from the failed cluster
            
            # Create a new cluster in the same region
            new_cluster_name = f"{cluster_name}-recovery"
            
            self.logger.info(f"Creating recovery cluster {new_cluster_name} in {region}")
            
            # Create a new cluster
            result = self.cluster_api_manager.create_workload_cluster(
                name=new_cluster_name,
                provider=provider,
                region=region,
                flavor="ml-llm",
                # Increase resources for resiliency
                control_plane_machine_count=3,
                worker_machine_count=3,
                gpu_machine_count=1
            )
            
            if "error" in result:
                self.logger.error(f"Failed to create recovery cluster: {result['error']}")
                return False
                
            self.logger.info(f"Recovery cluster {new_cluster_name} is being created")
            
            # In a real implementation, wait for the cluster to be ready
            # and migrate workloads to the new cluster
            
            # Update API Gateway routing to use the new cluster
            if self.api_gateway:
                self.logger.info(f"Updating API Gateway routing to use recovery cluster {new_cluster_name}")
                # Implementation would update routes to the new cluster
            
            self.logger.info(f"Recovery process initiated for failed cluster {cluster_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling cluster failure: {e}")
            return False
            
    def migrate_to_cluster_api(self, provider: str, region: str, create_new: bool = True) -> Dict[str, Any]:
        """Migrate existing infrastructure to Cluster API management.
        
        Args:
            provider: Cloud provider to migrate (aws, gcp, azure).
            region: Region to migrate.
            create_new: Whether to create a new cluster or import existing.
            
        Returns:
            Dictionary with migration details or error information.
        """
        if not self.cluster_api_enabled:
            self.logger.error("Cluster API is not enabled")
            return {"error": "Cluster API is not enabled"}
            
        if not self.cluster_api_manager or not self.cluster_api_manager.initialized:
            self.logger.error("Cluster API Manager is not initialized")
            return {"error": "Cluster API Manager is not initialized"}
            
        # Check if provider is available
        if provider not in self.cloud_providers or not self.cloud_providers[provider].is_enabled():
            self.logger.error(f"Provider {provider} is not available")
            return {"error": f"Provider {provider} is not available"}
            
        self.logger.info(f"Migrating infrastructure from {provider} in {region} to Cluster API management")
        
        try:
            # Generate a unique name for the migrated cluster
            cluster_name = f"migrated-{provider}-{region.replace('-', '')}"
            
            # For create_new=True, create a new cluster and migrate workloads
            if create_new:
                self.logger.info(f"Creating new CAPI-managed cluster {cluster_name}")
                
                # Create a new cluster using Cluster API
                result = self.cluster_api_manager.create_workload_cluster(
                    name=cluster_name,
                    provider=provider,
                    region=region,
                    flavor="ml-llm",
                    # Set appropriate machine counts
                    control_plane_machine_count=3,
                    worker_machine_count=3,
                    gpu_machine_count=1
                )
                
                if "error" in result:
                    self.logger.error(f"Failed to create new cluster: {result['error']}")
                    return {"error": f"Failed to create new cluster: {result['error']}"}
                    
                # In a real implementation, wait for the cluster to be ready
                # and then migrate all workloads from the old infrastructure
                
                self.logger.info(f"New CAPI-managed cluster {cluster_name} is being created for migration")
                return {
                    "status": "creating",
                    "cluster_name": cluster_name,
                    "provider": provider,
                    "region": region,
                    "migration_type": "new_cluster"
                }
                
            # For create_new=False, import the existing cluster into CAPI management
            else:
                self.logger.info(f"Importing existing infrastructure from {provider} in {region} to CAPI management")
                
                # In a real implementation, this would:
                # 1. Generate templates from the existing cluster
                # 2. Import it into CAPI management
                # 3. Verify the import was successful
                
                # For simplicity, we'll just return success
                return {
                    "status": "importing",
                    "cluster_name": cluster_name,
                    "provider": provider,
                    "region": region,
                    "migration_type": "import"
                }
                
        except Exception as e:
            self.logger.error(f"Error migrating to Cluster API: {e}")
            return {"error": f"Error migrating to Cluster API: {str(e)}"}