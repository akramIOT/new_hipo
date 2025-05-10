"""
Cluster API (CAPI) Manager for multi-cloud Kubernetes infrastructure.

This module provides implementation of the Cluster API for managing Kubernetes
clusters across multiple cloud providers using a declarative, Kubernetes-native approach.
"""
import logging
import os
import json
import yaml
import tempfile
import time
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ClusterAPIManager:
    """Manager for Cluster API operations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Cluster API Manager.
        
        Args:
            config: CAPI configuration dictionary.
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ClusterAPIManager")
        
        # Management cluster configuration
        self.management_context = config.get("management_context", "management-cluster")
        self.management_cluster_name = config.get("management_cluster_name", "hipo-management")
        self.management_cluster_namespace = config.get("management_cluster_namespace", "default")
        
        # Provider configurations
        self.providers = config.get("providers", {})
        self.templates_dir = config.get("templates_dir", os.path.join(os.path.dirname(__file__), "templates"))
        
        # CAPI version
        self.capi_version = config.get("capi_version", "1.4.0")
        
        # Status tracking
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the CAPI manager.
        
        This sets up the management cluster and installs necessary components.
        
        Returns:
            True if initialization is successful, False otherwise.
        """
        try:
            # Check if management cluster exists
            if self._check_management_cluster_exists():
                self.logger.info(f"Management cluster '{self.management_cluster_name}' already exists")
                self._switch_to_management_context()
            else:
                # Bootstrap a new management cluster if it doesn't exist
                self.logger.info(f"Management cluster '{self.management_cluster_name}' not found, bootstrapping...")
                if not self._bootstrap_management_cluster():
                    self.logger.error("Failed to bootstrap management cluster")
                    return False
            
            # Initialize provider credentials
            if not self._initialize_provider_credentials():
                self.logger.error("Failed to initialize provider credentials")
                return False
            
            # Install CAPI components
            if not self._install_capi_components():
                self.logger.error("Failed to install CAPI components")
                return False
            
            self.initialized = True
            self.logger.info("ClusterAPIManager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing ClusterAPIManager: {e}")
            return False
    
    def _check_management_cluster_exists(self) -> bool:
        """Check if the management cluster exists.
        
        Returns:
            True if the management cluster exists, False otherwise.
        """
        try:
            # Try to get kubeconfig contexts
            result = subprocess.run(
                ["kubectl", "config", "get-contexts", "-o", "name"],
                capture_output=True,
                text=True,
                check=True
            )
            contexts = result.stdout.strip().split('\n')
            
            # Check if management context exists
            return self.management_context in contexts
        except subprocess.CalledProcessError:
            return False
        except Exception as e:
            self.logger.error(f"Error checking if management cluster exists: {e}")
            return False
    
    def _switch_to_management_context(self) -> bool:
        """Switch to the management cluster context.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            subprocess.run(
                ["kubectl", "config", "use-context", self.management_context],
                check=True
            )
            self.logger.info(f"Switched to management context: {self.management_context}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error switching to management context: {e}")
            return False
    
    def _bootstrap_management_cluster(self) -> bool:
        """Bootstrap a new management cluster.
        
        Returns:
            True if successful, False otherwise.
        """
        self.logger.info("Bootstrapping management cluster using kind")
        
        try:
            # Create a kind configuration file
            kind_config = self._generate_kind_config()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
                kind_config_path = temp_file.name
                yaml.dump(kind_config, temp_file)
            
            # Create kind cluster
            subprocess.run(
                [
                    "kind", "create", "cluster",
                    "--name", self.management_cluster_name,
                    "--config", kind_config_path,
                    "--kubeconfig", f"{os.environ.get('HOME')}/.kube/config"
                ],
                check=True
            )
            
            # Clean up the temporary file
            os.unlink(kind_config_path)
            
            # Rename context to management context
            subprocess.run(
                [
                    "kubectl", "config", "rename-context",
                    f"kind-{self.management_cluster_name}",
                    self.management_context
                ],
                check=True
            )
            
            # Switch to management context
            self._switch_to_management_context()
            
            self.logger.info(f"Management cluster '{self.management_cluster_name}' bootstrapped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error bootstrapping management cluster: {e}")
            return False
    
    def _generate_kind_config(self) -> Dict[str, Any]:
        """Generate a kind configuration for the management cluster.
        
        Returns:
            Dictionary containing kind configuration.
        """
        return {
            "kind": "Cluster",
            "apiVersion": "kind.x-k8s.io/v1alpha4",
            "name": self.management_cluster_name,
            "nodes": [
                {
                    "role": "control-plane",
                    "kubeadmConfigPatches": [
                        "kind: InitConfiguration\nnodeRegistration:\n  kubeletExtraArgs:\n    node-labels: \"ingress-ready=true\"\n"
                    ],
                    "extraPortMappings": [
                        {"containerPort": 80, "hostPort": 80, "protocol": "TCP"},
                        {"containerPort": 443, "hostPort": 443, "protocol": "TCP"}
                    ]
                },
                {"role": "worker"},
                {"role": "worker"}
            ],
            "networking": {
                "podSubnet": "192.168.0.0/16",
                "serviceSubnet": "10.96.0.0/12"
            }
        }
    
    def _initialize_provider_credentials(self) -> bool:
        """Initialize credentials for cloud providers.
        
        Returns:
            True if successful, False otherwise.
        """
        success = True
        
        # Initialize AWS provider if configured
        if "aws" in self.providers and self.providers["aws"].get("enabled", False):
            aws_success = self._initialize_aws_credentials()
            success = success and aws_success
        
        # Initialize GCP provider if configured
        if "gcp" in self.providers and self.providers["gcp"].get("enabled", False):
            gcp_success = self._initialize_gcp_credentials()
            success = success and gcp_success
        
        # Initialize Azure provider if configured
        if "azure" in self.providers and self.providers["azure"].get("enabled", False):
            azure_success = self._initialize_azure_credentials()
            success = success and azure_success
        
        return success
    
    def _initialize_aws_credentials(self) -> bool:
        """Initialize AWS credentials for CAPA.
        
        Returns:
            True if successful, False otherwise.
        """
        self.logger.info("Initializing AWS credentials for CAPA")
        
        try:
            aws_config = self.providers["aws"]
            
            # Create clusterawsadm configuration
            clusterawsadm_config = {
                "apiVersion": "bootstrap.aws.infrastructure.cluster.x-k8s.io/v1beta1",
                "kind": "AWSIAMConfiguration",
                "spec": {
                    "bootstrapUser": {
                        "enable": True,
                        "userName": aws_config.get("bootstrap_user_name", "hipo-capi-bootstrap")
                    },
                    "eventBridge": {
                        "enable": True
                    },
                    "region": aws_config.get("region", "us-west-2")
                }
            }
            
            # Write config to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
                config_path = temp_file.name
                yaml.dump(clusterawsadm_config, temp_file)
            
            # Run clusterawsadm to generate CloudFormation template
            cf_template = subprocess.run(
                ["clusterawsadm", "bootstrap", "iam", "print-cloudformation", "--config", config_path],
                capture_output=True,
                text=True,
                check=True
            ).stdout
            
            # Clean up temporary file
            os.unlink(config_path)
            
            # Store the template for reference
            with open(os.path.join(self.templates_dir, "aws-cloudformation.yaml"), "w") as f:
                f.write(cf_template)
            
            # Create CloudFormation stack
            # Note: In a real implementation, we might want to show this to the user or provide
            # other ways to apply the CloudFormation template, especially for production environments
            self.logger.info("AWS CloudFormation template generated. Creating stack...")
            
            # Create AWS credentials secret in management cluster
            self._create_aws_credentials_secret()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing AWS credentials: {e}")
            return False
    
    def _create_aws_credentials_secret(self) -> bool:
        """Create Kubernetes secret with AWS credentials.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            aws_config = self.providers["aws"]
            access_key = aws_config.get("access_key")
            secret_key = aws_config.get("secret_key")
            
            if not access_key or not secret_key:
                self.logger.error("AWS access_key and secret_key must be provided")
                return False
            
            # Create secret manifest
            secret_manifest = {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {
                    "name": "capa-manager-bootstrap-credentials",
                    "namespace": self.management_cluster_namespace
                },
                "stringData": {
                    "credentials": f"[default]\naws_access_key_id = {access_key}\naws_secret_access_key = {secret_key}\n"
                }
            }
            
            # Apply secret
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
                secret_path = temp_file.name
                yaml.dump(secret_manifest, temp_file)
            
            subprocess.run(
                ["kubectl", "apply", "-f", secret_path],
                check=True
            )
            
            # Clean up temporary file
            os.unlink(secret_path)
            
            self.logger.info("AWS credentials secret created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating AWS credentials secret: {e}")
            return False
    
    def _initialize_gcp_credentials(self) -> bool:
        """Initialize GCP credentials for CAPG.
        
        Returns:
            True if successful, False otherwise.
        """
        self.logger.info("Initializing GCP credentials for CAPG")
        
        try:
            gcp_config = self.providers["gcp"]
            service_account_key = gcp_config.get("service_account_key")
            
            if not service_account_key:
                self.logger.error("GCP service_account_key must be provided")
                return False
            
            # Check if service_account_key is a file path or the key content
            if os.path.isfile(service_account_key):
                with open(service_account_key, "r") as f:
                    key_content = f.read()
            else:
                key_content = service_account_key
            
            # Create secret manifest
            secret_manifest = {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {
                    "name": "capg-manager-bootstrap-credentials",
                    "namespace": self.management_cluster_namespace
                },
                "stringData": {
                    "credentials.json": key_content
                }
            }
            
            # Apply secret
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
                secret_path = temp_file.name
                yaml.dump(secret_manifest, temp_file)
            
            subprocess.run(
                ["kubectl", "apply", "-f", secret_path],
                check=True
            )
            
            # Clean up temporary file
            os.unlink(secret_path)
            
            # Set environment variables for CAPG
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/kubernetes/capg-manager-bootstrap-credentials/credentials.json"
            
            self.logger.info("GCP credentials secret created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing GCP credentials: {e}")
            return False
    
    def _initialize_azure_credentials(self) -> bool:
        """Initialize Azure credentials for CAPZ.
        
        Returns:
            True if successful, False otherwise.
        """
        self.logger.info("Initializing Azure credentials for CAPZ")
        
        try:
            azure_config = self.providers["azure"]
            
            # Required credentials
            client_id = azure_config.get("client_id")
            client_secret = azure_config.get("client_secret")
            subscription_id = azure_config.get("subscription_id")
            tenant_id = azure_config.get("tenant_id")
            
            if not all([client_id, client_secret, subscription_id, tenant_id]):
                self.logger.error("Azure credentials must include client_id, client_secret, subscription_id, and tenant_id")
                return False
            
            # Create secret manifest
            secret_manifest = {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {
                    "name": "capz-manager-bootstrap-credentials",
                    "namespace": self.management_cluster_namespace
                },
                "stringData": {
                    "azure.json": json.dumps({
                        "clientId": client_id,
                        "clientSecret": client_secret,
                        "subscriptionId": subscription_id,
                        "tenantId": tenant_id
                    })
                }
            }
            
            # Apply secret
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
                secret_path = temp_file.name
                yaml.dump(secret_manifest, temp_file)
            
            subprocess.run(
                ["kubectl", "apply", "-f", secret_path],
                check=True
            )
            
            # Clean up temporary file
            os.unlink(secret_path)
            
            # Set environment variables for CAPZ
            os.environ["AZURE_SUBSCRIPTION_ID"] = subscription_id
            os.environ["AZURE_TENANT_ID"] = tenant_id
            os.environ["AZURE_CLIENT_ID"] = client_id
            os.environ["AZURE_CLIENT_SECRET"] = client_secret
            
            self.logger.info("Azure credentials secret created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing Azure credentials: {e}")
            return False
    
    def _install_capi_components(self) -> bool:
        """Install CAPI components in the management cluster.
        
        Returns:
            True if successful, False otherwise.
        """
        self.logger.info("Installing CAPI components in management cluster")
        
        try:
            # Install CAPI core components
            subprocess.run(
                [
                    "clusterctl", "init", "--infrastructure=none",
                    "--core", f"cluster-api:{self.capi_version}",
                    "--bootstrap", f"kubeadm:{self.capi_version}",
                    "--control-plane", f"kubeadm:{self.capi_version}"
                ],
                check=True
            )
            
            # Install provider components based on configuration
            if "aws" in self.providers and self.providers["aws"].get("enabled", False):
                self._install_aws_provider()
            
            if "gcp" in self.providers and self.providers["gcp"].get("enabled", False):
                self._install_gcp_provider()
            
            if "azure" in self.providers and self.providers["azure"].get("enabled", False):
                self._install_azure_provider()
            
            # Wait for components to be ready
            self.logger.info("Waiting for CAPI components to be ready...")
            time.sleep(30)  # Simple wait, in a real implementation we would poll the API
            
            self.logger.info("CAPI components installed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error installing CAPI components: {e}")
            return False
    
    def _install_aws_provider(self) -> bool:
        """Install AWS provider (CAPA) components.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            aws_config = self.providers["aws"]
            region = aws_config.get("region", "us-west-2")
            
            # Install CAPA provider
            subprocess.run(
                [
                    "clusterctl", "init", "--infrastructure", "aws",
                    "--bootstrap", "kubeadm",
                    "--control-plane", "kubeadm"
                ],
                env={**os.environ, "AWS_REGION": region, "EXP_MACHINE_POOL": "true"},
                check=True
            )
            
            self.logger.info("AWS provider (CAPA) installed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error installing AWS provider: {e}")
            return False
    
    def _install_gcp_provider(self) -> bool:
        """Install GCP provider (CAPG) components.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            gcp_config = self.providers["gcp"]
            project_id = gcp_config.get("project_id")
            region = gcp_config.get("region", "us-central1")
            
            if not project_id:
                self.logger.error("GCP project_id must be provided")
                return False
            
            # Install CAPG provider
            subprocess.run(
                [
                    "clusterctl", "init", "--infrastructure", "gcp",
                    "--bootstrap", "kubeadm",
                    "--control-plane", "kubeadm"
                ],
                env={
                    **os.environ,
                    "GOOGLE_PROJECT": project_id,
                    "GOOGLE_REGION": region,
                    "EXP_MACHINE_POOL": "true"
                },
                check=True
            )
            
            self.logger.info("GCP provider (CAPG) installed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error installing GCP provider: {e}")
            return False
    
    def _install_azure_provider(self) -> bool:
        """Install Azure provider (CAPZ) components.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            azure_config = self.providers["azure"]
            location = azure_config.get("location", "eastus")
            
            # Install CAPZ provider
            subprocess.run(
                [
                    "clusterctl", "init", "--infrastructure", "azure",
                    "--bootstrap", "kubeadm",
                    "--control-plane", "kubeadm"
                ],
                env={
                    **os.environ,
                    "AZURE_LOCATION": location,
                    "EXP_MACHINE_POOL": "true"
                },
                check=True
            )
            
            self.logger.info("Azure provider (CAPZ) installed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error installing Azure provider: {e}")
            return False
    
    def create_workload_cluster(self, 
                              name: str,
                              provider: str,
                              region: str,
                              flavor: str = "ml-llm",
                              control_plane_machine_count: int = 3,
                              worker_machine_count: int = 3,
                              gpu_machine_count: int = 0,
                              gpu_instance_type: str = None) -> Dict[str, Any]:
        """Create a new workload cluster with CAPI.
        
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
        if not self.initialized:
            self.logger.error("ClusterAPIManager not initialized")
            return {"error": "ClusterAPIManager not initialized"}
        
        if provider not in ["aws", "gcp", "azure"]:
            self.logger.error(f"Invalid provider: {provider}")
            return {"error": f"Invalid provider: {provider}"}
        
        if provider not in self.providers or not self.providers[provider].get("enabled", False):
            self.logger.error(f"Provider {provider} is not enabled in configuration")
            return {"error": f"Provider {provider} is not enabled in configuration"}
        
        try:
            # Generate cluster manifest
            manifest = self._generate_cluster_manifest(
                name=name,
                provider=provider,
                region=region,
                flavor=flavor,
                control_plane_machine_count=control_plane_machine_count,
                worker_machine_count=worker_machine_count,
                gpu_machine_count=gpu_machine_count,
                gpu_instance_type=gpu_instance_type
            )
            
            # Apply manifest to management cluster
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
                manifest_path = temp_file.name
                yaml.dump_all(manifest, temp_file)
            
            subprocess.run(
                ["kubectl", "apply", "-f", manifest_path],
                check=True
            )
            
            # Clean up temporary file
            os.unlink(manifest_path)
            
            # Wait for cluster to become available (simplified)
            self.logger.info(f"Cluster {name} creation initiated. Waiting for control plane initialization...")
            
            # In a real implementation, we would poll the API to check cluster status
            # For demonstration, we'll just return the expected kubeconfig path
            kubeconfig_path = f"~/.kube/cluster-{name}.kubeconfig"
            
            return {
                "name": name,
                "provider": provider,
                "region": region,
                "status": "creating",
                "kubeconfig_path": kubeconfig_path
            }
            
        except Exception as e:
            self.logger.error(f"Error creating workload cluster: {e}")
            return {"error": f"Error creating workload cluster: {str(e)}"}
    
    def _generate_cluster_manifest(self,
                                 name: str,
                                 provider: str,
                                 region: str,
                                 flavor: str,
                                 control_plane_machine_count: int,
                                 worker_machine_count: int,
                                 gpu_machine_count: int,
                                 gpu_instance_type: str) -> List[Dict[str, Any]]:
        """Generate cluster manifest based on provider and configuration.
        
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
            List of Kubernetes manifests for the cluster.
        """
        # Get provider-specific configuration
        provider_config = self.providers[provider]
        
        # Generate base cluster manifest
        cluster_manifest = {
            "apiVersion": "cluster.x-k8s.io/v1beta1",
            "kind": "Cluster",
            "metadata": {
                "name": name,
                "namespace": self.management_cluster_namespace,
                "labels": {
                    "capi.hipo.io/provider": provider,
                    "capi.hipo.io/flavor": flavor
                }
            },
            "spec": {
                "clusterNetwork": {
                    "pods": {
                        "cidrBlocks": ["192.168.0.0/16"]
                    },
                    "services": {
                        "cidrBlocks": ["10.128.0.0/12"]
                    }
                }
            }
        }
        
        manifests = []
        
        # Add provider-specific manifests
        if provider == "aws":
            manifests.extend(self._generate_aws_manifests(
                name=name,
                region=region,
                control_plane_machine_count=control_plane_machine_count,
                worker_machine_count=worker_machine_count,
                gpu_machine_count=gpu_machine_count,
                gpu_instance_type=gpu_instance_type or "g4dn.xlarge"
            ))
        elif provider == "gcp":
            manifests.extend(self._generate_gcp_manifests(
                name=name,
                region=region,
                project_id=provider_config.get("project_id"),
                control_plane_machine_count=control_plane_machine_count,
                worker_machine_count=worker_machine_count,
                gpu_machine_count=gpu_machine_count,
                gpu_instance_type=gpu_instance_type or "n1-standard-4"
            ))
        elif provider == "azure":
            manifests.extend(self._generate_azure_manifests(
                name=name,
                location=region,
                control_plane_machine_count=control_plane_machine_count,
                worker_machine_count=worker_machine_count,
                gpu_machine_count=gpu_machine_count,
                gpu_instance_type=gpu_instance_type or "Standard_NC6s_v3"
            ))
        
        # Add cluster manifest at the beginning
        manifests.insert(0, cluster_manifest)
        
        return manifests
    
    def _generate_aws_manifests(self,
                             name: str,
                             region: str,
                             control_plane_machine_count: int,
                             worker_machine_count: int,
                             gpu_machine_count: int,
                             gpu_instance_type: str) -> List[Dict[str, Any]]:
        """Generate AWS-specific manifests for a CAPI cluster.
        
        Args:
            name: Cluster name.
            region: AWS region.
            control_plane_machine_count: Number of control plane machines.
            worker_machine_count: Number of worker machines.
            gpu_machine_count: Number of GPU machines.
            gpu_instance_type: AWS instance type for GPU machines.
            
        Returns:
            List of AWS-specific Kubernetes manifests.
        """
        manifests = []
        
        # AWSCluster
        aws_cluster = {
            "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
            "kind": "AWSCluster",
            "metadata": {
                "name": name,
                "namespace": self.management_cluster_namespace
            },
            "spec": {
                "region": region,
                "sshKeyName": self.providers["aws"].get("ssh_key_name", "")
            }
        }
        manifests.append(aws_cluster)
        
        # Add cluster reference to main cluster
        manifests[0]["spec"]["infrastructureRef"] = {
            "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
            "kind": "AWSCluster",
            "name": name,
            "namespace": self.management_cluster_namespace
        }
        
        # Control plane
        if control_plane_machine_count > 0:
            # KubeadmControlPlane
            kubeadm_control_plane = {
                "apiVersion": "controlplane.cluster.x-k8s.io/v1beta1",
                "kind": "KubeadmControlPlane",
                "metadata": {
                    "name": f"{name}-control-plane",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "replicas": control_plane_machine_count,
                    "machineTemplate": {
                        "infrastructureRef": {
                            "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
                            "kind": "AWSMachineTemplate",
                            "name": f"{name}-control-plane",
                            "namespace": self.management_cluster_namespace
                        }
                    },
                    "kubeadmConfigSpec": {
                        "initConfiguration": {
                            "nodeRegistration": {
                                "name": "{{ ds.meta_data.local_hostname }}",
                                "kubeletExtraArgs": {
                                    "cloud-provider": "aws"
                                }
                            }
                        },
                        "clusterConfiguration": {
                            "apiServer": {
                                "extraArgs": {
                                    "cloud-provider": "aws"
                                }
                            },
                            "controllerManager": {
                                "extraArgs": {
                                    "cloud-provider": "aws"
                                }
                            }
                        },
                        "joinConfiguration": {
                            "nodeRegistration": {
                                "name": "{{ ds.meta_data.local_hostname }}",
                                "kubeletExtraArgs": {
                                    "cloud-provider": "aws"
                                }
                            }
                        }
                    },
                    "version": "v1.25.0"
                }
            }
            manifests.append(kubeadm_control_plane)
            
            # AWSMachineTemplate for control plane
            aws_control_plane_machine_template = {
                "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
                "kind": "AWSMachineTemplate",
                "metadata": {
                    "name": f"{name}-control-plane",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "template": {
                        "spec": {
                            "instanceType": "t3.large",
                            "iamInstanceProfile": "control-plane.cluster-api-provider-aws.sigs.k8s.io",
                            "sshKeyName": self.providers["aws"].get("ssh_key_name", ""),
                            "rootVolume": {
                                "size": 80,
                                "type": "gp2"
                            }
                        }
                    }
                }
            }
            manifests.append(aws_control_plane_machine_template)
            
            # Add control plane reference to main cluster
            manifests[0]["spec"]["controlPlaneRef"] = {
                "apiVersion": "controlplane.cluster.x-k8s.io/v1beta1",
                "kind": "KubeadmControlPlane",
                "name": f"{name}-control-plane",
                "namespace": self.management_cluster_namespace
            }
        
        # Worker nodes
        if worker_machine_count > 0:
            # KubeadmConfigTemplate for workers
            kubeadm_config_template = {
                "apiVersion": "bootstrap.cluster.x-k8s.io/v1beta1",
                "kind": "KubeadmConfigTemplate",
                "metadata": {
                    "name": f"{name}-md-0",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "template": {
                        "spec": {
                            "joinConfiguration": {
                                "nodeRegistration": {
                                    "name": "{{ ds.meta_data.local_hostname }}",
                                    "kubeletExtraArgs": {
                                        "cloud-provider": "aws"
                                    }
                                }
                            }
                        }
                    }
                }
            }
            manifests.append(kubeadm_config_template)
            
            # AWSMachineTemplate for workers
            aws_machine_template = {
                "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
                "kind": "AWSMachineTemplate",
                "metadata": {
                    "name": f"{name}-md-0",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "template": {
                        "spec": {
                            "instanceType": "t3.large",
                            "iamInstanceProfile": "nodes.cluster-api-provider-aws.sigs.k8s.io",
                            "sshKeyName": self.providers["aws"].get("ssh_key_name", ""),
                            "rootVolume": {
                                "size": 80,
                                "type": "gp2"
                            }
                        }
                    }
                }
            }
            manifests.append(aws_machine_template)
            
            # MachineDeployment for workers
            machine_deployment = {
                "apiVersion": "cluster.x-k8s.io/v1beta1",
                "kind": "MachineDeployment",
                "metadata": {
                    "name": f"{name}-md-0",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "clusterName": name,
                    "replicas": worker_machine_count,
                    "selector": {
                        "matchLabels": {
                            "cluster.x-k8s.io/cluster-name": name,
                            "node-type": "worker"
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "cluster.x-k8s.io/cluster-name": name,
                                "node-type": "worker"
                            }
                        },
                        "spec": {
                            "clusterName": name,
                            "bootstrap": {
                                "configRef": {
                                    "apiVersion": "bootstrap.cluster.x-k8s.io/v1beta1",
                                    "kind": "KubeadmConfigTemplate",
                                    "name": f"{name}-md-0",
                                    "namespace": self.management_cluster_namespace
                                }
                            },
                            "infrastructureRef": {
                                "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
                                "kind": "AWSMachineTemplate",
                                "name": f"{name}-md-0",
                                "namespace": self.management_cluster_namespace
                            }
                        }
                    }
                }
            }
            manifests.append(machine_deployment)
        
        # GPU nodes
        if gpu_machine_count > 0:
            # KubeadmConfigTemplate for GPU nodes
            gpu_kubeadm_config_template = {
                "apiVersion": "bootstrap.cluster.x-k8s.io/v1beta1",
                "kind": "KubeadmConfigTemplate",
                "metadata": {
                    "name": f"{name}-gpu-md-0",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "template": {
                        "spec": {
                            "joinConfiguration": {
                                "nodeRegistration": {
                                    "name": "{{ ds.meta_data.local_hostname }}",
                                    "kubeletExtraArgs": {
                                        "cloud-provider": "aws",
                                        "node-labels": "node-type=gpu"
                                    }
                                }
                            }
                        }
                    }
                }
            }
            manifests.append(gpu_kubeadm_config_template)
            
            # AWSMachineTemplate for GPU nodes
            aws_gpu_machine_template = {
                "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
                "kind": "AWSMachineTemplate",
                "metadata": {
                    "name": f"{name}-gpu-md-0",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "template": {
                        "spec": {
                            "instanceType": gpu_instance_type,
                            "iamInstanceProfile": "nodes.cluster-api-provider-aws.sigs.k8s.io",
                            "sshKeyName": self.providers["aws"].get("ssh_key_name", ""),
                            "rootVolume": {
                                "size": 100,
                                "type": "gp2"
                            }
                        }
                    }
                }
            }
            manifests.append(aws_gpu_machine_template)
            
            # MachineDeployment for GPU nodes
            gpu_machine_deployment = {
                "apiVersion": "cluster.x-k8s.io/v1beta1",
                "kind": "MachineDeployment",
                "metadata": {
                    "name": f"{name}-gpu-md-0",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "clusterName": name,
                    "replicas": gpu_machine_count,
                    "selector": {
                        "matchLabels": {
                            "cluster.x-k8s.io/cluster-name": name,
                            "node-type": "gpu"
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "cluster.x-k8s.io/cluster-name": name,
                                "node-type": "gpu"
                            }
                        },
                        "spec": {
                            "clusterName": name,
                            "bootstrap": {
                                "configRef": {
                                    "apiVersion": "bootstrap.cluster.x-k8s.io/v1beta1",
                                    "kind": "KubeadmConfigTemplate",
                                    "name": f"{name}-gpu-md-0",
                                    "namespace": self.management_cluster_namespace
                                }
                            },
                            "infrastructureRef": {
                                "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
                                "kind": "AWSMachineTemplate",
                                "name": f"{name}-gpu-md-0",
                                "namespace": self.management_cluster_namespace
                            }
                        }
                    }
                }
            }
            manifests.append(gpu_machine_deployment)
        
        return manifests
    
    def _generate_gcp_manifests(self,
                             name: str,
                             region: str,
                             project_id: str,
                             control_plane_machine_count: int,
                             worker_machine_count: int,
                             gpu_machine_count: int,
                             gpu_instance_type: str) -> List[Dict[str, Any]]:
        """Generate GCP-specific manifests for a CAPI cluster.
        
        Args:
            name: Cluster name.
            region: GCP region.
            project_id: GCP project ID.
            control_plane_machine_count: Number of control plane machines.
            worker_machine_count: Number of worker machines.
            gpu_machine_count: Number of GPU machines.
            gpu_instance_type: GCP instance type for GPU machines.
            
        Returns:
            List of GCP-specific Kubernetes manifests.
        """
        manifests = []
        
        # GCPCluster
        gcp_cluster = {
            "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
            "kind": "GCPCluster",
            "metadata": {
                "name": name,
                "namespace": self.management_cluster_namespace
            },
            "spec": {
                "project": project_id,
                "region": region,
                "network": {
                    "name": self.providers["gcp"].get("network_name", "default")
                }
            }
        }
        manifests.append(gcp_cluster)
        
        # Add cluster reference to main cluster
        manifests[0]["spec"]["infrastructureRef"] = {
            "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
            "kind": "GCPCluster",
            "name": name,
            "namespace": self.management_cluster_namespace
        }
        
        # Control plane
        if control_plane_machine_count > 0:
            # KubeadmControlPlane
            kubeadm_control_plane = {
                "apiVersion": "controlplane.cluster.x-k8s.io/v1beta1",
                "kind": "KubeadmControlPlane",
                "metadata": {
                    "name": f"{name}-control-plane",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "replicas": control_plane_machine_count,
                    "machineTemplate": {
                        "infrastructureRef": {
                            "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
                            "kind": "GCPMachineTemplate",
                            "name": f"{name}-control-plane",
                            "namespace": self.management_cluster_namespace
                        }
                    },
                    "kubeadmConfigSpec": {
                        "initConfiguration": {
                            "nodeRegistration": {
                                "name": "{{ ds.meta_data.local_hostname }}",
                                "kubeletExtraArgs": {
                                    "cloud-provider": "gce"
                                }
                            }
                        },
                        "clusterConfiguration": {
                            "apiServer": {
                                "extraArgs": {
                                    "cloud-provider": "gce"
                                }
                            },
                            "controllerManager": {
                                "extraArgs": {
                                    "cloud-provider": "gce"
                                }
                            }
                        },
                        "joinConfiguration": {
                            "nodeRegistration": {
                                "name": "{{ ds.meta_data.local_hostname }}",
                                "kubeletExtraArgs": {
                                    "cloud-provider": "gce"
                                }
                            }
                        }
                    },
                    "version": "v1.25.0"
                }
            }
            manifests.append(kubeadm_control_plane)
            
            # GCPMachineTemplate for control plane
            gcp_control_plane_machine_template = {
                "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
                "kind": "GCPMachineTemplate",
                "metadata": {
                    "name": f"{name}-control-plane",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "template": {
                        "spec": {
                            "machineType": "n1-standard-2",
                            "rootDeviceSize": 80,
                            "serviceAccounts": {
                                "email": "default",
                                "scopes": [
                                    "https://www.googleapis.com/auth/cloud-platform"
                                ]
                            }
                        }
                    }
                }
            }
            manifests.append(gcp_control_plane_machine_template)
            
            # Add control plane reference to main cluster
            manifests[0]["spec"]["controlPlaneRef"] = {
                "apiVersion": "controlplane.cluster.x-k8s.io/v1beta1",
                "kind": "KubeadmControlPlane",
                "name": f"{name}-control-plane",
                "namespace": self.management_cluster_namespace
            }
        
        # Worker nodes
        if worker_machine_count > 0:
            # KubeadmConfigTemplate for workers
            kubeadm_config_template = {
                "apiVersion": "bootstrap.cluster.x-k8s.io/v1beta1",
                "kind": "KubeadmConfigTemplate",
                "metadata": {
                    "name": f"{name}-md-0",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "template": {
                        "spec": {
                            "joinConfiguration": {
                                "nodeRegistration": {
                                    "name": "{{ ds.meta_data.local_hostname }}",
                                    "kubeletExtraArgs": {
                                        "cloud-provider": "gce"
                                    }
                                }
                            }
                        }
                    }
                }
            }
            manifests.append(kubeadm_config_template)
            
            # GCPMachineTemplate for workers
            gcp_machine_template = {
                "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
                "kind": "GCPMachineTemplate",
                "metadata": {
                    "name": f"{name}-md-0",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "template": {
                        "spec": {
                            "machineType": "n1-standard-2",
                            "rootDeviceSize": 80,
                            "serviceAccounts": {
                                "email": "default",
                                "scopes": [
                                    "https://www.googleapis.com/auth/cloud-platform"
                                ]
                            }
                        }
                    }
                }
            }
            manifests.append(gcp_machine_template)
            
            # MachineDeployment for workers
            machine_deployment = {
                "apiVersion": "cluster.x-k8s.io/v1beta1",
                "kind": "MachineDeployment",
                "metadata": {
                    "name": f"{name}-md-0",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "clusterName": name,
                    "replicas": worker_machine_count,
                    "selector": {
                        "matchLabels": {
                            "cluster.x-k8s.io/cluster-name": name,
                            "node-type": "worker"
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "cluster.x-k8s.io/cluster-name": name,
                                "node-type": "worker"
                            }
                        },
                        "spec": {
                            "clusterName": name,
                            "bootstrap": {
                                "configRef": {
                                    "apiVersion": "bootstrap.cluster.x-k8s.io/v1beta1",
                                    "kind": "KubeadmConfigTemplate",
                                    "name": f"{name}-md-0",
                                    "namespace": self.management_cluster_namespace
                                }
                            },
                            "infrastructureRef": {
                                "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
                                "kind": "GCPMachineTemplate",
                                "name": f"{name}-md-0",
                                "namespace": self.management_cluster_namespace
                            }
                        }
                    }
                }
            }
            manifests.append(machine_deployment)
        
        # GPU nodes
        if gpu_machine_count > 0:
            # KubeadmConfigTemplate for GPU nodes
            gpu_kubeadm_config_template = {
                "apiVersion": "bootstrap.cluster.x-k8s.io/v1beta1",
                "kind": "KubeadmConfigTemplate",
                "metadata": {
                    "name": f"{name}-gpu-md-0",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "template": {
                        "spec": {
                            "joinConfiguration": {
                                "nodeRegistration": {
                                    "name": "{{ ds.meta_data.local_hostname }}",
                                    "kubeletExtraArgs": {
                                        "cloud-provider": "gce",
                                        "node-labels": "node-type=gpu"
                                    }
                                }
                            }
                        }
                    }
                }
            }
            manifests.append(gpu_kubeadm_config_template)
            
            # GCPMachineTemplate for GPU nodes
            gcp_gpu_machine_template = {
                "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
                "kind": "GCPMachineTemplate",
                "metadata": {
                    "name": f"{name}-gpu-md-0",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "template": {
                        "spec": {
                            "machineType": gpu_instance_type,
                            "rootDeviceSize": 100,
                            "serviceAccounts": {
                                "email": "default",
                                "scopes": [
                                    "https://www.googleapis.com/auth/cloud-platform"
                                ]
                            },
                            "acceleratorType": "nvidia-tesla-t4",
                            "acceleratorCount": 1
                        }
                    }
                }
            }
            manifests.append(gcp_gpu_machine_template)
            
            # MachineDeployment for GPU nodes
            gpu_machine_deployment = {
                "apiVersion": "cluster.x-k8s.io/v1beta1",
                "kind": "MachineDeployment",
                "metadata": {
                    "name": f"{name}-gpu-md-0",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "clusterName": name,
                    "replicas": gpu_machine_count,
                    "selector": {
                        "matchLabels": {
                            "cluster.x-k8s.io/cluster-name": name,
                            "node-type": "gpu"
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "cluster.x-k8s.io/cluster-name": name,
                                "node-type": "gpu"
                            }
                        },
                        "spec": {
                            "clusterName": name,
                            "bootstrap": {
                                "configRef": {
                                    "apiVersion": "bootstrap.cluster.x-k8s.io/v1beta1",
                                    "kind": "KubeadmConfigTemplate",
                                    "name": f"{name}-gpu-md-0",
                                    "namespace": self.management_cluster_namespace
                                }
                            },
                            "infrastructureRef": {
                                "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
                                "kind": "GCPMachineTemplate",
                                "name": f"{name}-gpu-md-0",
                                "namespace": self.management_cluster_namespace
                            }
                        }
                    }
                }
            }
            manifests.append(gpu_machine_deployment)
        
        return manifests
    
    def _generate_azure_manifests(self,
                               name: str,
                               location: str,
                               control_plane_machine_count: int,
                               worker_machine_count: int,
                               gpu_machine_count: int,
                               gpu_instance_type: str) -> List[Dict[str, Any]]:
        """Generate Azure-specific manifests for a CAPI cluster.
        
        Args:
            name: Cluster name.
            location: Azure location.
            control_plane_machine_count: Number of control plane machines.
            worker_machine_count: Number of worker machines.
            gpu_machine_count: Number of GPU machines.
            gpu_instance_type: Azure VM size for GPU machines.
            
        Returns:
            List of Azure-specific Kubernetes manifests.
        """
        manifests = []
        
        # Resource group
        resource_group = self.providers["azure"].get("resource_group", f"hipo-{name}")
        
        # AzureCluster
        azure_cluster = {
            "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
            "kind": "AzureCluster",
            "metadata": {
                "name": name,
                "namespace": self.management_cluster_namespace
            },
            "spec": {
                "resourceGroup": resource_group,
                "location": location,
                "networkSpec": {
                    "vnet": {
                        "name": f"{name}-vnet"
                    }
                }
            }
        }
        manifests.append(azure_cluster)
        
        # Add cluster reference to main cluster
        manifests[0]["spec"]["infrastructureRef"] = {
            "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
            "kind": "AzureCluster",
            "name": name,
            "namespace": self.management_cluster_namespace
        }
        
        # Control plane
        if control_plane_machine_count > 0:
            # KubeadmControlPlane
            kubeadm_control_plane = {
                "apiVersion": "controlplane.cluster.x-k8s.io/v1beta1",
                "kind": "KubeadmControlPlane",
                "metadata": {
                    "name": f"{name}-control-plane",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "replicas": control_plane_machine_count,
                    "machineTemplate": {
                        "infrastructureRef": {
                            "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
                            "kind": "AzureMachineTemplate",
                            "name": f"{name}-control-plane",
                            "namespace": self.management_cluster_namespace
                        }
                    },
                    "kubeadmConfigSpec": {
                        "initConfiguration": {
                            "nodeRegistration": {
                                "name": "{{ ds.meta_data.local_hostname }}",
                                "kubeletExtraArgs": {
                                    "cloud-provider": "azure"
                                }
                            }
                        },
                        "clusterConfiguration": {
                            "apiServer": {
                                "extraArgs": {
                                    "cloud-provider": "azure"
                                }
                            },
                            "controllerManager": {
                                "extraArgs": {
                                    "cloud-provider": "azure"
                                }
                            }
                        },
                        "joinConfiguration": {
                            "nodeRegistration": {
                                "name": "{{ ds.meta_data.local_hostname }}",
                                "kubeletExtraArgs": {
                                    "cloud-provider": "azure"
                                }
                            }
                        }
                    },
                    "version": "v1.25.0"
                }
            }
            manifests.append(kubeadm_control_plane)
            
            # AzureMachineTemplate for control plane
            azure_control_plane_machine_template = {
                "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
                "kind": "AzureMachineTemplate",
                "metadata": {
                    "name": f"{name}-control-plane",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "template": {
                        "spec": {
                            "vmSize": "Standard_D2s_v3",
                            "osDisk": {
                                "osType": "Linux",
                                "diskSizeGB": 80,
                                "managedDisk": {
                                    "storageAccountType": "Premium_LRS"
                                }
                            },
                            "sshPublicKey": self.providers["azure"].get("ssh_public_key", "")
                        }
                    }
                }
            }
            manifests.append(azure_control_plane_machine_template)
            
            # Add control plane reference to main cluster
            manifests[0]["spec"]["controlPlaneRef"] = {
                "apiVersion": "controlplane.cluster.x-k8s.io/v1beta1",
                "kind": "KubeadmControlPlane",
                "name": f"{name}-control-plane",
                "namespace": self.management_cluster_namespace
            }
        
        # Worker nodes
        if worker_machine_count > 0:
            # KubeadmConfigTemplate for workers
            kubeadm_config_template = {
                "apiVersion": "bootstrap.cluster.x-k8s.io/v1beta1",
                "kind": "KubeadmConfigTemplate",
                "metadata": {
                    "name": f"{name}-md-0",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "template": {
                        "spec": {
                            "joinConfiguration": {
                                "nodeRegistration": {
                                    "name": "{{ ds.meta_data.local_hostname }}",
                                    "kubeletExtraArgs": {
                                        "cloud-provider": "azure"
                                    }
                                }
                            }
                        }
                    }
                }
            }
            manifests.append(kubeadm_config_template)
            
            # AzureMachineTemplate for workers
            azure_machine_template = {
                "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
                "kind": "AzureMachineTemplate",
                "metadata": {
                    "name": f"{name}-md-0",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "template": {
                        "spec": {
                            "vmSize": "Standard_D2s_v3",
                            "osDisk": {
                                "osType": "Linux",
                                "diskSizeGB": 80,
                                "managedDisk": {
                                    "storageAccountType": "Premium_LRS"
                                }
                            },
                            "sshPublicKey": self.providers["azure"].get("ssh_public_key", "")
                        }
                    }
                }
            }
            manifests.append(azure_machine_template)
            
            # MachineDeployment for workers
            machine_deployment = {
                "apiVersion": "cluster.x-k8s.io/v1beta1",
                "kind": "MachineDeployment",
                "metadata": {
                    "name": f"{name}-md-0",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "clusterName": name,
                    "replicas": worker_machine_count,
                    "selector": {
                        "matchLabels": {
                            "cluster.x-k8s.io/cluster-name": name,
                            "node-type": "worker"
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "cluster.x-k8s.io/cluster-name": name,
                                "node-type": "worker"
                            }
                        },
                        "spec": {
                            "clusterName": name,
                            "bootstrap": {
                                "configRef": {
                                    "apiVersion": "bootstrap.cluster.x-k8s.io/v1beta1",
                                    "kind": "KubeadmConfigTemplate",
                                    "name": f"{name}-md-0",
                                    "namespace": self.management_cluster_namespace
                                }
                            },
                            "infrastructureRef": {
                                "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
                                "kind": "AzureMachineTemplate",
                                "name": f"{name}-md-0",
                                "namespace": self.management_cluster_namespace
                            }
                        }
                    }
                }
            }
            manifests.append(machine_deployment)
        
        # GPU nodes
        if gpu_machine_count > 0:
            # KubeadmConfigTemplate for GPU nodes
            gpu_kubeadm_config_template = {
                "apiVersion": "bootstrap.cluster.x-k8s.io/v1beta1",
                "kind": "KubeadmConfigTemplate",
                "metadata": {
                    "name": f"{name}-gpu-md-0",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "template": {
                        "spec": {
                            "joinConfiguration": {
                                "nodeRegistration": {
                                    "name": "{{ ds.meta_data.local_hostname }}",
                                    "kubeletExtraArgs": {
                                        "cloud-provider": "azure",
                                        "node-labels": "node-type=gpu"
                                    }
                                }
                            }
                        }
                    }
                }
            }
            manifests.append(gpu_kubeadm_config_template)
            
            # AzureMachineTemplate for GPU nodes
            azure_gpu_machine_template = {
                "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
                "kind": "AzureMachineTemplate",
                "metadata": {
                    "name": f"{name}-gpu-md-0",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "template": {
                        "spec": {
                            "vmSize": gpu_instance_type,
                            "osDisk": {
                                "osType": "Linux",
                                "diskSizeGB": 100,
                                "managedDisk": {
                                    "storageAccountType": "Premium_LRS"
                                }
                            },
                            "sshPublicKey": self.providers["azure"].get("ssh_public_key", "")
                        }
                    }
                }
            }
            manifests.append(azure_gpu_machine_template)
            
            # MachineDeployment for GPU nodes
            gpu_machine_deployment = {
                "apiVersion": "cluster.x-k8s.io/v1beta1",
                "kind": "MachineDeployment",
                "metadata": {
                    "name": f"{name}-gpu-md-0",
                    "namespace": self.management_cluster_namespace
                },
                "spec": {
                    "clusterName": name,
                    "replicas": gpu_machine_count,
                    "selector": {
                        "matchLabels": {
                            "cluster.x-k8s.io/cluster-name": name,
                            "node-type": "gpu"
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "cluster.x-k8s.io/cluster-name": name,
                                "node-type": "gpu"
                            }
                        },
                        "spec": {
                            "clusterName": name,
                            "bootstrap": {
                                "configRef": {
                                    "apiVersion": "bootstrap.cluster.x-k8s.io/v1beta1",
                                    "kind": "KubeadmConfigTemplate",
                                    "name": f"{name}-gpu-md-0",
                                    "namespace": self.management_cluster_namespace
                                }
                            },
                            "infrastructureRef": {
                                "apiVersion": "infrastructure.cluster.x-k8s.io/v1beta1",
                                "kind": "AzureMachineTemplate",
                                "name": f"{name}-gpu-md-0",
                                "namespace": self.management_cluster_namespace
                            }
                        }
                    }
                }
            }
            manifests.append(gpu_machine_deployment)
        
        return manifests
    
    def get_workload_clusters(self) -> List[Dict[str, Any]]:
        """Get all workload clusters managed by CAPI.
        
        Returns:
            List of workload cluster details.
        """
        if not self.initialized:
            self.logger.error("ClusterAPIManager not initialized")
            return []
        
        try:
            # Get all clusters in the management cluster
            result = subprocess.run(
                [
                    "kubectl", "get", "clusters",
                    "-A",
                    "-o", "json"
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            clusters_json = json.loads(result.stdout)
            
            # Format cluster details
            clusters = []
            for item in clusters_json.get("items", []):
                metadata = item.get("metadata", {})
                spec = item.get("spec", {})
                status = item.get("status", {})
                
                # Skip management cluster
                if metadata.get("name") == self.management_cluster_name:
                    continue
                
                # Extract provider from labels or infrastructure ref
                provider = "unknown"
                if "infrastructureRef" in spec:
                    infra_ref = spec["infrastructureRef"]
                    if "aws" in infra_ref.get("kind", "").lower():
                        provider = "aws"
                    elif "gcp" in infra_ref.get("kind", "").lower():
                        provider = "gcp"
                    elif "azure" in infra_ref.get("kind", "").lower():
                        provider = "azure"
                
                # Extract region/location
                region = "unknown"
                if "controlPlaneEndpoint" in status:
                    region = status.get("controlPlaneEndpoint", {}).get("host", "").split(".")[0]
                
                clusters.append({
                    "name": metadata.get("name"),
                    "namespace": metadata.get("namespace"),
                    "provider": provider,
                    "region": region,
                    "status": status.get("phase", "unknown"),
                    "control_plane_ready": status.get("controlPlaneReady", False),
                    "infrastructure_ready": status.get("infrastructureReady", False),
                    "created_at": metadata.get("creationTimestamp")
                })
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error getting workload clusters: {e}")
            return []
    
    def delete_workload_cluster(self, name: str) -> bool:
        """Delete a workload cluster.
        
        Args:
            name: Name of the cluster to delete.
            
        Returns:
            True if deletion was initiated successfully, False otherwise.
        """
        if not self.initialized:
            self.logger.error("ClusterAPIManager not initialized")
            return False
        
        try:
            # Delete the cluster
            subprocess.run(
                [
                    "kubectl", "delete", "cluster",
                    name,
                    "-n", self.management_cluster_namespace
                ],
                check=True
            )
            
            self.logger.info(f"Cluster {name} deletion initiated")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting workload cluster {name}: {e}")
            return False
    
    def get_kubeconfig(self, cluster_name: str) -> str:
        """Get kubeconfig for a workload cluster.
        
        Args:
            cluster_name: Name of the cluster.
            
        Returns:
            Kubeconfig content as string if successful, empty string otherwise.
        """
        if not self.initialized:
            self.logger.error("ClusterAPIManager not initialized")
            return ""
        
        try:
            # Get kubeconfig from the cluster
            result = subprocess.run(
                [
                    "clusterctl", "get", "kubeconfig",
                    cluster_name,
                    "-n", self.management_cluster_namespace
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            return result.stdout
            
        except Exception as e:
            self.logger.error(f"Error getting kubeconfig for cluster {cluster_name}: {e}")
            return ""
    
    def scale_node_group(self, cluster_name: str, node_group_name: str, desired_size: int) -> bool:
        """Scale a node group in a workload cluster.
        
        Args:
            cluster_name: Name of the cluster.
            node_group_name: Name of the node group (machine deployment).
            desired_size: Desired number of nodes.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.initialized:
            self.logger.error("ClusterAPIManager not initialized")
            return False
        
        try:
            # Scale the machine deployment
            subprocess.run(
                [
                    "kubectl", "scale", "machinedeployment",
                    node_group_name,
                    "--replicas", str(desired_size),
                    "-n", self.management_cluster_namespace
                ],
                check=True
            )
            
            self.logger.info(f"Scaled node group {node_group_name} to {desired_size} nodes")
            return True
            
        except Exception as e:
            self.logger.error(f"Error scaling node group {node_group_name}: {e}")
            return False
    
    def get_node_groups(self, cluster_name: str) -> List[Dict[str, Any]]:
        """Get node groups for a workload cluster.
        
        Args:
            cluster_name: Name of the cluster.
            
        Returns:
            List of node group details.
        """
        if not self.initialized:
            self.logger.error("ClusterAPIManager not initialized")
            return []
        
        try:
            # Get all machine deployments for the cluster
            result = subprocess.run(
                [
                    "kubectl", "get", "machinedeployment",
                    "-n", self.management_cluster_namespace,
                    "-l", f"cluster.x-k8s.io/cluster-name={cluster_name}",
                    "-o", "json"
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            deployments_json = json.loads(result.stdout)
            
            # Format node group details
            node_groups = []
            for item in deployments_json.get("items", []):
                metadata = item.get("metadata", {})
                spec = item.get("spec", {})
                
                # Extract node type from labels
                node_type = "worker"
                template_metadata = spec.get("template", {}).get("metadata", {})
                labels = template_metadata.get("labels", {})
                if labels.get("node-type") == "gpu":
                    node_type = "gpu"
                
                # Extract instance type from infrastructure ref
                instance_type = "unknown"
                if "template" in spec and "spec" in spec["template"]:
                    template_spec = spec["template"]["spec"]
                    if "infrastructureRef" in template_spec:
                        infra_ref = template_spec["infrastructureRef"]
                        machine_template_name = infra_ref.get("name")
                        
                        # Get machine template details
                        machine_type_result = subprocess.run(
                            [
                                "kubectl", "get", infra_ref.get("kind"),
                                machine_template_name,
                                "-n", self.management_cluster_namespace,
                                "-o", "jsonpath={.spec.template.spec.instanceType}"
                            ],
                            capture_output=True,
                            text=True
                        )
                        
                        if machine_type_result.returncode == 0 and machine_type_result.stdout:
                            instance_type = machine_type_result.stdout
                
                node_groups.append({
                    "name": metadata.get("name"),
                    "cluster_name": spec.get("clusterName"),
                    "type": node_type,
                    "replicas": spec.get("replicas", 0),
                    "instance_type": instance_type,
                    "created_at": metadata.get("creationTimestamp")
                })
            
            return node_groups
            
        except Exception as e:
            self.logger.error(f"Error getting node groups for cluster {cluster_name}: {e}")
            return []