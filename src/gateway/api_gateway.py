"""
API Gateway module for multi-cloud Kubernetes infrastructure.

This module provides a unified API Gateway for routing traffic to LLM services
deployed across multiple cloud providers (AWS and GCP).
"""
import logging
import json
import uuid
import time
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import threading

from src.cloud.provider import CloudProvider
from src.utils.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class APIGateway:
    """API Gateway for multi-cloud infrastructure.
    
    Provides a unified interface for routing requests to LLM services
    deployed across multiple cloud providers and regions.
    """

    def __init__(self, config: Dict[str, Any], cloud_providers: Dict[str, CloudProvider]):
        """Initialize API Gateway.

        Args:
            config: API Gateway configuration.
            cloud_providers: Dictionary of cloud providers.
        """
        self.config = config
        self.cloud_providers = cloud_providers
        self.global_lb_config = config.get("global_lb", {})
        self.routing_policy = self.global_lb_config.get("routing_policy", "latency")
        self.health_check_interval = self.global_lb_config.get("health_check_interval", 30)  # seconds
        self.failover_threshold = self.global_lb_config.get("failover_threshold", 3)
        self.logger = logging.getLogger(f"{__name__}.APIGateway")

        # Security configuration
        self.security_config = config.get("security", {})
        
        # Traffic metrics
        self.metrics = {
            "requests_count": 0,
            "requests_per_provider": {},
            "latency_per_provider": {},
            "errors_per_provider": {},
            "provider_health": {}
        }
        
        # Endpoint registry
        self.endpoints = {}
        
        # Circuit breakers for providers
        self.circuit_breakers = {}
        for provider_name in self.cloud_providers:
            self.circuit_breakers[provider_name] = CircuitBreaker(
                name=f"api-gateway-{provider_name}",
                failure_threshold=self.failover_threshold,
                recovery_timeout=60,  # seconds
                timeout=5  # seconds
            )
            self.metrics["requests_per_provider"][provider_name] = 0
            self.metrics["latency_per_provider"][provider_name] = []
            self.metrics["errors_per_provider"][provider_name] = 0
            self.metrics["provider_health"][provider_name] = True

        # Dictionary to track cloud-specific API gateways
        self.cloud_gateways = {}

        # Initialize cloud-specific API gateways
        self._init_cloud_gateways()
        
        # Start health check thread
        self.running = True
        self.health_check_thread = threading.Thread(target=self._health_check_loop)
        self.health_check_thread.daemon = True
        self.health_check_thread.start()

    def __del__(self):
        """Clean up resources."""
        self.running = False
        if hasattr(self, 'health_check_thread') and self.health_check_thread:
            self.health_check_thread.join(timeout=3.0)

    def _init_cloud_gateways(self) -> None:
        """Initialize cloud-specific API gateways."""
        for provider_name, provider in self.cloud_providers.items():
            if not provider.is_enabled():
                continue

            gateway_config = self.config.get(f"{provider_name}_api_gateway", {})
            if not gateway_config:
                self.logger.warning(f"No API gateway configuration for {provider_name}")
                continue

            gateway_type = gateway_config.get("type", "")
            gateway_name = f"llm-gateway-{provider_name}"

            try:
                gateway_id = provider.create_api_gateway(
                    name=gateway_name, description=f"LLM Service API Gateway for {provider_name}"
                )

                if gateway_id:
                    self.cloud_gateways[provider_name] = {
                        "id": gateway_id,
                        "type": gateway_type,
                        "name": gateway_name,
                        "provider": provider_name,
                        "config": gateway_config,
                        "routes": {},
                        "health": {
                            "status": "healthy",
                            "last_check": datetime.now().isoformat(),
                            "consecutive_failures": 0
                        }
                    }
                    self.logger.info(f"Created API gateway {gateway_name} for {provider_name}")
                else:
                    self.logger.error(f"Failed to create API gateway for {provider_name}")
            except Exception as e:
                self.logger.error(f"Error creating API gateway for {provider_name}: {e}")

    def _health_check_loop(self) -> None:
        """Health check loop for all providers."""
        while self.running:
            for provider_name, gateway_info in self.cloud_gateways.items():
                try:
                    self._check_provider_health(provider_name)
                except Exception as e:
                    self.logger.error(f"Error checking health for provider {provider_name}: {e}")
            
            time.sleep(self.health_check_interval)

    def _check_provider_health(self, provider_name: str) -> None:
        """Check health of a specific provider.
        
        Args:
            provider_name: Cloud provider name.
        """
        if provider_name not in self.cloud_gateways:
            return

        gateway_info = self.cloud_gateways[provider_name]
        
        try:
            provider = self.cloud_providers.get(provider_name)
            if not provider or not provider.is_enabled():
                gateway_info["health"]["status"] = "disabled"
                self.metrics["provider_health"][provider_name] = False
                return

            # In a real implementation, this would use the cloud provider's API
            # to check the health of the API gateway
            
            # For demonstration, simulate a health check
            response_time = 150  # milliseconds
            health_status = "healthy"  # or "unhealthy"
            
            # Update health status
            if health_status == "healthy":
                gateway_info["health"]["consecutive_failures"] = 0
                gateway_info["health"]["status"] = "healthy"
                self.metrics["provider_health"][provider_name] = True
                
                # Reset circuit breaker if it was open
                if self.circuit_breakers[provider_name].is_open():
                    self.circuit_breakers[provider_name].reset()
            else:
                gateway_info["health"]["consecutive_failures"] += 1
                
                # Mark as unhealthy if threshold exceeded
                if gateway_info["health"]["consecutive_failures"] >= self.failover_threshold:
                    gateway_info["health"]["status"] = "unhealthy"
                    self.metrics["provider_health"][provider_name] = False
                    
                    # Trip circuit breaker
                    self.circuit_breakers[provider_name].trip()
            
            gateway_info["health"]["last_check"] = datetime.now().isoformat()
            gateway_info["health"]["response_time"] = response_time
            
            self.logger.debug(f"Health check for {provider_name}: {health_status}, response time: {response_time}ms")
        except Exception as e:
            gateway_info["health"]["consecutive_failures"] += 1
            gateway_info["health"]["status"] = "error"
            gateway_info["health"]["last_error"] = str(e)
            gateway_info["health"]["last_check"] = datetime.now().isoformat()
            
            if gateway_info["health"]["consecutive_failures"] >= self.failover_threshold:
                self.metrics["provider_health"][provider_name] = False
                self.circuit_breakers[provider_name].trip()
            
            self.logger.error(f"Health check error for {provider_name}: {e}")

    def register_route(self, path: str, method: str, service_name: str, service_port: int, 
                       provider_name: Optional[str] = None) -> bool:
        """Register a route in cloud-specific API gateways.

        Args:
            path: Route path.
            method: HTTP method.
            service_name: Kubernetes service name.
            service_port: Kubernetes service port.
            provider_name: Specific provider to register with. If None, register with all.

        Returns:
            True if successful, False otherwise.
        """
        success = True
        route_id = str(uuid.uuid4())
        
        # Register endpoint
        self.endpoints[path] = {
            "id": route_id,
            "path": path,
            "method": method,
            "service_name": service_name,
            "service_port": service_port,
            "providers": {}
        }

        # Determine which cloud gateways to register with
        gateways_to_register = {}
        if provider_name:
            if provider_name in self.cloud_gateways:
                gateways_to_register[provider_name] = self.cloud_gateways[provider_name]
            else:
                self.logger.error(f"Provider {provider_name} not found for route registration")
                return False
        else:
            gateways_to_register = self.cloud_gateways

        for gw_name, gateway_info in gateways_to_register.items():
            try:
                provider = self.cloud_providers[gw_name]
                gateway_client = provider.get_api_gateway()

                if not gateway_client:
                    self.logger.error(f"No API gateway client for {gw_name}")
                    success = False
                    continue

                # In a real implementation, this would use cloud-specific API Gateway clients
                # to register routes
                self.logger.info(
                    f"Registering route {method} {path} for service {service_name}:{service_port} in {gw_name}"
                )

                # Store route in gateway info
                gateway_info["routes"][path] = {
                    "id": route_id,
                    "method": method,
                    "service_name": service_name,
                    "service_port": service_port,
                    "created_at": datetime.now().isoformat()
                }
                
                # Store provider in endpoint info
                self.endpoints[path]["providers"][gw_name] = {
                    "endpoint": f"https://{service_name}.{gw_name}.example.com{path}",
                    "region": provider.region,
                    "health": "healthy"
                }
            except Exception as e:
                self.logger.error(f"Error registering route in {gw_name}: {e}")
                success = False

        return success

    def deploy(self) -> bool:
        """Deploy API Gateway configuration to all cloud providers.

        Returns:
            True if successful, False otherwise.
        """
        success = True

        for provider_name, gateway_info in self.cloud_gateways.items():
            try:
                provider = self.cloud_providers[provider_name]
                gateway_client = provider.get_api_gateway()

                if not gateway_client:
                    self.logger.error(f"No API gateway client for {provider_name}")
                    success = False
                    continue

                # In a real implementation, this would use cloud-specific API Gateway clients
                # to deploy the API Gateway configuration
                self.logger.info(f"Deploying API gateway for {provider_name}")

                # Simulated deployment
                gateway_info["deployed"] = True
                gateway_info["deployment_time"] = datetime.now().isoformat()
                gateway_info["endpoint"] = f"https://api-{provider_name}.example.com"
            except Exception as e:
                self.logger.error(f"Error deploying API gateway for {provider_name}: {e}")
                success = False

        if success:
            # After deployment, set up global load balancer
            self.setup_global_load_balancer()
            
            # Configure security
            self.configure_security()

        return success

    def setup_global_load_balancer(self) -> bool:
        """Set up global load balancer for all cloud-specific API gateways.

        Returns:
            True if successful, False otherwise.
        """
        lb_type = self.global_lb_config.get("type", "cloudflare")

        # In a real implementation, this would use the specified global load balancer
        # provider to set up a global load balancer
        self.logger.info(f"Setting up global load balancer of type {lb_type}")

        # Configure routing policy
        routing_config = {"policy": self.routing_policy, "endpoints": []}

        for provider_name, gateway_info in self.cloud_gateways.items():
            if gateway_info.get("deployed", False):
                # Get region for provider
                region = self.cloud_providers[provider_name].region
                endpoint_url = gateway_info.get("endpoint", f"https://api-{provider_name}.example.com")

                routing_config["endpoints"].append(
                    {
                        "provider": provider_name,
                        "url": endpoint_url,
                        "region": region,
                        "weight": 1.0,
                    }
                )

        self.logger.info(f"Global load balancer configuration: {routing_config}")
        self.global_config = routing_config

        # Simulated success for demonstration
        return True

    def get_routing_info(self) -> Dict[str, Any]:
        """Get routing information for the API Gateway.

        Returns:
            Routing information.
        """
        # Calculate metrics for reporting
        total_requests = max(1, self.metrics["requests_count"])  # Avoid division by zero
        provider_percentages = {}
        for provider, count in self.metrics["requests_per_provider"].items():
            provider_percentages[provider] = (count / total_requests) * 100
        
        # Calculate average latency per provider
        avg_latency = {}
        for provider, latencies in self.metrics["latency_per_provider"].items():
            if latencies:
                avg_latency[provider] = sum(latencies) / len(latencies)
            else:
                avg_latency[provider] = 0
        
        # Current health status
        health_status = {
            provider: {
                "healthy": self.metrics["provider_health"].get(provider, False),
                "circuit_breaker": "open" if self.circuit_breakers.get(provider, CircuitBreaker("dummy")).is_open() else "closed",
                "errors": self.metrics["errors_per_provider"].get(provider, 0)
            }
            for provider in self.cloud_providers
        }
        
        return {
            "policy": self.routing_policy,
            "global_endpoint": "https://api.llm-service.example.com",
            "gateways": [
                {
                    "provider": provider_name,
                    "region": self.cloud_providers[provider_name].region,
                    "deployed": gateway_info.get("deployed", False),
                    "health": gateway_info.get("health", {}).get("status", "unknown"),
                    "routes_count": len(gateway_info.get("routes", {})),
                    "request_percentage": provider_percentages.get(provider_name, 0),
                    "avg_latency": avg_latency.get(provider_name, 0)
                }
                for provider_name, gateway_info in self.cloud_gateways.items()
            ],
            "provider_health": health_status,
            "total_requests": total_requests,
            "routing_metrics": {
                "latency_based": {
                    "avg_latency_threshold": 200  # ms
                },
                "geo_based": {
                    "regions": [r for p in self.cloud_providers.values() for r in [p.region] + p.secondary_regions]
                },
                "weighted": {
                    "weights": {p: 1.0 for p in self.cloud_providers}
                }
            }
        }

    def configure_security(self) -> bool:
        """Configure security for all cloud-specific API gateways.

        Returns:
            True if successful, False otherwise.
        """
        auth_type = self.security_config.get("auth_type", "oauth2")

        # In a real implementation, this would use cloud-specific API Gateway clients
        # to configure security settings
        self.logger.info(f"Configuring API gateway security with auth type {auth_type}")

        # Configure CORS
        cors_enabled = self.security_config.get("cors_enabled", True)
        allowed_origins = self.security_config.get("allowed_origins", ["*"])

        # Configure TLS
        ssl_enabled = self.security_config.get("ssl_enabled", True)
        min_tls_version = self.security_config.get("min_tls_version", "1.2")

        # Configure WAF
        waf_enabled = self.security_config.get("waf_enabled", True)

        security_settings = {
            "auth_type": auth_type,
            "cors_enabled": cors_enabled,
            "allowed_origins": allowed_origins,
            "ssl_enabled": ssl_enabled,
            "min_tls_version": min_tls_version,
            "waf_enabled": waf_enabled,
        }

        for provider_name, gateway_info in self.cloud_gateways.items():
            try:
                # In a real implementation, this would use cloud-specific API Gateway clients
                # to apply security settings
                self.logger.info(f"Applying security settings to {provider_name} API gateway")

                # Simulated success for demonstration
                gateway_info["security"] = security_settings
            except Exception as e:
                self.logger.error(f"Error configuring security for {provider_name}: {e}")
                return False

        return True

    def route_request(self, path: str, method: str, data: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], int]:
        """Route a request to the appropriate LLM service.
        
        Args:
            path: Request path.
            method: HTTP method.
            data: Request data.
            
        Returns:
            Tuple of (response, status_code).
        """
        self.metrics["requests_count"] += 1
        
        # Check if path is registered
        if path not in self.endpoints:
            return {"error": f"No route registered for path: {path}"}, 404
        
        endpoint = self.endpoints[path]
        
        # Select provider based on routing policy
        provider_name = self._select_provider(path)
        if not provider_name:
            return {"error": "No healthy providers available"}, 503
        
        # Update metrics
        self.metrics["requests_per_provider"][provider_name] = \
            self.metrics["requests_per_provider"].get(provider_name, 0) + 1
        
        try:
            # Check circuit breaker
            if self.circuit_breakers[provider_name].is_open():
                self.logger.warning(f"Circuit breaker open for {provider_name}, failing fast")
                return {"error": f"Service unavailable via {provider_name}"}, 503
            
            # In a real implementation, this would forward the request to the actual LLM service
            # For demonstration, we'll simulate a response
            start_time = time.time()
            
            # Simulate processing time
            time.sleep(0.1)
            
            response = {
                "provider": provider_name,
                "region": self.cloud_providers[provider_name].region,
                "service": endpoint["service_name"],
                "timestamp": datetime.now().isoformat(),
                "result": "Simulated response from LLM service"
            }
            
            # Calculate latency
            latency = (time.time() - start_time) * 1000  # convert to ms
            
            # Update latency metrics
            provider_latencies = self.metrics["latency_per_provider"].get(provider_name, [])
            provider_latencies.append(latency)
            
            # Keep only the last 100 latencies
            if len(provider_latencies) > 100:
                provider_latencies = provider_latencies[-100:]
            
            self.metrics["latency_per_provider"][provider_name] = provider_latencies
            
            return response, 200
        except Exception as e:
            # Update error metrics
            self.metrics["errors_per_provider"][provider_name] = \
                self.metrics["errors_per_provider"].get(provider_name, 0) + 1
            
            # Record failure in circuit breaker
            self.circuit_breakers[provider_name].record_failure()
            
            self.logger.error(f"Error routing request to {provider_name}: {e}")
            return {"error": f"Error processing request: {str(e)}"}, 500

    def _select_provider(self, path: str) -> Optional[str]:
        """Select a provider based on the routing policy.
        
        Args:
            path: Request path.
            
        Returns:
            Selected provider name or None if no healthy providers available.
        """
        endpoint = self.endpoints.get(path)
        if not endpoint:
            return None
        
        available_providers = [
            provider for provider, info in endpoint["providers"].items()
            if self.metrics["provider_health"].get(provider, False) and 
               not self.circuit_breakers[provider].is_open()
        ]
        
        if not available_providers:
            self.logger.warning(f"No available providers for path {path}")
            return None
        
        # Apply routing policy
        if self.routing_policy == "latency":
            return self._select_by_latency(available_providers)
        elif self.routing_policy == "geo":
            return self._select_by_geo(available_providers)
        elif self.routing_policy == "weighted":
            return self._select_by_weight(available_providers)
        else:
            # Default to random selection
            import random
            return random.choice(available_providers)

    def _select_by_latency(self, providers: List[str]) -> str:
        """Select provider with lowest latency.
        
        Args:
            providers: List of available providers.
            
        Returns:
            Selected provider name.
        """
        if not providers:
            return None
        
        # Find provider with lowest average latency
        avg_latencies = {}
        for provider in providers:
            latencies = self.metrics["latency_per_provider"].get(provider, [])
            if latencies:
                avg_latencies[provider] = sum(latencies) / len(latencies)
            else:
                avg_latencies[provider] = float('inf')  # No data means high latency
        
        # Sort by latency
        sorted_providers = sorted(avg_latencies.items(), key=lambda x: x[1])
        if sorted_providers:
            return sorted_providers[0][0]
        
        # Fallback to first provider
        return providers[0]

    def _select_by_geo(self, providers: List[str]) -> str:
        """Select provider based on geographic proximity.
        
        Args:
            providers: List of available providers.
            
        Returns:
            Selected provider name.
        """
        # In a real implementation, this would determine the client's location
        # and select the closest provider
        
        # For demonstration, just select the first available provider
        return providers[0]

    def _select_by_weight(self, providers: List[str]) -> str:
        """Select provider based on weights.
        
        Args:
            providers: List of available providers.
            
        Returns:
            Selected provider name.
        """
        import random
        
        # In a real implementation, this would use configured weights
        # For demonstration, use equal weights
        weights = {provider: 1.0 for provider in providers}
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {p: w/total_weight for p, w in weights.items()}
        
        # Cumulative distribution
        items = sorted(normalized_weights.items())
        cum_weights = []
        cum_sum = 0
        for item, weight in items:
            cum_sum += weight
            cum_weights.append((item, cum_sum))
        
        # Select based on weights
        x = random.random()
        for item, cum_weight in cum_weights:
            if x <= cum_weight:
                return item
        
        # Fallback
        return providers[0]

    def reset_metrics(self) -> None:
        """Reset gateway metrics."""
        self.metrics = {
            "requests_count": 0,
            "requests_per_provider": {provider: 0 for provider in self.cloud_providers},
            "latency_per_provider": {provider: [] for provider in self.cloud_providers},
            "errors_per_provider": {provider: 0 for provider in self.cloud_providers},
            "provider_health": self.metrics["provider_health"]  # Preserve health status
        }
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get gateway metrics.
        
        Returns:
            Dictionary of metrics.
        """
        # Calculate additional metrics
        total_requests = self.metrics["requests_count"]
        provider_percentages = {}
        error_rates = {}
        
        for provider in self.cloud_providers:
            provider_requests = self.metrics["requests_per_provider"].get(provider, 0)
            provider_errors = self.metrics["errors_per_provider"].get(provider, 0)
            
            # Calculate percentage of total requests
            if total_requests > 0:
                provider_percentages[provider] = (provider_requests / total_requests) * 100
            else:
                provider_percentages[provider] = 0
                
            # Calculate error rate
            if provider_requests > 0:
                error_rates[provider] = (provider_errors / provider_requests) * 100
            else:
                error_rates[provider] = 0
        
        # Calculate average latency
        avg_latencies = {}
        for provider, latencies in self.metrics["latency_per_provider"].items():
            if latencies:
                avg_latencies[provider] = sum(latencies) / len(latencies)
            else:
                avg_latencies[provider] = 0
        
        return {
            "total_requests": total_requests,
            "provider_distribution": provider_percentages,
            "error_rates": error_rates,
            "average_latencies": avg_latencies,
            "provider_health": self.metrics["provider_health"]
        }