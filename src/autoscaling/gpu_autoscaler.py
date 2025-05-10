"""
GPU Autoscaler for multi-cloud Kubernetes infrastructure.

This module provides autoscaling functionality for GPU-based Kubernetes nodes
across multiple cloud providers, optimizing for performance, cost, and availability.
"""
import logging
import time
import threading
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import heapq
import statistics

from src.cloud.provider import CloudProvider
from src.utils.circuit_breaker import CircuitBreaker, retry_with_backoff

logger = logging.getLogger(__name__)


class GPUAutoscaler:
    """GPU Autoscaler for multi-cloud Kubernetes infrastructure.
    
    Provides intelligent autoscaling for GPU resources across multiple cloud providers
    based on utilization metrics, request queue length, response latency, cost, and
    other factors.
    """

    def __init__(self, config: Dict[str, Any], cloud_providers: Dict[str, CloudProvider]):
        """Initialize GPU Autoscaler.

        Args:
            config: Autoscaling configuration.
            cloud_providers: Dictionary of cloud providers.
        """
        self.config = config
        self.cloud_providers = cloud_providers
        self.metrics_config = config.get("metrics", {})
        self.scaling_config = config.get("scaling_config", {})

        # Initialize metrics collection configuration
        self.collection_interval = self.metrics_config.get("collection_interval", 15)  # seconds
        self.logger = logging.getLogger(f"{__name__}.GPUAutoscaler")

        # Initialize metrics thresholds
        self.gpu_metrics_thresholds = self._get_metric_thresholds("gpu_metrics")
        self.queue_metrics_thresholds = self._get_metric_thresholds("queue_metrics")
        self.response_metrics_thresholds = self._get_metric_thresholds("response_metrics")
        self.cost_metrics_thresholds = self._get_metric_thresholds("cost_metrics")

        # Scaling configuration
        self.min_replicas = self.scaling_config.get("min_replicas", 1)
        self.max_replicas = self.scaling_config.get("max_replicas", 20)
        self.target_gpu_utilization = self.scaling_config.get("target_gpu_utilization", 70)
        self.target_queue_length = self.scaling_config.get("target_queue_length", 50)
        self.scale_up_factor = self.scaling_config.get("scale_up_factor", 2)
        self.scale_down_factor = self.scaling_config.get("scale_down_factor", 0.5)
        self.stabilization_window_up = self.scaling_config.get("stabilization_window_up", 60)  # seconds
        self.stabilization_window_down = self.scaling_config.get("stabilization_window_down", 300)  # seconds
        self.enable_cost_based_scaling = self.scaling_config.get("enable_cost_based_scaling", True)
        self.max_cost_per_hour = self.scaling_config.get("max_cost_per_hour", 100)  # dollars
        
        # Multi-cloud coordination settings
        self.cross_cloud_coordination = self.scaling_config.get("cross_cloud_coordination", True)
        self.cloud_weight_strategy = self.scaling_config.get("cloud_weight_strategy", "cost")  # cost, latency, or balanced
        self.failover_enabled = self.scaling_config.get("failover_enabled", True)
        self.reserve_capacity = self.scaling_config.get("reserve_capacity", 20)  # percentage
        
        # Load patterns detection
        self.load_pattern_detection = self.scaling_config.get("load_pattern_detection", {
            "enabled": True,
            "window_size": 24,  # hours
            "pattern_threshold": 0.7  # correlation coefficient threshold
        })
        
        # Predictive scaling
        self.predictive_scaling = self.scaling_config.get("predictive_scaling", {
            "enabled": True,
            "forecast_window": 30,  # minutes
            "min_data_points": 60,  # minimum data points needed for prediction
            "confidence_threshold": 0.8  # confidence threshold for predictions
        })

        # State
        self.running = False
        self.collection_thread = None
        self.scaling_thread = None
        self.last_scale_up_time = {}
        self.last_scale_down_time = {}
        self.current_metrics = {}
        self.metrics_history = {}
        self.node_group_sizes = {}
        self.scaling_decisions = []  # keep track of scaling decisions
        self.scaling_events = []  # keep track of scaling events
        self.cloud_provider_weight = self._calculate_cloud_weights()
        
        # Circuit breakers for cloud providers
        self.circuit_breakers = {}
        for provider_name in self.cloud_providers:
            self.circuit_breakers[provider_name] = CircuitBreaker(
                name=f"gpu-autoscaler-{provider_name}",
                failure_threshold=3,
                recovery_timeout=300,  # 5 minutes
                timeout=30  # seconds
            )
            
        # Initialize metrics history storage
        self._init_metrics_storage()

    def _init_metrics_storage(self) -> None:
        """Initialize metrics history storage.
        
        Creates the directory for storing metrics history if it doesn't exist.
        """
        self.metrics_dir = os.path.join("data", "autoscaling", "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        
    def _get_metric_thresholds(self, metric_type: str) -> Dict[str, float]:
        """Get metric thresholds from config.

        Args:
            metric_type: Metric type.

        Returns:
            Dictionary of metric thresholds.
        """
        thresholds = {}

        for metric_config in self.metrics_config.get(metric_type, []):
            name = metric_config.get("name")
            threshold = metric_config.get("threshold")

            if name and threshold is not None:
                thresholds[name] = threshold

        return thresholds

    def start(self) -> None:
        """Start the GPU Autoscaler."""
        if self.running:
            self.logger.warning("GPU Autoscaler is already running")
            return

        self.running = True
        
        # Start metrics collection thread
        self.collection_thread = threading.Thread(target=self._metrics_collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        # Start scaling decision thread
        self.scaling_thread = threading.Thread(target=self._scaling_decision_loop)
        self.scaling_thread.daemon = True
        self.scaling_thread.start()

        self.logger.info("GPU Autoscaler started")

    def stop(self) -> None:
        """Stop the GPU Autoscaler."""
        self.running = False

        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
            self.collection_thread = None
            
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5.0)
            self.scaling_thread = None

        self.logger.info("GPU Autoscaler stopped")
        
        # Save current metrics to disk
        self._save_metrics_history()

    def _metrics_collection_loop(self) -> None:
        """Metrics collection loop."""
        while self.running:
            try:
                # Collect metrics from all providers
                self._collect_metrics()
                
                # Save metrics to history
                self._update_metrics_history()
                
                # Periodically save metrics to disk
                self._save_metrics_if_needed()
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")

            time.sleep(self.collection_interval)
            
    def _scaling_decision_loop(self) -> None:
        """Scaling decision loop.
        
        This runs separately from metrics collection to allow for different frequency
        and to ensure that scaling decisions are made with the most recent metrics.
        """
        # Add a slight delay to ensure metrics are collected first
        time.sleep(self.collection_interval * 1.5)
        
        # Scaling interval is typically longer than metrics collection interval
        scaling_interval = max(self.collection_interval * 2, 30)  # minimum 30 seconds
        
        while self.running:
            try:
                # Evaluate scaling based on current metrics
                self._evaluate_scaling()
            except Exception as e:
                self.logger.error(f"Error in scaling decision loop: {e}")
                
            time.sleep(scaling_interval)

    def _collect_metrics(self) -> None:
        """Collect metrics from all cloud providers."""
        timestamp = datetime.now().isoformat()

        # Collect metrics from each cloud provider
        for provider_name, provider in self.cloud_providers.items():
            if not provider.is_enabled():
                continue

            try:
                # Check if circuit breaker is open
                if self.circuit_breakers[provider_name].is_open():
                    self.logger.warning(f"Circuit breaker open for {provider_name}, skipping metrics collection")
                    continue
                
                # Use retry with exponential backoff for reliability
                self._collect_provider_metrics(provider_name, provider, timestamp)
            except Exception as e:
                self.logger.error(f"Error collecting metrics for {provider_name}: {e}")
                # Record failure in circuit breaker
                self.circuit_breakers[provider_name].record_failure()

    @retry_with_backoff(max_retries=3, base_delay=1.0, backoff_factor=2.0)
    def _collect_provider_metrics(self, provider_name: str, provider: CloudProvider, timestamp: str) -> None:
        """Collect metrics from a specific provider with retry support.
        
        Args:
            provider_name: Name of the cloud provider.
            provider: Cloud provider instance.
            timestamp: ISO format timestamp for the metrics.
        """
        # Get GPU metrics
        gpu_metrics = provider.get_gpu_metrics()

        # Collect queue metrics from the LLM service
        # In a real implementation, this would get metrics from a queue monitoring service
        queue_metrics = self._collect_queue_metrics(provider_name)
        
        # Collect response metrics from the LLM service
        # In a real implementation, this would get metrics from a service monitoring system
        response_metrics = self._collect_response_metrics(provider_name)

        # Get cost metrics
        cost_metrics = provider.get_cost_metrics(timeframe="hourly")

        # Get current node group sizes
        node_groups = provider.get_node_groups()
        self.node_group_sizes[provider_name] = {
            node_group.get("name"): node_group.get("desiredCapacity", 0) 
            if "desiredCapacity" in node_group 
            else node_group.get("initialNodeCount", 0) 
            for node_group in node_groups
        }

        # Store metrics
        self.current_metrics[provider_name] = {
            "timestamp": timestamp,
            "gpu_metrics": gpu_metrics,
            "queue_metrics": queue_metrics,
            "response_metrics": response_metrics,
            "cost_metrics": cost_metrics,
            "node_groups": node_groups
        }
        
        # Record success in circuit breaker
        self.circuit_breakers[provider_name].record_success()
        
        self.logger.debug(f"Collected metrics for {provider_name}")

    def _collect_queue_metrics(self, provider_name: str) -> Dict[str, Any]:
        """Collect queue metrics for a specific provider.
        
        Args:
            provider_name: Name of the cloud provider.
            
        Returns:
            Dictionary of queue metrics.
        """
        # In a real implementation, this would collect metrics from a queue monitoring service
        # For demonstration, we'll simulate queue metrics
        
        # Check if we have previous metrics to simulate a realistic pattern
        if provider_name in self.metrics_history and self.metrics_history[provider_name]:
            # Get the most recent metrics
            last_metrics = self.metrics_history[provider_name][-1]
            last_queue_metrics = last_metrics.get("queue_metrics", {})
            
            # Simulate some variation
            import random
            last_queue_length = last_queue_metrics.get("queue_length", 50)
            variation = random.uniform(-10, 10)
            new_queue_length = max(0, last_queue_length + variation)
            
            last_queue_latency = last_queue_metrics.get("queue_latency", 500)
            latency_variation = random.uniform(-100, 100)
            new_queue_latency = max(10, last_queue_latency + latency_variation)
            
            return {
                "queue_length": new_queue_length,
                "queue_latency": new_queue_latency  # ms
            }
        else:
            # Initial simulation
            return {"queue_length": 50, "queue_latency": 500}  # ms

    def _collect_response_metrics(self, provider_name: str) -> Dict[str, Any]:
        """Collect response metrics for a specific provider.
        
        Args:
            provider_name: Name of the cloud provider.
            
        Returns:
            Dictionary of response metrics.
        """
        # In a real implementation, this would collect metrics from a service monitoring system
        # For demonstration, we'll simulate response metrics
        
        # Check if we have previous metrics to simulate a realistic pattern
        if provider_name in self.metrics_history and self.metrics_history[provider_name]:
            # Get the most recent metrics
            last_metrics = self.metrics_history[provider_name][-1]
            last_response_metrics = last_metrics.get("response_metrics", {})
            
            # Simulate some variation
            import random
            last_p50 = last_response_metrics.get("latency_p50", 200)
            p50_variation = random.uniform(-20, 20)
            new_p50 = max(50, last_p50 + p50_variation)
            
            last_p95 = last_response_metrics.get("latency_p95", 500)
            p95_variation = random.uniform(-50, 50)
            new_p95 = max(new_p50, last_p95 + p95_variation)
            
            last_p99 = last_response_metrics.get("latency_p99", 800)
            p99_variation = random.uniform(-80, 80)
            new_p99 = max(new_p95, last_p99 + p99_variation)
            
            last_error_rate = last_response_metrics.get("error_rate", 1.5)
            error_variation = random.uniform(-0.5, 0.5)
            new_error_rate = max(0, min(10, last_error_rate + error_variation))
            
            return {
                "latency_p50": new_p50,  # ms
                "latency_p95": new_p95,  # ms
                "latency_p99": new_p99,  # ms
                "error_rate": new_error_rate,  # percentage
            }
        else:
            # Initial simulation
            return {
                "latency_p50": 200,  # ms
                "latency_p95": 500,  # ms
                "latency_p99": 800,  # ms
                "error_rate": 1.5,  # percentage
            }

    def _update_metrics_history(self) -> None:
        """Update metrics history with current metrics."""
        # For each provider, add current metrics to history
        for provider_name, metrics in self.current_metrics.items():
            if provider_name not in self.metrics_history:
                self.metrics_history[provider_name] = []

            self.metrics_history[provider_name].append(metrics)

            # Trim history to keep only recent data
            max_history_entries = 1000  # increased to support pattern detection
            if len(self.metrics_history[provider_name]) > max_history_entries:
                self.metrics_history[provider_name] = self.metrics_history[provider_name][-max_history_entries:]

    def _save_metrics_if_needed(self) -> None:
        """Save metrics to disk if enough time has passed.
        
        We don't want to save too frequently to avoid disk I/O overhead,
        so we'll save every hour.
        """
        # Get the current hour
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        # Check if we have a last saved timestamp
        last_saved_file = os.path.join(self.metrics_dir, "last_saved.txt")
        should_save = False
        
        if os.path.exists(last_saved_file):
            try:
                with open(last_saved_file, "r") as f:
                    last_saved = datetime.fromisoformat(f.read().strip())
                    # Save if more than an hour has passed
                    if current_hour > last_saved:
                        should_save = True
            except Exception as e:
                self.logger.error(f"Error reading last saved timestamp: {e}")
                should_save = True
        else:
            # If no last saved timestamp, save now
            should_save = True
            
        if should_save:
            self._save_metrics_history()
            
            # Update last saved timestamp
            try:
                with open(last_saved_file, "w") as f:
                    f.write(current_hour.isoformat())
            except Exception as e:
                self.logger.error(f"Error writing last saved timestamp: {e}")

    def _save_metrics_history(self) -> None:
        """Save metrics history to disk."""
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save metrics history for each provider
            for provider_name, metrics_list in self.metrics_history.items():
                filename = os.path.join(self.metrics_dir, f"{provider_name}_metrics_{timestamp}.json")
                
                with open(filename, "w") as f:
                    json.dump(metrics_list, f, indent=2)
                    
            self.logger.info(f"Saved metrics history to {self.metrics_dir}")
        except Exception as e:
            self.logger.error(f"Error saving metrics history: {e}")

    def _evaluate_scaling(self) -> None:
        """Evaluate if scaling is needed based on collected metrics."""
        # If cross-cloud coordination is enabled, we need to make a holistic decision
        if self.cross_cloud_coordination:
            self._evaluate_cross_cloud_scaling()
            return
            
        # Otherwise, evaluate each provider independently
        for provider_name, metrics in self.current_metrics.items():
            if provider_name not in self.cloud_providers:
                continue

            provider = self.cloud_providers[provider_name]
            if not provider.is_enabled():
                continue
                
            # Skip providers with open circuit breakers
            if self.circuit_breakers[provider_name].is_open():
                self.logger.warning(f"Circuit breaker open for {provider_name}, skipping scaling evaluation")
                continue

            try:
                # Check if we're in stabilization window
                now = datetime.now()

                last_scale_up = self.last_scale_up_time.get(provider_name)
                if last_scale_up and (now - last_scale_up).total_seconds() < self.stabilization_window_up:
                    self.logger.info(
                        f"In scale-up stabilization window for {provider_name}, skipping scaling evaluation"
                    )
                    continue

                last_scale_down = self.last_scale_down_time.get(provider_name)
                if last_scale_down and (now - last_scale_down).total_seconds() < self.stabilization_window_down:
                    self.logger.info(
                        f"In scale-down stabilization window for {provider_name}, skipping scaling evaluation"
                    )
                    continue

                # Get node groups for this provider
                node_groups = metrics.get("node_groups", [])

                for node_group in node_groups:
                    node_group_name = node_group.get("name")

                    # Skip if not a GPU node group
                    if not self._is_gpu_node_group(node_group):
                        continue

                    # Get current size
                    current_size = self._get_node_group_size(node_group)

                    # Calculate desired size based on metrics
                    desired_size = self._calculate_desired_size(provider_name, node_group_name, current_size)

                    # Apply min/max constraints
                    desired_size = max(self.min_replicas, min(desired_size, self.max_replicas))

                    # If no change, continue
                    if desired_size == current_size:
                        self.logger.info(f"No scaling needed for {provider_name}/{node_group_name}")
                        continue

                    # Check cost constraints if scaling up
                    if desired_size > current_size and self.enable_cost_based_scaling:
                        cost_metrics = metrics.get("cost_metrics", {})
                        current_cost = cost_metrics.get("total_cost", 0.0)
                        # Estimate cost increase
                        cost_per_node = current_cost / max(1, current_size)  # avoid division by zero
                        additional_nodes = desired_size - current_size
                        additional_cost = cost_per_node * additional_nodes
                        estimated_cost = current_cost + additional_cost
                        
                        if estimated_cost > self.max_cost_per_hour:
                            self.logger.warning(
                                f"Cost constraint violated for {provider_name}/{node_group_name}: "
                                f"estimated cost ${estimated_cost:.2f} > ${self.max_cost_per_hour:.2f}"
                            )
                            # Scale to maximum affordable size
                            affordable_size = int(current_size * (self.max_cost_per_hour / current_cost))
                            desired_size = max(current_size, min(desired_size, affordable_size))

                    # Record scaling decision
                    scaling_decision = {
                        "timestamp": datetime.now().isoformat(),
                        "provider": provider_name,
                        "node_group": node_group_name,
                        "current_size": current_size,
                        "desired_size": desired_size,
                        "metrics": {
                            "gpu_utilization": self._get_average_gpu_utilization(metrics),
                            "queue_length": metrics.get("queue_metrics", {}).get("queue_length", 0),
                            "latency_p95": metrics.get("response_metrics", {}).get("latency_p95", 0),
                            "cost_per_hour": metrics.get("cost_metrics", {}).get("total_cost", 0)
                        },
                        "reason": self._determine_scaling_reason(
                            provider_name, current_size, desired_size, metrics
                        )
                    }
                    self.scaling_decisions.append(scaling_decision)
                    
                    # Apply scaling
                    self.logger.info(
                        f"Scaling {provider_name}/{node_group_name} from {current_size} to {desired_size} - "
                        f"Reason: {scaling_decision['reason']}"
                    )
                    
                    if self._scale_node_group(provider, node_group_name, desired_size):
                        # Record the scaling event
                        scaling_event = {
                            "timestamp": datetime.now().isoformat(),
                            "provider": provider_name,
                            "node_group": node_group_name,
                            "from_size": current_size,
                            "to_size": desired_size,
                            "successful": True,
                            "metrics": scaling_decision["metrics"],
                            "reason": scaling_decision["reason"]
                        }
                        self.scaling_events.append(scaling_event)
                        
                        # Update timestamps
                        if desired_size > current_size:
                            self.last_scale_up_time[provider_name] = datetime.now()
                        else:
                            self.last_scale_down_time[provider_name] = datetime.now()

                        self.logger.info(f"Successfully scaled {provider_name}/{node_group_name} to {desired_size}")
                    else:
                        # Record failed scaling event
                        scaling_event = {
                            "timestamp": datetime.now().isoformat(),
                            "provider": provider_name,
                            "node_group": node_group_name,
                            "from_size": current_size,
                            "to_size": desired_size,
                            "successful": False,
                            "metrics": scaling_decision["metrics"],
                            "reason": scaling_decision["reason"]
                        }
                        self.scaling_events.append(scaling_event)
                        self.logger.error(f"Failed to scale {provider_name}/{node_group_name}")
            except Exception as e:
                self.logger.error(f"Error evaluating scaling for {provider_name}: {e}")

    def _evaluate_cross_cloud_scaling(self) -> None:
        """Evaluate scaling across all cloud providers.
        
        This method coordinates scaling decisions across all providers to optimize
        for global performance, cost, and availability.
        """
        # Step 1: Collect global metrics and capacity information
        global_metrics = self._collect_global_metrics()
        if not global_metrics:
            self.logger.warning("No metrics available for cross-cloud scaling evaluation")
            return
            
        # Step 2: Identify the providers and node groups that need scaling
        scaling_needs = self._identify_scaling_needs(global_metrics)
        if not scaling_needs:
            self.logger.info("No cross-cloud scaling needed")
            return
            
        # Step 3: Allocate capacity across providers based on weights
        capacity_allocation = self._allocate_capacity(scaling_needs, global_metrics)
        
        # Step 4: Apply scaling decisions to each provider
        self._apply_cross_cloud_scaling(capacity_allocation)
        
    def _collect_global_metrics(self) -> Dict[str, Any]:
        """Collect global metrics across all providers.
        
        Returns:
            Dictionary of global metrics.
        """
        global_metrics = {
            "timestamp": datetime.now().isoformat(),
            "providers": {},
            "total_gpu_nodes": 0,
            "avg_gpu_utilization": 0,
            "total_queue_length": 0,
            "avg_latency_p95": 0,
            "total_cost_per_hour": 0
        }
        
        provider_count = 0
        for provider_name, metrics in self.current_metrics.items():
            if provider_name not in self.cloud_providers:
                continue
                
            provider = self.cloud_providers[provider_name]
            if not provider.is_enabled():
                continue
                
            # Skip providers with open circuit breakers
            if self.circuit_breakers[provider_name].is_open():
                continue
                
            # Count total GPU nodes
            node_groups = metrics.get("node_groups", [])
            gpu_node_count = 0
            for node_group in node_groups:
                if self._is_gpu_node_group(node_group):
                    gpu_node_count += self._get_node_group_size(node_group)
            
            # Get key metrics
            gpu_utilization = self._get_average_gpu_utilization(metrics)
            queue_length = metrics.get("queue_metrics", {}).get("queue_length", 0)
            latency_p95 = metrics.get("response_metrics", {}).get("latency_p95", 0)
            cost_per_hour = metrics.get("cost_metrics", {}).get("total_cost", 0)
            
            # Add to global metrics
            global_metrics["providers"][provider_name] = {
                "gpu_node_count": gpu_node_count,
                "gpu_utilization": gpu_utilization,
                "queue_length": queue_length,
                "latency_p95": latency_p95,
                "cost_per_hour": cost_per_hour,
                "weight": self.cloud_provider_weight.get(provider_name, 1.0)
            }
            
            global_metrics["total_gpu_nodes"] += gpu_node_count
            global_metrics["avg_gpu_utilization"] += gpu_utilization
            global_metrics["total_queue_length"] += queue_length
            global_metrics["avg_latency_p95"] += latency_p95
            global_metrics["total_cost_per_hour"] += cost_per_hour
            
            provider_count += 1
            
        # Calculate averages
        if provider_count > 0:
            global_metrics["avg_gpu_utilization"] /= provider_count
            global_metrics["avg_latency_p95"] /= provider_count
            
        return global_metrics
            
    def _identify_scaling_needs(self, global_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Identify providers and node groups that need scaling.
        
        Args:
            global_metrics: Dictionary of global metrics.
            
        Returns:
            Dictionary of scaling needs.
        """
        scaling_needs = {
            "scale_up_providers": [],
            "scale_down_providers": [],
            "stable_providers": [],
            "desired_total_nodes": 0
        }
        
        # Check if global scaling is needed
        avg_gpu_utilization = global_metrics["avg_gpu_utilization"]
        total_queue_length = global_metrics["total_queue_length"]
        current_total_nodes = global_metrics["total_gpu_nodes"]
        
        # Determine if global scaling is needed
        if avg_gpu_utilization > self.target_gpu_utilization * 1.1 or total_queue_length > self.target_queue_length * 1.1:
            # Scale up needed
            utilization_ratio = avg_gpu_utilization / self.target_gpu_utilization
            queue_ratio = total_queue_length / self.target_queue_length if self.target_queue_length > 0 else 1.0
            
            # Take the maximum ratio for scaling
            scale_ratio = max(utilization_ratio, queue_ratio)
            desired_total_nodes = int(current_total_nodes * min(scale_ratio, self.scale_up_factor))
            
            # Apply min/max constraints
            desired_total_nodes = max(self.min_replicas, min(desired_total_nodes, self.max_replicas))
            
            scaling_needs["desired_total_nodes"] = desired_total_nodes
            
            # Identify which providers should scale up based on their metrics
            for provider_name, provider_metrics in global_metrics["providers"].items():
                if provider_metrics["gpu_utilization"] > self.target_gpu_utilization or \
                   provider_metrics["queue_length"] > self.target_queue_length / len(global_metrics["providers"]):
                    scaling_needs["scale_up_providers"].append({
                        "provider": provider_name,
                        "current_nodes": provider_metrics["gpu_node_count"],
                        "utilization": provider_metrics["gpu_utilization"],
                        "queue_length": provider_metrics["queue_length"],
                        "latency": provider_metrics["latency_p95"],
                        "cost": provider_metrics["cost_per_hour"],
                        "weight": provider_metrics["weight"]
                    })
                else:
                    scaling_needs["stable_providers"].append(provider_name)
                    
        elif avg_gpu_utilization < self.target_gpu_utilization * 0.7 and total_queue_length < self.target_queue_length * 0.7:
            # Scale down needed
            utilization_ratio = avg_gpu_utilization / self.target_gpu_utilization
            queue_ratio = total_queue_length / self.target_queue_length if self.target_queue_length > 0 else 0.5
            
            # Take the maximum ratio for less aggressive scaling
            scale_ratio = max(utilization_ratio, queue_ratio)
            desired_total_nodes = max(self.min_replicas, int(current_total_nodes * max(scale_ratio, self.scale_down_factor)))
            
            scaling_needs["desired_total_nodes"] = desired_total_nodes
            
            # Identify which providers should scale down based on their metrics
            for provider_name, provider_metrics in global_metrics["providers"].items():
                if provider_metrics["gpu_utilization"] < self.target_gpu_utilization * 0.7 and \
                   provider_metrics["queue_length"] < self.target_queue_length / len(global_metrics["providers"]) * 0.7:
                    scaling_needs["scale_down_providers"].append({
                        "provider": provider_name,
                        "current_nodes": provider_metrics["gpu_node_count"],
                        "utilization": provider_metrics["gpu_utilization"],
                        "queue_length": provider_metrics["queue_length"],
                        "latency": provider_metrics["latency_p95"],
                        "cost": provider_metrics["cost_per_hour"],
                        "weight": provider_metrics["weight"]
                    })
                else:
                    scaling_needs["stable_providers"].append(provider_name)
        else:
            # No scaling needed
            scaling_needs["desired_total_nodes"] = current_total_nodes
            for provider_name in global_metrics["providers"]:
                scaling_needs["stable_providers"].append(provider_name)
                
        return scaling_needs
                
    def _allocate_capacity(self, scaling_needs: Dict[str, Any], global_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate capacity across providers based on weights.
        
        Args:
            scaling_needs: Dictionary of scaling needs.
            global_metrics: Dictionary of global metrics.
            
        Returns:
            Dictionary of capacity allocation.
        """
        capacity_allocation = {}
        desired_total_nodes = scaling_needs["desired_total_nodes"]
        current_total_nodes = global_metrics["total_gpu_nodes"]
        
        # If no change in total nodes, no allocation needed
        if desired_total_nodes == current_total_nodes:
            return {}
            
        # Calculate weighted allocation
        if desired_total_nodes > current_total_nodes:
            # Scale up: allocate based on weights
            providers_to_scale = scaling_needs["scale_up_providers"]
            if not providers_to_scale:
                # If no specific providers to scale up, consider all providers
                providers_to_scale = [
                    {
                        "provider": provider_name,
                        "current_nodes": provider_metrics["gpu_node_count"],
                        "weight": provider_metrics["weight"]
                    }
                    for provider_name, provider_metrics in global_metrics["providers"].items()
                ]
                
            # Calculate total weight
            total_weight = sum(provider["weight"] for provider in providers_to_scale)
            
            # Allocate additional nodes based on weights
            additional_nodes = desired_total_nodes - current_total_nodes
            for provider in providers_to_scale:
                provider_name = provider["provider"]
                current_nodes = provider["current_nodes"]
                weight = provider["weight"]
                
                # Calculate weighted share of additional nodes
                weighted_share = int(additional_nodes * (weight / total_weight)) if total_weight > 0 else 0
                desired_nodes = current_nodes + weighted_share
                
                # Apply min/max constraints per provider
                desired_nodes = max(self.min_replicas, min(desired_nodes, self.max_replicas))
                
                capacity_allocation[provider_name] = {
                    "current_nodes": current_nodes,
                    "desired_nodes": desired_nodes,
                    "action": "scale_up" if desired_nodes > current_nodes else "no_change",
                    "delta": desired_nodes - current_nodes
                }
                
        else:
            # Scale down: allocate based on inverse of weights (scale down lower-weighted providers first)
            providers_to_scale = scaling_needs["scale_down_providers"]
            if not providers_to_scale:
                # If no specific providers to scale down, consider all providers
                providers_to_scale = [
                    {
                        "provider": provider_name,
                        "current_nodes": provider_metrics["gpu_node_count"],
                        "weight": provider_metrics["weight"]
                    }
                    for provider_name, provider_metrics in global_metrics["providers"].items()
                ]
                
            # Sort providers by weight (lower weights first for scale down)
            providers_to_scale.sort(key=lambda p: p["weight"])
            
            # Calculate total nodes to remove
            nodes_to_remove = current_total_nodes - desired_total_nodes
            remaining_to_remove = nodes_to_remove
            
            # First pass: allocate removals based on inverse weight
            for provider in providers_to_scale:
                provider_name = provider["provider"]
                current_nodes = provider["current_nodes"]
                
                # Calculate maximum nodes that can be removed
                max_removable = max(0, current_nodes - self.min_replicas)
                
                if max_removable > 0 and remaining_to_remove > 0:
                    # Remove nodes, but don't go below min_replicas
                    nodes_to_remove_from_provider = min(max_removable, remaining_to_remove)
                    desired_nodes = current_nodes - nodes_to_remove_from_provider
                    
                    capacity_allocation[provider_name] = {
                        "current_nodes": current_nodes,
                        "desired_nodes": desired_nodes,
                        "action": "scale_down" if desired_nodes < current_nodes else "no_change",
                        "delta": desired_nodes - current_nodes
                    }
                    
                    remaining_to_remove -= nodes_to_remove_from_provider
                else:
                    # No change for this provider
                    capacity_allocation[provider_name] = {
                        "current_nodes": current_nodes,
                        "desired_nodes": current_nodes,
                        "action": "no_change",
                        "delta": 0
                    }
                    
        return capacity_allocation
                
    def _apply_cross_cloud_scaling(self, capacity_allocation: Dict[str, Any]) -> None:
        """Apply scaling decisions across cloud providers.
        
        Args:
            capacity_allocation: Dictionary of capacity allocation.
        """
        if not capacity_allocation:
            return
            
        now = datetime.now()
        
        # Apply scaling to each provider
        for provider_name, allocation in capacity_allocation.items():
            if provider_name not in self.cloud_providers:
                continue
                
            provider = self.cloud_providers[provider_name]
            if not provider.is_enabled():
                continue
                
            # Skip providers with open circuit breakers
            if self.circuit_breakers[provider_name].is_open():
                self.logger.warning(f"Circuit breaker open for {provider_name}, skipping scaling")
                continue
                
            # Check if we're in stabilization window
            action = allocation["action"]
            if action == "scale_up":
                last_scale_up = self.last_scale_up_time.get(provider_name)
                if last_scale_up and (now - last_scale_up).total_seconds() < self.stabilization_window_up:
                    self.logger.info(
                        f"In scale-up stabilization window for {provider_name}, skipping scaling"
                    )
                    continue
            elif action == "scale_down":
                last_scale_down = self.last_scale_down_time.get(provider_name)
                if last_scale_down and (now - last_scale_down).total_seconds() < self.stabilization_window_down:
                    self.logger.info(
                        f"In scale-down stabilization window for {provider_name}, skipping scaling"
                    )
                    continue
            elif action == "no_change":
                # No scaling needed
                continue
                
            # Get node groups for this provider
            metrics = self.current_metrics.get(provider_name, {})
            node_groups = metrics.get("node_groups", [])
            
            # Find GPU node groups
            gpu_node_groups = [ng for ng in node_groups if self._is_gpu_node_group(ng)]
            if not gpu_node_groups:
                self.logger.warning(f"No GPU node groups found for {provider_name}")
                continue
                
            # Calculate size per node group
            current_nodes = allocation["current_nodes"]
            desired_nodes = allocation["desired_nodes"]
            delta = allocation["delta"]
            
            # If multiple GPU node groups, distribute nodes proportionally
            if len(gpu_node_groups) > 1:
                self._distribute_nodes_to_groups(provider, gpu_node_groups, current_nodes, desired_nodes)
            else:
                # Only one GPU node group
                node_group = gpu_node_groups[0]
                node_group_name = node_group.get("name")
                current_size = self._get_node_group_size(node_group)
                desired_size = current_size + delta
                
                # Apply min/max constraints
                desired_size = max(self.min_replicas, min(desired_size, self.max_replicas))
                
                # Check if scaling is needed
                if desired_size != current_size:
                    reason = self._determine_cross_cloud_scaling_reason(
                        provider_name, current_size, desired_size, metrics
                    )
                    
                    # Record scaling decision
                    scaling_decision = {
                        "timestamp": datetime.now().isoformat(),
                        "provider": provider_name,
                        "node_group": node_group_name,
                        "current_size": current_size,
                        "desired_size": desired_size,
                        "metrics": {
                            "gpu_utilization": self._get_average_gpu_utilization(metrics),
                            "queue_length": metrics.get("queue_metrics", {}).get("queue_length", 0),
                            "latency_p95": metrics.get("response_metrics", {}).get("latency_p95", 0),
                            "cost_per_hour": metrics.get("cost_metrics", {}).get("total_cost", 0)
                        },
                        "reason": reason,
                        "cross_cloud": True
                    }
                    self.scaling_decisions.append(scaling_decision)
                    
                    # Apply scaling
                    self.logger.info(
                        f"Cross-cloud scaling: {provider_name}/{node_group_name} from {current_size} to {desired_size} - "
                        f"Reason: {reason}"
                    )
                    
                    if self._scale_node_group(provider, node_group_name, desired_size):
                        # Record the scaling event
                        scaling_event = {
                            "timestamp": datetime.now().isoformat(),
                            "provider": provider_name,
                            "node_group": node_group_name,
                            "from_size": current_size,
                            "to_size": desired_size,
                            "successful": True,
                            "metrics": scaling_decision["metrics"],
                            "reason": reason,
                            "cross_cloud": True
                        }
                        self.scaling_events.append(scaling_event)
                        
                        # Update timestamps
                        if desired_size > current_size:
                            self.last_scale_up_time[provider_name] = now
                        else:
                            self.last_scale_down_time[provider_name] = now
                            
                        self.logger.info(f"Successfully scaled {provider_name}/{node_group_name} to {desired_size}")
                    else:
                        # Record failed scaling event
                        scaling_event = {
                            "timestamp": datetime.now().isoformat(),
                            "provider": provider_name,
                            "node_group": node_group_name,
                            "from_size": current_size,
                            "to_size": desired_size,
                            "successful": False,
                            "metrics": scaling_decision["metrics"],
                            "reason": reason,
                            "cross_cloud": True
                        }
                        self.scaling_events.append(scaling_event)
                        self.logger.error(f"Failed to scale {provider_name}/{node_group_name}")
                    
    def _distribute_nodes_to_groups(self, provider: CloudProvider, 
                                   node_groups: List[Dict[str, Any]],
                                   current_total: int, desired_total: int) -> None:
        """Distribute nodes across multiple node groups.
        
        Args:
            provider: Cloud provider.
            node_groups: List of node groups.
            current_total: Current total number of nodes.
            desired_total: Desired total number of nodes.
        """
        # Get current sizes
        current_sizes = {}
        for node_group in node_groups:
            name = node_group.get("name")
            size = self._get_node_group_size(node_group)
            current_sizes[name] = size
            
        # Calculate distribution ratio based on current sizes
        total_current = sum(current_sizes.values())
        if total_current == 0:
            # Equal distribution if all groups are empty
            ratios = {name: 1.0 / len(node_groups) for name in current_sizes}
        else:
            ratios = {name: size / total_current for name, size in current_sizes.items()}
            
        # Distribute delta nodes according to ratios
        delta = desired_total - current_total
        desired_sizes = {}
        remaining_delta = delta
        
        if delta > 0:
            # Scale up: distribute according to ratios
            for name, ratio in ratios.items():
                group_delta = int(delta * ratio)
                desired_sizes[name] = current_sizes[name] + group_delta
                remaining_delta -= group_delta
        else:
            # Scale down: distribute according to ratios, but ensure min_replicas
            # Sort by size (larger first)
            sorted_groups = sorted(current_sizes.items(), key=lambda x: x[1], reverse=True)
            
            for name, size in sorted_groups:
                # Calculate max nodes we can remove
                max_removable = max(0, size - self.min_replicas)
                # Calculate scaled-down size
                if remaining_delta < 0 and max_removable > 0:
                    group_delta = max(remaining_delta, -max_removable)
                    desired_sizes[name] = size + group_delta  # group_delta is negative
                    remaining_delta -= group_delta  # remaining_delta increases (becomes less negative)
                else:
                    desired_sizes[name] = size
                    
        # Distribute any remaining delta to the largest group
        if remaining_delta != 0:
            largest_group = max(current_sizes.items(), key=lambda x: x[1])[0]
            desired_sizes[largest_group] += remaining_delta
            
        # Apply min/max constraints and scale each group
        provider_name = provider.__class__.__name__.replace("Provider", "").lower()
        now = datetime.now()
        metrics = self.current_metrics.get(provider_name, {})
        
        for node_group in node_groups:
            name = node_group.get("name")
            current_size = current_sizes[name]
            desired_size = max(self.min_replicas, min(desired_sizes[name], self.max_replicas))
            
            if desired_size != current_size:
                reason = self._determine_cross_cloud_scaling_reason(
                    provider_name, current_size, desired_size, metrics
                )
                
                # Record scaling decision
                scaling_decision = {
                    "timestamp": now.isoformat(),
                    "provider": provider_name,
                    "node_group": name,
                    "current_size": current_size,
                    "desired_size": desired_size,
                    "metrics": {
                        "gpu_utilization": self._get_average_gpu_utilization(metrics),
                        "queue_length": metrics.get("queue_metrics", {}).get("queue_length", 0),
                        "latency_p95": metrics.get("response_metrics", {}).get("latency_p95", 0),
                        "cost_per_hour": metrics.get("cost_metrics", {}).get("total_cost", 0)
                    },
                    "reason": reason,
                    "cross_cloud": True
                }
                self.scaling_decisions.append(scaling_decision)
                
                # Apply scaling
                self.logger.info(
                    f"Cross-cloud scaling (distributed): {provider_name}/{name} from {current_size} to {desired_size} - "
                    f"Reason: {reason}"
                )
                
                if self._scale_node_group(provider, name, desired_size):
                    # Record the scaling event
                    scaling_event = {
                        "timestamp": now.isoformat(),
                        "provider": provider_name,
                        "node_group": name,
                        "from_size": current_size,
                        "to_size": desired_size,
                        "successful": True,
                        "metrics": scaling_decision["metrics"],
                        "reason": reason,
                        "cross_cloud": True
                    }
                    self.scaling_events.append(scaling_event)
                    
                    # Update timestamps
                    if desired_size > current_size:
                        self.last_scale_up_time[provider_name] = now
                    else:
                        self.last_scale_down_time[provider_name] = now
                        
                    self.logger.info(f"Successfully scaled {provider_name}/{name} to {desired_size}")
                else:
                    # Record failed scaling event
                    scaling_event = {
                        "timestamp": now.isoformat(),
                        "provider": provider_name,
                        "node_group": name,
                        "from_size": current_size,
                        "to_size": desired_size,
                        "successful": False,
                        "metrics": scaling_decision["metrics"],
                        "reason": reason,
                        "cross_cloud": True
                    }
                    self.scaling_events.append(scaling_event)
                    self.logger.error(f"Failed to scale {provider_name}/{name}")

    def _scale_node_group(self, provider: CloudProvider, node_group_id: str, desired_size: int) -> bool:
        """Scale a node group with error handling.
        
        Args:
            provider: Cloud provider.
            node_group_id: Node group ID.
            desired_size: Desired size of the node group.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            return provider.scale_node_group(node_group_id, desired_size)
        except Exception as e:
            # Record failure in circuit breaker
            provider_name = provider.__class__.__name__.replace("Provider", "").lower()
            if provider_name in self.circuit_breakers:
                self.circuit_breakers[provider_name].record_failure()
                
            self.logger.error(f"Error scaling node group {node_group_id}: {e}")
            return False

    def _is_gpu_node_group(self, node_group: Dict[str, Any]) -> bool:
        """Check if a node group is a GPU node group.

        Args:
            node_group: Node group configuration.

        Returns:
            True if the node group is a GPU node group, False otherwise.
        """
        # Explicitly marked as GPU
        if "is_gpu" in node_group:
            return node_group["is_gpu"]
            
        # Check for GPU instance types (simplified for demonstration)
        instance_type = node_group.get("instance_type", "")
        if isinstance(instance_type, str) and any(gpu_type in instance_type.lower() for gpu_type in ["gpu", "g4", "g5", "p2", "p3", "p4"]):
            return True

        # Check for machine type in GCP
        machine_type = node_group.get("machineType", "")
        if isinstance(machine_type, str) and any(gpu_type in machine_type.lower() for gpu_type in ["gpu", "a100", "t4", "v100", "p100"]):
            return True

        # Check for GPU in labels
        labels = node_group.get("labels", {})
        if labels.get("node-type") == "gpu" or labels.get("accelerator") == "gpu":
            return True

        # Check for GPU accelerators (GCP)
        accelerators = node_group.get("accelerators", [])
        if accelerators:
            return True
            
        # Check for config with accelerators
        config = node_group.get("config", {})
        if config.get("accelerators"):
            return True

        return False

    def _get_node_group_size(self, node_group: Dict[str, Any]) -> int:
        """Get the current size of a node group.

        Args:
            node_group: Node group configuration.

        Returns:
            Current size of the node group.
        """
        # AWS-style
        if "desiredCapacity" in node_group:
            return node_group["desiredCapacity"]
        # GCP-style
        elif "autoscaling" in node_group and "minNodeCount" in node_group["autoscaling"]:
            # Use current size if available, otherwise initial count
            return node_group.get("currentNodeCount", node_group.get("initialNodeCount", 0))
        # Default to initialNodeCount if available
        elif "initialNodeCount" in node_group:
            return node_group["initialNodeCount"]
        # No size information
        else:
            return 0

    def _calculate_desired_size(self, provider_name: str, node_group_name: str, current_size: int) -> int:
        """Calculate the desired size of a node group based on metrics.

        Args:
            provider_name: Cloud provider name.
            node_group_name: Node group name.
            current_size: Current size of the node group.

        Returns:
            Desired size of the node group.
        """
        metrics = self.current_metrics.get(provider_name, {})
        gpu_metrics = metrics.get("gpu_metrics", {})
        queue_metrics = metrics.get("queue_metrics", {})
        response_metrics = metrics.get("response_metrics", {})
        
        # Check if predictive scaling is enabled
        if self.predictive_scaling.get("enabled", False) and len(self.metrics_history.get(provider_name, [])) >= self.predictive_scaling.get("min_data_points", 60):
            # Try predictive scaling first
            predicted_size = self._calculate_predictive_size(provider_name, current_size)
            if predicted_size is not None:
                return predicted_size
        
        # Fall back to reactive scaling based on current metrics
        gpu_based_size = self._calculate_gpu_based_size(gpu_metrics, current_size)
        queue_based_size = self._calculate_queue_based_size(queue_metrics, current_size)
        latency_based_size = self._calculate_latency_based_size(response_metrics, current_size)

        # Take the maximum of all calculated sizes
        desired_size = max(gpu_based_size, queue_based_size, latency_based_size)

        self.logger.debug(
            f"Scaling calculation for {provider_name}/{node_group_name}: "
            f"current={current_size}, gpu={gpu_based_size}, queue={queue_based_size}, "
            f"latency={latency_based_size}, desired={desired_size}"
        )

        return desired_size

    def _calculate_predictive_size(self, provider_name: str, current_size: int) -> Optional[int]:
        """Calculate desired size based on predictive scaling.
        
        Args:
            provider_name: Cloud provider name.
            current_size: Current size of the node group.
            
        Returns:
            Predicted desired size, or None if prediction is not confident.
        """
        try:
            # Extract metrics history for this provider
            history = self.metrics_history.get(provider_name, [])
            if not history:
                return None
                
            # Check if we have enough data points
            min_data_points = self.predictive_scaling.get("min_data_points", 60)
            if len(history) < min_data_points:
                return None
                
            # Extract utilization and queue length time series
            timestamps = []
            utilizations = []
            queue_lengths = []
            
            for entry in history[-min_data_points:]:
                timestamp = datetime.fromisoformat(entry["timestamp"])
                timestamps.append(timestamp)
                
                # Get GPU utilization
                utilization = self._get_average_gpu_utilization(entry)
                utilizations.append(utilization)
                
                # Get queue length
                queue_length = entry.get("queue_metrics", {}).get("queue_length", 0)
                queue_lengths.append(queue_length)
                
            # Check for cyclic patterns in utilization
            if self.load_pattern_detection.get("enabled", True):
                pattern_detected, cycle_length = self._detect_load_pattern(utilizations)
                if pattern_detected:
                    # Use pattern to predict future load
                    forecast_window = self.predictive_scaling.get("forecast_window", 30)  # minutes
                    predicted_utilization = self._predict_utilization(utilizations, cycle_length, forecast_window)
                    
                    if predicted_utilization is not None:
                        # Calculate size based on predicted utilization
                        if predicted_utilization > self.target_gpu_utilization * 1.1:
                            # Scale up based on predicted utilization
                            utilization_ratio = predicted_utilization / self.target_gpu_utilization
                            return int(current_size * min(utilization_ratio, self.scale_up_factor))
                        elif predicted_utilization < self.target_gpu_utilization * 0.7:
                            # Scale down based on predicted utilization
                            utilization_ratio = predicted_utilization / self.target_gpu_utilization
                            return max(self.min_replicas, int(current_size * max(utilization_ratio, self.scale_down_factor)))
            
            # If no strong pattern detected or prediction not confident, return None
            # to fall back to reactive scaling
            return None
        except Exception as e:
            self.logger.error(f"Error in predictive scaling calculation: {e}")
            return None

    def _detect_load_pattern(self, values: List[float]) -> Tuple[bool, int]:
        """Detect cyclic patterns in load metrics.
        
        Args:
            values: List of metric values.
            
        Returns:
            Tuple of (pattern_detected, cycle_length).
        """
        try:
            # Simple autocorrelation to detect patterns
            if len(values) < 24:
                return False, 0
                
            # Calculate autocorrelation for different lags
            max_lag = min(len(values) // 2, 24 * 60 // self.collection_interval)  # max 24 hours
            correlations = []
            
            for lag in range(1, max_lag + 1):
                # Calculate correlation between series and lagged series
                series1 = values[:-lag]
                series2 = values[lag:]
                
                # Use numpy if available for efficient calculation
                try:
                    import numpy as np
                    correlation = np.corrcoef(series1, series2)[0, 1]
                except ImportError:
                    # Fall back to manual calculation
                    mean1 = sum(series1) / len(series1)
                    mean2 = sum(series2) / len(series2)
                    numerator = sum((a - mean1) * (b - mean2) for a, b in zip(series1, series2))
                    denominator = (
                        (sum((a - mean1) ** 2 for a in series1) * sum((b - mean2) ** 2 for b in series2)) ** 0.5
                    )
                    correlation = numerator / denominator if denominator != 0 else 0
                    
                correlations.append(correlation)
                
            # Find peaks in correlation
            peaks = []
            for i in range(1, len(correlations) - 1):
                if correlations[i] > correlations[i-1] and correlations[i] > correlations[i+1]:
                    peaks.append((i + 1, correlations[i]))
                    
            # Find strongest peak above threshold
            pattern_threshold = self.load_pattern_detection.get("pattern_threshold", 0.7)
            strong_peaks = [(lag, corr) for lag, corr in peaks if corr > pattern_threshold]
            
            if strong_peaks:
                # Return the lag with highest correlation
                strongest_peak = max(strong_peaks, key=lambda x: x[1])
                return True, strongest_peak[0]
                
            return False, 0
        except Exception as e:
            self.logger.error(f"Error detecting load pattern: {e}")
            return False, 0

    def _predict_utilization(self, utilizations: List[float], cycle_length: int, 
                            forecast_window: int) -> Optional[float]:
        """Predict future utilization based on cyclic pattern.
        
        Args:
            utilizations: List of utilization values.
            cycle_length: Detected cycle length in data points.
            forecast_window: Minutes to forecast ahead.
            
        Returns:
            Predicted utilization or None if prediction is not confident.
        """
        try:
            # Convert forecast window from minutes to data points
            forecast_points = forecast_window // (self.collection_interval // 60)
            
            # If cycle is shorter than forecast window, use the pattern
            if cycle_length <= forecast_points:
                # Calculate position in cycle for forecast point
                current_position = len(utilizations) % cycle_length
                forecast_position = (current_position + forecast_points) % cycle_length
                
                # Use the historical data from the same position in previous cycles
                cycle_values = []
                for i in range(forecast_position, len(utilizations), cycle_length):
                    if i < len(utilizations):
                        cycle_values.append(utilizations[i])
                        
                # Calculate average value at this position in the cycle
                if cycle_values:
                    predicted_value = sum(cycle_values) / len(cycle_values)
                    confidence = 0.8 if len(cycle_values) >= 3 else 0.5
                    
                    # Return prediction if confidence is high enough
                    if confidence >= self.predictive_scaling.get("confidence_threshold", 0.8):
                        return predicted_value
                    
            return None
        except Exception as e:
            self.logger.error(f"Error predicting utilization: {e}")
            return None
            
    def _calculate_gpu_based_size(self, gpu_metrics: Dict[str, Any], current_size: int) -> int:
        """Calculate the desired size based on GPU utilization metrics.
        
        Args:
            gpu_metrics: GPU metrics dictionary.
            current_size: Current size of the node group.
            
        Returns:
            Desired size based on GPU utilization.
        """
        avg_gpu_utilization = self._get_average_metric(gpu_metrics, "gpu_utilization")
        duty_cycle_threshold = self.gpu_metrics_thresholds.get("duty_cycle", 80)
        
        if avg_gpu_utilization > duty_cycle_threshold:
            # Scale up based on GPU utilization
            utilization_ratio = avg_gpu_utilization / self.target_gpu_utilization
            return int(current_size * min(utilization_ratio, self.scale_up_factor))
        elif avg_gpu_utilization < 0.7 * self.target_gpu_utilization:
            # Scale down based on GPU utilization
            utilization_ratio = avg_gpu_utilization / self.target_gpu_utilization
            return max(self.min_replicas, int(current_size * max(utilization_ratio, self.scale_down_factor)))
        else:
            # Utilization is within target range
            return current_size
            
    def _calculate_queue_based_size(self, queue_metrics: Dict[str, Any], current_size: int) -> int:
        """Calculate the desired size based on queue metrics.
        
        Args:
            queue_metrics: Queue metrics dictionary.
            current_size: Current size of the node group.
            
        Returns:
            Desired size based on queue length.
        """
        queue_length = queue_metrics.get("queue_length", 0)
        queue_length_threshold = self.queue_metrics_thresholds.get("queue_length", 100)
        
        if queue_length > queue_length_threshold:
            # Scale up based on queue length
            queue_ratio = queue_length / self.target_queue_length
            return int(current_size * min(queue_ratio, self.scale_up_factor))
        elif queue_length < 0.3 * self.target_queue_length:
            # Scale down based on queue length
            return max(self.min_replicas, int(current_size * self.scale_down_factor))
        else:
            # Queue length is within target range
            return current_size
            
    def _calculate_latency_based_size(self, response_metrics: Dict[str, Any], current_size: int) -> int:
        """Calculate the desired size based on response latency metrics.
        
        Args:
            response_metrics: Response metrics dictionary.
            current_size: Current size of the node group.
            
        Returns:
            Desired size based on response latency.
        """
        latency_p95 = response_metrics.get("latency_p95", 0)
        latency_threshold = self.response_metrics_thresholds.get("latency_p95", 1000)
        
        if latency_p95 > latency_threshold:
            # Scale up based on latency
            latency_ratio = latency_p95 / latency_threshold
            return int(current_size * min(latency_ratio, self.scale_up_factor))
        else:
            # Latency is within target range
            return current_size

    def _get_average_gpu_utilization(self, metrics: Dict[str, Any]) -> float:
        """Get the average GPU utilization across all nodes.
        
        Args:
            metrics: Dictionary of metrics.
            
        Returns:
            Average GPU utilization.
        """
        gpu_metrics = metrics.get("gpu_metrics", {})
        return self._get_average_metric(gpu_metrics, "gpu_utilization")

    def _get_average_metric(self, metrics: Dict[str, Dict[str, float]], metric_name: str) -> float:
        """Get the average value of a metric across all nodes.

        Args:
            metrics: Dictionary of metrics.
            metric_name: Metric name.

        Returns:
            Average value of the metric.
        """
        if not metrics or metric_name not in metrics:
            return 0.0

        values = metrics[metric_name].values()
        if not values:
            return 0.0

        return sum(values) / len(values)

    def _determine_scaling_reason(self, provider_name: str, current_size: int, 
                                desired_size: int, metrics: Dict[str, Any]) -> str:
        """Determine the reason for scaling.
        
        Args:
            provider_name: Cloud provider name.
            current_size: Current size of the node group.
            desired_size: Desired size of the node group.
            metrics: Metrics dictionary.
            
        Returns:
            Scaling reason.
        """
        if desired_size > current_size:
            # Scale up reasons
            avg_gpu_utilization = self._get_average_gpu_utilization(metrics)
            duty_cycle_threshold = self.gpu_metrics_thresholds.get("duty_cycle", 80)
            
            queue_length = metrics.get("queue_metrics", {}).get("queue_length", 0)
            queue_length_threshold = self.queue_metrics_thresholds.get("queue_length", 100)
            
            latency_p95 = metrics.get("response_metrics", {}).get("latency_p95", 0)
            latency_threshold = self.response_metrics_thresholds.get("latency_p95", 1000)
            
            reasons = []
            
            if avg_gpu_utilization > duty_cycle_threshold:
                reasons.append(f"GPU utilization {avg_gpu_utilization:.1f}% > threshold {duty_cycle_threshold}%")
                
            if queue_length > queue_length_threshold:
                reasons.append(f"Queue length {queue_length} > threshold {queue_length_threshold}")
                
            if latency_p95 > latency_threshold:
                reasons.append(f"P95 latency {latency_p95}ms > threshold {latency_threshold}ms")
                
            if not reasons:
                reasons.append("Predictive scaling based on load pattern")
                
            return f"Scale up: {', '.join(reasons)}"
        
        elif desired_size < current_size:
            # Scale down reasons
            avg_gpu_utilization = self._get_average_gpu_utilization(metrics)
            queue_length = metrics.get("queue_metrics", {}).get("queue_length", 0)
            
            reasons = []
            
            if avg_gpu_utilization < 0.7 * self.target_gpu_utilization:
                reasons.append(f"GPU utilization {avg_gpu_utilization:.1f}% < target {0.7 * self.target_gpu_utilization:.1f}%")
                
            if queue_length < 0.3 * self.target_queue_length:
                reasons.append(f"Queue length {queue_length} < target {0.3 * self.target_queue_length:.1f}")
                
            if not reasons:
                reasons.append("Cost optimization")
                
            return f"Scale down: {', '.join(reasons)}"
        
        else:
            return "No change needed"
            
    def _determine_cross_cloud_scaling_reason(self, provider_name: str, current_size: int, 
                                           desired_size: int, metrics: Dict[str, Any]) -> str:
        """Determine the reason for cross-cloud scaling.
        
        Args:
            provider_name: Cloud provider name.
            current_size: Current size of the node group.
            desired_size: Desired size of the node group.
            metrics: Metrics dictionary.
            
        Returns:
            Scaling reason.
        """
        if desired_size > current_size:
            return f"Cross-cloud scale up: Balancing load across cloud providers based on {self.cloud_weight_strategy} weights"
        elif desired_size < current_size:
            return f"Cross-cloud scale down: Optimizing distribution across cloud providers based on {self.cloud_weight_strategy} weights"
        else:
            return "No change needed in cross-cloud allocation"

    def _calculate_cloud_weights(self) -> Dict[str, float]:
        """Calculate weights for each cloud provider based on strategy.
        
        Returns:
            Dictionary of cloud provider weights.
        """
        weights = {}
        
        if self.cloud_weight_strategy == "cost":
            # Lower cost gets higher weight
            for provider_name, provider in self.cloud_providers.items():
                # Initially assign equal weights
                weights[provider_name] = 1.0
                
                # In a real implementation, this would calculate weights based on
                # historical cost data from each provider
                if provider_name == "aws":
                    weights[provider_name] = 0.8  # example: AWS is more expensive
                elif provider_name == "gcp":
                    weights[provider_name] = 1.2  # example: GCP is cheaper
        
        elif self.cloud_weight_strategy == "latency":
            # Lower latency gets higher weight
            for provider_name, provider in self.cloud_providers.items():
                # Initially assign equal weights
                weights[provider_name] = 1.0
                
                # In a real implementation, this would calculate weights based on
                # historical latency data from each provider
        
        elif self.cloud_weight_strategy == "balanced":
            # Balance between cost, latency, and availability
            for provider_name, provider in self.cloud_providers.items():
                # Initially assign equal weights
                weights[provider_name] = 1.0
                
                # In a real implementation, this would calculate weights based on
                # multiple factors
        
        else:
            # Default: equal weights
            for provider_name in self.cloud_providers:
                weights[provider_name] = 1.0
                
        return weights

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics.

        Returns:
            Dictionary of current metrics.
        """
        return self.current_metrics

    def get_scaling_history(self) -> Dict[str, Any]:
        """Get scaling history.

        Returns:
            Dictionary of scaling history.
        """
        history = {}

        for provider_name, metrics_list in self.metrics_history.items():
            if provider_name not in history:
                history[provider_name] = []

            for metrics in metrics_list:
                history[provider_name].append(
                    {
                        "timestamp": metrics["timestamp"],
                        "gpu_utilization": self._get_average_gpu_utilization(metrics),
                        "queue_length": metrics.get("queue_metrics", {}).get("queue_length", 0),
                        "latency_p95": metrics.get("response_metrics", {}).get("latency_p95", 0),
                        "node_group_sizes": self.node_group_sizes.get(provider_name, {})
                    }
                )

        return history
        
    def get_scaling_decisions(self) -> List[Dict[str, Any]]:
        """Get scaling decisions history.
        
        Returns:
            List of scaling decisions.
        """
        return self.scaling_decisions
        
    def get_scaling_events(self) -> List[Dict[str, Any]]:
        """Get scaling events history.
        
        Returns:
            List of scaling events.
        """
        return self.scaling_events
        
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get status of circuit breakers.
        
        Returns:
            Dictionary of circuit breaker status.
        """
        status = {}
        
        for provider_name, circuit_breaker in self.circuit_breakers.items():
            status[provider_name] = {
                "state": circuit_breaker.state.value,
                "failure_count": circuit_breaker.failure_count,
                "last_failure": circuit_breaker.last_failure_time.isoformat() 
                    if circuit_breaker.last_failure_time else None
            }
            
        return status