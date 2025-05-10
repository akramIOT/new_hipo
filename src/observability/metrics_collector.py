"""
Metrics collector for multi-cloud Kubernetes infrastructure.
"""
import logging
import time
import threading
from typing import Dict, Any, Optional
from datetime import datetime
import requests
import os
from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server
import socket

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Metrics collector for centralized observability across multi-cloud environments."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize MetricsCollector.

        Args:
            config: Metrics configuration.
        """
        self.config = config
        self.collection_interval = config.get("collection_interval", 15)  # seconds
        self.metrics_port = config.get("metrics_port", 8000)
        self.prometheus_enabled = config.get("prometheus", {}).get("enabled", True)
        self.opentelemetry_enabled = config.get("opentelemetry", {}).get("enabled", False)
        self.export_metrics = config.get("export_metrics", True)
        self.export_endpoint = config.get("export_endpoint", "")

        self.logger = logging.getLogger(f"{__name__}.MetricsCollector")

        # State
        self.running = False
        self.collection_thread = None
        self.hostname = socket.gethostname()

        # Initialize Prometheus metrics
        if self.prometheus_enabled:
            self._init_prometheus_metrics()

        # Initialize OpenTelemetry if enabled
        if self.opentelemetry_enabled:
            self._init_opentelemetry()

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        # System metrics
        self.system_metrics = {
            "cpu_usage": Gauge("system_cpu_usage", "CPU usage in percent", ["host"]),
            "memory_usage": Gauge("system_memory_usage", "Memory usage in percent", ["host"]),
            "disk_usage": Gauge("system_disk_usage", "Disk usage in percent", ["host", "mount"]),
            "network_received": Counter(
                "system_network_received_bytes", "Network bytes received", ["host", "interface"]
            ),
            "network_sent": Counter("system_network_sent_bytes", "Network bytes sent", ["host", "interface"]),
        }

        # GPU metrics
        self.gpu_metrics = {
            "gpu_utilization": Gauge("gpu_utilization", "GPU utilization in percent", ["host", "gpu_id", "provider"]),
            "gpu_memory_used": Gauge(
                "gpu_memory_used_bytes", "GPU memory used in bytes", ["host", "gpu_id", "provider"]
            ),
            "gpu_power_draw": Gauge("gpu_power_draw_watts", "GPU power draw in watts", ["host", "gpu_id", "provider"]),
            "gpu_temperature": Gauge(
                "gpu_temperature_celsius", "GPU temperature in Celsius", ["host", "gpu_id", "provider"]
            ),
        }

        # API metrics
        self.api_metrics = {
            "request_count": Counter(
                "api_request_count", "Number of API requests", ["host", "endpoint", "method", "status_code"]
            ),
            "request_latency": Histogram(
                "api_request_latency_seconds", "API request latency in seconds", ["host", "endpoint", "method"]
            ),
            "request_size": Histogram("api_request_size_bytes", "API request size in bytes", ["host", "endpoint"]),
            "response_size": Histogram("api_response_size_bytes", "API response size in bytes", ["host", "endpoint"]),
        }

        # Model metrics
        self.model_metrics = {
            "inference_count": Counter(
                "model_inference_count", "Number of model inferences", ["host", "model", "version"]
            ),
            "inference_latency": Histogram(
                "model_inference_latency_seconds", "Model inference latency in seconds", ["host", "model", "version"]
            ),
            "inference_memory": Gauge(
                "model_inference_memory_bytes", "Model inference memory usage in bytes", ["host", "model", "version"]
            ),
            "prediction_distribution": Counter(
                "model_prediction_distribution", "Distribution of model predictions", ["host", "model", "class"]
            ),
        }

        # Queue metrics
        self.queue_metrics = {
            "queue_length": Gauge("queue_length", "Number of items in queue", ["host", "queue_name"]),
            "queue_latency": Gauge("queue_latency_seconds", "Queue latency in seconds", ["host", "queue_name"]),
            "queue_errors": Counter("queue_errors", "Number of queue errors", ["host", "queue_name", "error_type"]),
        }

        # Cost metrics
        self.cost_metrics = {
            "cloud_cost": Gauge("cloud_cost_usd", "Cloud cost in USD", ["provider", "service", "timeframe"]),
        }

        # Infrastructure metrics
        self.infra_metrics = {
            "node_count": Gauge(
                "kubernetes_node_count", "Number of Kubernetes nodes", ["provider", "zone", "node_type"]
            ),
            "pod_count": Gauge(
                "kubernetes_pod_count", "Number of Kubernetes pods", ["provider", "namespace", "status"]
            ),
            "deployment_status": Gauge(
                "kubernetes_deployment_status", "Kubernetes deployment status", ["provider", "namespace", "deployment"]
            ),
            "resource_quota": Gauge(
                "kubernetes_resource_quota",
                "Kubernetes resource quota usage percent",
                ["provider", "namespace", "resource_type"],
            ),
        }

        # Business metrics
        self.business_metrics = {
            "active_users": Gauge("business_active_users", "Number of active users", ["timeframe"]),
            "inference_per_user": Gauge("business_inference_per_user", "Number of inferences per user", ["user_tier"]),
            "api_errors_rate": Gauge("business_api_error_rate", "API error rate in percent", []),
            "model_usage": Counter("business_model_usage", "Model usage", ["model_name", "user_tier"]),
        }

        # Component status
        self.status_metrics = {
            "component_status": Gauge("component_status", "Component status (1=up, 0=down)", ["component", "version"]),
            "component_uptime": Gauge(
                "component_uptime_seconds", "Component uptime in seconds", ["component", "version"]
            ),
        }

        # System info
        self.system_info = Info("system_information", "System information")
        self.system_info.info(
            {
                "hostname": self.hostname,
                "version": os.environ.get("APP_VERSION", "dev"),
                "environment": os.environ.get("ENVIRONMENT", "development"),
            }
        )

        # Start metrics server
        start_http_server(self.metrics_port)
        self.logger.info(f"Started Prometheus metrics server on port {self.metrics_port}")

    def _init_opentelemetry(self):
        """Initialize OpenTelemetry."""
        try:
            # This is a placeholder - in a real implementation,
            # you would initialize OpenTelemetry here
            self.logger.info("Initializing OpenTelemetry")
            # from opentelemetry import trace
            # from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            # from opentelemetry.sdk.resources import SERVICE_NAME, Resource
            # from opentelemetry.sdk.trace import TracerProvider
            # from opentelemetry.sdk.trace.export import BatchSpanProcessor

            # Configure the tracer provider
            # resource = Resource(attributes={
            #     SERVICE_NAME: "ml-infrastructure"
            # })
            # trace.set_tracer_provider(TracerProvider(resource=resource))
            # tracer = trace.get_tracer(__name__)

            # Configure the exporter
            # endpoint = self.config.get('opentelemetry', {}).get('endpoint', 'localhost:4317')
            # otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
            # span_processor = BatchSpanProcessor(otlp_exporter)
            # trace.get_tracer_provider().add_span_processor(span_processor)
        except Exception as e:
            self.logger.error(f"Error initializing OpenTelemetry: {e}")

    def start(self) -> None:
        """Start the MetricsCollector."""
        if self.running:
            self.logger.warning("MetricsCollector is already running")
            return

        self.running = True
        self.collection_thread = threading.Thread(target=self._metrics_collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()

        self.logger.info("MetricsCollector started")

    def stop(self) -> None:
        """Stop the MetricsCollector."""
        self.running = False

        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
            self.collection_thread = None

        self.logger.info("MetricsCollector stopped")

    def _metrics_collection_loop(self) -> None:
        """Metrics collection loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                self._collect_gpu_metrics()
                self._collect_kubernetes_metrics()
                self._collect_api_metrics()
                self._collect_model_metrics()
                self._collect_cost_metrics()

                if self.export_metrics and self.export_endpoint:
                    self._export_metrics()
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")

            time.sleep(self.collection_interval)

    def _collect_system_metrics(self) -> None:
        """Collect system metrics."""
        try:
            # This is simplified for demonstration
            # In a real implementation, this would use psutil or similar to get metrics

            # Update CPU usage
            cpu_usage = 50.0  # Example value
            self.system_metrics["cpu_usage"].labels(host=self.hostname).set(cpu_usage)

            # Update memory usage
            memory_usage = 60.0  # Example value
            self.system_metrics["memory_usage"].labels(host=self.hostname).set(memory_usage)

            # Update disk usage
            disk_usage = 70.0  # Example value
            self.system_metrics["disk_usage"].labels(host=self.hostname, mount="/").set(disk_usage)

            # Update network metrics
            network_received = 1024  # Example value
            network_sent = 512  # Example value
            self.system_metrics["network_received"].labels(host=self.hostname, interface="eth0").inc(network_received)
            self.system_metrics["network_sent"].labels(host=self.hostname, interface="eth0").inc(network_sent)

            self.logger.debug("Collected system metrics")
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")

    def _collect_gpu_metrics(self) -> None:
        """Collect GPU metrics."""
        try:
            # This is simplified for demonstration
            # In a real implementation, this would use NVML or similar to get GPU metrics

            # Update GPU metrics for each GPU
            for gpu_id in range(2):  # Example with 2 GPUs
                provider = "aws"  # Example value

                # Update GPU utilization
                gpu_utilization = 80.0  # Example value
                self.gpu_metrics["gpu_utilization"].labels(
                    host=self.hostname, gpu_id=str(gpu_id), provider=provider
                ).set(gpu_utilization)

                # Update GPU memory used
                gpu_memory_used = 8 * 1024 * 1024 * 1024  # Example value (8 GB in bytes)
                self.gpu_metrics["gpu_memory_used"].labels(
                    host=self.hostname, gpu_id=str(gpu_id), provider=provider
                ).set(gpu_memory_used)

                # Update GPU power draw
                gpu_power_draw = 150.0  # Example value (in watts)
                self.gpu_metrics["gpu_power_draw"].labels(
                    host=self.hostname, gpu_id=str(gpu_id), provider=provider
                ).set(gpu_power_draw)

                # Update GPU temperature
                gpu_temperature = 75.0  # Example value (in Celsius)
                self.gpu_metrics["gpu_temperature"].labels(
                    host=self.hostname, gpu_id=str(gpu_id), provider=provider
                ).set(gpu_temperature)

            self.logger.debug("Collected GPU metrics")
        except Exception as e:
            self.logger.error(f"Error collecting GPU metrics: {e}")

    def _collect_kubernetes_metrics(self) -> None:
        """Collect Kubernetes metrics."""
        try:
            # This is simplified for demonstration
            # In a real implementation, this would use the Kubernetes API to get metrics

            # Update node count
            self.infra_metrics["node_count"].labels(provider="aws", zone="us-west-2a", node_type="gpu").set(5)

            # Update pod count
            self.infra_metrics["pod_count"].labels(provider="aws", namespace="llm-serving", status="Running").set(10)

            # Update deployment status
            self.infra_metrics["deployment_status"].labels(
                provider="aws", namespace="llm-serving", deployment="llama2-7b"
            ).set(
                1
            )  # 1 = healthy

            # Update resource quota
            self.infra_metrics["resource_quota"].labels(
                provider="aws", namespace="llm-serving", resource_type="cpu"
            ).set(75.0)

            self.logger.debug("Collected Kubernetes metrics")
        except Exception as e:
            self.logger.error(f"Error collecting Kubernetes metrics: {e}")

    def _collect_api_metrics(self) -> None:
        """Collect API metrics."""
        try:
            # This is simplified for demonstration
            # In a real implementation, this would use middleware or similar to collect API metrics

            # These metrics are typically updated in real-time as API requests are processed,
            # not in the collection loop. This is just for demonstration.

            # Example of updating API metrics
            self.api_metrics["request_count"].labels(
                host=self.hostname, endpoint="/api/models/predict", method="POST", status_code="200"
            ).inc()

            self.logger.debug("Collected API metrics")
        except Exception as e:
            self.logger.error(f"Error collecting API metrics: {e}")

    def _collect_model_metrics(self) -> None:
        """Collect model metrics."""
        try:
            # This is simplified for demonstration
            # In a real implementation, this would collect metrics from model servers

            # These metrics are typically updated in real-time as model inferences are processed,
            # not in the collection loop. This is just for demonstration.

            # Example of updating model metrics
            self.model_metrics["inference_count"].labels(host=self.hostname, model="llama2-7b", version="1.0").inc()

            self.logger.debug("Collected model metrics")
        except Exception as e:
            self.logger.error(f"Error collecting model metrics: {e}")

    def _collect_cost_metrics(self) -> None:
        """Collect cost metrics."""
        try:
            # This is simplified for demonstration
            # In a real implementation, this would use cloud provider APIs to get cost metrics

            # Update cost metrics
            self.cost_metrics["cloud_cost"].labels(provider="aws", service="ec2", timeframe="daily").set(100.0)

            self.cost_metrics["cloud_cost"].labels(provider="aws", service="s3", timeframe="daily").set(20.0)

            self.logger.debug("Collected cost metrics")
        except Exception as e:
            self.logger.error(f"Error collecting cost metrics: {e}")

    def _export_metrics(self) -> None:
        """Export metrics to external system."""
        try:
            # This is simplified for demonstration
            # In a real implementation, this would format and send metrics to the export endpoint

            # Example of exporting metrics
            if self.export_endpoint:
                requests.post(
                    self.export_endpoint,
                    json={
                        "timestamp": datetime.now().isoformat(),
                        "hostname": self.hostname,
                        # Add metrics data here
                    },
                    timeout=5,
                )

            self.logger.debug("Exported metrics")
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")

    def record_request(
        self, endpoint: str, method: str, status_code: str, latency: float, request_size: int, response_size: int
    ) -> None:
        """Record API request metrics.

        Args:
            endpoint: API endpoint.
            method: HTTP method.
            status_code: HTTP status code.
            latency: Request latency in seconds.
            request_size: Request size in bytes.
            response_size: Response size in bytes.
        """
        if not self.prometheus_enabled:
            return

        try:
            self.api_metrics["request_count"].labels(
                host=self.hostname, endpoint=endpoint, method=method, status_code=status_code
            ).inc()

            self.api_metrics["request_latency"].labels(host=self.hostname, endpoint=endpoint, method=method).observe(
                latency
            )

            self.api_metrics["request_size"].labels(host=self.hostname, endpoint=endpoint).observe(request_size)

            self.api_metrics["response_size"].labels(host=self.hostname, endpoint=endpoint).observe(response_size)
        except Exception as e:
            self.logger.error(f"Error recording request metrics: {e}")

    def record_inference(
        self, model: str, version: str, latency: float, memory_used: int, prediction_class: Optional[str] = None
    ) -> None:
        """Record model inference metrics.

        Args:
            model: Model name.
            version: Model version.
            latency: Inference latency in seconds.
            memory_used: Memory used in bytes.
            prediction_class: Prediction class (for classification models).
        """
        if not self.prometheus_enabled:
            return

        try:
            self.model_metrics["inference_count"].labels(host=self.hostname, model=model, version=version).inc()

            self.model_metrics["inference_latency"].labels(host=self.hostname, model=model, version=version).observe(
                latency
            )

            self.model_metrics["inference_memory"].labels(host=self.hostname, model=model, version=version).set(
                memory_used
            )

            if prediction_class is not None:
                self.model_metrics["prediction_distribution"].labels(
                    host=self.hostname, model=model, prediction_class=prediction_class
                ).inc()
        except Exception as e:
            self.logger.error(f"Error recording inference metrics: {e}")

    def record_component_status(self, component: str, version: str, is_up: bool, uptime: float) -> None:
        """Record component status metrics.

        Args:
            component: Component name.
            version: Component version.
            is_up: True if component is up, False otherwise.
            uptime: Component uptime in seconds.
        """
        if not self.prometheus_enabled:
            return

        try:
            self.status_metrics["component_status"].labels(component=component, version=version).set(1 if is_up else 0)

            self.status_metrics["component_uptime"].labels(component=component, version=version).set(uptime)
        except Exception as e:
            self.logger.error(f"Error recording component status metrics: {e}")

    def record_business_metrics(
        self,
        active_users: Dict[str, int],
        inference_per_user: Dict[str, float],
        api_error_rate: float,
        model_usage: Dict[str, Dict[str, int]],
    ) -> None:
        """Record business metrics.

        Args:
            active_users: Dictionary mapping timeframe to number of active users.
            inference_per_user: Dictionary mapping user tier to number of inferences per user.
            api_error_rate: API error rate in percent.
            model_usage: Dictionary mapping model name to dictionary mapping user tier to usage.
        """
        if not self.prometheus_enabled:
            return

        try:
            # Update active users
            for timeframe, count in active_users.items():
                self.business_metrics["active_users"].labels(timeframe=timeframe).set(count)

            # Update inference per user
            for user_tier, count in inference_per_user.items():
                self.business_metrics["inference_per_user"].labels(user_tier=user_tier).set(count)

            # Update API error rate
            self.business_metrics["api_errors_rate"].set(api_error_rate)

            # Update model usage
            for model_name, tiers in model_usage.items():
                for user_tier, usage in tiers.items():
                    self.business_metrics["model_usage"].labels(model_name=model_name, user_tier=user_tier).inc(usage)
        except Exception as e:
            self.logger.error(f"Error recording business metrics: {e}")

    def _get_time_since_epoch(self) -> float:
        """Get time since epoch in seconds.

        Returns:
            Time since epoch in seconds.
        """
        return time.time()

    def record_queue_metrics(
        self, queue_name: str, queue_length: int, queue_latency: float, error_type: Optional[str] = None
    ) -> None:
        """Record queue metrics.

        Args:
            queue_name: Queue name.
            queue_length: Queue length.
            queue_latency: Queue latency in seconds.
            error_type: Error type if an error occurred, None otherwise.
        """
        if not self.prometheus_enabled:
            return

        try:
            self.queue_metrics["queue_length"].labels(host=self.hostname, queue_name=queue_name).set(queue_length)

            self.queue_metrics["queue_latency"].labels(host=self.hostname, queue_name=queue_name).set(queue_latency)

            if error_type is not None:
                self.queue_metrics["queue_errors"].labels(
                    host=self.hostname, queue_name=queue_name, error_type=error_type
                ).inc()
        except Exception as e:
            self.logger.error(f"Error recording queue metrics: {e}")
