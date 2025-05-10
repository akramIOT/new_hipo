"""
Distributed tracing for multi-cloud infrastructure.
"""
import logging
import time
import os
import json
import functools
import threading
from typing import Dict, List, Any, Optional, Callable
import uuid
import contextvars
from datetime import datetime

logger = logging.getLogger(__name__)

# Context variables to maintain trace context across async functions
trace_id_var = contextvars.ContextVar("trace_id", default=None)
span_id_var = contextvars.ContextVar("span_id", default=None)


class Span:
    """A single operation within a trace."""

    def __init__(self, name: str, trace_id: str = None, parent_id: str = None):
        """Initialize a span.

        Args:
            name: Name of the span.
            trace_id: ID of the trace this span belongs to. If None, a new ID is generated.
            parent_id: ID of the parent span. If None, this is a root span.
        """
        self.name = name
        self.span_id = str(uuid.uuid4())
        self.trace_id = trace_id or str(uuid.uuid4())
        self.parent_id = parent_id
        self.start_time = time.time()
        self.end_time = None
        self.tags = {}
        self.logs = []
        self.status = "OK"
        self.service_name = os.environ.get("SERVICE_NAME", "unknown")
        self.hostname = os.environ.get("HOSTNAME", "localhost")

    def set_tag(self, key: str, value: Any) -> "Span":
        """Set a tag on the span.

        Args:
            key: Tag key.
            value: Tag value.

        Returns:
            Self for chaining.
        """
        self.tags[key] = value
        return self

    def log_event(self, event: str, payload: Dict[str, Any] = None) -> "Span":
        """Log an event during the span.

        Args:
            event: Event name.
            payload: Additional information about the event.

        Returns:
            Self for chaining.
        """
        self.logs.append({"timestamp": time.time(), "event": event, "payload": payload or {}})
        return self

    def set_error(self, exception: Exception = None) -> "Span":
        """Mark the span as failed.

        Args:
            exception: Exception that caused the failure.

        Returns:
            Self for chaining.
        """
        self.status = "ERROR"

        if exception:
            self.set_tag("error", True)
            self.set_tag("error.message", str(exception))
            self.set_tag("error.type", type(exception).__name__)

        return self

    def finish(self) -> None:
        """Finish the span."""
        self.end_time = time.time()

        # Publish the completed span
        TraceExporter.get_instance().export_span(self)

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary.

        Returns:
            Dictionary representation of the span.
        """
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": (self.end_time - self.start_time) if self.end_time else None,
            "tags": self.tags,
            "logs": self.logs,
            "status": self.status,
            "service": self.service_name,
            "hostname": self.hostname,
        }

    def __enter__(self):
        """Start the span as a context manager."""
        # Save previous context
        self._prev_trace_id = trace_id_var.get()
        self._prev_span_id = span_id_var.get()

        # Set current context
        trace_id_var.set(self.trace_id)
        span_id_var.set(self.span_id)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the span when exiting the context manager."""
        if exc_val:
            self.set_error(exc_val)

        self.finish()

        # Restore previous context
        trace_id_var.set(self._prev_trace_id)
        span_id_var.set(self._prev_span_id)


class TraceExporter:
    """Singleton class to export traces to various backends."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern implementation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TraceExporter, cls).__new__(cls)
                cls._instance._init()

        return cls._instance

    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance.

        Returns:
            Singleton instance.
        """
        return cls()

    def _init(self):
        """Initialize the exporter."""
        self.logger = logging.getLogger(f"{__name__}.TraceExporter")
        self.exporters = []
        self.is_enabled = True

        # Configure exporters based on environment variables
        if os.environ.get("JAEGER_ENABLED", "false").lower() == "true":
            self._init_jaeger_exporter()

        if os.environ.get("ZIPKIN_ENABLED", "false").lower() == "true":
            self._init_zipkin_exporter()

        if os.environ.get("CONSOLE_TRACING_ENABLED", "false").lower() == "true":
            self.exporters.append(self._export_to_console)

        if os.environ.get("FILE_TRACING_ENABLED", "false").lower() == "true":
            file_path = os.environ.get("TRACE_FILE_PATH", "/tmp/traces.json")
            self.exporters.append(lambda span: self._export_to_file(span, file_path))

    def _init_jaeger_exporter(self):
        """Initialize Jaeger exporter.

        In a real implementation, this would initialize a Jaeger client.
        For simplicity, we just log a message.
        """
        self.logger.info("Initializing Jaeger exporter")
        # In a real implementation, this would initialize a Jaeger client
        # self.exporters.append(self._export_to_jaeger)

    def _init_zipkin_exporter(self):
        """Initialize Zipkin exporter.

        In a real implementation, this would initialize a Zipkin client.
        For simplicity, we just log a message.
        """
        self.logger.info("Initializing Zipkin exporter")
        # In a real implementation, this would initialize a Zipkin client
        # self.exporters.append(self._export_to_zipkin)

    def export_span(self, span: Span) -> None:
        """Export a span to all configured exporters.

        Args:
            span: Span to export.
        """
        if not self.is_enabled:
            return

        for exporter in self.exporters:
            try:
                exporter(span)
            except Exception as e:
                self.logger.error(f"Error exporting span: {e}")

    def _export_to_console(self, span: Span) -> None:
        """Export a span to the console.

        Args:
            span: Span to export.
        """
        span_dict = span.to_dict()
        self.logger.info(f"Span: {json.dumps(span_dict)}")

    def _export_to_file(self, span: Span, file_path: str) -> None:
        """Export a span to a file.

        Args:
            span: Span to export.
            file_path: Path to file.
        """
        span_dict = span.to_dict()

        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "a") as f:
                f.write(json.dumps(span_dict) + "\n")
        except Exception as e:
            self.logger.error(f"Error writing span to file: {e}")

    def _export_to_jaeger(self, span: Span) -> None:
        """Export a span to Jaeger.

        In a real implementation, this would send the span to Jaeger.
        For simplicity, we just log a message.

        Args:
            span: Span to export.
        """
        self.logger.debug(f"Exporting span to Jaeger: {span.name}")
        # In a real implementation, this would send the span to Jaeger

    def _export_to_zipkin(self, span: Span) -> None:
        """Export a span to Zipkin.

        In a real implementation, this would send the span to Zipkin.
        For simplicity, we just log a message.

        Args:
            span: Span to export.
        """
        self.logger.debug(f"Exporting span to Zipkin: {span.name}")
        # In a real implementation, this would send the span to Zipkin


def create_span(name: str) -> Span:
    """Create a new span.

    Args:
        name: Name of the span.

    Returns:
        New span.
    """
    # Get current trace context
    parent_trace_id = trace_id_var.get()
    parent_span_id = span_id_var.get()

    # Create new span
    return Span(name, trace_id=parent_trace_id, parent_id=parent_span_id)


def trace(name: str = None):
    """Decorator to trace a function.

    Args:
        name: Name of the span. If None, the function name is used.

    Returns:
        Decorator function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            span_name = name or func.__name__
            with create_span(span_name) as span:
                # Add function arguments as tags
                # Be careful not to add sensitive information
                span.set_tag("function", func.__name__)
                span.set_tag("module", func.__module__)

                # Execute function
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_error(e)
                    raise

        return wrapper

    return decorator


def get_current_trace_context() -> Dict[str, str]:
    """Get the current trace context.

    Returns:
        Dictionary containing trace_id and span_id if available.
    """
    trace_id = trace_id_var.get()
    span_id = span_id_var.get()

    if trace_id and span_id:
        return {"trace_id": trace_id, "span_id": span_id}

    return {}


def extract_trace_context_from_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Extract trace context from HTTP headers.

    Args:
        headers: HTTP headers.

    Returns:
        Dictionary containing trace_id and span_id if available.
    """
    context = {}

    if "X-Trace-ID" in headers:
        context["trace_id"] = headers["X-Trace-ID"]

    if "X-Span-ID" in headers:
        context["span_id"] = headers["X-Span-ID"]

    return context


def inject_trace_context_to_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Inject trace context into HTTP headers.

    Args:
        headers: HTTP headers to inject into.

    Returns:
        Updated HTTP headers.
    """
    context = get_current_trace_context()

    if context:
        headers["X-Trace-ID"] = context.get("trace_id")
        headers["X-Span-ID"] = context.get("span_id")

    return headers
