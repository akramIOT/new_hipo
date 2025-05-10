"""
Middleware for API monitoring and observability.
"""
import time
import logging
import json
import functools
import traceback
from typing import Dict, Any, Callable, Optional
from flask import request, Response, g, Flask
import uuid
import os

from src.observability.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class APIMiddleware:
    """Middleware for API monitoring and observability."""

    def __init__(self, app: Flask, metrics_collector: Optional[MetricsCollector] = None):
        """Initialize API middleware.

        Args:
            app: Flask application.
            metrics_collector: Metrics collector for observability.
        """
        self.app = app
        self.metrics_collector = metrics_collector
        self.request_id_header = os.environ.get("REQUEST_ID_HEADER", "X-Request-ID")
        self.environment = os.environ.get("ENVIRONMENT", "development")

        # Register middleware
        self._register_middleware()

    def _register_middleware(self) -> None:
        """Register middleware with Flask application."""

        # Register before_request handler
        @self.app.before_request
        def before_request() -> None:
            """Handler to execute before each request."""
            # Set request start time
            g.start_time = time.time()

            # Generate or retrieve request ID
            if self.request_id_header in request.headers:
                g.request_id = request.headers[self.request_id_header]
            else:
                g.request_id = str(uuid.uuid4())

            # Log incoming request
            self._log_request()

        # Register after_request handler
        @self.app.after_request
        def after_request(response: Response) -> Response:
            """Handler to execute after each request.

            Args:
                response: Flask response.

            Returns:
                Modified Flask response.
            """
            # Set request ID in response headers
            response.headers[self.request_id_header] = g.get("request_id", "unknown")

            # Record metrics
            self._record_metrics(response)

            # Log outgoing response
            self._log_response(response)

            return response

        # Register error handler
        @self.app.errorhandler(Exception)
        def handle_exception(error: Exception) -> Response:
            """Handler for exceptions.

            Args:
                error: Exception that was raised.

            Returns:
                JSON error response.
            """
            # Log error
            self._log_error(error)

            # Create error response
            status_code = getattr(error, "code", 500)

            # In production, don't reveal detailed error information
            if self.environment == "production":
                error_message = "An unexpected error occurred"
            else:
                error_message = str(error)

            response = self.app.response_class(
                response=json.dumps(
                    {"status": "error", "message": error_message, "request_id": g.get("request_id", "unknown")}
                ),
                status=status_code,
                mimetype="application/json",
            )

            # Record metrics
            self._record_metrics(response, error=error)

            return response

    def _log_request(self) -> None:
        """Log incoming request."""
        log_data = {
            "timestamp": time.time(),
            "request_id": g.get("request_id", "unknown"),
            "method": request.method,
            "path": request.path,
            "remote_addr": request.remote_addr,
            "user_agent": request.headers.get("User-Agent", "unknown"),
            "content_length": request.content_length or 0,
        }

        # Add query parameters if present, but exclude sensitive information
        if request.args:
            sanitized_args = self._sanitize_data(request.args.to_dict())
            log_data["query_params"] = sanitized_args

        # Add request body if present, but exclude sensitive information
        if request.is_json and request.json:
            sanitized_body = self._sanitize_data(request.json)
            log_data["body"] = sanitized_body

        logger.info(f"Request: {json.dumps(log_data)}")

    def _log_response(self, response: Response) -> None:
        """Log outgoing response.

        Args:
            response: Flask response.
        """
        # Calculate request duration
        duration = time.time() - g.get("start_time", time.time())

        log_data = {
            "timestamp": time.time(),
            "request_id": g.get("request_id", "unknown"),
            "method": request.method,
            "path": request.path,
            "status_code": response.status_code,
            "duration": duration,
            "content_length": response.content_length or 0,
        }

        logger.info(f"Response: {json.dumps(log_data)}")

    def _log_error(self, error: Exception) -> None:
        """Log error.

        Args:
            error: Exception that was raised.
        """
        # Calculate request duration
        duration = time.time() - g.get("start_time", time.time())

        log_data = {
            "timestamp": time.time(),
            "request_id": g.get("request_id", "unknown"),
            "method": request.method,
            "path": request.path,
            "error": str(error),
            "error_type": type(error).__name__,
            "duration": duration,
            "traceback": traceback.format_exc(),
        }

        logger.error(f"Error: {json.dumps(log_data)}")

    def _record_metrics(self, response: Response, error: Optional[Exception] = None) -> None:
        """Record metrics.

        Args:
            response: Flask response.
            error: Exception that was raised, if any.
        """
        if not self.metrics_collector:
            return

        try:
            # Calculate request duration
            duration = time.time() - g.get("start_time", time.time())

            # Get request size
            request_size = request.content_length or 0

            # Get response size
            response_size = response.content_length or 0

            # Record request metrics
            self.metrics_collector.record_request(
                endpoint=request.path,
                method=request.method,
                status_code=str(response.status_code),
                latency=duration,
                request_size=request_size,
                response_size=response_size,
            )

            # If error occurred, increment error counter
            if error:
                self.metrics_collector.record_api_error(
                    endpoint=request.path, method=request.method, error_type=type(error).__name__
                )
        except Exception as e:
            logger.error(f"Error recording metrics: {e}")

    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize data to remove sensitive information.

        Args:
            data: Data to sanitize.

        Returns:
            Sanitized data.
        """
        # Make a copy of the data to avoid modifying the original
        sanitized = data.copy() if isinstance(data, dict) else dict(data)

        # List of keys that might contain sensitive information
        sensitive_keys = [
            "password",
            "token",
            "api_key",
            "secret",
            "credential",
            "key",
            "auth",
            "authorization",
            "passphrase",
            "private",
            "credit_card",
            "cc",
            "ssn",
            "social_security",
            "account",
            "email",
            "phone",
            "address",
            "zip",
            "postal",
            "dob",
            "birth",
            "license",
        ]

        # Recursively sanitize nested dictionaries
        for key, value in sanitized.items():
            if isinstance(value, dict):
                sanitized[key] = self._sanitize_data(value)
            elif isinstance(value, list):
                sanitized[key] = [self._sanitize_data(item) if isinstance(item, dict) else item for item in value]
            elif any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                sanitized[key] = "[REDACTED]"

        return sanitized


def require_api_key(func: Callable) -> Callable:
    """Decorator to require API key for endpoint.

    Args:
        func: Function to decorate.

    Returns:
        Decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return {"status": "error", "message": "API key is required"}, 401

        # In a real implementation, this would validate the API key
        # against a database or other storage

        # For now, just check if the key exists
        if api_key != os.environ.get("API_KEY", "dev-key"):
            return {"status": "error", "message": "Invalid API key"}, 401

        return func(*args, **kwargs)

    return wrapper


def rate_limit(limit: int, period: int) -> Callable:
    """Decorator to apply rate limiting to endpoint.

    Args:
        limit: Maximum number of requests allowed in the period.
        period: Time period in seconds.

    Returns:
        Decorator function.
    """

    def decorator(func: Callable) -> Callable:
        # In a real implementation, this would use Redis or similar to track
        # request counts for each client

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get client IP for rate limiting
            # This would be used in a real implementation to track request counts per client
            _ = request.remote_addr

            # Check if client has exceeded rate limit
            # This is a placeholder - in a real implementation, this would
            # check the request count for the client

            # For now, just allow all requests
            return func(*args, **kwargs)

        return wrapper

    return decorator
