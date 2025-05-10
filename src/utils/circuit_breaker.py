"""
Circuit breaker pattern implementation for resilient service calls.
"""
import logging
import time
import functools
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, Any, Optional
import threading

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "CLOSED"  # Normal operation, requests are allowed
    OPEN = "OPEN"  # Circuit is open, requests are not allowed
    HALF_OPEN = "HALF_OPEN"  # Limited number of requests are allowed to test the service


class CircuitBreakerOpenException(Exception):
    """Exception raised when a circuit is open."""

    pass


class CircuitBreaker:
    """Circuit breaker implementation for resilient service calls."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 1,
        exception_types: tuple = (Exception,),
        name: str = "default",
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before opening the circuit.
            recovery_timeout: Number of seconds to wait before trying to recover.
            half_open_max_calls: Maximum number of calls allowed when circuit is half-open.
            exception_types: Types of exceptions to count as failures.
            name: Name of this circuit breaker.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.exception_types = exception_types
        self.name = name

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0

        self.lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.CircuitBreaker.{name}")

    def get_state(self) -> CircuitState:
        """Get current circuit state with recovery check.

        Returns:
            Current circuit state.
        """
        with self.lock:
            # If circuit is open, check if recovery timeout has elapsed
            if self.state == CircuitState.OPEN:
                if self._recovery_timeout_elapsed():
                    self._transition_to_half_open()

            return self.state

    def record_success(self) -> None:
        """Record a successful call."""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1

                # If enough successful calls in half-open state, close the circuit
                if self.half_open_calls >= self.half_open_max_calls:
                    self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        with self.lock:
            self.last_failure_time = datetime.now()

            if self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state opens the circuit again
                self._transition_to_open()
            elif self.state == CircuitState.CLOSED:
                self.failure_count += 1

                # If too many consecutive failures, open the circuit
                if self.failure_count >= self.failure_threshold:
                    self._transition_to_open()

    def _transition_to_open(self) -> None:
        """Transition from current state to OPEN."""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.logger.warning(f"Circuit {self.name} transitioned from {old_state.value} to {self.state.value}")

    def _transition_to_half_open(self) -> None:
        """Transition from current state to HALF_OPEN."""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.logger.info(f"Circuit {self.name} transitioned from {old_state.value} to {self.state.value}")

    def _transition_to_closed(self) -> None:
        """Transition from current state to CLOSED."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_calls = 0
        self.logger.info(f"Circuit {self.name} transitioned from {old_state.value} to {self.state.value}")

    def _recovery_timeout_elapsed(self) -> bool:
        """Check if recovery timeout has elapsed.

        Returns:
            True if recovery timeout has elapsed, False otherwise.
        """
        if self.last_failure_time is None:
            return True

        return (datetime.now() - self.last_failure_time).total_seconds() >= self.recovery_timeout

    def __call__(self, func):
        """Decorator to apply circuit breaker to a function.

        Args:
            func: Function to decorate.

        Returns:
            Decorated function.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_state = self.get_state()

            if current_state == CircuitState.OPEN:
                self.logger.warning(f"Circuit {self.name} is OPEN - call to {func.__name__} rejected")
                raise CircuitBreakerOpenException(f"Circuit {self.name} is OPEN - call to {func.__name__} rejected")

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except self.exception_types as e:
                self.record_failure()
                self.logger.error(f"Circuit {self.name} - call to {func.__name__} failed: {str(e)}")
                raise

        return wrapper


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern implementation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CircuitBreakerRegistry, cls).__new__(cls)
                cls._instance._init()

        return cls._instance

    def _init(self):
        """Initialize registry."""
        self.circuit_breakers = {}
        self.logger = logging.getLogger(f"{__name__}.CircuitBreakerRegistry")

    def get_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 1,
        exception_types: tuple = (Exception,),
    ) -> CircuitBreaker:
        """Get a circuit breaker by name, creating it if it doesn't exist.

        Args:
            name: Name of the circuit breaker.
            failure_threshold: Number of consecutive failures before opening the circuit.
            recovery_timeout: Number of seconds to wait before trying to recover.
            half_open_max_calls: Maximum number of calls allowed when circuit is half-open.
            exception_types: Types of exceptions to count as failures.

        Returns:
            Circuit breaker instance.
        """
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                half_open_max_calls=half_open_max_calls,
                exception_types=exception_types,
                name=name,
            )

        return self.circuit_breakers[name]

    def get_all_circuits(self) -> Dict[str, CircuitBreaker]:
        """Get all circuit breakers.

        Returns:
            Dictionary of circuit breaker names to instances.
        """
        return self.circuit_breakers.copy()

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers.

        Returns:
            Dictionary mapping circuit breaker names to their status.
        """
        status = {}

        for name, circuit in self.circuit_breakers.items():
            status[name] = {
                "state": circuit.state.value,
                "failure_count": circuit.failure_count,
                "last_failure_time": circuit.last_failure_time,
                "failure_threshold": circuit.failure_threshold,
                "recovery_timeout": circuit.recovery_timeout,
            }

        return status


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    half_open_max_calls: int = 1,
    exception_types: tuple = (Exception,),
):
    """Decorator factory to apply circuit breaker pattern to a function.

    Args:
        name: Name of the circuit breaker.
        failure_threshold: Number of consecutive failures before opening the circuit.
        recovery_timeout: Number of seconds to wait before trying to recover.
        half_open_max_calls: Maximum number of calls allowed when circuit is half-open.
        exception_types: Types of exceptions to count as failures.

    Returns:
        Decorator function.
    """
    registry = CircuitBreakerRegistry()
    circuit = registry.get_circuit_breaker(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        half_open_max_calls=half_open_max_calls,
        exception_types=exception_types,
    )

    return circuit


def with_fallback(fallback_function: Optional[Callable] = None):
    """Decorator to add fallback for circuit breaker failures.

    Args:
        fallback_function: Function to call if circuit is open or call fails.

    Returns:
        Decorator function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except CircuitBreakerOpenException as e:
                if fallback_function is not None:
                    return fallback_function(*args, **kwargs)
                raise

        return wrapper

    return decorator


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, backoff_factor: float = 2.0):
    """Decorator to retry a function with exponential backoff.

    Args:
        max_retries: Maximum number of retries.
        base_delay: Base delay time in seconds.
        backoff_factor: Backoff factor for delay calculation.

    Returns:
        Decorator function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for retry in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if retry >= max_retries:
                        raise

                    # Calculate delay for exponential backoff
                    delay = base_delay * (backoff_factor**retry)

                    # Add jitter to prevent synchronized retries
                    import random

                    jitter = random.uniform(0, 0.1 * delay)
                    delay += jitter

                    logger.warning(
                        f"Retrying {func.__name__} after error: {str(e)}. "
                        f"Retry {retry + 1}/{max_retries}, delay: {delay:.2f}s"
                    )

                    time.sleep(delay)

        return wrapper

    return decorator
