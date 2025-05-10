"""
Fault tolerance utilities for resilient ML infrastructure.
"""
import logging
import time
import random
import functools
import asyncio
import inspect
from typing import Callable, Dict, Any, Optional, Union, List, Tuple, Type
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class FailureHandler:
    """Handler for managing failure modes and recovery strategies."""

    def __init__(self):
        """Initialize FailureHandler."""
        self.logger = logging.getLogger(f"{__name__}.FailureHandler")
        self.recovery_strategies = {}
        self.failure_counts = {}
        self.last_failures = {}
        self.lock = threading.RLock()

    def register_strategy(self, failure_type: Type[Exception], strategy: Callable[[Exception, Dict[str, Any]], Any]):
        """Register a recovery strategy for a failure type.

        Args:
            failure_type: Type of exception to handle.
            strategy: Function to call when the failure occurs. Takes the exception and context dict.
        """
        self.recovery_strategies[failure_type] = strategy

    def handle_failure(self, exc: Exception, context: Dict[str, Any] = None) -> Any:
        """Handle a failure using registered recovery strategies.

        Args:
            exc: Exception that occurred.
            context: Additional context information.

        Returns:
            Result of recovery strategy, or raises the exception if no strategy is found.
        """
        context = context or {}

        with self.lock:
            # Update failure statistics
            exc_type = type(exc)
            self.failure_counts[exc_type] = self.failure_counts.get(exc_type, 0) + 1
            self.last_failures[exc_type] = datetime.now()

        # Find the most specific matching strategy
        for failure_type, strategy in self.recovery_strategies.items():
            if isinstance(exc, failure_type):
                self.logger.info(f"Applying recovery strategy for {failure_type.__name__}")
                try:
                    return strategy(exc, context)
                except Exception as recovery_exc:
                    self.logger.error(f"Recovery strategy for {failure_type.__name__} failed: {str(recovery_exc)}")
                    raise exc

        # No matching strategy found
        self.logger.warning(f"No recovery strategy found for {exc_type.__name__}")
        raise exc

    def get_failure_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get failure statistics.

        Returns:
            Dictionary mapping exception types to statistics.
        """
        stats = {}

        with self.lock:
            for exc_type, count in self.failure_counts.items():
                stats[exc_type.__name__] = {"count": count, "last_failure": self.last_failures.get(exc_type)}

        return stats


# Global default failure handler
default_failure_handler = FailureHandler()


def with_fallback(fallback_value: Any = None, fallback_func: Optional[Callable] = None):
    """Decorator to provide a fallback value or function when an exception occurs.

    Args:
        fallback_value: Value to return if an exception occurs.
        fallback_func: Function to call if an exception occurs. Takes same arguments as decorated function.

    Returns:
        Decorator function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Function {func.__name__} failed: {str(e)}. Using fallback.")
                if fallback_func is not None:
                    return fallback_func(*args, **kwargs)
                return fallback_value

        return wrapper

    return decorator


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """Decorator to retry a function on failure with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts.
        delay: Initial delay between attempts in seconds.
        backoff_factor: Factor to increase delay by after each attempt.
        jitter: Whether to add random jitter to delay.
        exceptions: Tuple of exceptions to catch and retry.

    Returns:
        Decorator function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay

            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"All {max_attempts} retry attempts for {func.__name__} failed: {str(e)}")
                        raise

                    # Calculate delay with optional jitter
                    sleep_time = current_delay
                    if jitter:
                        sleep_time = sleep_time * (1 + random.uniform(-0.1, 0.1))

                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} for {func.__name__} failed: {str(e)}. "
                        f"Retrying in {sleep_time:.2f}s"
                    )

                    time.sleep(sleep_time)
                    current_delay *= backoff_factor

        return wrapper

    return decorator


def timeout(seconds: float):
    """Decorator to apply a timeout to a function.

    Args:
        seconds: Timeout in seconds.

    Returns:
        Decorator function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")

            # Set the timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))

            try:
                result = func(*args, **kwargs)
            finally:
                # Cancel the timeout
                signal.alarm(0)

            return result

        return wrapper

    return decorator


def async_timeout(seconds: float):
    """Decorator to apply a timeout to an async function.

    Args:
        seconds: Timeout in seconds.

    Returns:
        Decorator function.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Async function {func.__name__} timed out after {seconds} seconds")

        return wrapper

    return decorator


def debounce(wait_time: float):
    """Decorator to debounce a function (only execute after a period of inactivity).

    Args:
        wait_time: Time to wait in seconds.

    Returns:
        Decorator function.
    """

    def decorator(func):
        last_call = [None]
        timer = [None]
        func_result = [None]
        lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                last_call[0] = time.time()

                def call_function():
                    current_time = time.time()
                    time_since_last_call = current_time - last_call[0]

                    if time_since_last_call < wait_time:
                        # Schedule another call
                        delay = wait_time - time_since_last_call
                        timer[0] = threading.Timer(delay, call_function)
                        timer[0].start()
                    else:
                        # Execute the function
                        with lock:
                            func_result[0] = func(*args, **kwargs)

                # Cancel any existing timer
                if timer[0] is not None:
                    timer[0].cancel()

                # Schedule a new call
                timer[0] = threading.Timer(wait_time, call_function)
                timer[0].start()

            return func_result[0]

        return wrapper

    return decorator


def rate_limit(calls: int, period: float):
    """Decorator to rate limit a function.

    Args:
        calls: Maximum number of calls allowed in the period.
        period: Time period in seconds.

    Returns:
        Decorator function.
    """

    def decorator(func):
        timestamps = []
        lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                now = time.time()

                # Remove timestamps older than the period
                while timestamps and now - timestamps[0] > period:
                    timestamps.pop(0)

                # Check if we've exceeded the rate limit
                if len(timestamps) >= calls:
                    wait_time = timestamps[0] + period - now
                    if wait_time > 0:
                        time.sleep(wait_time)
                        now = time.time()

                # Add current timestamp
                timestamps.append(now)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def bulkhead(max_concurrent: int, max_queue: int = 0):
    """Decorator to limit concurrent execution of a function (bulkhead pattern).

    Args:
        max_concurrent: Maximum number of concurrent executions.
        max_queue: Maximum size of queue for pending executions. 0 means no queue.

    Returns:
        Decorator function.
    """

    def decorator(func):
        semaphore = threading.Semaphore(max_concurrent)
        queue_size = threading.Semaphore(max_concurrent + max_queue)
        active = 0
        queued = 0
        lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal active, queued

            with lock:
                # Check if we can queue this execution
                if not queue_size.acquire(blocking=False):
                    raise RuntimeError(f"Bulkhead queue full for {func.__name__}")
                queued += 1

            try:
                # Wait for semaphore (may block if max_concurrent is reached)
                semaphore.acquire()

                with lock:
                    active += 1
                    queued -= 1

                try:
                    # Execute function
                    return func(*args, **kwargs)
                finally:
                    with lock:
                        active -= 1
                    semaphore.release()
            finally:
                queue_size.release()

        return wrapper

    return decorator


def cache(ttl: float = 60.0):
    """Decorator to cache function results for a specified time period.

    Args:
        ttl: Time to live for cache entries in seconds.

    Returns:
        Decorator function.
    """

    def decorator(func):
        cache_data = {}
        lock = threading.RLock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key based on function arguments
            key = str((args, frozenset(sorted(kwargs.items()))))

            with lock:
                now = time.time()

                # Check if we have a valid cached result
                if key in cache_data:
                    result, timestamp = cache_data[key]
                    if now - timestamp < ttl:
                        return result

                # Execute function and cache result
                result = func(*args, **kwargs)
                cache_data[key] = (result, now)

                # Clean up expired entries
                expired_keys = [k for k, (_, ts) in cache_data.items() if now - ts >= ttl]
                for k in expired_keys:
                    del cache_data[k]

                return result

        return wrapper

    return decorator


def parallel_map(max_workers: int = None):
    """Decorator to convert a function to a parallel mapping function.

    Args:
        max_workers: Maximum number of worker threads. None uses default ThreadPoolExecutor.

    Returns:
        Decorator function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(items, *args, **kwargs):
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                return list(executor.map(lambda item: func(item, *args, **kwargs), items))

        return wrapper

    return decorator


class CircuitBreaker:
    """Implementation of circuit breaker pattern."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit.
            recovery_timeout: Time to wait before attempting to close circuit.
            exceptions: Exceptions to count as failures.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.exceptions = exceptions

        self.failure_count = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = None

        self.lock = threading.RLock()

    def __call__(self, func):
        """Decorator implementation.

        Args:
            func: Function to decorate.

        Returns:
            Decorated function.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                if self.state == "OPEN":
                    if self._is_recovery_timeout_elapsed():
                        self.state = "HALF_OPEN"
                    else:
                        raise RuntimeError(f"Circuit breaker open for {func.__name__}")

            try:
                result = func(*args, **kwargs)

                with self.lock:
                    if self.state == "HALF_OPEN":
                        self._reset()

                return result
            except self.exceptions as e:
                with self.lock:
                    self.last_failure_time = time.time()

                    if self.state == "CLOSED":
                        self.failure_count += 1
                        if self.failure_count >= self.failure_threshold:
                            self.state = "OPEN"
                    elif self.state == "HALF_OPEN":
                        self.state = "OPEN"

                raise

        return wrapper

    def _reset(self):
        """Reset the circuit breaker state."""
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = None

    def _is_recovery_timeout_elapsed(self):
        """Check if recovery timeout has elapsed.

        Returns:
            True if timeout has elapsed, False otherwise.
        """
        if self.last_failure_time is None:
            return True

        return (time.time() - self.last_failure_time) >= self.recovery_timeout
