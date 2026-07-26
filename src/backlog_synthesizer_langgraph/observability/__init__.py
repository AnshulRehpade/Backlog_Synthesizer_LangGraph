"""Observability module — provides tracing, metrics, and structured logging.

When OTEL_ENABLED=false (or opentelemetry not installed), all functions return
no-op implementations that add zero overhead.
"""
import os
import logging

_OTEL_ENABLED = os.environ.get("OTEL_ENABLED", "true").lower() in ("true", "1", "yes")

# Try importing opentelemetry; if not available, use no-ops
try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry import metrics as _otel_metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

_tracer = None
_meter = None
_metric_reader = None


def init_telemetry(service_name: str = "backlog-synthesizer-langgraph") -> None:
    """Initialize OpenTelemetry tracing and metrics providers."""
    global _tracer, _meter, _metric_reader

    if not _OTEL_ENABLED or not _OTEL_AVAILABLE:
        return

    # Tracing
    provider = TracerProvider()
    _otel_trace.set_tracer_provider(provider)
    _tracer = _otel_trace.get_tracer(service_name)

    # Metrics
    _metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[_metric_reader])
    _otel_metrics.set_meter_provider(meter_provider)
    _meter = _otel_metrics.get_meter(service_name)


def get_tracer():
    """Get the configured tracer (or a no-op tracer if disabled)."""
    if not _OTEL_ENABLED or not _OTEL_AVAILABLE:
        if _OTEL_AVAILABLE:
            return _otel_trace.get_tracer("no-op")
        return None
    if _tracer is None:
        init_telemetry()
    return _tracer or _otel_trace.get_tracer("backlog-synthesizer-langgraph")


def get_meter():
    """Get the configured meter (or a no-op meter if disabled)."""
    if not _OTEL_ENABLED or not _OTEL_AVAILABLE:
        if _OTEL_AVAILABLE:
            return _otel_metrics.get_meter("no-op")
        return None
    if _meter is None:
        init_telemetry()
    return _meter or _otel_metrics.get_meter("backlog-synthesizer-langgraph")


def get_metric_reader():
    """Get the InMemoryMetricReader for testing (or None if disabled)."""
    return _metric_reader


def is_enabled() -> bool:
    """Check if telemetry is enabled."""
    return _OTEL_ENABLED and _OTEL_AVAILABLE
