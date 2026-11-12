"""OpenTelemetry tracing.

Off by default. TRACING=console prints spans; TRACING=otlp exports to any OTLP backend
(Jaeger, Grafana Tempo, Langfuse, Honeycomb) configured with the standard
OTEL_EXPORTER_OTLP_ENDPOINT / OTEL_EXPORTER_OTLP_HEADERS environment variables.
Spans carry hashes, sizes, tokens, cost and latency; never raw user text.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SpanExporter

SERVICE = "semantic-cache-faq-bot"
_configured = False


def setup_tracing(mode: str, service: str = SERVICE) -> bool:
    """Install a tracer provider once. Returns True if tracing was turned on by this call."""
    global _configured
    if mode == "off" or _configured:
        return False
    exporter: SpanExporter
    if mode == "console":
        exporter = ConsoleSpanExporter()
    else:  # pragma: no cover - needs the [otel] extra and a collector
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _configured = True
    return True


def tracer() -> trace.Tracer:
    return trace.get_tracer("semcache")
