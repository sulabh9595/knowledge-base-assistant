from app.services.langfuse_service import LangfuseService


class DummyTrace:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.updated = {}

    def update(self, **kwargs):
        self.updated.update(kwargs)


class DummyLangfuseClient:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.trace_calls = []
        self.generation_calls = []

    def trace(self, **kwargs):
        self.trace_calls.append(kwargs)
        return DummyTrace(**kwargs)

    def generation(self, **kwargs):
        self.generation_calls.append(kwargs)
        return DummyTrace(**kwargs)

    def flush(self):
        return None


def test_langfuse_service_skips_when_disabled():
    service = LangfuseService(enabled=False)

    assert service.is_enabled() is False


def test_langfuse_service_records_rag_trace(monkeypatch):
    client = DummyLangfuseClient()

    monkeypatch.setattr(
        "app.services.langfuse_service._build_langfuse_client",
        lambda **kwargs: client,
    )

    service = LangfuseService(enabled=True, public_key="pk", secret_key="sk")

    with service.trace_rag_query(question="What is this?", model="demo-model") as trace:
        assert trace is not None
        trace.update(output={"answer": "demo answer"})

    assert client.trace_calls[0]["name"] == "rag_query"
    assert client.trace_calls[0]["input"]["question"] == "What is this?"
    assert client.trace_calls[0]["metadata"] == {}
