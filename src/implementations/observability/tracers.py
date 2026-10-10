import time
from langchain_core.callbacks import BaseCallbackHandler
from src.interfaces.observability import BaseTracer

class SessionTracer(BaseTracer, BaseCallbackHandler):
    def __init__(self):
        """Thread-local / session-local pipeline execution tracer compatible with LangChain."""
        super().__init__()
        self.traces = []
        self.active_spans = {}
        self.run_to_span = {}

    def start_span(self, name: str) -> None:
        self.active_spans[name] = {
            'name': name,
            'start_time': time.time(),
            'inputs': None,
            'outputs': None,
            'error': None
        }

    def end_span(self, name: str, inputs: dict = None, outputs: dict = None, error: str = None) -> None:
        span = self.active_spans.pop(name, None)
        if span:
            span['end_time'] = time.time()
            span['duration_ms'] = round((span['end_time'] - span['start_time']) * 1000, 2)
            if inputs is not None:
                span['inputs'] = inputs
            if outputs is not None:
                span['outputs'] = outputs
            if error is not None:
                span['error'] = error
            self.traces.append(span)

    def get_traces(self) -> list[dict]:
        return self.traces

    def clear(self) -> None:
        self.traces = []
        self.active_spans = {}
        self.run_to_span = {}

    # LangChain Callbacks
    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        span_name = f"llm_generation_{len(self.run_to_span) + 1}"
        self.run_to_span[run_id] = span_name
        self.start_span(span_name)
        if span_name in self.active_spans:
            self.active_spans[span_name]['inputs'] = {'prompts': prompts}

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        span_name = self.run_to_span.pop(run_id, None)
        if span_name:
            output_text = ""
            try:
                output_text = response.generations[0][0].text
            except:
                pass
            self.end_span(span_name, outputs={'response': output_text})

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        span_name = self.run_to_span.pop(run_id, None)
        if span_name:
            self.end_span(span_name, error=str(error))

class ConsoleTracer(BaseTracer, BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.active_spans = {}

    def start_span(self, name: str) -> None:
        self.active_spans[name] = time.time()
        print(f"[TRACE START] {name}")

    def end_span(self, name: str, inputs: dict = None, outputs: dict = None, error: str = None) -> None:
        start_time = self.active_spans.pop(name, None)
        if start_time:
            duration = round((time.time() - start_time) * 1000, 2)
            err_str = f" | ERROR: {error}" if error else ""
            print(f"[TRACE END] {name} | Duration: {duration}ms{err_str}")

    def get_traces(self) -> list[dict]:
        return []

    def clear(self) -> None:
        self.active_spans = {}

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        self.start_span(f"llm_call_{run_id}")

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        self.end_span(f"llm_call_{run_id}")

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self.end_span(f"llm_call_{run_id}", error=str(error))
