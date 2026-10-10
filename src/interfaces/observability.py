from abc import ABC, abstractmethod

class BaseTracer(ABC):
    @abstractmethod
    def start_span(self, name: str) -> None:
        """
        Start an execution span.

        Args:
            name: Name of the span (e.g. 'intent_classifier', 'vector_search').
        """
        pass

    @abstractmethod
    def end_span(self, name: str, inputs: dict = None, outputs: dict = None, error: str = None) -> None:
        """
        End an execution span and save execution metrics.

        Args:
            name: Name of the span matching start_span.
            inputs: Dictionary of input parameters.
            outputs: Dictionary of outputs/results.
            error: Optional error string if execution failed.
        """
        pass

    @abstractmethod
    def get_traces(self) -> list[dict]:
        """
        Retrieve all recorded spans for the current run.

        Returns:
            list[dict]: List of trace dictionaries:
                        [{
                            'name': str,
                            'duration_ms': float,
                            'inputs': dict,
                            'outputs': dict,
                            'error': str
                        }]
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear the current trace memory."""
        pass
