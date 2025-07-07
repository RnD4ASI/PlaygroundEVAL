from abc import ABC, abstractmethod

class AbstractBenchmark(ABC):
    """
    Abstract base class for all benchmarks.
    Each benchmark implementation should inherit from this class.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the unique name of the benchmark."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Returns a brief description of the benchmark."""
        pass

    @abstractmethod
    def run(self, model_pipeline, model_name: str, **kwargs) -> dict:
        """
        Runs the benchmark evaluation.

        Args:
            model_pipeline: The pre-loaded model pipeline (e.g., from transformers.pipeline).
            model_name: The name or path of the model being evaluated.
            **kwargs: Additional arguments that might be needed by specific benchmarks.

        Returns:
            A dictionary containing the benchmark results.
            The structure can vary but should be serializable (e.g., for JSON output).
            Example: {"accuracy": 0.85, "f1": 0.82}
        """
        pass

    def get_config(self) -> dict:
        """
        Returns any specific configuration for this benchmark.
        Can be overridden by subclasses if they have specific configs.
        """
        return {}
