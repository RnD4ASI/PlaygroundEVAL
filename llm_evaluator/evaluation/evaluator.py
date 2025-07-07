import os
import importlib
import inspect
from ..benchmarks.abstract_benchmark import AbstractBenchmark
from ..utils.file_utils import load_json
from typing import List, Dict, Any

class Evaluator:
    """
    Discovers and runs benchmarks, then collects their results.
    """
    def __init__(self, benchmarks_dir: str = "llm_evaluator/benchmarks", category_config_path: str = "config/category.json"):
        self.benchmarks_dir = benchmarks_dir
        self.category_config_path = category_config_path
        self.benchmark_categories = self._load_benchmark_categories()
        self.benchmarks = self._discover_benchmarks()

    def _load_benchmark_categories(self) -> Dict[str, str]:
        """Loads benchmark categories from the category config file."""
        categories_map = {}
        try:
            config = load_json(self.category_config_path)
            for category, benchmark_names in config.items():
                for name in benchmark_names:
                    categories_map[name] = category
        except FileNotFoundError:
            print(f"Warning: Category config file not found at {self.category_config_path}. Benchmarks will not be categorized.")
        except Exception as e:
            print(f"Error loading category config: {e}. Benchmarks will not be categorized.")
        return categories_map

    def _discover_benchmarks(self) -> Dict[str, AbstractBenchmark]:
        """
        Discovers benchmark classes within the benchmarks directory.
        A module in the benchmarks directory is considered a benchmark provider
        if it contains one or more classes inheriting from AbstractBenchmark.
        """
        benchmarks: Dict[str, AbstractBenchmark] = {}
        if not os.path.exists(self.benchmarks_dir):
            print(f"Warning: Benchmarks directory '{self.benchmarks_dir}' not found.")
            return benchmarks

        for filename in os.listdir(self.benchmarks_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = filename[:-3]
                # Construct the full module path relative to the package root
                # Assumes 'llm_evaluator' is the top-level package.
                # This needs to be robust if the directory structure changes or if run from different locations.
                # A common way is to use the package structure: llm_evaluator.benchmarks.module_name
                full_module_path = f"llm_evaluator.benchmarks.{module_name}"
                try:
                    module = importlib.import_module(full_module_path)
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and issubclass(obj, AbstractBenchmark) and obj is not AbstractBenchmark:
                            try:
                                instance = obj()
                                if instance.name in benchmarks:
                                    print(f"Warning: Duplicate benchmark name '{instance.name}' found. Overwriting.")
                                benchmarks[instance.name] = instance
                                print(f"Discovered benchmark: {instance.name}")
                            except TypeError as e:
                                print(f"Error instantiating benchmark {name} from {module_name}: {e}. Does it have a constructor requiring arguments?")
                            except Exception as e:
                                print(f"Error processing benchmark class {name} in {module_name}: {e}")
                except ImportError as e:
                    print(f"Error importing benchmark module {full_module_path}: {e}")
                except Exception as e:
                    print(f"Unexpected error discovering benchmarks in {module_name}: {e}")

        if not benchmarks:
            print("No benchmarks discovered. Ensure benchmark classes inherit from AbstractBenchmark and are in the benchmarks directory.")
        return benchmarks

    def list_benchmarks(self) -> List[str]:
        """Returns a list of available benchmark names."""
        return list(self.benchmarks.keys())

    def run_benchmarks(self, model_pipeline, model_name: str, benchmark_names: List[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Runs specified benchmarks or all available benchmarks if none are specified.

        Args:
            model_pipeline: The pre-loaded model pipeline.
            model_name (str): The name of the model being evaluated.
            benchmark_names (List[str], optional): A list of benchmark names to run.
                                                  If None, all discovered benchmarks are run.
            **kwargs: Additional arguments to pass to each benchmark's run method.


        Returns:
            A list of dictionaries, where each dictionary contains the results of a benchmark.
            Each result dictionary includes 'benchmark_name', 'category', and 'metrics'.
        """
        results = []
        benchmarks_to_run = benchmark_names or self.list_benchmarks()

        print(f"\nRunning evaluation for model: {model_name}")
        print(f"Target benchmarks: {', '.join(benchmarks_to_run) if benchmarks_to_run else 'All available'}")

        for name in benchmarks_to_run:
            benchmark = self.benchmarks.get(name)
            if benchmark:
                print(f"\n--- Running benchmark: {benchmark.name} ---")
                try:
                    benchmark_result = benchmark.run(model_pipeline, model_name, **kwargs)
                    category = self.benchmark_categories.get(benchmark.name, "Uncategorized")
                    results.append({
                        "benchmark_name": benchmark.name,
                        "category": category,
                        "description": benchmark.description,
                        "metrics": benchmark_result
                    })
                    print(f"--- Completed benchmark: {benchmark.name} ---")
                except Exception as e:
                    print(f"Error running benchmark {benchmark.name}: {e}")
                    results.append({
                        "benchmark_name": benchmark.name,
                        "category": self.benchmark_categories.get(benchmark.name, "Uncategorized"),
                        "description": benchmark.description,
                        "metrics": None,
                        "error": str(e)
                    })
            else:
                print(f"Warning: Benchmark '{name}' not found.")
                results.append({
                    "benchmark_name": name,
                    "category": "Unknown",
                    "description": "Benchmark not found during execution.",
                    "metrics": None,
                    "error": "Benchmark not found."
                })

        return results

    def get_benchmark_instance(self, name: str) -> AbstractBenchmark:
        """Returns a benchmark instance by name."""
        return self.benchmarks.get(name)

if __name__ == '__main__':
    # Example usage (assuming you have some benchmarks in the ./benchmarks folder)
    # This part is for quick testing of the Evaluator itself.
    # Create a dummy benchmark file for this test to work:
    # llm_evaluator/benchmarks/dummy_benchmark.py
    # from .abstract_benchmark import AbstractBenchmark
    # class DummyBenchmark(AbstractBenchmark):
    #     @property
    #     def name(self): return "dummy_test"
    #     @property
    #     def description(self): return "A dummy test benchmark."
    #     def run(self, model_pipeline, model_name, **kwargs): return {"score": 1.0}

    # Create a dummy category file for this test to work:
    # config/category.json
    # {
    #   "General": ["dummy_test"]
    # }

    print("Testing Evaluator (run this script directly from the project root for paths to work)")
    # Adjust paths if running from a different directory or within a package structure
    # For example, if 'llm_evaluator' is a package, paths might need to be relative to that.
    # The import paths like `from ..benchmarks.abstract_benchmark import AbstractBenchmark`
    # assume this file is part of the `llm_evaluator.evaluation` package.

    # Create dummy files for testing if they don't exist
    if not os.path.exists("llm_evaluator/benchmarks/dummy_benchmark.py"):
        os.makedirs("llm_evaluator/benchmarks", exist_ok=True)
        with open("llm_evaluator/benchmarks/dummy_benchmark.py", "w") as f:
            f.write("from .abstract_benchmark import AbstractBenchmark\n")
            f.write("class DummyBenchmark(AbstractBenchmark):\n")
            f.write("    @property\n")
            f.write("    def name(self):\n")
            f.write("        return \"dummy_test\"\n")
            f.write("    @property\n")
            f.write("    def description(self):\n")
            f.write("        return \"A dummy test benchmark.\"\n")
            f.write("    def run(self, model_pipeline, model_name, **kwargs):\n")
            f.write("        print(f\"Dummy benchmark run for {model_name} with pipeline {type(model_pipeline)}\")\n")
            f.write("        return {\"score\": 1.0}\n")

    if not os.path.exists("config/category.json"):
        os.makedirs("config", exist_ok=True)
        with open("config/category.json", "w") as f:
            f.write("{\n  \"General\": [\"dummy_test\"]\n}\n")

    # The paths for Evaluator assume it's being run from the root of the project.
    # If your CWD is `llm_evaluator`, then `benchmarks_dir` should be `benchmarks`
    # and `category_config_path` should be `../config/category.json`.
    # For now, let's assume CWD is project root.
    evaluator = Evaluator(benchmarks_dir="llm_evaluator/benchmarks", category_config_path="config/category.json")

    print("\nAvailable benchmarks:")
    for b_name in evaluator.list_benchmarks():
        print(f"- {b_name}")
        benchmark_instance = evaluator.get_benchmark_instance(b_name)
        if benchmark_instance:
            print(f"  Description: {benchmark_instance.description}")
            print(f"  Category: {evaluator.benchmark_categories.get(b_name, 'Uncategorized')}")

    # Dummy model pipeline for testing run_benchmarks
    class DummyPipeline:
        def __init__(self, task): self.task = task
        def __call__(self, *args, **kwargs): return [{"label": "TEST", "score": 0.99}]

    dummy_pipeline = DummyPipeline(task="text-classification")

    print("\nRunning dummy_test benchmark:")
    results = evaluator.run_benchmarks(model_pipeline=dummy_pipeline, model_name="dummy_model", benchmark_names=["dummy_test"])
    print("\nResults:")
    for res in results:
        print(res)

    # Clean up dummy files
    # os.remove("llm_evaluator/benchmarks/dummy_benchmark.py")
    # if not os.listdir("llm_evaluator/benchmarks"): # remove dir if empty, except __init__.py
    #     # os.rmdir("llm_evaluator/benchmarks") # Careful with this
    #     pass
    # os.remove("config/category.json")
    # if not os.listdir("config"):
    #     # os.rmdir("config")
    #     pass
    print("\nEvaluator test finished. Note: Dummy files were created and may need manual cleanup if issues occurred.")
