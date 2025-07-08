from typing import List, Dict, Any
import json

class ResultsAggregator:
    """
    Aggregates results from multiple benchmarks and provides a summary.
    """
    def __init__(self, results: List[Dict[str, Any]]):
        """
        Initializes the aggregator with a list of benchmark results.

        Args:
            results (List[Dict[str, Any]]): A list of result dictionaries,
                                           where each dict is typically from Evaluator.run_benchmarks.
                                           Expected structure:
                                           [
                                               {
                                                   "benchmark_name": "...",
                                                   "category": "...",
                                                   "description": "...",
                                                   "metrics": {"metric1": value1, ...} or None,
                                                   "error": "..." (optional)
                                               },
                                               ...
                                           ]
        """
        self.results = results

    def get_summary(self) -> Dict[str, Any]:
        """
        Generates a summarized report of all benchmark results, categorized.

        Returns:
            A dictionary where keys are categories, and values are lists of benchmark
            results belonging to that category. Also includes an overall summary.
        """
        categorized_results: Dict[str, List[Dict[str, Any]]] = {}
        all_metrics_summary: Dict[str, List[float]] = {} # To calculate overall averages
        successful_benchmarks = 0
        failed_benchmarks = 0

self.results = results

    def get_summary(self) -> Dict[str, Any]:
        """
        Generates a summarized report of all benchmark results, categorized.

        Returns:
            A dictionary where keys are categories, and values are lists of benchmark
            results belonging to that category. Also includes an overall summary.
        """
        categorized_results: Dict[str, List[Dict[str, Any]]] = {}
        all_metrics_summary: Dict[str, List[float]] = {}  # To calculate overall averages
        successful_benchmarks = 0
        failed_benchmarks = 0

        for res in self.results:
            category = res.get("category", "Uncategorized")
            if category not in categorized_results:
                categorized_results[category] = []

            benchmark_data = {
                "name": res.get("benchmark_name"),
                "description": res.get("description"),
                "metrics": res.get("metrics"),
                "error": res.get("error")
            }
            categorized_results[category].append(benchmark_data)

            if res.get("metrics") and not res.get("error"):
                successful_benchmarks += 1
                # For overall summary, collect all numeric metrics
                for metric_name, metric_value in res["metrics"].items():
                    if isinstance(metric_value, (int, float)):  # Only numeric metrics for averaging
                        if metric_name not in all_metrics_summary:
                            all_metrics_summary[metric_name] = []
                        all_metrics_summary[metric_name].append(metric_value)
            else:
                failed_benchmarks += 1

        # Calculate average for overall metrics
        overall_average_metrics = {
            metric: sum(values) / len(values) if values else "N/A (no numeric values)"
            for metric, values in all_metrics_summary.items()
        }

        summary = {
            "overall_summary": {
                "total_benchmarks_run": len(self.results),
                "successful_benchmarks": successful_benchmarks,
                "failed_benchmarks": failed_benchmarks,
                "average_metrics": overall_average_metrics  # Average across all successful benchmarks
            },
            "categorized_results": categorized_results
        }
        return summary

    def print_summary(self, summary_data: Dict[str, Any] = None):
        """Prints the summary in a human-readable format."""
        if summary_data is None:
            summary_data = self.get_summary()

        print("
===== Evaluation Summary =====")
            category = res.get("category", "Uncategorized")
            if category not in categorized_results:
                categorized_results[category] = []

            benchmark_data = {
                "name": res.get("benchmark_name"),
                "description": res.get("description"),
                "metrics": res.get("metrics"),
                "error": res.get("error")
            }
            categorized_results[category].append(benchmark_data)

            if res.get("metrics") and not res.get("error"):
                successful_benchmarks += 1
                # For overall summary, collect all numeric metrics
                for metric_name, metric_value in res["metrics"].items():
                    if isinstance(metric_value, (int, float)): # Only numeric metrics for averaging
                        if metric_name not in all_metrics_summary:
                            all_metrics_summary[metric_name] = []
                        all_metrics_summary[metric_name].append(metric_value)
            else:
                failed_benchmarks +=1

        # Calculate average for overall metrics
        overall_average_metrics = {}
        for metric, values in all_metrics_summary.items():
            if values: # Ensure list is not empty
                overall_average_metrics[metric] = sum(values) / len(values)
            else:
                overall_average_metrics[metric] = "N/A (no numeric values)"


        summary = {
            "overall_summary": {
                "total_benchmarks_run": len(self.results),
                "successful_benchmarks": successful_benchmarks,
                "failed_benchmarks": failed_benchmarks,
                "average_metrics": overall_average_metrics # Average across all successful benchmarks
            },
            "categorized_results": categorized_results
        }
        return summary

    def print_summary(self, summary_data: Dict[str, Any] = None):
        """Prints the summary in a human-readable format."""
        if summary_data is None:
            summary_data = self.get_summary()

        print("\n===== Evaluation Summary =====")
        overall = summary_data["overall_summary"]
        print(f"Total Benchmarks Run: {overall['total_benchmarks_run']}")
        print(f"Successful: {overall['successful_benchmarks']}, Failed: {overall['failed_benchmarks']}")

        if overall['average_metrics']:
            print("\nOverall Average Metrics (across successful benchmarks):")
            for metric, value in overall['average_metrics'].items():
                if isinstance(value, float):
                    print(f"  - {metric}: {value:.4f}")
                else:
                    print(f"  - {metric}: {value}")

        print("\n--- Categorized Results ---")
        for category, benchmarks in summary_data["categorized_results"].items():
            print(f"\nCategory: {category}")
            for benchmark in benchmarks:
                print(f"  Benchmark: {benchmark['name']}")
                if benchmark['metrics']:
                    print("    Metrics:")
                    for metric, value in benchmark['metrics'].items():
                        if isinstance(value, float):
                             print(f"      - {metric}: {value:.4f}")
                        else:
                            print(f"      - {metric}: {value}")
                if benchmark['error']:
                    print(f"    Error: {benchmark['error']}")
        print("============================")

    def save_summary_json(self, filepath: str, summary_data: Dict[str, Any] = None):
        """Saves the summary to a JSON file."""
        if summary_data is None:
            summary_data = self.get_summary()

        try:
# Import os.path for secure path handling
    # Import pathlib for additional path validation
    def save_summary_json(self, filepath: str, summary_data: Dict[str, Any] = None):
        """Saves the summary to a JSON file."""
        if summary_data is None:
            summary_data = self.get_summary()

        try:
            safe_path = os.path.abspath(os.path.normpath(filepath))
            if not safe_path.startswith(os.path.abspath(os.getcwd())):
                raise ValueError("Invalid file path")
            
            with open(safe_path, 'w') as f:
                json.dump(summary_data, f, indent=4)
            print(f"Summary saved to {safe_path}")
        except Exception as e:
            print(f"Error saving summary to {filepath}: {e}")
                json.dump(summary_data, f, indent=4)
            print(f"Summary saved to {filepath}")
        except Exception as e:
            print(f"Error saving summary to {filepath}: {e}")


if __name__ == '__main__':
    # Example Usage:
    dummy_results = [
        {
            "benchmark_name": "TruthfulQA_MCQ",
            "category": "Truthfulness",
            "description": "Measures truthfulness in multiple-choice questions.",
            "metrics": {"mc1_acc": 0.7523, "mc2_f1": 0.6891},
        },
        {
            "benchmark_name": "GLUE_MRPC",
            "category": "NLU",
            "description": "Paraphrase detection.",
            "metrics": {"accuracy": 0.8876, "f1": 0.9123},
        },
        {
            "benchmark_name": "GLUE_CoLA",
            "category": "NLU",
            "description": "Linguistic acceptability.",
            "metrics": {"matthews_correlation": 0.6011},
        },
        {
            "benchmark_name": "HellaSwag",
            "category": "Commonsense Reasoning",
            "description": "Commonsense NLI.",
            "metrics": {"accuracy": 0.8055},
        },
        {
            "benchmark_name": "BrokenBenchmark",
            "category": "NLU",
            "description": "A benchmark designed to fail.",
            "metrics": None,
            "error": "Simulated benchmark failure: Dataset not found."
        }
    ]

    aggregator = ResultsAggregator(dummy_results)
    summary = aggregator.get_summary()
    aggregator.print_summary(summary)
    aggregator.save_summary_json("sample_summary_output.json", summary)
    print("\nResultsAggregator test finished. Check 'sample_summary_output.json'.")
