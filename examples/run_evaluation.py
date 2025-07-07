import argparse
import json
import os
import sys

# Ensure the llm_evaluator package is in the Python path
# This is often needed when running scripts from a subdirectory (like examples/)
# that need to import modules from a parent directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root) # Add project_root to the Python path

from llm_evaluator.models.loader import load_model_and_tokenizer
from llm_evaluator.evaluation.evaluator import Evaluator
from llm_evaluator.evaluation.results_aggregator import ResultsAggregator
from llm_evaluator.utils.file_utils import save_json

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM Evaluation Benchmarks")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Name or path of the Hugging Face model to evaluate (e.g., 'gpt2', './my_finetuned_model')."
    )
    parser.add_argument(
        "--model_task",
        type=str,
        default="text-classification", # Default task
        help="The primary task for the model (e.g., 'text-classification', 'text-generation', 'question-answering'). This guides model loading and applicable benchmarks."
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="*", # 0 or more arguments
        help="Optional: List of specific benchmark names to run. If not provided, all discovered benchmarks will be run. (e.g., ExampleNLUBenchmark GLUE_MRPC)"
    )
    parser.add_argument(
        "--benchmark_params",
        type=str,
        help="Optional: JSON string or path to a JSON file containing parameters for specific benchmarks. "
             "Format: '{\"BenchmarkName1\": {\"param1\": \"value1\"}, \"BenchmarkName2\": {\"param2\": \"value2\"}}'."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save the evaluation results."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None, # Auto-detect
        help="Device to run the model on (e.g., 'cpu', 'cuda', 'cuda:0'). Defaults to 'cuda' if available, else 'cpu'."
    )
    parser.add_argument(
        "--benchmarks_dir",
        type=str,
        default=os.path.join(project_root, "llm_evaluator", "benchmarks"), # Default path from project root
        help="Directory where benchmark modules are located."
    )
    parser.add_argument(
        "--category_config_path",
        type=str,
        default=os.path.join(project_root, "config", "category.json"), # Default path from project root
        help="Path to the benchmark category configuration JSON file."
    )
    return parser.parse_args()

def main():
    args = parse_arguments()

    print(f"Starting evaluation for model: {args.model_name_or_path}")
    print(f"Model task: {args.model_task}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer
    try:
        print(f"Loading model '{args.model_name_or_path}' for task '{args.model_task}' on device '{args.device or 'auto'}'...")
        # The pipeline itself will be passed to benchmarks
        model_pipeline = load_model_and_tokenizer(
            args.model_name_or_path,
            task=args.model_task,
            device=args.device
        )
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Initialize evaluator
    print(f"Initializing evaluator with benchmarks from: {args.benchmarks_dir}")
    print(f"Using category config: {args.category_config_path}")
    evaluator = Evaluator(
        benchmarks_dir=args.benchmarks_dir,
        category_config_path=args.category_config_path
    )

    available_benchmarks = evaluator.list_benchmarks()
    if not available_benchmarks:
        print("No benchmarks discovered. Please check your benchmarks directory and implementations.")
        print(f"Looked in: {args.benchmarks_dir}")
        # Create a dummy benchmark for demonstration if none are found
        # This part is mainly for making sure the script can run end-to-end initially
        # In a real scenario, users must provide their benchmarks.
        dummy_benchmark_path = os.path.join(args.benchmarks_dir, "example_nlu_benchmark.py")
        if not os.path.exists(dummy_benchmark_path):
             print(f"\nINFO: No benchmarks found. Consider creating an example benchmark like 'example_nlu_benchmark.py' in '{args.benchmarks_dir}'.")
             print("The script will exit as no benchmarks can be run.")
             sys.exit(0) # Exit gracefully if no benchmarks, after providing info.
        else:
             print(f"Found {dummy_benchmark_path}, attempting to re-discover.")
             evaluator = Evaluator( # Re-initialize to pick up any newly created file (though this is tricky for dynamic creation)
                benchmarks_dir=args.benchmarks_dir,
                category_config_path=args.category_config_path
            )
             available_benchmarks = evaluator.list_benchmarks()
             if not available_benchmarks:
                 print("Still no benchmarks found after attempting to locate the example. Exiting.")
                 sys.exit(1)


    print(f"\nAvailable benchmarks: {', '.join(available_benchmarks)}")

    benchmarks_to_run = args.benchmarks
    if not benchmarks_to_run: # If no specific benchmarks are requested, run all
        print("No specific benchmarks requested, will attempt to run all available benchmarks.")
        benchmarks_to_run = available_benchmarks
    else: # Validate requested benchmarks
        valid_requested_benchmarks = []
        for b_name in benchmarks_to_run:
            if b_name in available_benchmarks:
                valid_requested_benchmarks.append(b_name)
            else:
                print(f"Warning: Requested benchmark '{b_name}' not found among available benchmarks. It will be skipped.")
        benchmarks_to_run = valid_requested_benchmarks
        if not benchmarks_to_run:
            print("None of the requested benchmarks are available. Exiting.")
            sys.exit(1)


    # Parse benchmark parameters
    benchmark_params = {}
    if args.benchmark_params:
        if os.path.exists(args.benchmark_params):
            try:
                with open(args.benchmark_params, 'r') as f:
                    benchmark_params = json.load(f)
            except Exception as e:
                print(f"Error loading benchmark parameters from file {args.benchmark_params}: {e}")
        else:
            try:
                benchmark_params = json.loads(args.benchmark_params)
            except json.JSONDecodeError as e:
                print(f"Error parsing benchmark parameters JSON string: {e}")

    print(f"\nRunning benchmarks: {', '.join(benchmarks_to_run)}")
    if benchmark_params:
        print(f"With parameters: {benchmark_params}")

    # Run benchmarks
    # The model_pipeline is passed directly, benchmarks are expected to use it.
    raw_results = evaluator.run_benchmarks(
        model_pipeline=model_pipeline,
        model_name=args.model_name_or_path, # Pass model name for logging/reference
        benchmark_names=benchmarks_to_run,
        # Pass benchmark-specific params. Each benchmark's `run` method
        # will receive `**kwargs`. It can then look for its own params.
        # Example: `params = kwargs.get(self.name, {})`
        benchmark_specific_params=benchmark_params
    )

    # Aggregate and print results
    aggregator = ResultsAggregator(raw_results)
    summary = aggregator.get_summary()
    aggregator.print_summary(summary)

    # Save results
    model_name_safe = args.model_name_or_path.replace('/', '_') # Sanitize model name for filename
    results_filename = f"results_{model_name_safe}.json"
    results_filepath = os.path.join(args.output_dir, results_filename)

    # Save the detailed raw results as well for more in-depth analysis
    raw_results_filename = f"raw_results_{model_name_safe}.json"
    raw_results_filepath = os.path.join(args.output_dir, raw_results_filename)

    save_json(raw_results, raw_results_filepath)
    print(f"\nRaw results saved to: {raw_results_filepath}")

    save_json(summary, results_filepath) # This saves the aggregated summary
    print(f"Aggregated summary saved to: {results_filepath}")

    print("\nEvaluation finished.")

if __name__ == "__main__":
    # To run this script (after creating at least one benchmark, e.g., example_nlu_benchmark.py):
    # Ensure your CWD is the project root (e.g., `path/to/llm-benchmark-evaluator`)
    #
    # Example command (replace with a real model and ensure 'example_nlu_benchmark.py' exists):
    # python examples/run_evaluation.py --model_name_or_path "distilbert-base-uncased" --model_task "text-classification" --benchmarks "ExampleNLUBenchmark"
    #
    # If 'example_nlu_benchmark.py' is not yet created, the script will guide you.
    # For the script to find 'llm_evaluator' module, ensure project_root is correctly added to sys.path.
    main()
