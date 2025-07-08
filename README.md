# LLM Benchmark Evaluator

A Python repository for evaluating fine-tuned Hugging Face language models across various benchmarks.
This framework is designed to be modular, allowing easy integration of additional benchmarks.

## Features

-   Load Hugging Face compatible models.
-   Run evaluations on a suite of benchmarks.
-   Categorize benchmarks for organized reporting (via `config/category.json`).
-   Summarize performance metrics across different benchmarks and categories.
-   Modular design for easy addition of new benchmarks.

## Project Structure

```
llm-benchmark-evaluator/
├── config/
│   └── category.json           # Configuration for benchmark categories
├── examples/
│   └── run_evaluation.py       # Example script to run evaluations
├── llm_evaluator/
│   ├── __init__.py
│   ├── benchmarks/             # Benchmark implementations
│   │   ├── __init__.py
│   │   ├── abstract_benchmark.py # Abstract base class for benchmarks
│   │   └── example_nlu_benchmark.py # Example benchmark (to be created)
│   │   └── utils.py
│   ├── evaluation/             # Evaluation orchestration
│   │   ├── __init__.py
│   │   ├── evaluator.py          # Discovers and runs benchmarks
│   │   └── results_aggregator.py # Aggregates and summarizes results
│   ├── models/                 # Model loading utilities
│   │   ├── __init__.py
│   │   └── loader.py             # Loads Hugging Face models
│   └── utils/                  # Common utility functions
│       ├── __init__.py
│       └── file_utils.py       # File I/O utilities (JSON, etc.)
├── tests/                      # Unit and integration tests
│   └── __init__.py
├── requirements.txt            # Python package dependencies
└── README.md                   # This file
```

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd llm-benchmark-evaluator
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    If you have a specific backend in mind (e.g., PyTorch with CUDA), ensure you have the correct version installed. For PyTorch:
    ```bash
    # Example for CUDA 11.8 (see https://pytorch.org/get-started/locally/ for other versions)
    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

## Running Evaluations

The primary way to run evaluations is through the `examples/run_evaluation.py` script.

```bash
python examples/run_evaluation.py --model_name_or_path <your_model_name_or_path> [options]
```

**Key Arguments:**

*   `--model_name_or_path`: (Required) The Hugging Face model identifier (e.g., `gpt2`, `bert-base-uncased`) or path to a local model directory.
*   `--model_task`: The primary task of the model (e.g., `text-classification`, `text-generation`, `question-answering`). Defaults to `text-classification`. This helps in loading the correct model head and pipeline.
*   `--benchmarks`: (Optional) A space-separated list of specific benchmark names to run (e.g., `ExampleNLUBenchmark GLUE_MRPC`). If not provided, all discovered benchmarks will be attempted.
*   `--output_dir`: Directory to save evaluation results (JSON files). Defaults to `evaluation_results/`.
*   `--device`: Device to run the model on (e.g., `cpu`, `cuda`, `cuda:0`). Defaults to auto-detection.
*   `--benchmark_params`: JSON string or path to a JSON file with parameters for specific benchmarks.
    Example JSON string: `'{"BenchmarkName1": {"param1": "value1"}, "BenchmarkName2": {"param2": "value2"}}'`
*   `--benchmarks_dir`: Path to your benchmarks directory. Defaults to `llm_evaluator/benchmarks/`.
*   `--category_config_path`: Path to your category configuration. Defaults to `config/category.json`.

**Example Command:**

```bash
# Make sure you are in the project root directory
# Replace "distilbert-base-uncased" with your model and "ExampleNLUBenchmark" with actual benchmark names
python examples/run_evaluation.py \
    --model_name_or_path "distilbert-base-uncased" \
    --model_task "text-classification" \
    --benchmarks "ExampleNLUBenchmark" \
    --output_dir "my_evaluation_output"
```

## Adding New Benchmarks

1.  **Create a new Python file** in the `llm_evaluator/benchmarks/` directory (e.g., `my_custom_benchmark.py`).
2.  **Define a class** that inherits from `AbstractBenchmark` (from `llm_evaluator.benchmarks.abstract_benchmark`).
3.  **Implement the required properties and methods:**
    *   `name` (property): A unique string name for your benchmark (e.g., `"MyCustomBenchmark"`). This name is used to identify the benchmark in `config/category.json` and when specifying benchmarks to run.
    *   `description` (property): A brief string description of what the benchmark evaluates.
    *   `run(self, model_pipeline, model_name: str, **kwargs) -> dict`: This method contains the core logic for your benchmark.
        *   `model_pipeline`: The Hugging Face pipeline object (e.g., `transformers.TextClassificationPipeline`) for the model being evaluated. Use this to get predictions.
        *   `model_name`: The name/path of the model, for reference or logging.
        *   `**kwargs`: Can receive additional parameters, including `benchmark_specific_params` passed from `run_evaluation.py`. You can access parameters for your benchmark like this: `params = kwargs.get("benchmark_specific_params", {}).get(self.name, {})`.
        *   It should return a dictionary of metrics (e.g., `{"accuracy": 0.95, "f1_score": 0.92}`).

    **Example `my_custom_benchmark.py`:**
    ```python
    from llm_evaluator.benchmarks.abstract_benchmark import AbstractBenchmark
    # You might need 'evaluate' for metrics, 'datasets' to load data
    import evaluate
    import datasets

    class MyCustomBenchmark(AbstractBenchmark):
        @property
        def name(self) -> str:
            return "MyCustomBenchmark"

        @property
        def description(self) -> str:
            return "Evaluates X, Y, and Z aspects of a model."

        def run(self, model_pipeline, model_name: str, **kwargs) -> dict:
            print(f"Running {self.name} for model {model_name} using pipeline: {type(model_pipeline)}")

            # Access benchmark-specific parameters if provided
            # benchmark_params = kwargs.get("benchmark_specific_params", {}).get(self.name, {})
            # num_samples = benchmark_params.get("num_samples", 100) # Example parameter

            # 1. Load your dataset (e.g., using `datasets.load_dataset`)
            # For this example, let's assume a simple list of inputs and references
            # In a real benchmark, you'd load a proper dataset.
            # e.g. dataset = datasets.load_dataset("glue", "mrpc", split="validation[:1%]") # small sample
            # inputs = [item['sentence1'] + " [SEP] " + item['sentence2'] for item in dataset]
            # references = [item['label'] for item in dataset]

            # Dummy data for illustration
            inputs = ["This is a positive sentence.", "This is a negative sentence."]
            references = [1, 0] # Assuming 1 for positive, 0 for negative

            # 2. Get predictions from the model
            # The model_pipeline is already configured for the task (e.g., text-classification)
            # Its output format depends on the pipeline type.
            # For TextClassificationPipeline: [{'label': 'LABEL_X', 'score': 0.X}, ...]
            try:
                # Note: Ensure your input format matches what the pipeline expects.
                # For sentence-pair tasks, concatenate them or pass them as tuples if pipeline supports.
                raw_predictions = model_pipeline(inputs)
            except Exception as e:
                print(f"Error during model prediction in {self.name}: {e}")
                return {"error": str(e)}

            # 3. Post-process predictions to match reference format
            # This step is highly dependent on your model output and benchmark requirements.
            # For a text classification pipeline, you might need to map labels like 'LABEL_1' to integers.
            # For this dummy example, let's assume a simple mapping if needed, or direct use if labels are already 0/1.
            # This is a placeholder; actual label mapping will be specific to the model.

            # Example for a pipeline that returns {'label': 'POSITIVE', 'score': ...}
            # label_map = {"POSITIVE": 1, "NEGATIVE": 0}
            # predictions = [label_map.get(pred[0]['label'].upper(), -1) for pred in raw_predictions]

            # If the pipeline directly gives numerical labels compatible with references:
            # This is highly dependent on the tokenizer and model config.
            # The example below assumes the label from pipeline can be directly used or mapped.
            # It's often more robust to configure the pipeline's model with id2label and label2id.
            predictions = []
            for i, item_input in enumerate(inputs):
                try:
                    # Forcing single item prediction to simplify processing, batching is better for performance.
                    pred_output = model_pipeline(item_input)
                    # Assuming standard output: [{'label': 'LABEL_X', 'score': Y}]
                    # You need to map 'LABEL_X' to your reference format (e.g., 0 or 1)
                    # This is a placeholder for actual label conversion logic
                    # For instance, if model outputs 'LABEL_1', 'LABEL_0'
                    if pred_output[0]['label'] == 'LABEL_1' or pred_output[0]['label'] == 1: # Adjust based on actual output
                        predictions.append(1)
                    elif pred_output[0]['label'] == 'LABEL_0' or pred_output[0]['label'] == 0: # Adjust based on actual output
                        predictions.append(0)
                    else: # Fallback or error if label is unexpected
                        print(f"Warning: Unexpected label '{pred_output[0]['label']}' from model for input '{item_input}'. Appending -1.")
                        predictions.append(-1) # Or handle as an error
                except Exception as e:
                    print(f"Error processing prediction for input '{item_input}': {e}")
                    predictions.append(-1) # Or handle error appropriately

            if len(predictions) != len(references):
                return {"error": f"Mismatch in number of predictions ({len(predictions)}) and references ({len(references)})."}

            # 4. Compute metrics (e.g., using `evaluate.load`)
            accuracy_metric = evaluate.load("accuracy")
            f1_metric = evaluate.load("f1")

            results = {}
            try:
                results["accuracy"] = accuracy_metric.compute(predictions=predictions, references=references)["accuracy"]
                results["f1"] = f1_metric.compute(predictions=predictions, references=references, average="weighted")["f1"] # Use 'weighted' for multi-class if applicable
            except Exception as e:
                print(f"Error computing metrics in {self.name}: {e}")
                results["metrics_error"] = str(e)

            return results
    ```

4.  **Add to `config/category.json` (Optional but Recommended):**
    Open `config/category.json` and add your new benchmark's name (the one returned by the `name` property) to an appropriate category. If the category doesn't exist, you can create it.
    ```json
    {
      "Natural Language Understanding (NLU)": [
        "ExampleNLUBenchmark",
        "MyCustomBenchmark"
      ],
      // ... other categories
    }
    ```
    If your benchmark is not listed in `category.json`, it will be assigned to "Uncategorized" in the results summary.

5.  **Run it!**
    The `run_evaluation.py` script will automatically discover any new benchmark classes in the `llm_evaluator/benchmarks/` directory. You can run it by its name:
    ```bash
    python examples/run_evaluation.py --model_name_or_path "your-model" --benchmarks "MyCustomBenchmark"
    ```

## Modifying Benchmark Categories

To change how benchmarks are grouped in the final report:

1.  Edit the `config/category.json` file.
2.  This file is a dictionary where keys are category names (strings) and values are lists of benchmark names (strings, matching the `name` property of your benchmark classes).

    ```json
    {
      "Category A": ["Benchmark1", "Benchmark2"],
      "Category B": ["Benchmark3"]
    }
    ```

## Contributing

(Details on how to contribute, if this were a public project - e.g., fork, branch, PR, coding standards.)

## License

(Specify a license, e.g., MIT, Apache 2.0 - for now, this is placeholder.)

This project is licensed under the MIT License. See the LICENSE file for details.
```
