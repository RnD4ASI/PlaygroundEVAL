## Agent Instructions for `llm-benchmark-evaluator`

Welcome, agent! This file provides guidance for working on the `llm-benchmark-evaluator` repository.

### Core Principles:

1.  **Modularity:** The system is designed to be highly modular, especially concerning benchmarks. When adding new features or benchmarks, prioritize designs that maintain or enhance this modularity.
2.  **Hugging Face Ecosystem:** The evaluator is built around the Hugging Face ecosystem (`transformers`, `datasets`, `evaluate`). Leverage these libraries as much as possible.
3.  **Clarity and Readability:** Code should be well-documented and easy to understand. Python type hints are encouraged.
4.  **Configuration-driven:** Aspects like benchmark categorization are driven by configuration files (`config/category.json`). New, similar features should also consider a configuration-driven approach if it simplifies customization for the user.

### Working with Benchmarks:

*   **Abstract Base Class:** All benchmarks MUST inherit from `llm_evaluator.benchmarks.abstract_benchmark.AbstractBenchmark` and implement its abstract methods/properties (`name`, `description`, `run`).
*   **Discovery:** Benchmarks are automatically discovered by the `Evaluator` class from the `llm_evaluator/benchmarks/` directory. Ensure your new benchmark file is placed there and contains a class inheriting from `AbstractBenchmark`.
*   **`run` Method Signature:** The `run` method of a benchmark receives `model_pipeline` (a Hugging Face pipeline object), `model_name` (string), and `**kwargs`.
    *   The `model_pipeline` is pre-loaded and ready to use for inference. The task of this pipeline is determined by the `--model_task` argument to `run_evaluation.py`. Ensure your benchmark logic is compatible with the expected pipeline type.
    *   `**kwargs` can include `benchmark_specific_params`. If your benchmark needs custom parameters, it should look for them in `kwargs.get("benchmark_specific_params", {}).get(self.name, {})`.
*   **Return Value:** The `run` method must return a dictionary of metrics (e.g., `{"accuracy": 0.85, "f1_score": 0.90}`). These metrics should be numerical where possible for aggregation. If an error occurs, it can return a dictionary like `{"error": "description of error"}`.
*   **Dependencies:** If a benchmark requires specific Python packages not already in `requirements.txt`, add them.
*   **Data Handling:**
    *   Use the `datasets` library to load standard datasets.
    *   Handle data preprocessing within the benchmark's `run` method or in helper functions within the benchmark's module or `llm_evaluator.benchmarks.utils`.
    *   Be mindful of dataset sizes. For example benchmarks or tests, use small subsets or mock data.
*   **Metrics:** Use the `evaluate` library for common metrics.

### Configuration:

*   **`config/category.json`**: This file maps benchmark names (as returned by their `name` property) to user-defined categories. When adding a new benchmark, add its name to the appropriate category list in this file. If a suitable category doesn't exist, you can add a new one. Benchmarks not listed here will be grouped under "Uncategorized".

### Running and Testing:

*   The main entry point is `examples/run_evaluation.py`. Familiarize yourself with its command-line arguments.
*   When developing a new benchmark, test it by running `run_evaluation.py` and specifying your benchmark with the `--benchmarks` argument.
    ```bash
    python examples/run_evaluation.py --model_name_or_path "gpt2" --model_task "text-generation" --benchmarks "YourNewBenchmarkName"
    ```
    (Use a small, fast model like "gpt2" or "distilbert-base-uncased" for initial testing).
*   Add unit tests for new utility functions or complex logic within benchmarks to the `tests/` directory.

### Code Style:

*   Follow PEP 8 guidelines.
*   Use type hints.
*   Keep lines under a reasonable length (e.g., 100-120 characters).

### Committing and Submitting:

*   Ensure your changes are functional and tested.
*   If you add new dependencies, update `requirements.txt`.
*   Update `README.md` if your changes affect usage, add new features, or change the project structure.
*   If you add a new benchmark, ensure it's mentioned in `README.md` as an example if appropriate, and update `config/category.json`.

By following these guidelines, we can maintain a clean, extensible, and user-friendly evaluation framework. Thank you!
