from llm_evaluator.benchmarks.abstract_benchmark import AbstractBenchmark
import evaluate
import datasets
from datasets import Dataset
import random

class ExampleNLUBenchmark(AbstractBenchmark):
    """
    Example NLU (Natural Language Understanding) benchmark.
    This benchmark uses a tiny, fixed subset of data for demonstration purposes.
    It simulates a text classification task (e.g., sentiment analysis or paraphrase detection).
    """

    @property
    def name(self) -> str:
        return "ExampleNLUBenchmark"

    @property
    def description(self) -> str:
        return "A simple example NLU benchmark for text classification using a dummy dataset."

    def run(self, model_pipeline, model_name: str, **kwargs) -> dict:
        """
        Runs the NLU benchmark.

        Args:
            model_pipeline: The pre-loaded Hugging Face text classification pipeline.
            model_name (str): The name of the model being evaluated.
            **kwargs: Additional arguments. Expects `benchmark_specific_params` which can contain
                      parameters for this benchmark, e.g., `num_samples`.

        Returns:
            A dictionary containing the benchmark results (accuracy and F1 score).
        """
        print(f"\nRunning {self.name} for model: {model_name}")
        print(f"Using pipeline: {type(model_pipeline)}")

        # Access benchmark-specific parameters if provided
        benchmark_params = kwargs.get("benchmark_specific_params", {}).get(self.name, {})
        num_samples = benchmark_params.get("num_samples", 5) # Default to 5 samples for this example
        dataset_name = benchmark_params.get("dataset_name", "glue") # Example: allow choosing dataset via params
        dataset_config = benchmark_params.get("dataset_config", "mrpc") # Example: allow choosing config

        print(f"Starting {self.name}.run() for model {model_name}")
        # 1. Load or define a small dataset
        try:
            print(f"Step 1: Loading dataset ({dataset_name}/{dataset_config}, {num_samples} samples)")
            full_dataset = datasets.load_dataset(dataset_name, dataset_config, split="validation", streaming=True)

            print(f"Attempting to load {num_samples} samples from {dataset_name}/{dataset_config} validation split.")
            samples = []
            for i, example in enumerate(full_dataset):
                if i >= num_samples * 2:
                    break
                samples.append(example)

            if len(samples) > num_samples:
                 samples = random.sample(samples, num_samples)
            elif not samples:
                raise ValueError(f"No samples could be loaded from {dataset_name}/{dataset_config} with streaming.")

            if dataset_name == "glue" and dataset_config == "mrpc":
                processed_samples = {
                    "sentence1": [s["sentence1"] for s in samples],
                    "sentence2": [s["sentence2"] for s in samples],
                    "label": [s["label"] for s in samples], "idx": [s["idx"] for s in samples]
                }
                dataset = Dataset.from_dict(processed_samples)
                inputs = [item['sentence1'] + " [SEP] " + item['sentence2'] for item in dataset]
                references = [item['label'] for item in dataset]
            else:
                print(f"Warning: Specific processing for {dataset_name}/{dataset_config} not implemented. Assuming 'text' and 'label' columns.")
                processed_samples = {
                    "text": [s.get("text", s.get("sentence1", "")) for s in samples],
                    "label": [s["label"] for s in samples]
                }
                dataset = Dataset.from_dict(processed_samples)
                inputs = [item['text'] for item in dataset]
                references = [item['label'] for item in dataset]
            print(f"Successfully loaded {len(inputs)} inputs and {len(references)} references.")
        except Exception as e_data:
            print(f"CRITICAL: Error during dataset loading/processing: {e_data!r}")
            print("Falling back to a minimal, hardcoded dataset.")
            inputs = ["This is a fantastic product!", "I am not happy with this service.", "The movie was okay.", "Best experience ever.", "Terrible, refund now."]
            references = [1, 0, 0, 1, 0]
            if len(inputs) > num_samples:
                inputs = inputs[:num_samples]
                references = references[:num_samples]
            print(f"Using {len(inputs)} hardcoded samples.")

        # 2. Get predictions from the model
        print(f"Step 2: Getting predictions for {len(inputs)} inputs...")
        raw_predictions = []
        try:
            raw_predictions = model_pipeline(inputs, truncation=True, max_length=512)
            print(f"Successfully received {len(raw_predictions)} raw predictions.")
            if raw_predictions:
                print(f"Structure of raw_predictions[0]: {type(raw_predictions[0])}")
                print(f"Content of raw_predictions[0]: {raw_predictions[0]}")
            else:
                print("raw_predictions is empty.")
        except Exception as e_pred:
            print(f"CRITICAL: Error during model prediction in {self.name}: {e_pred!r}")
            return {"error": f"Model prediction failed: {str(e_pred)}", "details": repr(e_pred)}

        # 3. Post-process predictions
        print(f"Step 3: Post-processing {len(raw_predictions)} predictions...")
        predictions = []
        try:
            # Determine if raw_predictions is List[List[Dict]] or List[Dict]
            is_list_of_lists = False
            if raw_predictions and isinstance(raw_predictions[0], list):
                is_list_of_lists = True

            print(f"Processing logic assumes raw_predictions is: {'List[List[Dict]]' if is_list_of_lists else 'List[Dict]'}")

            for i, item_or_list in enumerate(raw_predictions):
                pred_dict = None
                current_input_text = inputs[i] # For debugging messages

                if is_list_of_lists:
                    if not item_or_list: # Inner list is empty
                        print(f"Warning: Model returned empty list for input '{current_input_text}'. Appending -1.")
                        predictions.append(-1)
                        continue
                    pred_dict = item_or_list[0] # Take the first dict from inner list
                else: # Assumed to be List[Dict], so item_or_list is the dict itself
                    pred_dict = item_or_list

                if not pred_dict or not isinstance(pred_dict, dict):
                    print(f"Warning: Prediction item for input '{current_input_text}' is not a valid dictionary: {pred_dict!r}. Appending -1.")
                    predictions.append(-1)
                    continue

                if 'label' not in pred_dict:
                    print(f"Warning: 'label' key missing in prediction dictionary for input '{current_input_text}': {pred_dict!r}. Appending -1.")
                    predictions.append(-1)
                    continue

                label = pred_dict['label']
                print(f"  Processing input '{current_input_text}': raw label = {label!r} (type: {type(label)})")

                if isinstance(label, int):
                    predictions.append(label)
                elif isinstance(label, str):
                    if label.isdigit():
                        predictions.append(int(label))
                    elif model_pipeline.model.config.label2id and label in model_pipeline.model.config.label2id:
                         predictions.append(model_pipeline.model.config.label2id[label])
                    elif label.upper() in ["POSITIVE", "ENTAILED", "LABEL_1", "1"]:
                        predictions.append(1)
                    elif label.upper() in ["NEGATIVE", "CONTRADICTION", "LABEL_0", "0"]:
                        predictions.append(0)
                    elif label.upper() in ["NEUTRAL", "LABEL_2", "2"]:
                        predictions.append(2 if max(references) > 1 else 0)
                    else:
                        numeric_part_default = -1
                        try:
                            numeric_part = label.split('_')[-1]
                            if numeric_part.isdigit():
                                numeric_part_default = int(numeric_part)
                        except: # Keep default -1 if parsing fails
                            pass

                        if numeric_part_default == -1:
                             print(f"Warning: Unhandled string label '{label}' for input '{inputs[i]}'. Appending -1.")
                        predictions.append(numeric_part_default)
                else:
                    print(f"Warning: Unexpected label type {type(label)}: '{label}'. Appending -1.")
                    predictions.append(-1)
            print(f"Successfully processed {len(predictions)} predictions.")
        except Exception as e_label:
            print(f"CRITICAL: Error during label processing: {e_label!r}")
            return {"error": f"Label processing failed: {str(e_label)}", "details": repr(e_label)}

        if len(predictions) != len(references):
            print(f"CRITICAL: Mismatch in number of processed predictions ({len(predictions)}) and references ({len(references)}).")
            return {"error": f"Prediction/reference count mismatch."}

        # Filter out error predictions (-1) if any, before metric calculation, or handle them as incorrect.
        # For simplicity, let's treat -1 as an incorrect prediction if the reference isn't also -1 (which it shouldn't be).
        # This step is important for metrics to compute correctly.
        # Alternatively, filter out pairs where prediction is -1.
        # For now, let's assume metrics can handle them or they are few.
        # A more robust way:
        valid_predictions = []
        valid_references = []
        for p, r in zip(predictions, references):
            if p != -1: # Or any other error indicator you used
                valid_predictions.append(p)
                valid_references.append(r)

        if not valid_predictions: # All predictions failed
            print("Error: No valid predictions were generated to compute metrics.")
            return {"error": "All predictions resulted in an error or unparseable format."}

        print(f"Computing metrics with {len(valid_predictions)} valid predictions and {len(valid_references)} valid references.")
        print(f"Sample valid predictions: {valid_predictions[:5]}")
        print(f"Sample valid references: {valid_references[:5]}")

        # 4. Compute metrics
        results = {}
        try:
            accuracy_metric = evaluate.load("accuracy")
            accuracy_result = accuracy_metric.compute(predictions=valid_predictions, references=valid_references)
            if accuracy_result is not None:
                results["accuracy"] = accuracy_result.get("accuracy")
            else:
                results["accuracy"] = None
                results["accuracy_error"] = "Accuracy computation returned None"
            print(f"Accuracy computation result: {accuracy_result}")

        except Exception as e:
            results["accuracy"] = None
            results["accuracy_error"] = str(e)
            print(f"Error computing accuracy: {e!r}") # Print representation of exception

        try:
            f1_metric = evaluate.load("f1")
            # Determine average strategy based on number of unique labels in references
            num_unique_labels = len(set(valid_references))
            average_strategy = "binary" if num_unique_labels <= 2 else "weighted"
            f1_result = f1_metric.compute(predictions=valid_predictions, references=valid_references, average=average_strategy)
            if f1_result is not None:
                results["f1"] = f1_result.get("f1")
            else:
                results["f1"] = None
                results["f1_error"] = "F1 computation returned None"
            print(f"F1 computation result (strategy: {average_strategy}): {f1_result}")

        except Exception as e:
            results["f1"] = None
            results["f1_error"] = str(e)
            print(f"Error computing F1 score: {e!r}") # Print representation of exception

        print(f"{self.name} results: {results}")
        return results

if __name__ == '__main__':
    # This is for direct testing of the benchmark class, not part of the framework execution.
    # You'd typically run this via examples/run_evaluation.py
    print("Testing ExampleNLUBenchmark directly...")

    # Mock a Hugging Face pipeline
    class MockTextClassificationPipeline:
        def __init__(self, model_config):
            self.task = "text-classification"
            self.model = type('model', (object,), {'config': model_config})()


        def __call__(self, inputs, **kwargs):
            # Simulate model output based on input text
            results = []
            for text_input in inputs if isinstance(inputs, list) else [inputs]:
                if "fantastic" in text_input or "best" in text_input :
                    results.append([{'label': 'LABEL_1', 'score': 0.9}]) # Simulate positive
                elif "not happy" in text_input or "terrible" in text_input:
                    results.append([{'label': 'LABEL_0', 'score': 0.85}]) # Simulate negative
                else:
                    results.append([{'label': 'LABEL_0', 'score': 0.6}]) # Simulate neutral/negative
            return results

    # Mock model config (id2label and label2id are important for label mapping)
    mock_config = {
        "id2label": {0: "LABEL_0", 1: "LABEL_1"},
        "label2id": {"LABEL_0": 0, "LABEL_1": 1}
    }

    mock_pipeline = MockTextClassificationPipeline(model_config=mock_config)

    benchmark = ExampleNLUBenchmark()

    # Test with default (hardcoded) data if dataset loading fails
    print("\n--- Test Case 1: Default parameters (likely fallback to hardcoded data) ---")
    # To force fallback, you could pass invalid dataset_name/config
    # For now, it will try to load glue/mrpc with 5 samples. If that fails, it uses hardcoded.
    results_default = benchmark.run(mock_pipeline, "mock_model_default_params")
    print(f"Results (default params): {results_default}")

    # Test with specific parameters (e.g., more samples, different dataset if available)
    print("\n--- Test Case 2: Custom parameters (e.g., more samples) ---")
    # This will still use the mock pipeline, but tests parameter passing for num_samples
    custom_params = {
        "ExampleNLUBenchmark": {
            "num_samples": 3 # Using a small number for the hardcoded dataset part
        }
    }
    results_custom = benchmark.run(mock_pipeline, "mock_model_custom_params", benchmark_specific_params=custom_params)
    print(f"Results (custom params, 3 samples): {results_custom}")

    # Test with a model that might produce integer labels directly (if pipeline configured that way)
    class MockIntLabelPipeline:
        def __init__(self, model_config):
            self.task = "text-classification"
            self.model = type('model', (object,), {'config': model_config})()

        def __call__(self, inputs, **kwargs):
            results = []
            for text_input in inputs if isinstance(inputs, list) else [inputs]:
                if "fantastic" in text_input or "best" in text_input :
                    results.append([{'label': 1, 'score': 0.9}])
                elif "not happy" in text_input or "terrible" in text_input:
                    results.append([{'label': 0, 'score': 0.85}])
                else:
                    results.append([{'label': 0, 'score': 0.6}])
            return results

    mock_int_pipeline = MockIntLabelPipeline(model_config=mock_config)
    print("\n--- Test Case 3: Pipeline with integer labels ---")
    results_int_labels = benchmark.run(mock_int_pipeline, "mock_model_int_labels")
    print(f"Results (integer labels): {results_int_labels}")

    print("\nExampleNLUBenchmark direct test finished.")
