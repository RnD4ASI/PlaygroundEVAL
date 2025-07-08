from llm_evaluator.benchmarks.abstract_benchmark import AbstractBenchmark
import evaluate # Using evaluate for ROUGE
import random

class ExampleTextGenBenchmark(AbstractBenchmark):
    """
    Example Text Generation benchmark.
    This benchmark uses a few hardcoded prompts and reference continuations.
    It calculates ROUGE scores for the generated text.
    """

    @property
    def name(self) -> str:
        return "ExampleTextGenBenchmark"

    @property
    def description(self) -> str:
        return "A simple example text generation benchmark using hardcoded prompts and ROUGE."

    def run(self, model_pipeline, model_name: str, **kwargs) -> dict:
        """
        Runs the Text Generation benchmark.

        Args:
            model_pipeline: The pre-loaded Hugging Face text-generation pipeline.
            model_name (str): The name of the model being evaluated.
            **kwargs: Additional arguments. Expects `benchmark_specific_params` which can contain
                      parameters for this benchmark, e.g., `num_samples`, `max_new_tokens`.

        Returns:
            A dictionary containing the benchmark results (e.g., ROUGE scores).
        """
        print(f"\nRunning {self.name} for model: {model_name}")
        print(f"Using pipeline: {type(model_pipeline)}")

        benchmark_params = kwargs.get("benchmark_specific_params", {}).get(self.name, {})
        num_samples_to_run = benchmark_params.get("num_samples", 3) # Default to 3 samples
        max_new_tokens = benchmark_params.get("max_new_tokens", 20) # Max tokens for generation

        # 1. Define prompts and reference continuations
        prompts_and_references = [
            {"prompt": "Once upon a time, in a land far away, there lived a",
             "reference": "brave knight who searched for a legendary dragon."},
            {"prompt": "The best way to learn programming is to",
             "reference": "practice coding every day and build projects."},
            {"prompt": "Artificial intelligence is rapidly changing the world by",
             "reference": "automating tasks, enabling new discoveries, and creating new challenges."},
            {"prompt": "To bake a delicious cake, you first need to",
             "reference": "gather all your ingredients, such as flour, sugar, eggs, and butter."},
            {"prompt": "The capital of France is",
             "reference": "Paris, a city known for its art, fashion, and culture."}
        ]

        # Select a subset of samples if num_samples_to_run is less than total
        if len(prompts_and_references) > num_samples_to_run:
            selected_samples = random.sample(prompts_and_references, num_samples_to_run)
        else:
            selected_samples = prompts_and_references
            num_samples_to_run = len(selected_samples) # Adjust if fewer samples available

        print(f"Selected {num_samples_to_run} samples for generation.")

        prompts = [item["prompt"] for item in selected_samples]
        reference_texts = [item["reference"] for item in selected_samples]
        generated_texts = []

        # 2. Generate text using the pipeline
        print(f"Generating text for {len(prompts)} prompts (max_new_tokens={max_new_tokens})...")
        try:
            # Text generation pipeline output is typically List[List[Dict[str, str]]]
            # Each inner list contains generated sequences for one prompt.
            # Each dict has 'generated_text'.
            # We need to ensure the pipeline doesn't include the prompt in the generated_text
            # or handle it appropriately. Some pipelines do, some don't by default.
            # Common params for text-generation pipeline:
            # - max_new_tokens
            # - num_return_sequences
            # -eos_token_id, pad_token_id (might be important for some models)

            # The default behavior for many text-generation pipelines is to include the prompt.
            # We need to extract only the newly generated part.
            pipeline_outputs = model_pipeline(
                prompts,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1, # We only want one generation per prompt for this example
                # Explicitly setting pad_token_id if tokenizer has it and it's not eos_token
                # This can prevent warnings or errors for some models.
                pad_token_id=model_pipeline.tokenizer.pad_token_id if model_pipeline.tokenizer.pad_token_id is not None else model_pipeline.tokenizer.eos_token_id
            )

            for i, output_list in enumerate(pipeline_outputs):
                if output_list and len(output_list) > 0:
                    full_generated_text = output_list[0]['generated_text']
                    # Remove the original prompt from the generated text
                    original_prompt = prompts[i]
                    if full_generated_text.startswith(original_prompt):
                        generated_continuation = full_generated_text[len(original_prompt):].strip()
                    else: # Pipeline might not have included prompt, or something else happened
                        print(f"Warning: Generated text for prompt '{original_prompt}' did not start with the prompt. Using full output.")
                        generated_continuation = full_generated_text.strip()
                    generated_texts.append(generated_continuation)
                    print(f"  Prompt: \"{original_prompt}\" -> Generated: \"{generated_continuation}\"")
                else:
                    print(f"Warning: No text generated for prompt: {prompts[i]}")
                    generated_texts.append("") # Append empty string if no generation

        except Exception as e:
            print(f"CRITICAL: Error during text generation: {e!r}")
            return {"error": f"Text generation failed: {str(e)}", "details": repr(e)}

        if len(generated_texts) != len(reference_texts):
            return {"error": "Mismatch in number of generated texts and reference texts."}

        # 3. Compute ROUGE scores
        results = {}
        print("Computing ROUGE scores...")
        try:
            rouge_metric = evaluate.load('rouge')
            # ROUGE expects lists of strings for predictions and references
            rouge_scores = rouge_metric.compute(predictions=generated_texts, references=reference_texts)

            # Flatten the ROUGE scores for easier reporting (e.g., rouge1, rouge2, rougeL, rougeLsum)
            if rouge_scores:
                for key, value in rouge_scores.items(): # value might be a float or a Score object
                    if hasattr(value, 'mid'): # Handle AggregateScore object from ROUGE for older versions
                        results[key + '_precision'] = value.mid.precision
                        results[key + '_recall'] = value.mid.recall
                        results[key + '_fmeasure'] = value.mid.fmeasure
                    elif isinstance(value, float): # Newer versions might return floats directly
                         results[key] = value
                    else: # Fallback if structure is unexpected
                        results[key] = str(value)
            else:
                results["rouge_error"] = "ROUGE computation returned empty or None."

            print(f"ROUGE scores: {results}")

        except Exception as e:
            print(f"CRITICAL: Error computing ROUGE scores: {e!r}")
            results["rouge_error"] = str(e)
            # Return any partial results if available, or just the error

        return results


if __name__ == '__main__':
    # This is for direct testing of the benchmark class.
    print("Testing ExampleTextGenBenchmark directly...")

    # Mock a Hugging Face text-generation pipeline
    class MockTextGenerationPipeline:
        def __init__(self, tokenizer=None): # Added tokenizer for pad_token_id
            self.task = "text-generation"
            self.tokenizer = tokenizer if tokenizer else self.MockTokenizer()

        class MockTokenizer:
            def __init__(self):
                self.pad_token_id = 0
                self.eos_token_id = 1

        def __call__(self, prompts, max_new_tokens=10, **kwargs):
            outputs = []
            for prompt_text in prompts if isinstance(prompts, list) else [prompts]:
                mock_continuation = " ".join(["mock"] * min(5, max_new_tokens)) # Generate some mock words
                outputs.append([{'generated_text': prompt_text + mock_continuation}])
            return outputs

    mock_pipeline = MockTextGenerationPipeline()
    benchmark = ExampleTextGenBenchmark()

    print("\n--- Test Case 1: Default parameters ---")
    results_default = benchmark.run(mock_pipeline, "mock_text_gen_model")
    print(f"Results (default params): {results_default}")

    print("\n--- Test Case 2: Custom parameters (1 sample, 5 tokens) ---")
    custom_params = {
        "ExampleTextGenBenchmark": {
            "num_samples": 1,
            "max_new_tokens": 5
        }
    }
    results_custom = benchmark.run(mock_pipeline, "mock_text_gen_model_custom", benchmark_specific_params=custom_params)
    print(f"Results (custom params): {results_custom}")

    print("\nExampleTextGenBenchmark direct test finished.")
