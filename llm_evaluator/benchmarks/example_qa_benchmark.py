from llm_evaluator.benchmarks.abstract_benchmark import AbstractBenchmark
import evaluate # For SQuAD metric or manual F1/EM
import random
import re
import string
from collections import Counter

class ExampleQABenchmark(AbstractBenchmark):
    """
    Example Question Answering (QA) benchmark.
    This benchmark uses a few hardcoded context-question-answer samples.
    It calculates F1 and Exact Match (EM) scores for the predicted answer spans.
    """

    @property
    def name(self) -> str:
        return "ExampleQABenchmark"

    @property
    def description(self) -> str:
        return "A simple example QA benchmark using hardcoded samples and F1/EM metrics."

    def _normalize_answer(self, s: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _compute_f1(self, prediction: str, ground_truth: str) -> float:
        pred_tokens = self._normalize_answer(prediction).split()
        truth_tokens = self._normalize_answer(ground_truth).split()

        if not pred_tokens or not truth_tokens:
            return 0.0

        common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
        num_same = sum(common_tokens.values())

        if num_same == 0:
            return 0.0

        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def _compute_exact_match(self, prediction: str, ground_truth: str) -> float:
        return float(self._normalize_answer(prediction) == self._normalize_answer(ground_truth))

    def run(self, model_pipeline, model_name: str, **kwargs) -> dict:
        """
        Runs the QA benchmark.

        Args:
            model_pipeline: The pre-loaded Hugging Face question-answering pipeline.
            model_name (str): The name of the model being evaluated.
            **kwargs: Additional arguments, e.g., `benchmark_specific_params`.

        Returns:
            A dictionary containing F1 and Exact Match scores.
        """
        print(f"\nRunning {self.name} for model: {model_name}")
        print(f"Using pipeline: {type(model_pipeline)}")

        benchmark_params = kwargs.get("benchmark_specific_params", {}).get(self.name, {})
        num_samples_to_run = benchmark_params.get("num_samples", 3)

        qa_samples = [
            {
                "context": "The Apollo 11 mission landed the first humans on the Moon. Neil Armstrong was the first man to walk on the moon, followed by Buzz Aldrin.",
                "question": "Who was the first person to walk on the Moon?",
                "answers": ["Neil Armstrong"] # For SQuAD-like metrics, this would be a list of possible answers
            },
            {
                "context": "Python is a versatile and widely used programming language, known for its readability and extensive libraries. It was created by Guido van Rossum.",
                "question": "Who created Python?",
                "answers": ["Guido van Rossum"]
            },
            {
                "context": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
                "question": "Where is the Eiffel Tower located?",
                "answers": ["Paris", "Champ de Mars in Paris, France", "Paris, France"]
            },
            {
                "context": "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. Its elevation is 8,848.86 metres.",
                "question": "What is the elevation of Mount Everest?",
                "answers": ["8,848.86 metres", "8,848.86 meters"]
            },
             {
                "context": "The quick brown fox jumps over the lazy dog. This sentence contains all letters of the alphabet.",
                "question": "What does the quick brown fox jump over?",
                "answers": ["the lazy dog", "lazy dog"]
            }
        ]

        if len(qa_samples) > num_samples_to_run:
            selected_samples = random.sample(qa_samples, num_samples_to_run)
        else:
            selected_samples = qa_samples
            num_samples_to_run = len(selected_samples)

        print(f"Selected {num_samples_to_run} QA samples.")

        all_f1_scores = []
        all_em_scores = []

        # The SQuAD metric from `evaluate` can handle this more robustly,
        # but for a simple example, we'll do manual calculation per prediction.
        # squad_metric = evaluate.load("squad") # Alternative

        print(f"Getting predictions for {len(selected_samples)} QA pairs...")
        for sample in selected_samples:
            context = sample["context"]
            question = sample["question"]
            reference_answers = sample["answers"] # List of acceptable answers

            try:
                # QA pipeline returns a dict like {'score': 0.99, 'start': 30, 'end': 42, 'answer': 'Neil Armstrong'}
                prediction_output = model_pipeline(question=question, context=context)
                predicted_answer = prediction_output['answer']
                print(f"  Q: \"{question}\" -> Predicted A: \"{predicted_answer}\" (Correct: \"{reference_answers[0]}\")")

                # For multiple reference answers, compute F1/EM against each and take the max.
                current_max_f1 = 0.0
                current_max_em = 0.0
                for ref_ans in reference_answers:
                    current_max_f1 = max(current_max_f1, self._compute_f1(predicted_answer, ref_ans))
                    current_max_em = max(current_max_em, self._compute_exact_match(predicted_answer, ref_ans))

                all_f1_scores.append(current_max_f1)
                all_em_scores.append(current_max_em)

            except Exception as e:
                print(f"CRITICAL: Error during QA prediction or processing for question '{question}': {e!r}")
                # Append 0 scores for this sample in case of error to maintain list lengths
                all_f1_scores.append(0.0)
                all_em_scores.append(0.0)

        results = {}
        if all_f1_scores: # Ensure not empty
            results["average_f1"] = sum(all_f1_scores) / len(all_f1_scores)
        else:
            results["average_f1"] = 0.0
            results["f1_error"] = "No F1 scores were computed."

        if all_em_scores: # Ensure not empty
            results["average_exact_match"] = sum(all_em_scores) / len(all_em_scores)
        else:
            results["average_exact_match"] = 0.0
            results["em_error"] = "No EM scores were computed."

        print(f"QA Benchmark results: {results}")
        return results

if __name__ == '__main__':
    print("Testing ExampleQABenchmark directly...")

    class MockQAPipeline:
        def __init__(self):
            self.task = "question-answering"

        def __call__(self, question, context, **kwargs):
            # Simple mock logic
            if "first person" in question and "Moon" in context:
                return {'score': 0.95, 'start': 50, 'end': 63, 'answer': 'Neil Armstrong'}
            elif "created Python" in question:
                return {'score': 0.92, 'start': 100, 'end': 115, 'answer': 'Guido van Rossum'}
            elif "Eiffel Tower located" in question:
                 return {'score': 0.88, 'start': 0, 'end': 5, 'answer': 'Paris'} # Exact match
            elif "elevation" in question and "Everest" in context:
                 return {'score': 0.90, 'start': 0, 'end': 10, 'answer': '8,848.86 metres'}
            else:
                return {'score': 0.1, 'start': 0, 'end': 10, 'answer': 'mock answer'}

    mock_pipeline = MockQAPipeline()
    benchmark = ExampleQABenchmark()

    print("\n--- Test Case 1: Default parameters (3 samples) ---")
    results_default = benchmark.run(mock_pipeline, "mock_qa_model")
    print(f"Results (default params): {results_default}")

    # Test normalization
    # print("\nNormalization test:")
    # print(f"'The   Best  answer.': {benchmark._normalize_answer('The   Best  answer.')}")
    # print(f"' an apple a day '   : {benchmark._normalize_answer(' an apple a day ')}")

    # Test F1/EM
    # print("\nF1/EM test:")
    # pred = "Neil Armstrong"
    # ref = "Neil Armstrong"
    # print(f"Pred: '{pred}', Ref: '{ref}' -> EM: {benchmark._compute_exact_match(pred, ref)}, F1: {benchmark._compute_f1(pred, ref)}")
    # pred = "armstrong"
    # print(f"Pred: '{pred}', Ref: '{ref}' -> EM: {benchmark._compute_exact_match(pred, ref)}, F1: {benchmark._compute_f1(pred, ref)}")
    # pred = "Neil   Armstrong."
    # print(f"Pred: '{pred}', Ref: '{ref}' -> EM: {benchmark._compute_exact_match(pred, ref)}, F1: {benchmark._compute_f1(pred, ref)}")
    # pred = "Someone else"
    # print(f"Pred: '{pred}', Ref: '{ref}' -> EM: {benchmark._compute_exact_match(pred, ref)}, F1: {benchmark._compute_f1(pred, ref)}")


    print("\nExampleQABenchmark direct test finished.")
