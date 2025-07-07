The standard way to judge a Hugging Face-style fine-tuned model is to run it on a public benchmark dataset that matches the task (e.g. GLUE, SQuAD, MTEB, TruthfulQA), compute task-appropriate metrics with a library such as 🤗 evaluate, lighteval, or lm-eval-harness, and then compare the scores to the published leaderboard or a strong baseline; the open-source repos listed below automate almost every part of that loop.

⸻

1  Pick the right benchmark dataset

Task	Widely-used benchmark	Typical citation
General NLU	GLUE / SuperGLUE	￼
Extractive QA	SQuAD v1/v2 (included in datasets)	￼
Text generation & reasoning	BIG-bench	￼
Safety, truthfulness	TruthfulQA, Risk Bench (Open-LLM leaderboard)	￼
Sentence/embedding quality	MTEB	￼
Holistic multi-metric view	HELM	￼

Choose the subset that mirrors your fine-tuning objective; e.g. a NER model belongs on CoNLL-2003, while an instruction-tuned LLM should hit multi-skill suites such as HELM or the Open-LLM Leaderboard.

⸻

2  Choose metrics and make them explicit

Below are the most common scalar metrics, with precise formulas (all counts over the evaluation set):

\text{Accuracy}= \frac{TP+TN}{TP+FP+TN+FN}

\text{Precision}= \frac{TP}{TP+FP},\qquad
\text{Recall}= \frac{TP}{TP+FN}

\text{F1}= \frac{2\,\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}}
	•	TP, FP, TN, FN – true/false positives/negatives.
	•	Generation tasks substitute token overlap metrics (e.g. BLEU, ROUGE-L) or log-likelihood/perplexity.
	•	Embedding tasks use Spearman/Pearson r or MRR.

All of these are implemented in the libraries below; for sequence labelling there is a ready-made seqeval metric  ￼, and for classical tasks you can fall back on scikit-learn.metrics  ￼ ￼.

⸻

3  Python tooling that does the heavy lifting

3.1 🤗 evaluate (and datasets)

import evaluate, datasets, transformers

metric = evaluate.load("glue", "mrpc")          # loads the GLUE MRPC scorer
dataset = datasets.load_dataset("glue", "mrpc", split="validation")
model = transformers.AutoModelForSequenceClassification.from_pretrained("your-finetuned-model")
pipe = transformers.TextClassificationPipeline(model=model, tokenizer="your-tokenizer", device=0)

def compute(preds, refs):
    return metric.compute(predictions=preds, references=refs)

preds = [pipe(q1 + " [SEP] " + q2)[0]['label'] for q1,q2 in zip(dataset['sentence1'],dataset['sentence2'])]
print(compute(preds, dataset['label']))

evaluate brings > 100 ready metrics, stream-safe logging, and seamless multiprocess sharding  ￼ ￼.

3.2 lighteval

A newer Hugging Face toolkit that wraps multiple back-ends (Transformers, vLLM, OpenAI API) and pushes results straight to the Hub with one CLI command  ￼ ￼:

lighteval accelerate "model_name=your-finetuned-model" "leaderboard|truthfulqa:mc|0|0"

3.3 lm-eval-harness

The backbone of the Open-LLM leaderboard; supports zero-, one- and few-shot templates, streaming tokens, and batched GPU inference  ￼ ￼:

python main.py \
  --model hf \
  --model_args pretrained=your-finetuned-model \
  --tasks hellaswag,arc_easy \
  --device cuda:0

3.4 OpenAI Evals

Task registry plus YAML-style prompts; useful if you want parity with GPT-x baselines  ￼.

3.5 HELM framework

If you need a multi-metric, long-form, human-readable report across dozens of scenarios, run the Stanford HELM pipeline (docker / Python)  ￼.

⸻

4  End-to-end evaluation workflow
	1.	Load the fine-tuned model (HF AutoModel* or API wrapper).
	2.	Select benchmark + split (validation/test).
	3.	Infer with deterministic decoding (e.g. temperature=0, top_p=1) to keep scores reproducible.
	4.	Compute metrics via one of the libraries above; log run-time and GPU hours as extra columns.
	5.	Compare against:
	•	the original base model,
	•	published leaderboard baselines, and
	•	a strong proprietary model (optional).
	6.	Track runs in MLFlow/W&B or push JSON to the Hugging Face Hub so others can reproduce.

⸻

5  Established open-source repos at a glance

Repo	Scope	Stars	Quick-start command
huggingface/evaluate	Metrics collection, any domain	3 k	evaluate.load("rouge")  ￼
huggingface/lighteval	Fast LLM evaluation, Hub integration	1 k	lighteval accelerate ...  ￼
EleutherAI/lm-eval-harness	Generative LMs, 100+ tasks	3 k	python main.py --tasks ...  ￼
stanford-crfm/helm	Holistic, multi-metric	2 k	python -m helm.benchmark.run  ￼
embeddings-benchmark/mteb	Embedding quality	2.7 k	python mteb.py --task all  ￼
openai/evals	Eval registry + chat models	8 k	oaiexec eval.yaml  ￼

All of them are pure-Python, pip-installable, and actively maintained in 2025.

⸻

Final checklist
	•	Match task ↔ metric (BLEU is meaningless for classification).
	•	Deterministic decoding for comparability.
	•	Document dataset version & prompt template.
	•	Report ≥ 2 complementary metrics (e.g. Accuracy and F1).
	•	Log runtime + hardware; a faster model with the same score may be preferable in production.

Follow this pattern and your fine-tuned Hugging Face model will be evaluated in a way the community recognises—and you can publish the numbers straight to a leaderboard.