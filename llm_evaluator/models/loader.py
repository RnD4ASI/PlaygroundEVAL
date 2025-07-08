# Handles loading Hugging Face models and tokenizers

from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, pipeline, AutoModelForCausalLM, AutoModelForQuestionAnswering
import torch

SUPPORTED_TASK_MODELS = {
    "text-classification": AutoModelForSequenceClassification,
    "text-generation": AutoModelForCausalLM,
    "question-answering": AutoModelForQuestionAnswering,
    # Add other task-specific AutoModel classes here
}

def load_model_and_tokenizer(model_name_or_path: str, task: str = "text-classification", device: str = None):
    """
    Loads a Hugging Face model and tokenizer.

    Args:
        model_name_or_path (str): The name or path of the Hugging Face model.
        task (str): The task for which the model is being loaded.
                      This helps in selecting the correct AutoModel class.
                      Supported tasks: "text-classification", "text-generation", "question-answering".
        device (str, optional): The device to load the model on (e.g., "cuda:0", "cpu").
                                If None, defaults to "cuda" if available, else "cpu".

    Returns:
        A Hugging Face pipeline object.

    Raises:
        ValueError: If the task is not supported.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {model_name_or_path} for task: {task} on device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    except Exception as e:
        print(f"Error loading tokenizer for {model_name_or_path}: {e}")
        raise

    model_class = SUPPORTED_TASK_MODELS.get(task)
    if not model_class:
        raise ValueError(
            f"Unsupported task: {task}. Supported tasks are: {list(SUPPORTED_TASK_MODELS.keys())}"
        )

    try:
        model = model_class.from_pretrained(model_name_or_path)
    except Exception as e:
        print(f"Error loading model {model_name_or_path} with {model_class}: {e}")
        # Fallback for models that might not strictly conform to AutoModelForSequenceClassification etc.
        # This is a common issue if the model config doesn't specify the model type correctly for the task.
        print(f"Attempting to load with AutoModel...")
        try:
            # This is a more generic loader but might not have task-specific heads.
            # For evaluation, specific heads are often necessary.
            model = AutoModel.from_pretrained(model_name_or_path)
        except Exception as auto_model_e:
            print(f"Error loading model with AutoModel as well: {auto_model_e}")
            raise auto_model_e


    # For pipelines, device can be specified as an integer for GPU index
    pipeline_device = -1
    if "cuda" in device:
        if ":" in device: # "cuda:0"
            pipeline_device = int(device.split(":")[1])
        else: # "cuda"
            pipeline_device = 0 # Default to GPU 0

    # Ensure tokenizer has a pad_token if it's missing (common for some models)
    if tokenizer.pad_token is None and task in ["text-classification", "text-generation"]: # Tasks that often require padding
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        print(f"Tokenizer pad_token was None. Set to eos_token: {tokenizer.eos_token}")


    try:
        # The pipeline handles moving the model to the specified device.
        model_pipeline = pipeline(
            task,
            model=model,
            tokenizer=tokenizer,
            device=pipeline_device if pipeline_device !=-1 else "cpu" # pipeline expects int for gpu, or string 'cpu'
        )
    except Exception as e:
        print(f"Error creating pipeline for task {task} with model {model_name_or_path}: {e}")
        print("This might be due to a mismatch between the model architecture and the specified task,")
        print("or the model not having a default head for the task.")
        print("For text-generation, ensure you're using a model like GPT-2, LLaMA, etc.")
        print("For classification, ensure the model has a sequence classification head.")
        raise

    print(f"Model {model_name_or_path} and tokenizer loaded successfully into a pipeline for task '{task}'.")
    return model_pipeline
