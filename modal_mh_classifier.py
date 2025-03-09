import modal
import os

# Define Modal app
app = modal.App("llama3-classifier")

# Reference a persistent volume
VOLUME_NAME = "MH-dataset"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Define Modal Image with necessary dependencies
image = ( 
#   modal.Image.from_registry("nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10") #from modal example
  modal.Image.from_registry("pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime")
  # modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04") #from GPT
  # modal.Image.debian_slim()
  .apt_install("gcc", "g++", "make")
  .pip_install_from_requirements("req.txt") 
)
# image = (
#     modal.Image.from_registry("pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel")  # Use 'devel' for NVCC support
#     .env({"CUDA_HOME": "/usr/local/cuda"})  # Set CUDA_HOME
#     .pip_install(
#         "torch==2.5.1",
#         "tensorboard",
#         "flash-attn",
#         "setuptools<71.0.0",
#         "scikit-learn",
#         "transformers",
#         "datasets",
#         "tokenizers",
#         "accelerate",
#         "hf-transfer",
#     )
# )

@app.function(
        image=image,
        volumes={f"/vol/{VOLUME_NAME}": volume},
        gpu="A100-80GB", 
        timeout=86400, 
        secrets=[modal.Secret.from_name("huggingface-secret")])
def train_model():
    from huggingface_hub import login, HfFolder
    from datasets import load_dataset, load_from_disk
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, BitsAndBytesConfig
    from huggingface_hub import HfFolder
    import numpy as np
    from sklearn.metrics import f1_score
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    #from trl import SFTTrainer
    import torch

    # Login to Hugging Face
    login(token=os.environ["HF_TOKEN"])

    # Function to download and store dataset in Modal Volume
    dataset_id = "youralien/feedback_qesconv_16wayclassification"
    
    # Define dataset storage path
    dataset_path = "/datasets/dataset.arrow"

    # Check if dataset already exists
    if not os.path.exists(dataset_path):
        print("Downloading dataset...")
        dataset = load_dataset(dataset_id, split="train")
        dataset.save_to_disk(dataset_path)  # Save dataset to volume
        print("Dataset saved to volume.")
    else:
        print("Dataset already stored in volume.")

    # Dataset id from huggingface.co/dataset
    # dataset_id = "youralien/feedback_qesconv_16wayclassification"

    # # Load raw dataset
    # raw_dataset = load_dataset(dataset_id, split="train") # happens to be called train

    # Load dataset from the volume instead of memory
    dataset_path = "/datasets/dataset.arrow"
    raw_dataset = load_from_disk(dataset_path)

    # print(f"Raw dataset size: {len(raw_dataset)}")
    split_dataset = raw_dataset.train_test_split(test_size=0.2) # 80-20 split
    # print(f"Train dataset size: {len(split_dataset['train'])}")
    # print(f"Test dataset size: {len(split_dataset['test'])}")
    # split_dataset['train'][0]

    def prepare_input_text(example):
        # Convert the last two items of input list to a single text
        return {
            'text': "\n".join(example['input'][-2:]),
            **{k:v for k,v in example.items() if k != 'input'}  # Keep other fields
        }

    # Apply the preprocessing
    split_dataset = split_dataset.map(prepare_input_text)
    split_dataset['train'][0]

    # Model id to load the tokenizer
    # model_id = "meta-llama/Llama-2-7b-hf"
    model_id = "meta-llama/Llama-3.1-8b"
    model_name = "Llama3.1-8b" # "Llama2-7b"

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.model_max_length = 512 # set model_max_length to 512 as prompts are not longer than 1024 tokens
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.pad_token_id = tokenizer.eos_token_id
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Tokenize helper function
    def tokenize(batch):
        # return tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt") 
        return tokenizer(batch['text'], padding='max_length', truncation=True, return_tensors="pt")


    which_class = "Empathy-goodareas"
    #which_class = "Questions-badareas" # "Suggestions-badareas" # "Reflections-badareas" # "Empathy-badareas"
    SKILL_OPTIONS = ["Reflections", "Validation", "Empathy", "Questions", "Suggestions", "Self-disclosure", "Structure", "Professionalism"]
    goodareas_to_ignore = [f"{skill}-goodareas" for skill in SKILL_OPTIONS if f"{skill}-goodareas" != which_class]
    badareas_to_ignore = [f"{skill}-badareas" for skill in SKILL_OPTIONS if f"{skill}-badareas" != which_class]
    cols_to_remove = ['conv_index', 'helper_index', 'input', 'text']
    cols_to_remove.extend(goodareas_to_ignore)
    cols_to_remove.extend(badareas_to_ignore)
    if which_class in split_dataset["train"].features.keys():
        split_dataset =  split_dataset.rename_column(which_class, "labels") # to match Trainer
    tokenized_dataset = split_dataset.map(tokenize, batched=True, remove_columns=cols_to_remove)

    tokenized_dataset["train"].features.keys()
    # dict_keys(['labels', 'input_ids', 'attention_mask'])

    # Prepare model labels - useful for inference
    labels = ["not selected", "selected"]
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Enable 4-bit quantization
        bnb_4bit_compute_dtype=torch.float16,  # Use fp16 for computation
        bnb_4bit_use_double_quant=True,  # Double quantization (saves memory)
        bnb_4bit_quant_type="nf4"  # Normalized Float 4 (better for LLaMA)
    )
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=False,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype="float16",
    # )
    # LoRA Configuration
    lora_config = LoraConfig(
        r=8,                      # Rank of LoRA matrices (trade-off: lower = less memory but worse adaptation)
        lora_alpha=16,            # Scaling factor
        lora_dropout=0.1,        # Regularization dropout
        bias="none",              # No bias tuning
        task_type=TaskType.SEQ_CLS       # Sequence Classification task
    )

    # Download the model from huggingface.co/models
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels, label2id=label2id, id2label=id2label, quantization_config=bnb_config, )
    #device_map="auto", load_in_8bit=True,)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)
    # print("device = ", device)
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Show trainable params

    model.config.pad_token_id = tokenizer.pad_token_id #model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))  # Adjust model embeddings for llama 2 for padding problems
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Metric helper method
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        score = f1_score(
                labels, predictions, labels=labels, pos_label=1, average="weighted"
            )
        print(f"Computed F1 Score: {score}")
        return {"eval_f1": float(score) if score == 1 else score}
    
    # Define training args
    training_args = TrainingArguments(
        output_dir= f"{model_name}-{which_class}-classifier",        
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4, # part of lora
        learning_rate=5e-5,
        num_train_epochs=5,
        bf16=True, # bfloat16 training #fp16=True,  # Use mixed precision
        optim="adamw_torch_fused", # improved optimizer
        # logging & evaluation strategies
        logging_strategy="steps",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        # use_mps_device=True, # mps device is a mac thing
        label_names=labels,
        metric_for_best_model="f1",
        # push to hub parameters
        report_to="tensorboard",
        push_to_hub=True,
        hub_strategy="every_save",
        hub_token=HfFolder.get_token(),        
    )
    # Create a Trainer instance
    print("Creating trainer instance")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )
    # trainer = SFTTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_dataset["train"],
    #     eval_dataset=tokenized_dataset["test"],
    #     compute_metrics=compute_metrics,
    #     tokenizer=tokenizer,
    #     peft_config=lora_config,
    # )
    print("Before Trainer.train()")
    trainer.train()
    tokenizer.save_pretrained(f"{model_name}-{which_class}-classifier")
    print("after trainer")
    trainer.create_model_card()
    trainer.push_to_hub()


# @app.function(image=image,gpu="A100-80GB")
# def classify_text(seeker: str, helper: str):
#     from transformers import pipeline

#     model_name = "Llama2-Empathy-goodareas-classifier"
#     classifier = pipeline("sentiment-analysis", model=model_name, device=0)

#     sample = f"Seeker: {seeker}\nHelper: {helper}"
#     pred = classifier(sample)
#     return int(pred[0]['label'] == 'selected')

if __name__ == "__main__":
    with app.run():
        train_model()