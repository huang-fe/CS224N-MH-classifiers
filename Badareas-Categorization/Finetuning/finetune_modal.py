import modal
import os
from transformers import AutoModelForSequenceClassification
import torch

# Define Modal app
app = modal.App("badareas-classifier")

image = ( 
  modal.Image.from_registry("pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime")
  .pip_install_from_requirements("requirements.txt") 
)

# class WeightedBERTForSequenceClassification(AutoModelForSequenceClassification):
#     def __init__(self, model_name, num_labels, class_weights):
        
#         # ✅ Load Pretrained Weights Separately
#         self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, torch_dtype="auto")
        
#         # ✅ Move class weights to the correct device
#         self.class_weights = class_weights.to(self.model.device)
        
#         # ✅ Define weighted loss function
#         self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
    
#     def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
#         outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs)
#         logits = outputs.logits
        
#         loss = None
#         if labels is not None:
#             loss = self.loss_fn(logits, labels)
        
#         return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


@app.function(
        image=image,
        gpu="A100-80GB", 
        timeout=86400,
        secrets=[modal.Secret.from_name("huggingface-secret")])
def train_model():
    from huggingface_hub import login, HfFolder
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback #BitsAndBytesConfig,
    from huggingface_hub import HfFolder
    import numpy as np
    #from sklearn.metrics import f1_score
    from transformers.utils import logging
    #from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    #from trl import SFTTrainer
    from sklearn.utils.class_weight import compute_class_weight
    from torch.nn import CrossEntropyLoss
    import torch
    import evaluate

    # Login to Hugging Face
    print("HF_TOKEN:", os.getenv("HF_TOKEN"))
    login(token= os.environ["HF_TOKEN"])
    print("Logged into Hugging Face.")

    logging.set_verbosity_info()
    logger = logging.get_logger("transformers")
    logger.info("LOGGER")

    def prepare_input_text(example):
        # Convert the last two items of input list to a single text
        return {
            'text': "Seeker: " + example["seeker-prompt"] + "\n Helper: " + example["last-helper-response"],
            **{k:v for k,v in example.items() if k != 'input'}  # Keep other fields
        }

    def tokenize_function(example):
        # !!!!!!!! "example["text"]" should be the last helper prompt !!!!!!!!!!
        # text = "Seeker: " + example["seeker-prompt"] + "\n Helper: " + example["last-helper-response"]
        # print("Within tokenize_function: ", example['text'])
        return tokenizer(example['text'], padding="max_length", truncation=True, max_length=512)

    # load data
    dataset_id = "huangfe/feedback_qesconv_badareas_questions_reflections" #"youralien/feedback_qesconv_16wayclassification"
    dataset = load_dataset(dataset_id, split="train")

    # Split dataset further (80%) and validation (20%)
    split_dataset = dataset.train_test_split(test_size=0.2)
    # Apply the preprocessing
    split_dataset = split_dataset.map(prepare_input_text)
    split_dataset['train'][0]
    print(f"Train dataset size: {len(split_dataset['train'])}")
    print(f"Test dataset size: {len(split_dataset['test'])}")
    # train_dataset = split_dataset['train']
    # eval_dataset = split_dataset['test']

    # apply tokenization
    # train_dataset = train_dataset.map(tokenize_function, batched=True)
    # eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    skill = "reflections"
    model_id = "bert-base-cased"
    model_name = "bert" # "Llama2-7b"

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.model_max_length = 512 # set model_max_length to 512 as prompts are not longer than 1024 tokens


    classifier_name = f"{skill}-badarea-suboptimal" #"{skill}-badareas-shouldHave" "{skill}-badareas-shouldNotHave"
    print("Classifier name: ", classifier_name)
    
    # !!!!! CHANGE THESE COLUMN NAMES BASED ON THE TABLE COLUMN NAMES !!!!!
    suffixes = ["-badarea-suboptimal", "-badarea-shouldhave", "-badarea-shouldnothave" ]
               #"Validation-badareas", "Empathy-badareas", "Questions-badareas", "Suggestions-badareas", "Self-disclosure-badareas", "Structure-badareas", "Professionalism-badareas"]
    skill_categories = [
        # "empathy", "validation", "suggestion", 
        "questions", "reflections"
        # "professionalism", "self-disclosure", "structure"
    ]
    columns = []
    for s in skill_categories:
        for suffix in suffixes:
            columns.append(s+suffix)
    print("all columns: ", columns)

    columns_to_remove = [f"{c}" for c in columns if c != classifier_name]
    cols_to_remove = ['Entry', 'alternative-response', 'seeker-prompt', 'last-helper-response']
    cols_to_remove.extend(columns_to_remove)
    print("columns to remove: ", columns)
    # if which_class in split_dataset["train"].features.keys():
    split_dataset =  split_dataset.rename_column(classifier_name, "labels") # to match Trainer
    tokenized_dataset = split_dataset.map(tokenize_function, batched=True, remove_columns=cols_to_remove)
    #check 
    # print("Example from Tokenized_dataset: ", tokenized_dataset["train"][0])
    # print("All Features: ", tokenized_dataset["train"].features) 

    # # Compute class weights DOESNT WORK
    # labels = np.array(tokenized_dataset["train"]["labels"]) 
    # class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    # class_weights = torch.tensor(class_weights, dtype=torch.float)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # class_weights = class_weights.to(device)
    # loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # load model    
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2, torch_dtype="auto")
    # weighted_model = WeightedBERTForSequenceClassification(
    #     model_name=model_id, 
    #     num_labels=2, 
    #     class_weights=class_weights
    # )
    model.config.pad_token_id = tokenizer.pad_token_id #model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    save_directory = f"{model_name}-{classifier_name}-classifier-class-weights"
    training_args = TrainingArguments(
        output_dir=save_directory,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=15,  
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=2,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # compute_loss_func=loss_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    )

    trainer.train()
    # model.save_pretrained(save_directory)
    # tokenizer.save_pretrained(save_directory)

    tokenizer.save_pretrained(save_directory)
    print("after trainer")
    trainer.create_model_card()
    trainer.push_to_hub()

    print("Model & tokenizer saved")

if __name__ == "__main__":
    with app.run():
        train_model()