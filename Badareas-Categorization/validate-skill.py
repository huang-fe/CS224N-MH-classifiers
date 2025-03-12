import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from PAIR_model import run_model
from cross_scorer_model import CrossScorerCrossEncoder
# from datasets import load_dataset, concatenate_datasets

device = "cuda" if torch.cuda.is_available() else "cpu"

# dataset_id = "youralien/feedback_qesconv_16wayclassification"
# raw_dataset = load_dataset(dataset_id, split="train")
# split_dataset = raw_dataset.train_test_split(test_size=0.20, seed=0)
# split_dataset['train'][0]

# PAIR Reflections Classification #

print("Load pre-trained weights")
scorer_path = "Weights/reflection_scorer_weight.pt"  # Your weight file
c_ckpt = torch.load(scorer_path, weights_only=True,map_location=torch.device(device))

print("Load tokenizer and model")
model_name = "roberta-base"
encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
tokenizer = AutoTokenizer.from_pretrained(model_name)
cross_scorer = CrossScorerCrossEncoder(encoder).to(device)
cross_scorer.load_state_dict(c_ckpt, strict=False) #["model_state_dict"])
cross_scorer.eval()  # Set model to evaluation mode

# def load_dataset(file_path: str):
#     """Load JSON dataset"""
#     with open(file_path, "r", encoding="utf-8") as f:
#         return json.load(f)

# Model inference for a single skill
def predict_skill(prompt: str, response: str, threshold: float = 0.4, percentile: float = 75):
    # score returned by PAIR = continuous between 0-1, 0: No reflection, 0.5: Simple reflection, 1: Complex reflection
    score = run_model(cross_scorer, tokenizer, prompt, response)
    return score>threshold

# Model validation using `predict_skill()`
def validate_model(dataset, percentile=75, label_smoothing=0.1):
    """Evaluate model's skill prediction accuracy with label smoothing."""

    metrics = {"TP": 0, "FP": 0, "FN": 0}

    for example in tqdm(dataset, desc="Processing examples", unit="example"):
        seeker_prompt = example['input'][-2:-1]
        helper_response = example['input'][-1]
        alternative_response = example['alternative']
        
        is_present_original = example['annotations']['original-hasreflections']
        is_present_alternative = example['annotations']['alternative-hasreflections']

        is_predicted = predict_skill(seeker_prompt, helper_response, percentile=percentile)

        # Track per-skill metrics with label smoothing
        if is_present_original and is_predicted:
            metrics["TP"] += 1  
        elif not is_present_original and is_predicted:
            metrics["FN"] += 1 - label_smoothing  # Reduce FN penalty
        elif is_present_original and not is_predicted:
            metrics["FP"] += 1 - label_smoothing  # Reduce FP penalty

        is_predicted = predict_skill(seeker_prompt, alternative_response, percentile=percentile)

        if is_present_alternative and is_predicted:
            metrics["TP"] += 1  
        elif not is_present_alternative and is_predicted:
            metrics["FN"] += 1 - label_smoothing  # Reduce FN penalty
        elif is_present_alternative and not is_predicted:
            metrics["FP"] += 1 - label_smoothing  # Reduce FP penalty

        # print(f"TEXT: {text[:50]}...")
        # print(f"Skill Predictions: {skill_scores}\n")

    tp, fp, fn = metrics["TP"], metrics["FP"], metrics["FN"]
    precision_skill = tp / (tp + fp) if (tp + fp) else 0
    recall_skill = tp / (tp + fn) if (tp + fn) else 0
    f1_skill = (2 * precision_skill * recall_skill) / (precision_skill + recall_skill) if (precision_skill + recall_skill) else 0
    return {"Precision": precision_skill, "Recall": recall_skill, "F1-score": f1_skill}


# -------------------- MAIN DRIVER CODE --------------------

if __name__ == "__main__":
    dataset = load_dataset("./test.json")
    formatted_dataset = format_dataset(dataset)

    # Run validation with the fine-tuned model
    results = validate_model(dataset=formatted_dataset, percentile=75)

    # Save results
    output_file = "skill_validation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\u2705 Validation results saved to {output_file}")