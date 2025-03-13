import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from PAIR_model import run_model
from cross_scorer_model import CrossScorerCrossEncoder
# from datasets import load_dataset, concatenate_datasets

def load_dataset(file_path: str):
    """Load JSON dataset"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Model inference for a single skill
def predict_skill(prompt: str, response: str, threshold: float = 0.1):
    # score returned by PAIR = continuous between 0-1, 0: No reflection, 0.5: Simple reflection, 1: Complex reflection
    score = run_model(prompt, response)
    return score[0] > threshold

# Model validation using `predict_skill()`
def validate_model(dataset, label_smoothing=0.1):
    """Evaluate model's skill prediction accuracy with label smoothing."""

    metrics = {"TP": 0, "FP": 0, "FN": 0}
    correct = 0

    for example in tqdm(dataset, desc="Processing examples", unit="example"):
        seeker_prompt = example['input'][-2].removeprefix("Seeker: ")
        helper_response = example['input'][-1].removeprefix("Helper: ")
        alternative_response = example['annotations']['alternative']
        
        is_present_original = example['annotations']['original-hasreflection']
        is_present_alternative = example['annotations']['alternative-hasreflection']

        is_predicted = predict_skill(seeker_prompt, helper_response)

        # Track per-skill metrics with label smoothing
        if is_present_original == is_predicted: correct+=1
        if is_present_original and is_predicted:
            metrics["TP"] += 1  
        elif not is_present_original and is_predicted:
            metrics["FP"] += 1 - label_smoothing  # Reduce FN penalty
        elif is_present_original and not is_predicted:
            metrics["FN"] += 1 - label_smoothing  # Reduce FP penalty

        is_predicted = predict_skill(seeker_prompt, alternative_response)

        if is_present_original == is_predicted: correct+=1
        if is_present_alternative and is_predicted:
            metrics["TP"] += 1  
        elif not is_present_alternative and is_predicted:
            metrics["FP"] += 1 - label_smoothing  # Reduce FN penalty
        elif is_present_alternative and not is_predicted:
            metrics["FN"] += 1 - label_smoothing  # Reduce FP penalty

        # print(f"TEXT: {text[:50]}...")
        # print(f"Skill Predictions: {skill_scores}\n")

    tp, fp, fn = metrics["TP"], metrics["FP"], metrics["FN"]
    precision_skill = tp / (tp + fp) if (tp + fp) else 0
    recall_skill = tp / (tp + fn) if (tp + fn) else 0
    f1_skill = (2 * precision_skill * recall_skill) / (precision_skill + recall_skill) if (precision_skill + recall_skill) else 0
    accuracy = correct / (len(dataset) *2)
    return {"Precision": precision_skill, "Recall": recall_skill, "Accuracy": accuracy, "F1-score": f1_skill}


# -------------------- MAIN DRIVER CODE --------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = load_dataset("validation_set_skills.json")
    # print(dataset[0])

    # Run validation with the fine-tuned model
    results = validate_model(dataset)

    # Save results
    output_file = "reflection_validation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\u2705 Validation results saved to {output_file}")