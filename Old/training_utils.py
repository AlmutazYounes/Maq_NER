import os
import json
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForTokenClassification


def create_experiment_directory(base_dir="experiments", experiment_name=None):
    """Create a timestamped experiment directory."""
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    exp_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def save_model_and_tokenizer(model, tokenizer, save_path):
    """Save model and tokenizer to the specified path."""
    print(f"Saving model and tokenizer to: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    
    # Save model
    model.save_pretrained(save_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(save_path)
    
    print(f"Model and tokenizer saved successfully!")


def save_training_results(results, save_path, dataset_name=""):
    """Save training results to JSON file."""
    results_file = os.path.join(save_path, f"training_results_{dataset_name}.json")
    
    # Add metadata
    results_with_metadata = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name,
        "results": results
    }
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_with_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {results_file}")


def save_training_config(config_dict, save_path, dataset_name=""):
    """Save training configuration to JSON file."""
    config_file = os.path.join(save_path, f"training_config_{dataset_name}.json")
    
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Training config saved to: {config_file}")


def save_label_mapping(label2id, id2label, save_path, dataset_name=""):
    """Save label mappings to JSON file."""
    mapping_file = os.path.join(save_path, f"label_mapping_{dataset_name}.json")
    
    mapping_data = {
        "label2id": label2id,
        "id2label": id2label,
        "unique_labels": list(label2id.keys()),
        "num_labels": len(label2id)
    }
    
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, indent=2, ensure_ascii=False)
    
    print(f"Label mapping saved to: {mapping_file}")


def log_training_summary(experiment_dir, dataset_name, num_train, num_val, results):
    """Create a summary log file for the training run."""
    summary_file = os.path.join(experiment_dir, f"training_summary_{dataset_name}.txt")
    
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"Training Summary for {dataset_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Training Samples: {num_train}\n")
        f.write(f"Validation Samples: {num_val}\n\n")
        f.write("Final Results:\n")
        f.write(f"Precision: {results.get('eval_precision', 'N/A'):.4f}\n")
        f.write(f"Recall: {results.get('eval_recall', 'N/A'):.4f}\n")
        f.write(f"F1-Score: {results.get('eval_f1', 'N/A'):.4f}\n")
        f.write(f"Accuracy: {results.get('eval_accuracy', 'N/A'):.4f}\n")
    
    print(f"Training summary saved to: {summary_file}")


def print_training_results(results, dataset_name):
    """Print formatted training results."""
    print(f"\n{'='*60}")
    print(f"TRAINING RESULTS FOR: {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Precision: {results.get('eval_precision', 'N/A'):.4f}")
    print(f"Recall: {results.get('eval_recall', 'N/A'):.4f}")
    print(f"F1-Score: {results.get('eval_f1', 'N/A'):.4f}")
    print(f"Accuracy: {results.get('eval_accuracy', 'N/A'):.4f}")
    print(f"{'='*60}\n") 