import random
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification, AutoConfig
import torch
import evaluate
from training_utils import (
    create_experiment_directory, save_model_and_tokenizer, save_training_results,
    save_training_config, save_label_mapping, log_training_summary, print_training_results
)

MODEL_NAME = "MutazYoune/Arabic-NER-PII"
HUB_MODEL_PREFIX = "MutazYoune/Arabic-NER-PII"

# Define datasets to train on
DATASETS = [
    {
        "name": "original", 
        "file": "conll_training_data.txt",
        "description": "Original data without patterns"
    },
    {
        "name": "patterns", 
        "file": "conll_training_data_patterns.txt",
        "description": "Data with patterns applied"
    }
]

def read_conll(file_path):
    """Read CoNLL format data."""
    sentences = []
    labels = []
    with open(file_path, encoding="utf-8") as f:
        tokens = []
        tags = []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens = []
                    tags = []
            else:
                if "\t" in line:
                    token, tag = line.split("\t")
                elif " " in line:
                    token, tag = line.split(" ", 1)
                else:
                    continue
                tokens.append(token)
                tags.append(tag)
        if tokens:
            sentences.append(tokens)
            labels.append(tags)
    return sentences, labels

def prepare_dataset(sentences, tags, tokenizer, label2id):
    """Prepare HuggingFace dataset from sentences and tags."""
    def to_hf_dict(sentences, tags):
        return [{"tokens": s, "ner_tags": t} for s, t in zip(sentences, tags)]
    
    def tokenize_and_align_labels(example):
        tokenized = tokenizer(
            example["tokens"], truncation=True, is_split_into_words=True, max_length=512
        )
        word_ids = tokenized.word_ids()
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[example["ner_tags"][word_idx]])
            else:
                label_ids.append(label2id[example["ner_tags"][word_idx]])
            previous_word_idx = word_idx
        tokenized["labels"] = label_ids
        return tokenized
    
    # Split data
    train_sents, val_sents, train_tags, val_tags = train_test_split(
        sentences, tags, test_size=0.1, random_state=42
    )
    
    # Create dataset
    dataset = DatasetDict({
        "train": Dataset.from_list(to_hf_dict(train_sents, train_tags)),
        "validation": Dataset.from_list(to_hf_dict(val_sents, val_tags)),
    })
    
    # Apply tokenization
    dataset = dataset.map(tokenize_and_align_labels, batched=False)
    
    return dataset, len(train_sents), len(val_sents)

def compute_metrics(metric, id2label):
    """Create compute_metrics function."""
    def _compute_metrics(p):
        predictions, labels = p
        predictions = predictions.argmax(axis=-1)
        true_labels = [[id2label[l] for l in sent if l != -100] for sent in labels]
        true_preds = [
            [id2label[pred] for pred, lab in zip(preds, labs) if lab != -100]
            for preds, labs in zip(predictions, labels)
        ]
        results = metric.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": results["overall_precision"], 
            "recall": results["overall_recall"], 
            "f1": results["overall_f1"], 
            "accuracy": results["overall_accuracy"]
        }
    return _compute_metrics

def train_model(dataset_config):
    """Train model on a specific dataset."""
    print(f"\n{'='*80}")
    print(f"TRAINING ON: {dataset_config['name'].upper()}")
    print(f"Description: {dataset_config['description']}")
    print(f"File: {dataset_config['file']}")
    print(f"{'='*80}")
    
    # Read data
    sentences, tags = read_conll(dataset_config['file'])
    print(f"Total sentences: {len(sentences)}")
    
    # Create label mapping
    unique_labels = sorted({l for tag_seq in tags for l in tag_seq})
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}
    
    print(f"Labels found: {unique_labels}")
    print(f"Number of labels: {len(unique_labels)}")
    
    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Prepare dataset
    dataset, num_train, num_val = prepare_dataset(sentences, tags, tokenizer, label2id)
    print(f"Training sentences: {num_train}")
    print(f"Validation sentences: {num_val}")
    
    # Create experiment directory
    exp_dir = create_experiment_directory("models", f"{dataset_config['name']}_ner")
    
    # Save training splits for reference
    train_file = f"{exp_dir}/train_{dataset_config['name']}.txt"
    val_file = f"{exp_dir}/val_{dataset_config['name']}.txt"
    
    with open(train_file, "w", encoding="utf-8") as f:
        for example in dataset["train"]:
            tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
            labels = [id2label[l] if l != -100 else "O" for l in example["labels"]]
            for t, l in zip(tokens, labels):
                if not t.startswith("##") and t not in ["[CLS]", "[SEP]", "[PAD]"]:
                    f.write(f"{t}\t{l}\n")
            f.write("\n")
    
    with open(val_file, "w", encoding="utf-8") as f:
        for example in dataset["validation"]:
            tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
            labels = [id2label[l] if l != -100 else "O" for l in example["labels"]]
            for t, l in zip(tokens, labels):
                if not t.startswith("##") and t not in ["[CLS]", "[SEP]", "[PAD]"]:
                    f.write(f"{t}\t{l}\n")
            f.write("\n")
    
    # Model setup
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = len(label2id)
    config.id2label = id2label
    config.label2id = label2id
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, config=config, ignore_mismatched_sizes=True
    )
    
    # Training setup
    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = evaluate.load("seqeval")

    # Calculate logging_steps for 0.5 epoch
    steps_per_epoch = max(1, num_train // 8)  # batch size is 8
    logging_steps = max(1, steps_per_epoch // 2)

    # Hugging Face Hub model id for this run
    hub_model_id = f"{HUB_MODEL_PREFIX}-{dataset_config['name']}"

    training_args = TrainingArguments(
        output_dir=f"{exp_dir}/checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",  # Save at each epoch
        save_total_limit=1,      # Only keep the best checkpoint
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=logging_steps,  # Log every 0.5 epoch
        logging_dir=f"{exp_dir}/logs",
        report_to="none",
        # Do not push to hub automatically, do it manually after training
        load_best_model_at_end=True,  # Load best model at end
        metric_for_best_model="eval_f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics(metric, id2label),
    )
    
    # Save training configuration
    training_config = {
        "model_name": MODEL_NAME,
        "dataset_file": dataset_config['file'],
        "dataset_name": dataset_config['name'],
        "num_train_samples": num_train,
        "num_val_samples": num_val,
        "unique_labels": unique_labels,
        "num_labels": len(unique_labels),
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "num_epochs": training_args.num_train_epochs,
        "max_length": 512
    }
    save_training_config(training_config, exp_dir, dataset_config['name'])
    
    # Save label mapping
    save_label_mapping(label2id, id2label, exp_dir, dataset_config['name'])
    
    print("Starting training...")
    trainer.train()
    print("Training completed!")
    
    print("Evaluating model...")
    results = trainer.evaluate()
    
    # Print and save results
    print_training_results(results, dataset_config['name'])
    save_training_results(results, exp_dir, dataset_config['name'])
    
    # Save final (best) model and tokenizer
    model_save_path = f"{exp_dir}/final_model"
    save_model_and_tokenizer(trainer.model, tokenizer, model_save_path)
    
    # Push only the model and tokenizer to Hugging Face Hub
    print(f"Pushing best model and tokenizer to Hugging Face Hub: {hub_model_id}")
    trainer.model.push_to_hub(hub_model_id)
    tokenizer.push_to_hub(hub_model_id)
    
    # Create training summary
    log_training_summary(exp_dir, dataset_config['name'], num_train, num_val, results)
    
    return results, exp_dir

# Main execution
if __name__ == "__main__":
    print("Starting Arabic NER PII Training on Multiple Datasets")
    print("=" * 80)
    
    all_results = {}
    all_dirs = {}
    
    # Train on each dataset
    for dataset_config in DATASETS:
        try:
            results, exp_dir = train_model(dataset_config)
            all_results[dataset_config['name']] = results
            all_dirs[dataset_config['name']] = exp_dir
            print(f"✅ Successfully trained model on {dataset_config['name']} dataset")
        except Exception as e:
            print(f"❌ Error training on {dataset_config['name']}: {str(e)}")
            continue
    
    # Summary of all training runs
    print(f"\n{'='*80}")
    print("FINAL SUMMARY - ALL TRAINING RUNS")
    print(f"{'='*80}")
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()} DATASET:")
        print(f"  Model saved in: {all_dirs[dataset_name]}")
        print(f"  Precision: {results.get('eval_precision', 'N/A'):.4f}")
        print(f"  Recall: {results.get('eval_recall', 'N/A'):.4f}")
        print(f"  F1-Score: {results.get('eval_f1', 'N/A'):.4f}")
        print(f"  Accuracy: {results.get('eval_accuracy', 'N/A'):.4f}")
    
    print(f"\n{'='*80}")
    print("Training completed! All models and results have been saved.")
    print(f"{'='*80}") 