import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification, AutoConfig
import evaluate

MODEL_NAME = "MutazYoune/Arabic-NER-PII"
HUB_MODEL_PREFIX = "MutazYoune/Arabic-NER-PII"

def read_conll(file_path):
    sentences, labels = [], []
    with open(file_path, encoding="utf-8") as f:
        tokens, tags = [], []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
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
    def tokenize_and_align_labels(example):
        tokenized = tokenizer(example["tokens"], truncation=True, is_split_into_words=True, max_length=512)
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
    
    train_sents, val_sents, train_tags, val_tags = train_test_split(sentences, tags, test_size=0.1, random_state=42)
    
    dataset = DatasetDict({
        "train": Dataset.from_list([{"tokens": s, "ner_tags": t} for s, t in zip(train_sents, train_tags)]),
        "validation": Dataset.from_list([{"tokens": s, "ner_tags": t} for s, t in zip(val_sents, val_tags)]),
    })
    
    dataset = dataset.map(tokenize_and_align_labels, batched=False)
    return dataset

def compute_metrics(metric, id2label):
    def _compute_metrics(p):
        predictions, labels = p
        predictions = predictions.argmax(axis=-1)
        true_labels = [[id2label[l] for l in sent if l != -100] for sent in labels]
        true_preds = [[id2label[pred] for pred, lab in zip(preds, labs) if lab != -100] for preds, labs in zip(predictions, labels)]
        results = metric.compute(predictions=true_preds, references=true_labels)
        return {"precision": results["overall_precision"], "recall": results["overall_recall"], 
                "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}
    return _compute_metrics

def train_ner_model(data_file, model_suffix):
    sentences, tags = read_conll(data_file)
    
    unique_labels = sorted({l for tag_seq in tags for l in tag_seq})
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = prepare_dataset(sentences, tags, tokenizer, label2id)
    
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = len(label2id)
    config.id2label = id2label
    config.label2id = label2id
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, config=config, ignore_mismatched_sizes=True)
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = evaluate.load("seqeval")
    
    hub_model_id = f"{HUB_MODEL_PREFIX}-{model_suffix}"
    
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        report_to="none",
        load_best_model_at_end=True,
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
    
    trainer.train()
    results = trainer.evaluate()
    
    trainer.model.push_to_hub(hub_model_id)
    tokenizer.push_to_hub(hub_model_id)
    
    return results

# Train on both datasets
datasets = [
    {"file": "conll_training_data.txt", "suffix": "original"},
    {"file": "conll_training_data_patterns.txt", "suffix": "patterns"}
]

for dataset in datasets:
    results = train_ner_model(dataset["file"], dataset["suffix"])
    print(f"{dataset['suffix']}: F1={results['eval_f1']:.4f}") 