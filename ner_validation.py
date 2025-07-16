import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import torch
import re
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sklearn.metrics import classification_report, precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")

patterns = {
    'Email': {'pattern': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', 'token': '[EMAIL]'},
    'URL_HTTP': {'pattern': r'https?://[^\s<>"{}|\\^`\[\]]+', 'token': '[URL]'},
    'Ethereum_Address': {'pattern': r'0x[a-fA-F0-9]{40}', 'token': '[CRYPTO]'},
    'Bitcoin_Address': {'pattern': r'[13][a-km-zA-HJ-NP-Z1-9]{25,34}', 'token': '[CRYPTO]'},
    'IBAN': {'pattern': r'[A-Z]{2}[0-9]{2}[A-Z0-9]{4,30}', 'token': '[IBAN]'},
    'Credit_Card_Visa': {'pattern': r'4[0-9]{12}(?:[0-9]{3})?', 'token': '[CREDIT_CARD]'},
    'Credit_Card_Mastercard': {'pattern': r'5[1-5][0-9]{14}', 'token': '[CREDIT_CARD]'},
    'SSN': {'pattern': r'\d{3}-\d{2}-\d{4}', 'token': '[SSN]'},
    'VIN': {'pattern': r'[A-HJ-NPR-Z0-9]{17}', 'token': '[VIN]'},
    'IPv4': {'pattern': r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)', 'token': '[IP]'},
    'IPv6': {'pattern': r'(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}', 'token': '[IP]'},
    'MAC_Address': {'pattern': r'([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})', 'token': '[MAC]'},
    'Phone_US': {'pattern': r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})(?:\s*x\d+)?', 'token': '[PHONE]'},
    'Date_MM_DD_YYYY': {'pattern': r'(0?[1-9]|1[0-2])/(0?[1-9]|[12][0-9]|3[01])/((19|20)\d{2})', 'token': '[DATE]'},
    'Date_DD_MM_YYYY': {'pattern': r'(0?[1-9]|[12][0-9]|3[01])/(0?[1-9]|1[0-2])/((19|20)\d{2})', 'token': '[DATE]'},
    'ISO_DateTime': {'pattern': r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z', 'token': '[DATETIME]'},
    'Time_24H': {'pattern': r'([01]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?', 'token': '[TIME]'},
    'Complex_ID': {'pattern': r'[A-Z0-9]{24,45}', 'token': '[ID]'},
    'Hash_Token_Long': {'pattern': r'[a-zA-Z0-9]{36,60}', 'token': '[HASH]'},
    'Hash_Token_Short': {'pattern': r'[a-zA-Z0-9]{20,35}', 'token': '[HASH]'},
    'Base58_Token': {'pattern': r'[1-9A-HJ-NP-Za-km-z]{25,60}', 'token': '[TOKEN]'},
    'Custom_Alphanum_ID': {'pattern': r'MK[A-Z0-9]{17}', 'token': '[ID]'},
    'Reference_Number': {'pattern': r'\d{2}-\d{6}-\d{6}-\d{1}', 'token': '[REF_NUM]'},
    'Custom_ID_Format': {'pattern': r'\d{4}\s\d{5}-\d{4}', 'token': '[CUSTOM_ID]'},
    'Custom_ID_Format_2': {'pattern': r'\d{4}\s\d{5}', 'token': '[CUSTOM_ID]'},
    'Custom_ID_Format_3': {'pattern': r'\d{3},\s\d{5}', 'token': '[CUSTOM_ID]'},
    'Custom_ID_Format_4': {'pattern': r'\d{5}،\s\d{4}', 'token': '[CUSTOM_ID]'},
    'Custom_ID_Format_5': {'pattern': r'\d{3}\s\d{2}\s\d{4}', 'token': '[CUSTOM_ID]'},
    'ZIP_Code_US': {'pattern': r'\d{5}(?:-\d{4})?', 'token': '[ZIP_CODE]'},
    'Numeric_ID_5Plus': {'pattern': r'\d{5,}', 'token': '[NUMERIC_ID]'},
    'URL_Domain': {'pattern': r'[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}', 'token': '[DOMAIN]'},
    'License_Plate': {'pattern': r'[A-Z]{2}[0-9]{2}[A-Z]{3}', 'token': '[LICENSE_PLATE]'},
    'IPv6_Prefix': {'pattern': r'IPV6:\s*[0-9a-fA-F:]+', 'token': '[IP]'},
    'Token_Parentheses': {'pattern': r'\([A-Z0-9]{10,}\)', 'token': '[TOKEN]'},
    'USER_AGENT_STRING': {'pattern': r'(?:Mozilla|Opera|Chrome|Safari|Edge|Firefox|MSIE|Trident|Googlebot|Bingbot|Slurp|DuckDuckBot|YandexBot|curl|Wget|PostmanRuntime|Dalvik|okhttp|AppleWebKit|python-requests|Java)/[a-zA-Z0-9\s\(\)\;\.\/\-\_:,rv=x_]+[a-zA-Z0-9\/]', 'token': '[AGENT]'},
}

class NERValidationEvaluator:
    def __init__(self, model_name: str, pattern_method: bool = False):
        self.model_name = model_name
        self.pattern_method = pattern_method
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline("ner", 
                                    model=self.model, 
                                    tokenizer=self.tokenizer,
                                    aggregation_strategy="simple",
                                    device=0 if torch.cuda.is_available() else -1)
        
        # Get all pattern tokens for later conversion to [MASK]
        if self.pattern_method:
            self.pattern_tokens = set(pattern_info['token'] for pattern_info in patterns.values())
        
    def read_conll(self, file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
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
    
    def get_validation_split(self, sentences: List[List[str]], tags: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
        _, val_sents, _, val_tags = train_test_split(sentences, tags, test_size=0.1, random_state=42)
        return val_sents, val_tags
    
    def apply_patterns_to_text(self, text: str) -> str:
        if not self.pattern_method:
            return text
        
        modified_text = text
        sorted_patterns = sorted(patterns.items(), 
                               key=lambda x: len(x[1]['pattern']), 
                               reverse=True)
        for pattern_name, pattern_info in sorted_patterns:
            pattern = pattern_info['pattern']
            token = pattern_info['token']
            try:
                modified_text = re.sub(pattern, token, modified_text)
            except re.error:
                continue
        return modified_text
    
    def convert_pattern_tokens_to_mask(self, tokens: List[str]) -> List[str]:
        if not self.pattern_method:
            return tokens
        mask_tokens = []
        for token in tokens:
            if token in self.pattern_tokens:
                mask_tokens.append('[MASK]')
            else:
                mask_tokens.append(token)
        return mask_tokens
    
    def predict_sentence(self, tokens: List[str]) -> List[str]:
        try:
            text = " ".join(tokens)
            if self.pattern_method:
                text = self.apply_patterns_to_text(text)
            predictions = self.ner_pipeline(text)
            predicted_tags = ["O"] * len(tokens)
            if self.pattern_method:
                modified_tokens = text.split()
                modified_predicted_tags = ["O"] * len(modified_tokens)
                for pred in predictions:
                    start_char = pred['start']
                    end_char = pred['end']
                    label = pred['entity_group']
                    char_pos = 0
                    for i, token in enumerate(modified_tokens):
                        token_start = char_pos
                        token_end = char_pos + len(token)
                        if (token_start < end_char and token_end > start_char):
                            if i == 0 or modified_predicted_tags[i-1] != label:
                                modified_predicted_tags[i] = f"B-{label}"
                            else:
                                modified_predicted_tags[i] = f"I-{label}"
                        char_pos = token_end + 1
                for i, token in enumerate(modified_tokens):
                    if token in self.pattern_tokens and modified_predicted_tags[i] == "O":
                        modified_predicted_tags[i] = "B-MASK"
                    elif token in self.pattern_tokens and modified_predicted_tags[i] != "O":
                        if modified_predicted_tags[i].startswith("B-"):
                            modified_predicted_tags[i] = "B-MASK"
                        elif modified_predicted_tags[i].startswith("I-"):
                            modified_predicted_tags[i] = "I-MASK"
                if len(modified_predicted_tags) <= len(predicted_tags):
                    predicted_tags[:len(modified_predicted_tags)] = modified_predicted_tags
                else:
                    predicted_tags = modified_predicted_tags[:len(predicted_tags)]
            else:
                for pred in predictions:
                    start_char = pred['start']
                    end_char = pred['end']
                    label = pred['entity_group']
                    char_pos = 0
                    for i, token in enumerate(tokens):
                        token_start = char_pos
                        token_end = char_pos + len(token)
                        if (token_start < end_char and token_end > start_char):
                            if i == 0 or predicted_tags[i-1] != label:
                                predicted_tags[i] = f"B-{label}"
                            else:
                                predicted_tags[i] = f"I-{label}"
                        char_pos = token_end + 1
            return predicted_tags
        except Exception as e:
            return ["O"] * len(tokens)
    
    def extract_entities(self, tokens: List[str], tags: List[str]) -> List[Tuple[int, int, str]]:
        entities = []
        current_entity = None
        for i, tag in enumerate(tags):
            if tag.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = (i, i, tag[2:])
            elif tag.startswith("I-") and current_entity and tag[2:] == current_entity[2]:
                current_entity = (current_entity[0], i, current_entity[2])
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity = None
        if current_entity:
            entities.append(current_entity)
        return entities
    
    def compute_iou(self, true_span: Tuple[int, int], pred_span: Tuple[int, int]) -> float:
        true_start, true_end = true_span
        pred_start, pred_end = pred_span
        intersection_start = max(true_start, pred_start)
        intersection_end = min(true_end, pred_end)
        intersection = max(0, intersection_end - intersection_start + 1)
        union = (true_end - true_start + 1) + (pred_end - pred_start + 1) - intersection
        if union == 0:
            return 0.0
        return intersection / union
    
    def compute_ner_metrics(self, true_entities: List[Tuple[int, int, str]], 
                           pred_entities: List[Tuple[int, int, str]]) -> Dict:
        if len(true_entities) == 0 and len(pred_entities) == 0:
            return {
                'exact_precision': 1.0, 'exact_recall': 1.0, 'exact_f1': 1.0,
                'partial_precision': 1.0, 'partial_recall': 1.0, 'partial_f1': 1.0,
                'iou50_precision': 1.0, 'iou50_recall': 1.0, 'iou50_f1': 1.0,
                'value_precision': 1.0, 'value_recall': 1.0, 'value_f1': 1.0
            }
        if len(pred_entities) == 0:
            return {
                'exact_precision': 0.0, 'exact_recall': 0.0, 'exact_f1': 0.0,
                'partial_precision': 0.0, 'partial_recall': 0.0, 'partial_f1': 0.0,
                'iou50_precision': 0.0, 'iou50_recall': 0.0, 'iou50_f1': 0.0,
                'value_precision': 0.0, 'value_recall': 0.0, 'value_f1': 0.0
            }
        if len(true_entities) == 0:
            return {
                'exact_precision': 0.0, 'exact_recall': 1.0, 'exact_f1': 0.0,
                'partial_precision': 0.0, 'partial_recall': 1.0, 'partial_f1': 0.0,
                'iou50_precision': 0.0, 'iou50_recall': 1.0, 'iou50_f1': 0.0,
                'value_precision': 0.0, 'value_recall': 1.0, 'value_f1': 0.0
            }
        exact_matches = 0
        partial_matches = 0
        iou50_matches = 0
        value_matches = 0
        true_exact_matched = set()
        true_partial_matched = set()
        true_iou50_matched = set()
        true_value_matched = set()
        pred_exact_matched = set()
        pred_partial_matched = set()
        pred_iou50_matched = set()
        pred_value_matched = set()
        for pred_idx, (pred_start, pred_end, pred_type) in enumerate(pred_entities):
            for true_idx, (true_start, true_end, true_type) in enumerate(true_entities):
                if (pred_start, pred_end, pred_type) == (true_start, true_end, true_type):
                    if pred_idx not in pred_exact_matched and true_idx not in true_exact_matched:
                        exact_matches += 1
                        pred_exact_matched.add(pred_idx)
                        true_exact_matched.add(true_idx)
                if pred_type == true_type:
                    overlap = max(0, min(pred_end, true_end) - max(pred_start, true_start) + 1)
                    if overlap > 0:
                        if pred_idx not in pred_partial_matched and true_idx not in true_partial_matched:
                            partial_matches += 1
                            pred_partial_matched.add(pred_idx)
                            true_partial_matched.add(true_idx)
                if pred_type == true_type:
                    iou = self.compute_iou((true_start, true_end), (pred_start, pred_end))
                    if iou >= 0.5:
                        if pred_idx not in pred_iou50_matched and true_idx not in true_iou50_matched:
                            iou50_matches += 1
                            pred_iou50_matched.add(pred_idx)
                            true_iou50_matched.add(true_idx)
                if pred_type == true_type:
                    if pred_idx not in pred_value_matched and true_idx not in true_value_matched:
                        value_matches += 1
                        pred_value_matched.add(pred_idx)
                        true_value_matched.add(true_idx)
        def safe_divide(numerator, denominator):
            return numerator / denominator if denominator > 0 else 0.0
        def calc_f1(precision, recall):
            return safe_divide(2 * precision * recall, precision + recall)
        exact_precision = safe_divide(exact_matches, len(pred_entities))
        exact_recall = safe_divide(exact_matches, len(true_entities))
        exact_f1 = calc_f1(exact_precision, exact_recall)
        partial_precision = safe_divide(partial_matches, len(pred_entities))
        partial_recall = safe_divide(partial_matches, len(true_entities))
        partial_f1 = calc_f1(partial_precision, partial_recall)
        iou50_precision = safe_divide(iou50_matches, len(pred_entities))
        iou50_recall = safe_divide(iou50_matches, len(true_entities))
        iou50_f1 = calc_f1(iou50_precision, iou50_recall)
        value_precision = safe_divide(value_matches, len(pred_entities))
        value_recall = safe_divide(value_matches, len(true_entities))
        value_f1 = calc_f1(value_precision, value_recall)
        return {
            'exact_precision': exact_precision,
            'exact_recall': exact_recall,
            'exact_f1': exact_f1,
            'partial_precision': partial_precision,
            'partial_recall': partial_recall,
            'partial_f1': partial_f1,
            'iou50_precision': iou50_precision,
            'iou50_recall': iou50_recall,
            'iou50_f1': iou50_f1,
            'value_precision': value_precision,
            'value_recall': value_recall,
            'value_f1': value_f1
        }
    
    def evaluate_single_sentence(self, tokens: List[str], true_tags: List[str]) -> Dict:
        pred_tags = self.predict_sentence(tokens)
        if self.pattern_method:
            original_text = " ".join(tokens)
            pattern_applied_text = self.apply_patterns_to_text(original_text)
            pattern_tokens = pattern_applied_text.split()
            true_tags_with_patterns = true_tags.copy()
            for i, token in enumerate(tokens):
                if i < len(pattern_tokens) and pattern_tokens[i] in self.pattern_tokens:
                    true_tags_with_patterns[i] = "B-MASK"
            true_tags = true_tags_with_patterns
        min_len = min(len(true_tags), len(pred_tags))
        true_tags = true_tags[:min_len]
        pred_tags = pred_tags[:min_len]
        true_entities = self.extract_entities(tokens[:min_len], true_tags)
        pred_entities = self.extract_entities(tokens[:min_len], pred_tags)
        ner_metrics = self.compute_ner_metrics(true_entities, pred_entities)
        return {
            'tokens': tokens[:min_len],
            'true_tags': true_tags,
            'pred_tags': pred_tags,
            'true_entities': true_entities,
            'pred_entities': pred_entities,
            **ner_metrics,
            'num_true_entities': len(true_entities),
            'num_pred_entities': len(pred_entities)
        }
    
    def evaluate_dataset(self, sentences: List[List[str]], tags: List[List[str]]) -> Tuple[Dict, List[Dict]]:
        results = []
        for i, (sentence, sentence_tags) in enumerate(zip(sentences, tags)):
            try:
                result = self.evaluate_single_sentence(sentence, sentence_tags)
                result['sentence_idx'] = i
                results.append(result)
            except Exception as e:
                continue
        if not results:
            return {}, []
        metrics = {
            'num_sentences': len(results),
            'exact_precision': np.mean([r['exact_precision'] for r in results]),
            'exact_recall': np.mean([r['exact_recall'] for r in results]),
            'exact_f1': np.mean([r['exact_f1'] for r in results]),
            'partial_precision': np.mean([r['partial_precision'] for r in results]),
            'partial_recall': np.mean([r['partial_recall'] for r in results]),
            'partial_f1': np.mean([r['partial_f1'] for r in results]),
            'iou50_precision': np.mean([r['iou50_precision'] for r in results]),
            'iou50_recall': np.mean([r['iou50_recall'] for r in results]),
            'iou50_f1': np.mean([r['iou50_f1'] for r in results]),
            'value_precision': np.mean([r['value_precision'] for r in results]),
            'value_recall': np.mean([r['value_recall'] for r in results]),
            'value_f1': np.mean([r['value_f1'] for r in results]),
            'total_true_entities': sum([r['num_true_entities'] for r in results]),
            'total_pred_entities': sum([r['num_pred_entities'] for r in results])
        }
        exact_f1_scores = [r['exact_f1'] for r in results]
        partial_f1_scores = [r['partial_f1'] for r in results]
        iou50_f1_scores = [r['iou50_f1'] for r in results]
        value_f1_scores = [r['value_f1'] for r in results]
        metrics['exact_f1_std'] = np.std(exact_f1_scores)
        metrics['exact_f1_median'] = np.median(exact_f1_scores)
        metrics['partial_f1_std'] = np.std(partial_f1_scores)
        metrics['partial_f1_median'] = np.median(partial_f1_scores)
        metrics['iou50_f1_std'] = np.std(iou50_f1_scores)
        metrics['iou50_f1_median'] = np.median(iou50_f1_scores)
        metrics['value_f1_std'] = np.std(value_f1_scores)
        metrics['value_f1_median'] = np.median(value_f1_scores)
        return metrics, results

def main():
    data_file = "conll_training_data.txt"
    models = [
        "MutazYoune/ARAB_BERT-original",
        "MutazYoune/ARAB_BERT-patterns",
        "MutazYoune/Arabic-NER-PII-patterns",
        "MutazYoune/Arabic-NER-PII-original"
    ]
    results_path = "ner_validation_results.xlsx"
    all_summaries = []
    all_details = []
    for model_name in models:
        pattern_method = 'pattern' in model_name.lower()
        try:
            evaluator = NERValidationEvaluator(model_name, pattern_method=pattern_method)
        except Exception as e:
            continue
        sentences, tags = evaluator.read_conll(data_file)
        val_sentences, val_tags = evaluator.get_validation_split(sentences, tags)
        metrics, detailed_results = evaluator.evaluate_dataset(val_sentences, val_tags)
        if detailed_results:
            # Add model info to summary
            summary_row = {
                'model_name': model_name,
                'data_file': data_file,
                'pattern_method': pattern_method,
                'num_sentences': metrics['num_sentences'],
                'exact_precision': metrics['exact_precision'],
                'exact_recall': metrics['exact_recall'],
                'exact_f1': metrics['exact_f1'],
                'partial_precision': metrics['partial_precision'],
                'partial_recall': metrics['partial_recall'],
                'partial_f1': metrics['partial_f1'],
                'iou50_precision': metrics['iou50_precision'],
                'iou50_recall': metrics['iou50_recall'],
                'iou50_f1': metrics['iou50_f1'],
                'value_precision': metrics['value_precision'],
                'value_recall': metrics['value_recall'],
                'value_f1': metrics['value_f1'],
                'total_true_entities': metrics['total_true_entities'],
                'total_pred_entities': metrics['total_pred_entities']
            }
            all_summaries.append(summary_row)
            # Add model info to each detailed row
            for result in detailed_results:
                row = {
                    'model_name': model_name,
                    'pattern_method': pattern_method,
                    'sentence_idx': result['sentence_idx'],
                    'sentence': ' '.join(result['tokens']),
                    'true_tags': ' '.join(result['true_tags']),
                    'pred_tags': ' '.join(result['pred_tags']),
                    'exact_precision': result['exact_precision'],
                    'exact_recall': result['exact_recall'],
                    'exact_f1': result['exact_f1'],
                    'partial_precision': result['partial_precision'],
                    'partial_recall': result['partial_recall'],
                    'partial_f1': result['partial_f1'],
                    'iou50_precision': result['iou50_precision'],
                    'iou50_recall': result['iou50_recall'],
                    'iou50_f1': result['iou50_f1'],
                    'value_precision': result['value_precision'],
                    'value_recall': result['value_recall'],
                    'value_f1': result['value_f1'],
                    'num_true_entities': result['num_true_entities'],
                    'num_pred_entities': result['num_pred_entities'],
                    'true_entities': str(result['true_entities']),
                    'pred_entities': str(result['pred_entities'])
                }
                all_details.append(row)
    if all_summaries or all_details:
        with pd.ExcelWriter(results_path, engine='openpyxl') as writer:
            pd.DataFrame(all_summaries).to_excel(writer, sheet_name='Summary', index=False)
            pd.DataFrame(all_details).to_excel(writer, sheet_name='Detailed_Results', index=False)

if __name__ == "__main__":
    main() 