import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import torch
import re
import logging
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

patterns_small = {
    'Email': {'pattern': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', 'token': '[EMAIL]'},
    'URL_HTTP': {'pattern': r'https?://[^\s<>"{}|\\^`\[\]]+', 'token': '[URL]'},
    'USER_AGENT_STRING': {'pattern': r'(?:Mozilla|Opera|Chrome|Safari|Edge|Firefox|MSIE|Trident|Googlebot|Bingbot|Slurp|DuckDuckBot|YandexBot|curl|Wget|PostmanRuntime|Dalvik|okhttp|AppleWebKit|python-requests|Java)/[a-zA-Z0-9\s\(\)\;\.\/\-\_:,rv=x_]+[a-zA-Z0-9\/]', 'token': '[AGENT]'},
}

class NERValidationEvaluator:
    def __init__(self, model_name: str, pattern_method: bool = False):
        self.model_name = model_name
        self.pattern_method = pattern_method
        
        # Select patterns based on model name
        if 'small' in model_name.lower():
            self.patterns = patterns_small
            logger.info(f"Using small patterns set for model: {model_name}")
        else:
            self.patterns = patterns
            logger.info(f"Using full patterns set for model: {model_name}")
        
        logger.info(f"Loading model: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.ner_pipeline = pipeline("ner", 
                                        model=self.model, 
                                        tokenizer=self.tokenizer,
                                        aggregation_strategy="simple",
                                        device=0 if torch.cuda.is_available() else -1)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
        
        if self.pattern_method:
            self.pattern_tokens = set(pattern_info['token'] for pattern_info in self.patterns.values())
            self.sorted_patterns = sorted(self.patterns.items(), key=lambda x: len(x[1]['pattern']), reverse=True)
        
    def read_conll(self, file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
        """Read CoNLL format file and return sentences and labels."""
        sentences, labels = [], []
        try:
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
                            token, tag = line.split("\t", 1)
                        elif " " in line:
                            token, tag = line.split(" ", 1)
                        else:
                            continue
                        tokens.append(token.strip())
                        tags.append(tag.strip())
                
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    
            logger.info(f"Read {len(sentences)} sentences from {file_path}")
            return sentences, labels
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise
    
    def get_validation_split(self, sentences: List[List[str]], tags: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
        """Get validation split from the data."""
        _, val_sents, _, val_tags = train_test_split(sentences, tags, test_size=0.1, random_state=42)
        logger.info(f"Validation split: {len(val_sents)} sentences")
        return val_sents, val_tags
    
    def apply_patterns_to_tokens(self, tokens: List[str]) -> List[str]:
        """Apply patterns at token level."""
        if not self.pattern_method:
            return tokens
        
        modified_tokens = []
        for token in tokens:
            matched = False
            for pattern_name, pattern_info in self.sorted_patterns:
                try:
                    if re.fullmatch(pattern_info['pattern'], token):
                        modified_tokens.append(pattern_info['token'])
                        matched = True
                        break
                except re.error:
                    continue
            
            if not matched:
                modified_tokens.append(token)
        
        return modified_tokens
    
    def predict_sentence(self, tokens: List[str]) -> List[str]:
        """Predict NER tags for a sentence."""
        try:
            if self.pattern_method:
                pattern_tokens = self.apply_patterns_to_tokens(tokens)
                text = " ".join(pattern_tokens)
            else:
                text = " ".join(tokens)
                pattern_tokens = tokens
            
            predictions = self.ner_pipeline(text)
            predicted_tags = ["O"] * len(pattern_tokens)
            
            # Map character positions to token positions
            char_to_token = {}
            char_pos = 0
            for token_idx, token in enumerate(pattern_tokens):
                for i in range(len(token)):
                    char_to_token[char_pos + i] = token_idx
                char_pos += len(token) + 1
            
            for pred in predictions:
                start_char = pred['start']
                end_char = pred['end']
                label = pred['entity_group']
                
                start_token = char_to_token.get(start_char, -1)
                end_token = char_to_token.get(end_char - 1, -1)
                
                if start_token != -1 and end_token != -1:
                    for token_idx in range(start_token, min(end_token + 1, len(pattern_tokens))):
                        if token_idx == start_token:
                            predicted_tags[token_idx] = f"B-{label}"
                        else:
                            predicted_tags[token_idx] = f"I-{label}"
            
            # Convert pattern tokens to PII labels
            if self.pattern_method:
                for i, token in enumerate(pattern_tokens):
                    if token in self.pattern_tokens:
                        if predicted_tags[i] == "O":
                            predicted_tags[i] = "B-PII"
                        elif predicted_tags[i].startswith("B-"):
                            predicted_tags[i] = "B-PII"
                        elif predicted_tags[i].startswith("I-"):
                            predicted_tags[i] = "I-PII"
            
            return predicted_tags
            
        except Exception as e:
            logger.warning(f"Error predicting sentence: {str(e)}")
            return ["O"] * len(tokens)
    
    def extract_entities(self, tokens: List[str], tags: List[str]) -> List[Tuple[int, int, str, str]]:
        """Extract entities from BIO tags. Returns (start, end, label, text)."""
        entities = []
        current_entity = None
        
        for i, tag in enumerate(tags):
            if tag.startswith("B-"):
                if current_entity:
                    start, end, label = current_entity
                    entity_text = " ".join(tokens[start:end+1])
                    entities.append((start, end, label, entity_text))
                current_entity = (i, i, tag[2:])
            elif tag.startswith("I-") and current_entity and tag[2:] == current_entity[2]:
                current_entity = (current_entity[0], i, current_entity[2])
            else:
                if current_entity:
                    start, end, label = current_entity
                    entity_text = " ".join(tokens[start:end+1])
                    entities.append((start, end, label, entity_text))
                current_entity = None
        
        if current_entity:
            start, end, label = current_entity
            entity_text = " ".join(tokens[start:end+1])
            entities.append((start, end, label, entity_text))
        
        return entities
    
    def compute_iou(self, true_span: Tuple[int, int], pred_span: Tuple[int, int]) -> float:
        """Compute IoU between two spans."""
        true_start, true_end = true_span
        pred_start, pred_end = pred_span
        
        intersection_start = max(true_start, pred_start)
        intersection_end = min(true_end, pred_end)
        intersection = max(0, intersection_end - intersection_start + 1)
        
        union = (true_end - true_start + 1) + (pred_end - pred_start + 1) - intersection
        
        if union == 0:
            return 0.0
        return intersection / union
    
    def compute_metrics(self, true_entities: List[Tuple[int, int, str, str]], 
                       pred_entities: List[Tuple[int, int, str, str]]) -> Dict:
        """Compute the 12 leaderboard metrics."""
        
        if len(true_entities) == 0 and len(pred_entities) == 0:
            return {
                'Exact Precision': 1.0, 'Exact Recall': 1.0, 'Exact F1': 1.0,
                'Partial Precision': 1.0, 'Partial Recall': 1.0, 'Partial F1': 1.0,
                'IoU50 Precision': 1.0, 'IoU50 Recall': 1.0, 'IoU50 F1': 1.0,
                'Value Precision': 1.0, 'Value Recall': 1.0, 'Value F1': 1.0
            }
        
        if len(pred_entities) == 0:
            return {
                'Exact Precision': 0.0, 'Exact Recall': 0.0, 'Exact F1': 0.0,
                'Partial Precision': 0.0, 'Partial Recall': 0.0, 'Partial F1': 0.0,
                'IoU50 Precision': 0.0, 'IoU50 Recall': 0.0, 'IoU50 F1': 0.0,
                'Value Precision': 0.0, 'Value Recall': 0.0, 'Value F1': 0.0
            }
        
        if len(true_entities) == 0:
            return {
                'Exact Precision': 0.0, 'Exact Recall': 1.0, 'Exact F1': 0.0,
                'Partial Precision': 0.0, 'Partial Recall': 1.0, 'Partial F1': 0.0,
                'IoU50 Precision': 0.0, 'IoU50 Recall': 1.0, 'IoU50 F1': 0.0,
                'Value Precision': 0.0, 'Value Recall': 1.0, 'Value F1': 0.0
            }
        
        exact_matches = 0
        partial_matches = 0
        iou50_matches = 0
        value_matches = 0
        
        true_matched = {'exact': set(), 'partial': set(), 'iou50': set(), 'value': set()}
        pred_matched = {'exact': set(), 'partial': set(), 'iou50': set(), 'value': set()}
        
        for pred_idx, (pred_start, pred_end, pred_type, pred_text) in enumerate(pred_entities):
            for true_idx, (true_start, true_end, true_type, true_text) in enumerate(true_entities):
                
                # Exact match
                if (pred_start, pred_end, pred_type) == (true_start, true_end, true_type):
                    if pred_idx not in pred_matched['exact'] and true_idx not in true_matched['exact']:
                        exact_matches += 1
                        pred_matched['exact'].add(pred_idx)
                        true_matched['exact'].add(true_idx)
                
                # Partial match
                if pred_type == true_type:
                    overlap = max(0, min(pred_end, true_end) - max(pred_start, true_start) + 1)
                    if overlap > 0:
                        if pred_idx not in pred_matched['partial'] and true_idx not in true_matched['partial']:
                            partial_matches += 1
                            pred_matched['partial'].add(pred_idx)
                            true_matched['partial'].add(true_idx)
                
                # IoU50 match
                if pred_type == true_type:
                    iou = self.compute_iou((true_start, true_end), (pred_start, pred_end))
                    if iou >= 0.5:
                        if pred_idx not in pred_matched['iou50'] and true_idx not in true_matched['iou50']:
                            iou50_matches += 1
                            pred_matched['iou50'].add(pred_idx)
                            true_matched['iou50'].add(true_idx)
                
                # Value match
                if pred_text.strip().lower() == true_text.strip().lower():
                    if pred_idx not in pred_matched['value'] and true_idx not in true_matched['value']:
                        value_matches += 1
                        pred_matched['value'].add(pred_idx)
                        true_matched['value'].add(true_idx)
        
        def safe_divide(a, b):
            return a / b if b > 0 else 0.0
        
        def calc_f1(p, r):
            return safe_divide(2 * p * r, p + r)
        
        exact_precision = safe_divide(exact_matches, len(pred_entities))
        exact_recall = safe_divide(exact_matches, len(true_entities))
        
        partial_precision = safe_divide(partial_matches, len(pred_entities))
        partial_recall = safe_divide(partial_matches, len(true_entities))
        
        iou50_precision = safe_divide(iou50_matches, len(pred_entities))
        iou50_recall = safe_divide(iou50_matches, len(true_entities))
        
        value_precision = safe_divide(value_matches, len(pred_entities))
        value_recall = safe_divide(value_matches, len(true_entities))
        
        return {
            'Exact Precision': exact_precision,
            'Exact Recall': exact_recall,
            'Exact F1': calc_f1(exact_precision, exact_recall),
            'Partial Precision': partial_precision,
            'Partial Recall': partial_recall,
            'Partial F1': calc_f1(partial_precision, partial_recall),
            'IoU50 Precision': iou50_precision,
            'IoU50 Recall': iou50_recall,
            'IoU50 F1': calc_f1(iou50_precision, iou50_recall),
            'Value Precision': value_precision,
            'Value Recall': value_recall,
            'Value F1': calc_f1(value_precision, value_recall)
        }
    
    def evaluate_dataset(self, sentences: List[List[str]], tags: List[List[str]]) -> Dict:
        """Evaluate entire dataset and return aggregated metrics."""
        all_true_entities = []
        all_pred_entities = []
        
        logger.info(f"Evaluating {len(sentences)} sentences...")
        
        for i, (sentence, sentence_tags) in enumerate(zip(sentences, tags)):
            if i > 0 and i % 100 == 0:
                logger.info(f"Processed {i}/{len(sentences)} sentences")
            
            try:
                pred_tags = self.predict_sentence(sentence)
                
                # Handle pattern method
                if self.pattern_method:
                    pattern_tokens = self.apply_patterns_to_tokens(sentence)
                    modified_true_tags = sentence_tags.copy()
                    
                    for j, (orig_token, pattern_token) in enumerate(zip(sentence, pattern_tokens)):
                        if j < len(modified_true_tags) and pattern_token in self.pattern_tokens:
                            if modified_true_tags[j] == "O":
                                modified_true_tags[j] = "B-PII"
                            elif modified_true_tags[j].startswith("B-"):
                                modified_true_tags[j] = "B-PII"
                            elif modified_true_tags[j].startswith("I-"):
                                modified_true_tags[j] = "I-PII"
                    
                    sentence_tags = modified_true_tags
                
                # Ensure same length
                min_len = min(len(sentence_tags), len(pred_tags))
                sentence_tags = sentence_tags[:min_len]
                pred_tags = pred_tags[:min_len]
                sentence = sentence[:min_len]
                
                # Extract entities
                true_entities = self.extract_entities(sentence, sentence_tags)
                pred_entities = self.extract_entities(sentence, pred_tags)
                
                # Adjust entity positions for global list
                offset = len(all_true_entities)
                true_entities = [(start + offset, end + offset, label, text) for start, end, label, text in true_entities]
                pred_entities = [(start + offset, end + offset, label, text) for start, end, label, text in pred_entities]
                
                all_true_entities.extend(true_entities)
                all_pred_entities.extend(pred_entities)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate sentence {i}: {str(e)}")
                continue
        
        # Compute final metrics
        metrics = self.compute_metrics(all_true_entities, all_pred_entities)
        logger.info(f"Evaluation complete")
        
        return metrics

def main():
    data_file = "conll_training_data.txt"
    models = [
        "MutazYoune/arabic-ner-masking-patterns",
        "MutazYoune/arabic-ner-masking-original", 
        "MutazYoune/ARAB_BERT-original",
        "MutazYoune/ARAB_BERT-patterns",
        "MutazYoune/Arabic-NER-PII-patterns",
        "MutazYoune/Arabic-NER-PII",
        "MutazYoune/Arabic-NER-PII-original",
        "MutazYoune/Arabic-NER-PII-patterns_small",
        "MutazYoune/ARAB_BERT-patterns_small",
        "MutazYoune/bert-base-arabic-camelbert-mix-ner-patterns_small",
        "MutazYoune/bert-base-arabic-camelbert-mix-ner-patterns",
        "MutazYoune/bert-base-arabic-camelbert-mix-ner-original"
    ]
    results_path = "ner_validation_results.xlsx"
    
    results = []
    
    logger.info(f"Starting evaluation of {len(models)} models")
    
    for model_idx, model_name in enumerate(models, 1):
        logger.info(f"Processing model {model_idx}/{len(models)}: {model_name}")
        
        pattern_method = 'pattern' in model_name.lower()
        
        try:
            evaluator = NERValidationEvaluator(model_name, pattern_method=pattern_method)
            sentences, tags = evaluator.read_conll(data_file)
            val_sentences, val_tags = evaluator.get_validation_split(sentences, tags)
            metrics = evaluator.evaluate_dataset(val_sentences, val_tags)
            
            result = {
                'Model': model_name,
                **metrics
            }
            results.append(result)
            
            logger.info(f"✓ {model_name}: Exact F1={metrics['Exact F1']:.4f}")
            
        except Exception as e:
            logger.error(f"✗ Failed to evaluate {model_name}: {str(e)}")
            continue
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('Exact F1', ascending=False)  # Sort by Exact F1
        df.to_excel(results_path, index=False)
        logger.info(f"Results saved to: {results_path}")
        
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        for _, row in df.iterrows():
            print(f"{row['Model']}")
            print(f"  Exact F1: {row['Exact F1']:.4f}")
            print(f"  Partial F1: {row['Partial F1']:.4f}")
            print(f"  IoU50 F1: {row['IoU50 F1']:.4f}")
            print(f"  Value F1: {row['Value F1']:.4f}")
            print()
    else:
        logger.error("No results to save")
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main() 