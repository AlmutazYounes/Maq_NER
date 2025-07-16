import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")

# Full patterns set
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

# Small patterns set
patterns_small = {
    'Email': {'pattern': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', 'token': '[EMAIL]'},
    'URL_HTTP': {'pattern': r'https?://[^\s<>"{}|\\^`\[\]]+', 'token': '[URL]'},
    'USER_AGENT_STRING': {'pattern': r'(?:Mozilla|Opera|Chrome|Safari|Edge|Firefox|MSIE|Trident|Googlebot|Bingbot|Slurp|DuckDuckBot|YandexBot|curl|Wget|PostmanRuntime|Dalvik|okhttp|AppleWebKit|python-requests|Java)/[a-zA-Z0-9\s\(\)\;\.\/\-\_:,rv=x_]+[a-zA-Z0-9\/]', 'token': '[AGENT]'},
}

def get_patterns_for_model(model_name: str) -> Dict:
    """Get the appropriate patterns set based on model name."""
    if 'small' in model_name.lower():
        return patterns_small
    else:
        return patterns

def uses_patterns(model_name: str) -> bool:
    """Check if model uses pattern replacement method."""
    return 'pattern' in model_name.lower()

def apply_patterns_to_text(text: str, patterns_dict: Dict) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Apply patterns to text and return modified text and list of (original, pattern_token) replacements.
    """
    # Sort patterns by length (longest first) to avoid partial matches
    sorted_patterns = sorted(patterns_dict.items(), key=lambda x: len(x[1]['pattern']), reverse=True)
    
    replacements = []
    modified_text = text
    
    for pattern_name, pattern_info in sorted_patterns:
        try:
            pattern = pattern_info['pattern']
            token = pattern_info['token']
            
            # Find all matches
            matches = list(re.finditer(pattern, modified_text))
            
            # Replace matches from right to left to maintain positions
            for match in reversed(matches):
                original = match.group()
                start, end = match.span()
                replacements.append((original, token))
                modified_text = modified_text[:start] + token + modified_text[end:]
                
        except re.error:
            continue
    
    return modified_text, replacements

def get_ner_pipeline(model_name: str):
    """Initialize NER pipeline for the given model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    ner_pipeline = pipeline("ner", 
                           model=model, 
                           tokenizer=tokenizer,
                           aggregation_strategy="simple",
                           device=0 if torch.cuda.is_available() else -1)
    
    return ner_pipeline

def mask_pii(sentence: str, ner_pipeline, model_name: str) -> str:
    """
    Main function to mask PII in a sentence.
    Handles both pattern-based and non-pattern-based models.
    """
    original_sentence = sentence
    pattern_tokens = set()
    
    # Step 1: Apply patterns if model uses them
    if uses_patterns(model_name):
        patterns_dict = get_patterns_for_model(model_name)
        pattern_tokens = set(pattern_info['token'] for pattern_info in patterns_dict.values())
        sentence, replacements = apply_patterns_to_text(sentence, patterns_dict)
    
    # Step 2: Run NER on the (possibly pattern-modified) sentence
    try:
        predictions = ner_pipeline(sentence)
    except Exception:
        # If NER fails, just replace pattern tokens with [MASK] if applicable
        if uses_patterns(model_name):
            for token in pattern_tokens:
                sentence = sentence.replace(token, '[MASK]')
        return sentence
    
    # Step 3: Create a mapping of character positions to mask
    char_to_mask = [False] * len(sentence)
    
    for pred in predictions:
        start_char = pred['start']
        end_char = pred['end']
        
        # Mark characters for masking
        for i in range(start_char, min(end_char, len(sentence))):
            char_to_mask[i] = True
    
    # Step 4: Also mark pattern tokens for masking
    if uses_patterns(model_name):
        for token in pattern_tokens:
            token_start = 0
            while True:
                pos = sentence.find(token, token_start)
                if pos == -1:
                    break
                for i in range(pos, pos + len(token)):
                    if i < len(char_to_mask):
                        char_to_mask[i] = True
                token_start = pos + 1
    
    # Step 5: Build the masked sentence
    result = []
    i = 0
    while i < len(sentence):
        if char_to_mask[i]:
            # Find the end of this masked region
            j = i
            while j < len(sentence) and char_to_mask[j]:
                j += 1
            result.append('[MASK]')
            i = j
        else:
            result.append(sentence[i])
            i += 1
    
    return ''.join(result) 