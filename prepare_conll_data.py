import pandas as pd
import re
import os
from typing import List, Tuple, Dict

def examine_data_structure(file_path: str):
    """Examine the structure of the Excel file"""
    print("Loading Excel file...")
    df = pd.read_excel(file_path)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nSample data:")
    for i in range(min(3, len(df))):
        print(f"\nRow {i+1}:")
        print(f"Source: {df.iloc[i]['source']}")
        print(f"Target: {df.iloc[i]['target']}")
        print(f"Masked values: {df.iloc[i]['masked_value']}")
    return df

def tokenize_simple(text: str) -> List[str]:
    """Simple tokenization - split by whitespace and punctuation"""
    # Split by whitespace first
    tokens = text.split()
    result = []
    
    for token in tokens:
        # Split punctuation from words
        # This is a simple approach - for production, use proper tokenizers
        words = re.findall(r'\w+|[^\w\s]', token)
        result.extend(words)
    
    return result

def align_tokens_with_masks(source_tokens: List[str], target_tokens: List[str], masked_values: List[str]) -> List[Tuple[str, str]]:
    """
    Align source tokens with target tokens to identify PII entities
    Returns list of (token, label) pairs
    """
    labels = []
    source_idx = 0
    target_idx = 0
    mask_idx = 0
    
    while source_idx < len(source_tokens) and target_idx < len(target_tokens):
        source_token = source_tokens[source_idx]
        target_token = target_tokens[target_idx]
        
        if target_token == '[MASK]':
            # This is a masked token - need to find corresponding PII in source
            if mask_idx < len(masked_values):
                masked_value = masked_values[mask_idx].strip()
                # Find how many source tokens make up this masked value
                
                # Simple approach: check if current source token is part of masked value
                if source_token.lower() in masked_value.lower():
                    # Mark as PII entity - using BIO tagging
                    if mask_idx == 0 or not any(source_tokens[source_idx-1].lower() in mv.lower() for mv in masked_values[:mask_idx]):
                        labels.append((source_token, 'B-PII'))
                    else:
                        labels.append((source_token, 'I-PII'))
                    source_idx += 1
                    
                    # Check if we've consumed all tokens for this masked value
                    remaining_masked = masked_value.lower().replace(source_token.lower(), '', 1).strip()
                    if not remaining_masked or remaining_masked in ['.', ',', '!', '?']:
                        mask_idx += 1
                        target_idx += 1
                else:
                    # This might be a multi-token PII entity
                    labels.append((source_token, 'I-PII'))
                    source_idx += 1
            else:
                # No more masked values, but still have [MASK] tokens
                labels.append((source_token, 'B-PII'))
                source_idx += 1
                target_idx += 1
        else:
            # Regular token - should match between source and target
            if source_token == target_token:
                labels.append((source_token, 'O'))
                source_idx += 1
                target_idx += 1
            else:
                # Tokens don't match - skip source token and mark as non-entity
                labels.append((source_token, 'O'))
                source_idx += 1
    
    # Handle remaining source tokens
    while source_idx < len(source_tokens):
        labels.append((source_tokens[source_idx], 'O'))
        source_idx += 1
    
    return labels

def create_better_alignment(source: str, target: str, masked_values_str: str) -> List[Tuple[str, str]]:
    """
    Better approach to align tokens using the mask positions
    """
    # Parse masked values
    if pd.isna(masked_values_str) or masked_values_str == '':
        masked_values = []
    else:
        masked_values = [val.strip() for val in str(masked_values_str).split('|') if val.strip()]
    
    # Tokenize source
    source_tokens = tokenize_simple(source)
    
    # For each masked value, find its position in source and mark as PII
    labels = ['O'] * len(source_tokens)
    
    for masked_value in masked_values:
        # Find the masked value in the source text
        masked_tokens = tokenize_simple(masked_value)
        
        # Look for sequence of tokens in source that match masked value
        for i in range(len(source_tokens) - len(masked_tokens) + 1):
            match = True
            for j, masked_token in enumerate(masked_tokens):
                if source_tokens[i + j].lower() != masked_token.lower():
                    match = False
                    break
            
            if match:
                # Mark these tokens as PII
                labels[i] = 'B-PII'
                for j in range(1, len(masked_tokens)):
                    labels[i + j] = 'I-PII'
                break
    
    return list(zip(source_tokens, labels))

def process_excel_to_conll(file_path: str, output_path: str):
    """Convert Excel data to CoNLL format"""
    print("Loading Excel file...")
    df = pd.read_excel(file_path)
    
    print(f"Processing {len(df)} sentences...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"Processed {idx} sentences...")
            
            source = str(row['source']).strip()
            target = str(row['target']).strip()
            masked_values_str = str(row['masked_value']).strip()
            
            # Skip empty or invalid rows
            if not source or source == 'nan':
                continue
            
            # Create token-label pairs
            token_labels = create_better_alignment(source, target, masked_values_str)
            
            # Write to CoNLL format
            for token, label in token_labels:
                f.write(f"{token}\t{label}\n")
            
            # Empty line to separate sentences
            f.write("\n")
    
    print(f"CoNLL data saved to {output_path}")

def validate_conll_data(file_path: str):
    """Validate the generated CoNLL data"""
    print(f"Validating CoNLL data in {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    sentences = []
    current_sentence = []
    entity_count = 0
    
    for line in lines:
        line = line.strip()
        if line == '':
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
        else:
            parts = line.split('\t')
            if len(parts) == 2:
                token, label = parts
                current_sentence.append((token, label))
                if label.startswith('B-'):
                    entity_count += 1
    
    if current_sentence:
        sentences.append(current_sentence)
    
    print(f"Total sentences: {len(sentences)}")
    print(f"Total entities: {entity_count}")
    
    # Show some examples
    print("\nFirst few sentences:")
    for i, sentence in enumerate(sentences[:3]):
        print(f"\nSentence {i+1}:")
        for token, label in sentence:
            print(f"{token}\t{label}")

if __name__ == "__main__":
    input_file = "Data/Cleaned_data_mask_only.xlsx"
    output_file = "conll_training_data.txt"
    
    # First examine the data structure
    print("=== Examining Data Structure ===")
    df = examine_data_structure(input_file)
    
    print("\n=== Converting to CoNLL Format ===")
    process_excel_to_conll(input_file, output_file)
    
    print("\n=== Validating Generated Data ===")
    validate_conll_data(output_file)
    
    print(f"\nTraining data has been saved to: {output_file}")
    print("The file is ready for BERT NER training!") 