import re
from patterns_small import patterns
import string

def load_patterns():
    compiled_patterns = []
    for name, info in patterns.items():
        compiled_patterns.append((re.compile(info['pattern']), info['token']))
    return compiled_patterns

def clean_entity_text(text):
    # Remove all punctuation except URL-relevant ones
    url_safe_punct = ':/. -'
    allowed = set(url_safe_punct)
    # Remove all punctuation except allowed
    cleaned = ''.join(c for c in text if c.isalnum() or c in allowed)
    # Remove extra spaces
    cleaned = re.sub(r'\s+', '', cleaned)
    return cleaned

def process_conll_file(input_path, output_path):
    compiled_patterns = load_patterns()
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        entity_tokens = []
        entity_labels = []
        for line in infile:
            line = line.rstrip('\n')
            if not line or line.isspace():
                # End of sentence or entity
                if entity_tokens:
                    process_entity(entity_tokens, entity_labels, compiled_patterns, outfile)
                    entity_tokens = []
                    entity_labels = []
                outfile.write('\n')
                continue

            if '\t' in line:
                token, label = line.split('\t')
            elif ' ' in line:
                token, label = line.split(' ', 1)
            else:
                continue  # skip malformed lines

            if label.startswith('B-'):
                if entity_tokens:
                    process_entity(entity_tokens, entity_labels, compiled_patterns, outfile)
                entity_tokens = [token]
                entity_labels = [label]
            elif label.startswith('I-') and entity_labels and (entity_labels[-1][2:] == label[2:]):
                entity_tokens.append(token)
                entity_labels.append(label)
            else:
                if entity_tokens:
                    process_entity(entity_tokens, entity_labels, compiled_patterns, outfile)
                entity_tokens = [token]
                entity_labels = [label]

        # Process any remaining entity at EOF
        if entity_tokens:
            process_entity(entity_tokens, entity_labels, compiled_patterns, outfile)

def process_entity(tokens, labels, compiled_patterns, outfile):
    entity_type = labels[0][2:] if labels[0].startswith(('B-', 'I-')) else None
    entity_text = ''.join(tokens) if entity_type else tokens[0]
    replaced = False
    if entity_type in ('PII', 'MASK'):
        for pattern, token in compiled_patterns:
            if pattern.search(entity_text):
                outfile.write(f"{token} O\n")
                replaced = True
                break
    if not replaced:
        for t, l in zip(tokens, labels):
            outfile.write(f"{t} {l}\n")

if __name__ == "__main__":
    process_conll_file('conll_training_data.txt', 'conll_training_data_patterns_small.txt') 