# Arabic PII Detection Competition - Model Submissions

This submission contains **16 different Arabic NER models** for PII (Personally Identifiable Information) detection, implementing three distinct masking strategies with robust character-level masking.

## 🏗️ Architecture Overview

### Core System
- **`submission_helper.py`** - Centralized masking engine with character-level precision
- **Pattern-First Approach** - Apply regex patterns before NER for enhanced detection
- **Hybrid Post-Processing** - NER followed by pattern validation
- **Robust Masking** - Character-level mapping prevents partial token issues

### Pattern Sets
- **Full Patterns (32+ patterns)** - Comprehensive PII detection including emails, URLs, IPs, credit cards, etc.
- **Small Patterns (3 patterns)** - Focused on Email, URL, and User-Agent strings
- **No Patterns** - Pure NER-based detection

## 📁 Model Categories

### 🟦 Original Models (5 models)
Pure NER-based detection without pattern preprocessing:
```
submission_ARAB_BERT_original.py
submission_arabic_ner_masking_original.py  
submission_Arabic_NER_PII_original.py
submission_Arabic_NER_PII.py
submission_bert_base_arabic_camelbert_mix_ner_original.py
```

### 🟨 Full Pattern Models (4 models)
Apply 32+ patterns before NER processing:
```
submission_ARAB_BERT_patterns.py
submission_arabic_ner_masking_patterns.py
submission_Arabic_NER_PII_patterns.py
submission_bert_base_arabic_camelbert_mix_ner_patterns.py
```

### 🟩 Small Pattern Models (3 models)
Apply 3 key patterns before NER processing:
```
submission_ARAB_BERT_patterns_small.py
submission_Arabic_NER_PII_patterns_small.py
submission_bert_base_arabic_camelbert_mix_ner_patterns_small.py
```

### 🟪 Hybrid Models (2 models)
NER first, then pattern validation for missed entities:
```
submission_Arabic_NER_PII_plus_patterns.py        # NER + Full Patterns
submission_Arabic_NER_PII_plus_patterns_small.py  # NER + Small Patterns
```

### 🟫 Base Models (2 models)
Standard implementations:
```
submission_Arabic_NER_PII.py
submission_Arabic_NER_PII_original.py
```

## 🔧 Technical Implementation

### Pattern-First Workflow
```python
# 1. Apply regex patterns to text
sentence = "Contact Ahmed at ahmed@example.com"
modified = "Contact Ahmed at [EMAIL]"

# 2. Run NER on pattern-modified text
ner_predictions = model(modified)  # Detects "Ahmed" as PERSON

# 3. Character-level masking
char_mask = [False] * len(original)
# Mark NER predictions
char_mask[8:13] = True   # "Ahmed"
# Mark pattern tokens  
char_mask[17:24] = True  # "[EMAIL]" -> original email position

# 4. Generate final output
result = "Contact [MASK] at [MASK]"
```

### Hybrid Post-Processing Workflow
```python
# 1. Run NER on original text
ner_predictions = model("Contact Ahmed at ahmed@example.com")

# 2. Apply patterns to original text
pattern_matches = find_patterns("Contact Ahmed at ahmed@example.com")

# 3. Combine both using character-level mapping
char_mask = mark_ner_regions(ner_predictions) | mark_pattern_regions(pattern_matches)

# 4. Generate final output with robust masking
result = apply_character_mask(original_text, char_mask)
```

## 🎯 Model Selection Strategy

| Use Case | Recommended Models |
|----------|-------------------|
| **Maximum Recall** | Full Pattern Models (32+ patterns) |
| **Balanced Performance** | Small Pattern Models (key patterns) |
| **Pure NER Evaluation** | Original Models (no preprocessing) |
| **Comprehensive Detection** | Hybrid Models (dual validation) |
| **Base Comparison** | Base Models (standard approach) |

## 🚀 Usage

### Basic Usage
```python
# Import any model
from submission_ARAB_BERT_patterns_small import run

# Input text with PII
text = "اسمي أحمد وإيميلي ahmed@example.com ورقمي 055-123-4567"

# Get masked output
masked = run(text)
print(masked)  # "اسمي [MASK] وإيميلي [MASK] ورقمي [MASK]"
```

### Advanced Example
```python
# Test different approaches
from submission_Arabic_NER_PII_original import run as run_original
from submission_Arabic_NER_PII_patterns import run as run_patterns  
from submission_Arabic_NER_PII_plus_patterns import run as run_hybrid

text = "Visit https://bank.com, call +1-800-555-0199, ID: 1234567890"

print("Original:", run_original(text))
print("Patterns:", run_patterns(text))  
print("Hybrid:", run_hybrid(text))
```

## 🛠️ Requirements

```python
torch>=1.9.0
transformers>=4.20.0
re  # Built-in
warnings  # Built-in
```

## 🧪 Testing

```bash
# Test individual models
python submission_ARAB_BERT_patterns_small.py

# Test with custom input
python -c "
from submission_helper import mask_pii, get_ner_pipeline
pipeline = get_ner_pipeline('MutazYoune/ARAB_BERT-patterns_small')
result = mask_pii('Your test text here', pipeline, 'MutazYoune/ARAB_BERT-patterns_small')
print(result)
"
```

## 📊 Performance Characteristics

- **Character-Level Precision** - No partial token masking issues
- **Overlap Handling** - Robust handling of overlapping NER + pattern detections  
- **Error Recovery** - Graceful fallback if NER fails
- **Memory Efficient** - Model caching prevents reloading
- **Extensible Patterns** - Easy to add new regex patterns

## 🏆 Competition Strategy

This submission provides comprehensive coverage with 16 different approaches, allowing the evaluation system to select the optimal model based on:
- Dataset characteristics
- Precision/recall requirements  
- Processing constraints
- Domain-specific PII patterns

---

**Ready for submission! 🚀** 