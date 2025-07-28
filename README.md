# Maq_NER - Arabic Named Entity Recognition for PII Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.20+-green.svg)](https://huggingface.co/transformers/)
[![Competition](https://img.shields.io/badge/Competition-Arabic%20PII%20Challenge-orange.svg)](https://huggingface.co/spaces/Maqsam/Arabic_PII_Leaderboard)

A comprehensive Arabic Named Entity Recognition (NER) system designed for the [Arabic PII Redaction Challenge](https://huggingface.co/spaces/Maqsam/Arabic_PII_Leaderboard) on Hugging Face. This project addresses the unique challenges of Arabic PII detection by implementing multiple approaches combining deep learning models with pattern-based detection to achieve robust PII masking in Arabic text.

## 🏆 Competition Context

This project was developed for the **Arabic PII Redaction Challenge** - a $1000 prize competition focused on detecting and masking Personally Identifiable Information (PII) in Arabic text. The challenge addresses critical challenges in Arabic PII detection:

- **Arabic names with semantic meanings** that are harder to distinguish from regular words
- **Absence of capitalization** (unlike English where it's a key feature for name detection)  
- **Complex morphological structure** of Arabic text
- **Critical applications** in healthcare, finance, and government services across MENA region

### Competition Requirements
- Process **single Arabic sentences** at a time
- Replace detected PII with `[MASKED]` tokens
- Preserve sentence structure
- Handle multiple PII types: personal names, phone numbers, emails, addresses, national IDs, bank info, dates of birth

### Evaluation Metrics
The competition uses token-level classification with a sophisticated scoring system:
- **Precision**: Accuracy of masking decisions
- **Recall**: Completeness of PII detection
- **Final Score**: `0.45 × P_avg + 0.45 × R_avg + 0.1 × (1/time_avg)`

## 🚀 Features

- **Multiple Model Architectures**: Support for various BERT-based models including Arabic-BERT, CAMeL-BERT, and custom Arabic NER models
- **Pattern-Enhanced Detection**: Integration of 32+ regex patterns for improved PII detection accuracy
- **Hybrid Approaches**: Combination of NER and pattern-based validation for comprehensive coverage
- **Competition-Optimized**: 16 different model variants designed for the Arabic PII Challenge
- **Speed-Conscious Design**: Models optimized for the competition's speed requirements
- **Easy Integration**: Simple API for text masking and PII detection

## 📋 Supported PII Types

The system can detect and mask various types of personally identifiable information:

- **Contact Information**: Email addresses, phone numbers, URLs
- **Financial Data**: Credit card numbers, IBAN codes, cryptocurrency addresses
- **Personal Identifiers**: Social Security Numbers, VIN numbers, custom IDs
- **Network Information**: IP addresses (IPv4/IPv6), MAC addresses
- **Temporal Data**: Dates, times, timestamps
- **Technical Identifiers**: Hash tokens, license plates, user agent strings

## 🏗️ Project Structure

```
Maq_NER/
├── kaggle_ner_training.py          # Main training script
├── ner_validation.py               # Model validation and testing
├── patterns.py                     # Full pattern definitions (32+ patterns)
├── patterns_small.py               # Simplified patterns (3 key patterns)
├── prepare_conll_data*.py          # Data preparation scripts
├── conll_training_data*.txt        # Training datasets
├── submission/                     # Competition submission models
│   ├── submission_helper.py        # Core masking logic
│   ├── submission_*_original.py    # Pure NER models
│   ├── submission_*_patterns.py    # Pattern-enhanced models
│   └── submission_*_plus_patterns.py # Hybrid models
└── Data/                           # Additional data files
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AlmutazYounes/Maq_NER.git
   cd Maq_NER
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install torch transformers datasets evaluate scikit-learn
   ```

## 🚀 Quick Start

### Basic Usage

```python
from submission.submission_Arabic_NER_PII_patterns import run

# Example Arabic text with PII (competition format)
text = "يعمل احمد محمد في شركة تقنية"

# Mask PII in the text
masked_text = run(text)
print(masked_text)
# Output: "يعمل [MASKED] [MASKED] في شركة تقنية"

# Another example with different PII types
text = "للتواصل مع سارة على الرقم 0501234567"
masked_text = run(text)
# Output: "للتواصل مع [MASKED] على الرقم [MASKED]"
```

### Training Your Own Model

```python
# Train on original dataset
python kaggle_ner_training.py

# The script will train models on both:
# - conll_training_data.txt (original)
# - conll_training_data_patterns.txt (pattern-enhanced)
```

## 📊 Model Variants

Our 16 model variants are strategically designed to maximize both precision and recall while considering speed requirements for the competition:

### Original Models (Pure NER)
- `submission_ARAB_BERT_original.py`
- `submission_arabic_ner_masking_original.py`
- `submission_Arabic_NER_PII_original.py`
- `submission_bert_base_arabic_camelbert_mix_ner_original.py`

### Pattern-Enhanced Models
**Full Patterns (32+ regex patterns)**:
- `submission_ARAB_BERT_patterns.py`
- `submission_arabic_ner_masking_patterns.py`
- `submission_Arabic_NER_PII_patterns.py`
- `submission_bert_base_arabic_camelbert_mix_ner_patterns.py`

**Small Patterns (3 key patterns)**:
- `submission_ARAB_BERT_patterns_small.py`
- `submission_Arabic_NER_PII_patterns_small.py`
- `submission_bert_base_arabic_camelbert_mix_ner_patterns_small.py`

### Hybrid Models (NER + Pattern Validation)
- `submission_Arabic_NER_PII_plus_patterns.py`
- `submission_Arabic_NER_PII_plus_patterns_small.py`

## 🔧 Configuration

### Training Parameters
- **Learning Rate**: 2e-5
- **Batch Size**: 8 (train), 16 (eval)
- **Epochs**: 3
- **Max Sequence Length**: 512 tokens
- **Evaluation Strategy**: End of epoch
- **Model Checkpointing**: Best model based on F1 score

### Pattern Categories
The system includes patterns for:
- Email addresses and URLs
- Financial information (credit cards, IBAN, crypto)
- Personal identifiers (SSN, VIN, custom IDs)
- Network addresses (IP, MAC)
- Temporal data (dates, times)
- Technical tokens (hashes, user agents)

## 📈 Performance

The models are evaluated using the competition's token-level classification metrics:

### Core Metrics
- **Precision**: Accuracy of masking decisions (`TP / (TP + FP)`)
- **Recall**: Completeness of PII detection (`TP / (TP + FN)`)
- **F1-Score**: Harmonic mean of precision and recall

### Competition Scoring
The final competition score is calculated as:
```
Score_final = 0.45 × P_avg + 0.45 × R_avg + 0.1 × (1/time_avg)
```

Where:
- `P_avg`: Average precision across all test sentences
- `R_avg`: Average recall across all test sentences  
- `time_avg`: Average processing time per sentence
