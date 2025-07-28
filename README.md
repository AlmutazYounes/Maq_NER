# Maq_NER - Arabic Named Entity Recognition for PII Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.20+-green.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive Arabic Named Entity Recognition (NER) system designed for Personally Identifiable Information (PII) detection. This project implements multiple approaches combining deep learning models with pattern-based detection to achieve robust PII masking in Arabic text.

## 🚀 Features

- **Multiple Model Architectures**: Support for various BERT-based models including Arabic-BERT, CAMeL-BERT, and custom Arabic NER models
- **Pattern-Enhanced Detection**: Integration of 32+ regex patterns for improved PII detection accuracy
- **Hybrid Approaches**: Combination of NER and pattern-based validation for comprehensive coverage
- **Competition-Ready**: 16 different model variants optimized for Kaggle competitions
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

# Example Arabic text with PII
text = "اسمي أحمد محمد وإيميلي ahmed@example.com ورقمي 123-45-6789"

# Mask PII in the text
masked_text = run(text)
print(masked_text)
# Output: "اسمي [MASK] [MASK] وإيميلي [MASK] ورقمي [MASK]"
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

The models are evaluated using standard NER metrics:
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall prediction accuracy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of Hugging Face Transformers library
- Uses Arabic-BERT and CAMeL-BERT models
- Inspired by Arabic NLP research and competitions

## 📞 Contact

- **Author**: Almutaz Younes
- **GitHub**: [@AlmutazYounes](https://github.com/AlmutazYounes)
- **Repository**: [Maq_NER](https://github.com/AlmutazYounes/Maq_NER)

---

⭐ **Star this repository if you find it useful!**
