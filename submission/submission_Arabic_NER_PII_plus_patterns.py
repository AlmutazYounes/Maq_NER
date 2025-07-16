"""
Submission script for MutazYoune/Arabic-NER-PII + Full Patterns
This model runs NER first, then applies full pattern checking to catch missed PII.
"""

from submission_helper import get_ner_pipeline, mask_pii, patterns, apply_patterns_to_text
import re

# Model configuration
MODEL_NAME = "MutazYoune/Arabic-NER-PII"

# Global pipeline (will be initialized once)
_ner_pipeline = None

def get_ner_pipeline_cached():
    """Get NER pipeline (cached)."""
    global _ner_pipeline
    if _ner_pipeline is None:
        _ner_pipeline = get_ner_pipeline(MODEL_NAME)
    return _ner_pipeline

def run(sentence):
    """
    Receives a list of input sentences and returns a list of masked sentences.
    Uses NER first, then applies full pattern checking to catch any missed PII.
    """
    # Step 1: Get NER predictions on original sentence
    ner_pipeline = get_ner_pipeline_cached()
    
    try:
        ner_predictions = ner_pipeline(sentence)
    except Exception:
        ner_predictions = []
    
    # Step 2: Create character mapping for masking
    char_to_mask = [False] * len(sentence)
    
    # Step 3: Mark NER-detected entities for masking
    for pred in ner_predictions:
        start_char = pred['start']
        end_char = pred['end']
        for i in range(start_char, min(end_char, len(sentence))):
            char_to_mask[i] = True
    
    # Step 4: Apply pattern matching and mark additional characters for masking
    for pattern_name, pattern_info in patterns.items():
        try:
            pattern = pattern_info['pattern']
            matches = list(re.finditer(pattern, sentence))
            
            for match in matches:
                start, end = match.span()
                for i in range(start, min(end, len(sentence))):
                    char_to_mask[i] = True
                    
        except re.error:
            continue
    
    # Step 5: Build final result with all masked regions
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

# Example usage (remove or comment out in production)
if __name__ == "__main__":
    example_sentences = "مرحباً، يمكنك التواصل مع أحمد عبر البريد Ahmad.AlSaeed99@example.com أو support_team123@mail.co.uk، أو الاتصال على 055-123-4567، +1 (800) 555-0199، و0100.555.7890، كما تم تسجيل الدخول من عناوين IP مثل 10.0.0.5 و172.16.254.1، ورقم الهوية الوطنية 9876543210123، ورقم جواز السفر P1234567، ورقم بطاقة الائتمان 4111-1111-1111-1111، مع تاريخ الميلاد 1990-05-31، ولا تنسى زيارة مواقع https://secure-bank.example.com وhttp://intranet.local لتحديث بيانات المستخدم user_2025 أو user.name_42."
    masked = run(example_sentences)
    print(f"Original: {example_sentences}\nMasked:   {masked}\n") 