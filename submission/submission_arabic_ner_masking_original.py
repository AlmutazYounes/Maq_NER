"""
Submission script for MutazYoune/arabic-ner-masking-original
This model does not use pattern replacement.
"""

from submission_helper import get_ner_pipeline, mask_pii

# Model configuration
MODEL_NAME = "MutazYoune/arabic-ner-masking-original"

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
    """
    ner_pipeline = get_ner_pipeline_cached()
    return mask_pii(sentence, ner_pipeline, MODEL_NAME)

# Example usage (remove or comment out in production)
if __name__ == "__main__":
    example_sentences = "مرحباً، يمكنك التواصل مع أحمد عبر البريد Ahmad.AlSaeed99@example.com أو support_team123@mail.co.uk، أو الاتصال على 055-123-4567، +1 (800) 555-0199، و0100.555.7890، كما تم تسجيل الدخول من عناوين IP مثل 10.0.0.5 و172.16.254.1، ورقم الهوية الوطنية 9876543210123، ورقم جواز السفر P1234567، ورقم بطاقة الائتمان 4111-1111-1111-1111، مع تاريخ الميلاد 1990-05-31، ولا تنسى زيارة مواقع https://secure-bank.example.com وhttp://intranet.local لتحديث بيانات المستخدم user_2025 أو user.name_42."
    masked = run(example_sentences)
    print(f"Original: {example_sentences}\nMasked:   {masked}\n") 