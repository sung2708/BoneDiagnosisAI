import re
from configs.config import MODEL_CONFIG

class QuestionValidator:
    def __init__(self):
        self.medical_keywords = MODEL_CONFIG['medical_keywords']
        self.question_patterns = [
            r"có.*không",
            r"dấu hiệu.*gì",
            r"nguyên nhân.*là",
            r"biểu hiện.*ra sao"
        ]

    def preprocess(self, text):
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def is_valid_medical_question(self, text):
        text = self.preprocess(text)

        # Kiểm tra từ khóa y tế
        has_medical_terms = any(
            keyword in text for keyword in self.medical_keywords
        )

        # Kiểm tra cấu trúc câu hỏi
        is_question = any(
            re.search(pattern, text) for pattern in self.question_patterns
        ) or text.endswith('?')

        return has_medical_terms and is_question
