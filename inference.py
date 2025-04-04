import torch
from transformers import AutoTokenizer
from PIL import Image
from torchvision import transforms
from models.classifier import DiseaseClassifier
from configs.config import MODEL_CONFIG

class BoneDiseaseInference:
    def __init__(self, model_path):
        self.device = MODEL_CONFIG['device']
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['text_model'])
        self.model = DiseaseClassifier().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Đảm bảo class_names theo đúng thứ tự training
        self.class_names = ["U_xuong", "Viem_nhiem", "Chan_thuong"]

        self.transform = transforms.Compose([
            transforms.Resize((MODEL_CONFIG['image_size'], MODEL_CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(MODEL_CONFIG['image_mean'], MODEL_CONFIG['image_std'])
        ])

    def validate_question(self, question):
        """Kiểm tra tính hợp lệ của câu hỏi"""
        required_keywords = ["xương", "X-quang", "chụp"]
        question_lower = question.lower()
        return any(kw in question_lower for kw in required_keywords)

    def process_inputs(self, image_path, question):
        # Xử lý ảnh
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)

        # Xử lý câu hỏi
        inputs = self.tokenizer(
            question,
            max_length=MODEL_CONFIG['max_len'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        return image, inputs.input_ids, inputs.attention_mask

    def predict(self, image_path, question):
        if not self.validate_question(question):
            return {
                "error": "Câu hỏi không phù hợp. Vui lòng hỏi về vấn đề X-quang/xương khớp."
            }

        image, input_ids, attention_mask = self.process_inputs(image_path, question)

        with torch.no_grad():
            outputs = self.model(image, input_ids, attention_mask)

        # Lấy kết quả
        probs = outputs['disease_probs'].squeeze().cpu().numpy()
        pred_class_idx = outputs['disease_pred'].item()

        return {
            "prediction": self.class_names[pred_class_idx],
            "confidence": float(outputs['confidence'].item()),
            "probabilities": {
                self.class_names[i]: float(probs[i])
                for i in range(len(self.class_names))
            },
            "question_analysis": {
                "is_valid": True,
                "keywords": self._extract_keywords(question)
            }
        }

    def _extract_keywords(self, question):
        medical_keywords = MODEL_CONFIG['medical_keywords']
        return [kw for kw in medical_keywords if kw in question.lower()]
