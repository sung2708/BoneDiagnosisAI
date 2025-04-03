import torch
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms
from models.classifier import DiseaseClassifier
from configs.config import MODEL_CONFIG
import matplotlib.pyplot as plt
import numpy as np

class ModelTester:
    def __init__(self, model_path):
        self.device = MODEL_CONFIG['device']
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['text_model'])
        self.model = DiseaseClassifier().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((MODEL_CONFIG['image_size'], MODEL_CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(MODEL_CONFIG['image_mean'], MODEL_CONFIG['image_std'])
        ])

    def preprocess_question(self, question):
        return self.tokenizer(
            question,
            max_length=MODEL_CONFIG['max_len'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

    def predict(self, image_path, question):
        # Xử lý ảnh
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Xử lý câu hỏi
        inputs = self.preprocess_question(question)

        # Dự đoán
        with torch.no_grad():
            outputs = self.model(img_tensor, inputs.input_ids, inputs.attention_mask)

        # Lấy kết quả
        probs = torch.softmax(outputs['disease_probs'], dim=1).squeeze().cpu().numpy()
        pred_class = outputs['disease_pred'].item()
        confidence = outputs['confidence'].item()

        # Visualize
        self._show_result(img, probs, pred_class, confidence, question)

        return {
            'prediction': MODEL_CONFIG['class_names'][pred_class],
            'confidence': float(confidence),
            'probabilities': {name: float(prob) for name, prob in zip(MODEL_CONFIG['class_names'], probs)}
        }

    def _show_result(self, img, probs, pred_class, confidence, question):
        plt.figure(figsize=(12, 6))

        # Hiển thị ảnh
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Prediction: {MODEL_CONFIG['class_names'][pred_class]}\nConfidence: {confidence:.2f}")
        plt.axis('off')

        # Hiển thị biểu đồ xác suất
        plt.subplot(1, 2, 2)
        y_pos = np.arange(len(MODEL_CONFIG['class_names']))
        plt.barh(y_pos, probs, align='center')
        plt.yticks(y_pos, MODEL_CONFIG['class_names'])
        plt.xlabel('Probability')
        plt.title('Class Probabilities')
        plt.xlim(0, 1)

        for i, prob in enumerate(probs):
            plt.text(prob + 0.02, i, f"{prob:.2%}", va='center')

        plt.suptitle(f"Q: {question}", y=0.98)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    tester = ModelTester("saved_models/best_model.pt")

    # Test với dữ liệu mẫu
    test_cases = [
        ("data/U_xuong/sample1.jpg", "Có dấu hiệu u xương không?"),
        ("data/Viem_nhiem/sample2.jpg", "Ảnh X-quang có viêm nhiễm không?"),
        ("data/Chan_thuong/sample3.jpg", "Có chấn thương xương nào không?")
    ]

    for img_path, question in test_cases:
        try:
            result = tester.predict(img_path, question)
            print("\nTest Result:")
            print(f"Image: {img_path}")
            print(f"Question: {question}")
            print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2%})")
            print("Probabilities:")
            for cls, prob in result['probabilities'].items():
                print(f"- {cls}: {prob:.2%}")
        except Exception as e:
            print(f"\nError processing {img_path}: {str(e)}")
