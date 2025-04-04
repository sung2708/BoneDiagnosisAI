import torch
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms
from models.classifier import DiseaseClassifier
from configs.config import MODEL_CONFIG
import matplotlib.pyplot as plt
import numpy as np
import re
from typing import Dict, List, Union

class MedicalQuestionValidator:
    def __init__(self):
        # Mở rộng từ vựng y khoa chuyên sâu
        self.medical_terms = {
            'anatomy': [
                'xương đùi', 'xương chày', 'xương mác', 'xương bánh chè',
                'xương cánh tay', 'xương quay', 'xương trụ', 'xương cổ tay',
                'xương bàn tay', 'xương đốt ngón', 'xương chậu', 'xương hông',
                'xương cùng', 'xương cụt', 'xương sườn', 'xương ức', 'xương bả vai',
                'xương đòn', 'xương sọ', 'xương hàm', 'xương sống', 'đốt sống cổ',
                'đốt sống ngực', 'đốt sống thắt lưng', 'đĩa đệm', 'tủy sống',
                'khớp gối', 'khớp háng', 'khớp vai', 'khớp khuỷu', 'khớp cổ tay',
                'sụn khớp', 'dây chằng', 'gân', 'bao hoạt dịch'
            ],
            'pathology': [
                'gãy xương', 'nứt xương', 'rạn xương', 'gãy hở', 'gãy kín',
                'gãy di lệch', 'gãy không di lệch', 'loãng xương', 'viêm xương',
                'viêm tủy xương', 'lao xương', 'u xương', 'ung thư xương',
                'osteosarcoma', 'chondrosarcoma', 'ewing sarcoma', 'hoại tử xương',
                'thoái hóa khớp', 'viêm khớp dạng thấp', 'gout', 'loạn sản xương',
                'thoát vị đĩa đệm', 'gãy Colles', 'gãy Pouteau', 'gãy xương đùi',
                'gãy xương cẳng tay', 'viêm cột sống dính khớp', 'loãng xương',
                'nhuyễn xương', 'xương thủy tinh', 'hoại tử vô khuẩn chỏm xương đùi',
                'u tế bào khổng lồ', 'u nguyên bào sụn', 'u mạch xương'
            ],
            'symptoms': [
                'đau nhức xương', 'sưng khớp', 'nóng đỏ khớp', 'cứng khớp',
                'hạn chế vận động', 'biến dạng xương', 'tiếng lạo xạo khớp',
                'teo cơ', 'yếu chi', 'tê bì', 'dị cảm', 'mất vận động',
                'khó vận động', 'đau tăng về đêm', 'đau khi vận động',
                'sốt', 'mệt mỏi', 'sụt cân', 'phù nề', 'bầm tím',
                'co cứng cơ', 'vẹo cột sống', 'gù lưng', 'dáng đi bất thường'
            ],
            'imaging': [
                'X-quang', 'CT scan', 'MRI', 'siêu âm', 'xạ hình xương',
                'chụp cắt lớp', 'cộng hưởng từ', 'PET-CT', 'DXA', 'đo mật độ xương',
                'chụp tủy đồ', 'nội soi khớp', 'chụp mạch', 'chụp bao hoạt dịch',
                'chụp khớp cản quang', 'chụp xương bằng Technetium'
            ],
            'treatments': [
                'bó bột', 'nẹp vít', 'đóng đinh nội tủy', 'thay khớp',
                'vật lý trị liệu', 'phẫu thuật', 'ghép xương', 'tạo hình xương',
                'dùng thuốc giảm đau', 'tiêm khớp', 'dùng corticoid',
                'bổ sung canxi', 'bổ sung vitamin D', 'chống hủy xương',
                'tập phục hồi chức năng', 'chườm lạnh', 'chườm nóng',
                'kéo giãn cột sống', 'điều trị bằng tế bào gốc', 'xạ trị',
                'hóa trị', 'dùng bisphosphonate', 'tiêm huyết tương giàu tiểu cầu'
            ]
        }

        # Các từ/cụm từ không phải y khoa
        self.non_medical_patterns = [
            r'thời tiết', r'ăn uống', r'giờ giấc', r'ngày tháng',
            r'chế độ ăn', r'thức ăn', r'công việc', r'làm việc',
            r'chơi thể thao', r'tập luyện', r'đi lại', r'đứng ngồi',
            r'nghỉ ngơi', r'ngủ nghê', r'tâm lý', r'cảm xúc',
            r'gia đình', r'công ty', r'trường học', r'du lịch'
        ]

    def is_medical_question(self, question: str) -> bool:
        """Kiểm tra câu hỏi có đủ yếu tố y khoa không"""
        question = question.lower().strip()

        # Loại bỏ câu hỏi chứa từ không phù hợp
        if any(re.search(pattern, question) for pattern in self.non_medical_patterns):
            return False

        # Đếm số từ y khoa
        medical_count = 0
        for terms in self.medical_terms.values():
            medical_count += sum(1 for term in terms if term in question)

        return medical_count >= 2  # Cần ít nhất 2 từ y khoa

    def get_suggestions(self) -> List[str]:
        """Gợi ý câu hỏi mẫu"""
        return [
            "Trên phim X-quang có dấu hiệu gãy xương nào không?",
            "Có tổn thương viêm hoặc u xương trên hình ảnh không?",
            "Đánh giá tình trạng loãng xương trên phim chụp?",
            "Có dấu hiệu thoái hóa khớp hoặc viêm khớp không?",
            "Nhận xét về mật độ xương và các bất thường khác?",
            "Có bằng chứng của chấn thương xương khớp nào không?",
            "Đánh giá tình trạng viêm tủy xương trên phim chụp?",
            "Có dấu hiệu gãy xương hở hoặc biến chứng nhiễm trùng không?"
        ]

class ModelTester:
    def __init__(self, model_path):
        self.device = MODEL_CONFIG['device']
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['text_model'])
        self.validator = MedicalQuestionValidator()

        # Load model
        self.model = DiseaseClassifier().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((MODEL_CONFIG['image_size'], MODEL_CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(MODEL_CONFIG['image_mean'], MODEL_CONFIG['image_std'])
        ])

    def preprocess_question(self, question):
        """Chuẩn bị câu hỏi cho model"""
        return self.tokenizer(
            question,
            max_length=MODEL_CONFIG['max_len'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

    def predict(self, image_path, question, show_result=True):
        """Thực hiện dự đoán và trả về kết quả"""
        # Validate câu hỏi
        if not self.validator.is_medical_question(question):
            return {
                'valid': False,
                'error': 'Câu hỏi không đủ yếu tố y khoa về xương khớp',
                'suggestion': self.validator.get_suggestions(),
                'medical_terms': self._extract_medical_terms(question)
            }

        try:
            # Xử lý ảnh
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # Xử lý câu hỏi
            inputs = self.preprocess_question(question)

            # Dự đoán
            with torch.no_grad():
                outputs = self.model(img_tensor, inputs.input_ids, inputs.attention_mask)
                probs = torch.softmax(outputs['disease_probs'], dim=1).squeeze().cpu().numpy()
                pred_class = outputs['disease_pred'].item()
                confidence = outputs['confidence'].item()

            # Hiển thị kết quả
            if show_result:
                self._show_result(img, probs, pred_class, confidence, question)

            return {
                'valid': True,
                'prediction': MODEL_CONFIG['class_names'][pred_class],
                'confidence': float(confidence),
                'probabilities': {name: float(prob) for name, prob in zip(MODEL_CONFIG['class_names'], probs)},
                'medical_terms': self._extract_medical_terms(question)
            }

        except Exception as e:
            return {
                'valid': False,
                'error': f'Lỗi khi dự đoán: {str(e)}',
                'suggestion': 'Vui lòng kiểm tra lại ảnh và câu hỏi'
            }

    def _extract_medical_terms(self, question: str) -> List[str]:
        """Trích xuất các từ y khoa từ câu hỏi"""
        question = question.lower()
        found_terms = []
        for terms in self.validator.medical_terms.values():
            found_terms.extend(term for term in terms if term in question)
        return found_terms

    def _show_result(self, img, probs, pred_class, confidence, question):
        """Hiển thị kết quả trực quan"""
        plt.figure(figsize=(14, 6))

        # Hiển thị ảnh
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(
            f"Kết quả: {MODEL_CONFIG['class_names'][pred_class]}\n"
            f"Độ tin cậy: {confidence:.2%}\n"
            f"Câu hỏi: {question[:100]}..."
        )
        plt.axis('off')

        # Hiển thị xác suất
        plt.subplot(1, 2, 2)
        y_pos = np.arange(len(MODEL_CONFIG['class_names']))
        plt.barh(y_pos, probs, align='center', color='skyblue')
        plt.yticks(y_pos, MODEL_CONFIG['class_names'])
        plt.xlabel('Xác suất')
        plt.xlim(0, 1)
        plt.title('Phân phối xác suất các lớp bệnh')

        # Thêm giá trị xác suất
        for i, prob in enumerate(probs):
            plt.text(prob + 0.01, i, f"{prob:.2%}", va='center')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Khởi tạo và test
    tester = ModelTester("saved_models/best_model.pt")

    test_cases = [
        ("data/sample1.jpg", "Có dấu hiệu gãy xương đùi không?"),
        ("data/sample2.jpg", "Tôi bị đau khi đi lại nhiều"),  # Câu hỏi không đạt
        ("data/sample3.jpg", "Đánh giá tình trạng viêm khớp gối trên phim X-quang")
    ]

    for img_path, question in test_cases:
        print(f"\nĐang xử lý: {img_path} | Câu hỏi: '{question}'")
        result = tester.predict(img_path, question)

        if result['valid']:
            print(f"Kết quả: {result['prediction']} (Độ tin cậy: {result['confidence']:.2%})")
            print("Xác suất chi tiết:")
            for cls, prob in result['probabilities'].items():
                print(f"  - {cls}: {prob:.2%}")
            print(f"Các thuật ngữ y khoa được dùng: {', '.join(result['medical_terms'])}")
        else:
            print(f"Lỗi: {result['error']}")
            print("Gợi ý câu hỏi tốt hơn:")
            for i, suggestion in enumerate(result['suggestion'], 1):
                print(f"  {i}. {suggestion}")
            if result['medical_terms']:
                print(f"Các thuật ngữ y khoa tìm thấy: {', '.join(result['medical_terms'])}")
