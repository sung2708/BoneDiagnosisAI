# BoneDiagnosisAI

## 📌 Giới thiệu  
**BoneDiagnosisAI** là một hệ thống chẩn đoán bệnh về xương bằng trí tuệ nhân tạo, kết hợp xử lý hình ảnh và ngôn ngữ tự nhiên. Hệ thống sử dụng **ResNet** để trích xuất đặc trưng từ ảnh X-quang và **BERT** để biểu diễn thông tin từ câu hỏi của bác sĩ/nhân viên y tế. Hai nguồn thông tin này được kết hợp bằng phương pháp **Bi-Pooling**, sau đó được đưa vào mạng phân loại để dự đoán loại bệnh.

## 🏥 Ứng dụng  
Hệ thống có thể hỗ trợ bác sĩ trong việc chẩn đoán và phân loại các bệnh liên quan đến xương dựa trên dữ liệu hình ảnh và mô tả triệu chứng từ bệnh nhân.  

Bộ dữ liệu huấn luyện gồm 3 nhóm bệnh chính:  
- **U xương**  
- **Viêm nhiễm**  
- **Chấn thương**  

## 🛠 Công nghệ sử dụng  
- **Xử lý hình ảnh:** ResNet152  
- **Xử lý ngôn ngữ:** BERT  
- **Kết hợp đặc trưng:** Bi-Pooling  
- **Phân loại bệnh:** Mạng nơ-ron nhiều lớp (MLP) 
