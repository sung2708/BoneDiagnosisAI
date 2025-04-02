import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
from transformers import AutoTokenizer
from torchvision import transforms
import json
import numpy as np
from models.classifier import DiseaseClassifier
from configs import MODEL_CONFIG

# Load model và tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiseaseClassifier().to(device)
model.load_state_dict(torch.load("path/to/best_model.pt", map_location=device))
model.eval()
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

# Transform ảnh
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Ánh xạ ID bệnh
disease_map = {
    0: "U xương",
    1: "Viêm nhiễm",
    2: "Chấn thương"
}

class MedicalVQAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chẩn đoán bệnh từ ảnh X-quang")
        self.root.geometry("900x700")

        # Biến lưu trữ
        self.image_path = ""
        self.mask_path = ""
        self.image_tk = None
        self.mask_tk = None
        self.use_mask = tk.BooleanVar(value=False)  # Mặc định không dùng mask

        # Giao diện
        self.setup_ui()

    def setup_ui(self):
        # Frame chế độ mask
        frame_mode = tk.Frame(self.root, padx=10, pady=5)
        frame_mode.pack(fill=tk.X)
        ttk.Checkbutton(frame_mode, text="Sử dụng mask", variable=self.use_mask,
                       command=self.toggle_mask_input).pack(side=tk.LEFT)

        # Frame upload ảnh
        frame_upload = tk.Frame(self.root, padx=10, pady=5)
        frame_upload.pack(fill=tk.X)

        tk.Button(frame_upload, text="Chọn ảnh X-quang", command=self.load_image).pack(side=tk.LEFT)
        self.label_image = tk.Label(frame_upload, text="Chưa chọn ảnh", width=40, anchor='w')
        self.label_image.pack(side=tk.LEFT, padx=10)

        # Frame upload mask (ẩn ban đầu)
        self.frame_mask = tk.Frame(self.root, padx=10, pady=5)
        tk.Button(self.frame_mask, text="Chọn mask", command=self.load_mask).pack(side=tk.LEFT)
        self.label_mask = tk.Label(self.frame_mask, text="Chưa chọn mask", width=40, anchor='w')
        self.label_mask.pack(side=tk.LEFT, padx=10)

        # Hiển thị ảnh và mask
        self.img_frame = tk.Frame(self.root)
        self.img_frame.pack(pady=10)

        self.canvas_image = tk.Canvas(self.img_frame, width=400, height=300, bg='white')
        self.canvas_image.pack(side=tk.LEFT, padx=10)

        self.canvas_mask = tk.Canvas(self.img_frame, width=400, height=300, bg='white')
        self.canvas_mask.pack(side=tk.LEFT, padx=10)

        # Nhập câu hỏi
        frame_question = tk.Frame(self.root, padx=10, pady=10)
        frame_question.pack(fill=tk.X)

        tk.Label(frame_question, text="Câu hỏi:").pack(side=tk.LEFT)
        self.entry_question = ttk.Entry(frame_question, width=70)
        self.entry_question.insert(0, "Bệnh nhân có triệu chứng gì trong ảnh X-quang này?")
        self.entry_question.pack(side=tk.LEFT, padx=10)

        # Nút dự đoán
        tk.Button(self.root, text="Chẩn đoán", command=self.predict,
                 bg="#4CAF50", fg="white", font=('Arial', 12)).pack(pady=10)

        # Hiển thị kết quả
        self.result_frame = tk.Frame(self.root)
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(self.result_frame, text="Kết quả chẩn đoán:", font=('Arial', 12, 'bold')).pack(anchor='w')

        self.result_text = tk.Text(self.result_frame, height=12, width=90, state=tk.DISABLED,
                                 font=('Consolas', 10), wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(self.result_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)

        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def toggle_mask_input(self):
        if self.use_mask.get():
            self.frame_mask.pack(fill=tk.X)
        else:
            self.frame_mask.pack_forget()
            self.mask_path = ""
            self.canvas_mask.delete("all")

    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Chọn ảnh X-quang",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if self.image_path:
            self.label_image.config(text=self.image_path.split('/')[-1])
            self.display_image(self.image_path, self.canvas_image)

    def load_mask(self):
        self.mask_path = filedialog.askopenfilename(
            title="Chọn ảnh mask",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if self.mask_path:
            self.label_mask.config(text=self.mask_path.split('/')[-1])
            self.display_image(self.mask_path, self.canvas_mask, is_mask=True)

    def display_image(self, path, canvas, is_mask=False):
        img = Image.open(path)
        img.thumbnail((400, 300))

        if is_mask:
            img = img.convert("L")  # Chuyển sang grayscale
            self.mask_tk = ImageTk.PhotoImage(img)
        else:
            self.image_tk = ImageTk.PhotoImage(img)

        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=self.mask_tk if is_mask else self.image_tk)

    def predict(self):
        if not self.image_path:
            messagebox.showerror("Lỗi", "Vui lòng chọn ảnh X-quang trước!")
            return

        question = self.entry_question.get()
        if not question:
            messagebox.showerror("Lỗi", "Vui lòng nhập câu hỏi!")
            return

        try:
            # Tiền xử lý ảnh
            image = Image.open(self.image_path).convert('RGB')
            image_tensor = image_transform(image).unsqueeze(0).to(device)

            # Tiền xử lý mask (nếu có)
            mask_tensor = None
            if self.use_mask.get() and self.mask_path:
                mask = Image.open(self.mask_path).convert('L')
                mask = mask.resize((224, 224))
                mask_tensor = transforms.ToTensor()(mask).unsqueeze(0).to(device)
            elif self.use_mask.get():
                # Tạo mask mặc định (toàn 1) nếu chọn chế độ mask nhưng không chọn file
                mask_tensor = torch.ones_like(image_tensor[:, :1, :, :]).to(device)

            # Tiền xử lý câu hỏi
            inputs = tokenizer(
                question,
                return_tensors="pt",
                padding='max_length',
                max_length=MODEL_CONFIG.max_len,
                truncation=True
            ).to(device)

            # Dự đoán
            with torch.no_grad():
                if mask_tensor is not None:
                    outputs = model(image_tensor, mask_tensor, inputs['input_ids'], inputs['attention_mask'])
                else:
                    outputs = model(image_tensor, inputs['input_ids'], inputs['attention_mask'])

                disease_probs = torch.softmax(outputs['disease'], dim=1)[0]
                confidence = outputs['confidence'].item() if 'confidence' in outputs else 1.0

            # Định dạng kết quả
            result = {
                "predictions": [
                    {"Bệnh": disease_map[i], "Xác suất": f"{prob.item()*100:.2f}%"}
                    for i, prob in enumerate(disease_probs)
                ],
                "Độ tin cậy": f"{confidence*100:.2f}%"
            }

            # Hiển thị kết quả
            self.show_result(result)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi dự đoán:\n{str(e)}")

    def show_result(self, result):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)

        # Hiển thị đẹp hơn
        text = "KẾT QUẢ CHẨN ĐOÁN:\n"
        text += "="*50 + "\n"

        for pred in result["predictions"]:
            text += f"- {pred['Bệnh']}: {pred['Xác suất']}\n"

        text += "\n" + "ĐỘ TIN CẬY: " + result["Độ tin cậy"]

        self.result_text.insert(tk.END, text)
        self.result_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalVQAApp(root)
    root.mainloop()
