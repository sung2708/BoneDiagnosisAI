import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from test_model import ModelTester
import os

class MedicalDiagnosisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chẩn Đoán Bệnh Xương")
        self.root.geometry("800x600")

        # Load model
        self.model = ModelTester("saved_models/best_model.pt")

        # GUI Components
        self.create_widgets()

    def create_widgets(self):
        # Image Frame
        self.img_frame = tk.LabelFrame(self.root, text="Ảnh X-quang")
        self.img_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.img_label = tk.Label(self.img_frame)
        self.img_label.pack(pady=20)

        # Question Frame
        self.q_frame = tk.LabelFrame(self.root, text="Câu hỏi chẩn đoán")
        self.q_frame.pack(pady=10, padx=10, fill="x")

        self.question_entry = tk.Entry(self.q_frame, width=60)
        self.question_entry.pack(pady=5, padx=5)
        self.question_entry.insert(0, "Có dấu hiệu bất thường gì trên ảnh X-quang này?")

        # Button Frame
        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.pack(pady=10)

        self.load_btn = tk.Button(self.btn_frame, text="Chọn Ảnh", command=self.load_image)
        self.load_btn.pack(side="left", padx=5)

        self.predict_btn = tk.Button(self.btn_frame, text="Chẩn Đoán", command=self.predict)
        self.predict_btn.pack(side="left", padx=5)

        # Result Frame
        self.result_frame = tk.LabelFrame(self.root, text="Kết Quả")
        self.result_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.result_text = tk.Text(self.result_frame, height=10, wrap="word")
        self.result_text.pack(pady=5, padx=5, fill="both", expand=True)

        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Sẵn sàng")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief="sunken", anchor="w")
        self.status_bar.pack(side="bottom", fill="x")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh X-quang",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )

        if file_path:
            try:
                self.current_image = file_path
                img = Image.open(file_path)
                img.thumbnail((400, 400))
                photo = ImageTk.PhotoImage(img)

                self.img_label.config(image=photo)
                self.img_label.image = photo
                self.status_var.set(f"Đã tải ảnh: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải ảnh: {str(e)}")

    def predict(self):
        if not hasattr(self, 'current_image'):
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước khi chẩn đoán")
            return

        question = self.question_entry.get().strip()
        if not question:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập câu hỏi chẩn đoán")
            return

        try:
            self.status_var.set("Đang xử lý...")
            self.root.update()

            result = self.model.predict(self.current_image, question)

            # Hiển thị kết quả
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"KẾT QUẢ CHẨN ĐOÁN\n{'='*30}\n")
            self.result_text.insert(tk.END, f"Ảnh: {os.path.basename(self.current_image)}\n")
            self.result_text.insert(tk.END, f"Câu hỏi: {question}\n\n")
            self.result_text.insert(tk.END, f"Kết luận: {result['prediction']}\n")
            self.result_text.insert(tk.END, f"Độ tin cậy: {result['confidence']:.2%}\n\n")
            self.result_text.insert(tk.END, "XÁC SUẤT CHI TIẾT:\n")

            for cls, prob in result['probabilities'].items():
                self.result_text.insert(tk.END, f"- {cls}: {prob:.2%}\n")

            self.status_var.set("Chẩn đoán hoàn tất")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi chẩn đoán: {str(e)}")
            self.status_var.set("Lỗi xảy ra")

if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalDiagnosisApp(root)
    root.mainloop()
