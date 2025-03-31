
import os
import torch
import numpy as np
from tqdm import tqdm

# Import các thành phần cần thiết từ train.py
from train import Config, get_answer, test_data

class Tester:
    def __init__(self, config=Config()):
        self.config = config
        self.device = config.device

    def load_model(self, model_class, model_path):
        """Load model từ file đã train"""
        model = model_class().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def predict_classification(self, model, test_samples, dict_op, max_len, yn_mode=False):
        """Dự đoán cho task classification"""
        results = {}
        gt_answers = []
        pred_answers = []

        for i in tqdm(range(len(test_samples)), desc="Predicting"):
            sample = test_samples[i]
            ques_text = sample['question']
            imag_name = sample['image']

            # Chuẩn bị dữ liệu đầu vào
            _, imag_name, tokens, segment_ids, input_mask = test_data(
                [imag_name], [ques_text], 0, max_len
            )

            # Dự đoán
            pred_answer = get_answer(model, dict_op, imag_name, tokens, segment_ids, input_mask)

            # Lưu kết quả
            results[str(i)] = {
                'question': ques_text,
                'image': imag_name,
                'pred_answer': pred_answer,
                'true_answer': sample['answer']
            }

            if (yn_mode and ques_text in mod_yn_ques + abn_yn_ques) or \
               (not yn_mode and ques_text not in mod_yn_ques + abn_yn_ques):
                gt_answers.append(sample['answer'])
                pred_answers.append(pred_answer)

        # Tính accuracy nếu có ground truth
        accuracy = None
        if gt_answers:
            accuracy = sum(1 for gt, pred in zip(gt_answers, pred_answers) if gt == pred) / len(gt_answers)
            print(f"Accuracy: {accuracy:.4f}")

        return results, accuracy

    def predict_generation(self, model, test_samples, tokenizer, max_len):
        """Dự đoán cho task generation"""
        results = {}

        for i in tqdm(range(len(test_samples)), desc="Generating"):
            sample = test_samples[i]
            ques_text = sample['question']
            imag_name = sample['image']

            # Chuẩn bị dữ liệu đầu vào
            part1 = [0 for _ in range(5)]
            part2 = token_id(tokenize(ques_text))
            tokens = token_id(['[CLS]']) + part1 + token_id(['[SEP]']) + part2 + token_id(['[SEP]']) + token_id(['[MASK]'])
            masked_pos = [len(part1 + part2) + 3]
            segment_ids = [0] * (len(part1) + 2) + [1] * (len(part2) + 1) + [2] * 1
            input_mask = [1] * (len(part1 + part2) + 4)

            n_pad = max_len - len(part1 + part2) - 4
            tokens.extend([0] * n_pad)
            segment_ids.extend([0] * n_pad)
            input_mask.extend([0] * n_pad)

            # Thực hiện generation
            output = []
            for k in range(n_pad):
                with torch.no_grad():
                    pred = model([imag_name], [tokens], [segment_ids], [input_mask], [masked_pos])
                    out = int(np.argsort((pred.cpu())[0][0])[-1])
                    output.append(tokenizer.vocab[out])

                    if out == 102:  # [SEP] token
                        break
                    else:
                        tokens[len(part1 + part2) + 3 + k] = out
                        tokens[len(part1 + part2) + 4 + k] = token_id(['[MASK]'])[0]
                        masked_pos = [len(part1 + part2) + 4 + k]
                        segment_ids = [0] * (len(part1) + 2) + [1] * (len(part2) + 1) + [2] * (2 + k)
                        input_mask = [1] * (len(part1 + part2) + 5 + k)
                        n_pad = n_pad - 1
                        segment_ids.extend([0] * n_pad)
                        input_mask.extend([0] * n_pad)

            generated_text = (' '.join(output)).replace(' ##', '').replace(' [SEP]', '').replace(', ', '')

            results[str(i)] = {
                'question': ques_text,
                'image': imag_name,
                'generated_text': generated_text,
                'true_answer': sample['answer']
            }

        return results

    def save_results(self, results, output_file):
        """Lưu kết quả dự đoán ra file"""
        with open(output_file, 'w') as f:
            f.write("ID\tQuestion\tImage\tPrediction\tGround Truth\n")
            for idx, result in results.items():
                pred = result.get('pred_answer', result.get('generated_text', ''))
                f.write(f"{idx}\t{result['question']}\t{result['image']}\t{pred}\t{result['true_answer']}\n")
        print(f"Results saved to {output_file}")

def main():
    # Khởi tạo tester
    tester = Tester()

    # Load dữ liệu test
    # Giả sử có hàm load_test_data trả về list các dict {'question': ..., 'image': ..., 'answer': ...}
    test_samples = load_test_data(test_text_file, 0, 6)

    # Đường dẫn đến các model đã train
    model_paths = {
        'mod': 'saved_models/mod_1_0.850_1.234.pt',
        'mod_yn': 'saved_models/mod_yn_1_0.900_1.123.pt',
        'pla': 'saved_models/pla_1_0.800_1.345.pt',
        'org': 'saved_models/org_1_0.750_1.456.pt',
        'abn': 'saved_models/abn_1.234.pt',
        'abn_yn': 'saved_models/abn_yn_1_0.850_1.123.pt'
    }

    # Load các model
    models = {}
    for name, path in model_paths.items():
        if name == 'mod':
            models[name] = tester.load_model(mod_model1, path)
        elif name == 'mod_yn':
            models[name] = tester.load_model(mod_yn_model1, path)
        elif name == 'pla':
            models[name] = tester.load_model(pla_model1, path)
        elif name == 'org':
            models[name] = tester.load_model(org_model1, path)
        elif name == 'abn':
            models[name] = tester.load_model(abn_model, path)
        elif name == 'abn_yn':
            models[name] = tester.load_model(abn_yn_model1, path)

    # Tạo các dictionary ánh xạ
    mod_dict_op = {value: key for key, value in mod_dict.items()}
    mod_yn_dict_op = {value: key for key, value in mod_yn_dict.items()}
    pla_dict_op = {value: key for key, value in pla_dict.items()}
    org_dict_op = {value: key for key, value in org_dict.items()}
    abn_yn_dict_op = {value: key for key, value in abn_yn_dict.items()}

    # Dự đoán cho từng mẫu test
    final_results = {}

    for i, sample in enumerate(test_samples):
        ques_text = sample['question']

        if ques_text in mod_yn_ques + abn_yn_ques:
            # Câu hỏi yes/no
            if ques_text in mod_yn_ques:
                # Sử dụng mod_yn model
                result, _ = tester.predict_classification(
                    models['mod_yn'], [sample], mod_yn_dict_op, Config.max_len, yn_mode=True
                )
            else:
                # Sử dụng abn_yn model
                result, _ = tester.predict_classification(
                    models['abn_yn'], [sample], abn_yn_dict_op, Config.max_len, yn_mode=True
                )
        elif ques_text in mod_ques:
            # Câu hỏi về modality
            result, _ = tester.predict_classification(
                models['mod'], [sample], mod_dict_op, Config.max_len
            )
        elif ques_text in pla_ques:
            # Câu hỏi về place
            result, _ = tester.predict_classification(
                models['pla'], [sample], pla_dict_op, Config.max_len
            )
        elif ques_text in org_ques:
            # Câu hỏi về organization
            result, _ = tester.predict_classification(
                models['org'], [sample], org_dict_op, Config.max_len
            )
        else:
            # Câu hỏi generation (abnormality)
            result = tester.predict_generation(
                models['abn'], [sample], tokenizer, Config.gen_len
            )

        final_results[str(i)] = result[str(i)]

    # Lưu kết quả cuối cùng
    tester.save_results(final_results, "final_predictions.tsv")

    # In một vài kết quả để kiểm tra
    print("\nSample predictions:")
    for i in range(min(3, len(final_results))):
        print(f"Question: {final_results[str(i)]['question']}")
        print(f"Prediction: {final_results[str(i)].get('pred_answer', final_results[str(i)].get('generated_text', ''))}")
        print(f"Ground Truth: {final_results[str(i)]['true_answer']}\n")

if __name__ == "__main__":
    # Giả sử các biến và hàm này đã được định nghĩa ở nơi khác hoặc import từ module khác
    from models import base_model as models
    from data_utils import load_test_data, token_id, tokenize
    from vocab import mod_dict, mod_yn_dict, pla_dict, org_dict, abn_yn_dict
    from config import mod_yn_ques, abn_yn_ques, mod_ques, pla_ques, org_ques, test_text_file

    main()
