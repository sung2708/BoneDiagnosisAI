import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

# Constants and configurations
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 1e-4
    batch_size = 32
    max_len = 128
    gen_len = 64
    epochs = 50
    clip = True
    save_dir = "./saved_models/"
    threshold = 2.0  # Loss threshold for saving models

    # Create save directory if not exists
    os.makedirs(save_dir, exist_ok=True)

# Model training functions
class Trainer:
    def __init__(self, model, optimizer, scheduler, config):
        self.model = model.to(config.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

    def get_loss(self, batch, mode='class'):
        if mode == 'class':
            imag_name, tokens, segment_ids, input_mask, classes = zip(*batch)
            logits_clsf = self.model(imag_name, tokens, segment_ids, input_mask)
            loss_clsf = nn.CrossEntropyLoss()(logits_clsf, torch.cuda.LongTensor(classes))
            return loss_clsf
        else:
            imag_name, tokens, segment_ids, input_mask, masked_tokens, masked_pos, masked_weights = zip(*batch)
            logits_lm = self.model(imag_name, tokens, segment_ids, input_mask, masked_pos)
            masked_tokens = torch.cuda.LongTensor(masked_tokens)
            masked_weights = torch.cuda.LongTensor(masked_weights)
            loss_lm = nn.CrossEntropyLoss(reduction='none')(logits_lm.transpose(1, 2), masked_tokens)
            return (loss_lm * masked_weights.float()).mean()

    def train_epoch(self, iterator, epoch, mode='class'):
        self.model.train()
        epoch_loss, count = 0, 0
        iter_bar = tqdm(iterator, desc=f'Training Epoch {epoch}')

        for batch in iter_bar:
            count += 1
            self.optimizer.zero_grad()

            loss = self.get_loss(batch, mode)
            loss = loss.mean()
            loss.backward()

            if self.config.clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            iter_bar.set_description(f'Iter (loss={loss.item():5.3f})')
            epoch_loss += loss.item()

        return epoch_loss / count

    def evaluate(self, iterator, epoch, mode='class'):
        self.model.eval()
        epoch_loss, count = 0, 0

        with torch.no_grad():
            iter_bar = tqdm(iterator, desc=f'Validation Epoch {epoch}')
            for batch in iter_bar:
                count += 1
                loss = self.get_loss(batch, mode)
                loss = loss.mean()
                iter_bar.set_description(f'Iter (loss={loss.item():5.3f})')
                epoch_loss += loss.item()

        return epoch_loss / count

# Training pipeline
def train_pipeline(model, train_data, valid_data, log_name, mode='class', dict_op=None,
                   start_idx=0, end_idx=2, yn_mode=False, config=Config()):
    """
    Main training pipeline for classification or generation models

    Args:
        model: The model to train
        train_data: Tuple of (images, questions, answers) for training
        valid_data: Tuple of (images, questions, answers) for validation
        log_name: Name for saving logs and models
        mode: 'class' for classification or 'gene' for generation
        dict_op: Dictionary for mapping output indices to answers
        start_idx: Start index for test samples
        end_idx: End index for test samples
        yn_mode: Whether it's a yes/no question model
        config: Configuration object
    """
    imag_t, ques_t, answ_t = train_data
    imag_v, ques_v, answ_v = valid_data

    # Initialize components
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True)
    trainer = Trainer(model, optimizer, scheduler, config)

    # Prepare log file
    log_file = os.path.join(config.save_dir, f"{log_name}.txt")
    with open(log_file, 'w') as log_f:
        log_f.write('epoch,train_loss,valid_loss\n')

    # Training loop
    best_loss = float('inf')
    stop_counter = 0
    loss_history = []
    result_dict = {}

    for epoch in range(config.epochs):
        print(f'\nEpoch: {epoch + 1}')

        # Create data loaders
        if mode == 'class':
            train_iterator = data_loader(imag_t, ques_t, answ_t, config.batch_size, config.max_len)
            valid_iterator = data_loader(imag_v, ques_v, answ_v, config.batch_size, config.max_len)
        else:
            train_iterator = gene_loader(imag_t, ques_t, answ_t, config.batch_size, config.gen_len)
            valid_iterator = gene_loader(imag_v, ques_v, answ_v, config.batch_size, config.gen_len)

        # Train and validate
        train_loss = trainer.train_epoch(train_iterator, epoch + 1, mode)
        valid_loss = trainer.evaluate(valid_iterator, epoch + 1, mode)
        scheduler.step(valid_loss)

        # Log results
        with open(log_file, 'a') as log_f:
            log_f.write(f'{epoch + 1},{train_loss:.5f},{valid_loss:.5f}\n')

        # Check for best model
        if valid_loss < best_loss and valid_loss < config.threshold:
            best_loss = valid_loss
            stop_counter = 0
            loss_history.append(valid_loss)

            # Save model and get predictions
            if mode == 'class':
                # Classification task
                gt_temp, answ_temp = [], []

                for i in range(start_idx, end_idx):
                    ques_name, imag_name, tokens, segment_ids, input_mask = test_data(
                        test_imag, test_ques, i)

                    if yn_mode:
                        if test_ques[i] in mod_yn_ques + abn_yn_ques:
                            result_dict[str(i)] = get_answer(
                                model, dict_op, imag_name, tokens, segment_ids, input_mask)
                            gt_temp.append(test_answ[i])
                            answ_temp.append(result_dict[str(i)])
                    else:
                        if test_ques[i] not in mod_yn_ques + abn_yn_ques:
                            result_dict[str(i)] = get_answer(
                                model, dict_op, imag_name, tokens, segment_ids, input_mask)
                            gt_temp.append(test_answ[i])
                            answ_temp.append(result_dict[str(i)])

                # Calculate accuracy
                if gt_temp:
                    accuracy = sum(1 for i in range(len(gt_temp)) if gt_temp[i] == answ_temp[i]) / len(gt_temp)
                    model_path = os.path.join(
                        config.save_dir,
                        f"{log_name}_{accuracy:.3f}_{valid_loss:.3f}.pt"
                    )
                    torch.save(model.state_dict(), model_path)
            else:
                # Generation task
                for i in range(start_idx, end_idx):
                    imag_name = test_imag[i]
                    part1 = [0 for _ in range(5)]
                    part2 = token_id(tokenize(test_ques[i]))
                    tokens = token_id(['[CLS]']) + part1 + token_id(['[SEP]']) + part2 + token_id(['[SEP]']) + token_id(['[MASK]'])
                    masked_pos = [len(part1 + part2) + 3]
                    segment_ids = [0] * (len(part1) + 2) + [1] * (len(part2) + 1) + [2] * 1
                    input_mask = [1] * (len(part1 + part2) + 4)

                    n_pad = config.gen_len - len(part1 + part2) - 4
                    tokens.extend([0] * n_pad)
                    segment_ids.extend([0] * n_pad)
                    input_mask.extend([0] * n_pad)

                    output = []
                    for k in range(n_pad):
                        with torch.no_grad():
                            pred = model([imag_name], [tokens], [segment_ids], [input_mask], [masked_pos])
                            out = int(np.argsort((pred.cpu())[0][0])[-1])
                            output.append(abn_dict_op[out])

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

                    result_dict[str(i)] = (' '.join(output)).replace(' ##', '').replace(' [SEP]', '').replace(', ', '')

                model_path = os.path.join(config.save_dir, f"{log_name}_{valid_loss:.3f}.pt")
                torch.save(model.state_dict(), model_path)
        else:
            stop_counter += 1
            if stop_counter > 10:
                print("Early stopping triggered")
                break

    return result_dict, min(loss_history) if loss_history else float('inf')

# Utility functions
def get_answer(model, dict_op, imag_name, tokens, segment_ids, input_mask):
    model.eval()
    with torch.no_grad():
        pred = model([imag_name], [tokens], [segment_ids], [input_mask])
    return dict_op[int(np.argsort(pred.cpu())[:, -1:][0][0])]

def test_data(test_imag, test_ques, i, max_len):
    imag_name = test_imag[i]
    part1 = [0 for _ in range(5)]
    part2 = token_id(tokenize(test_ques[i]))
    tokens = token_id(['[CLS]']) + part1 + token_id(['[SEP]']) + part2[:max_len - 8] + token_id(['[SEP]'])
    segment_ids = [0] * (len(part1) + 2) + [1] * (len(part2[:max_len - 8]) + 1)
    input_mask = [1] * len(tokens)

    n_pad = max_len - len(tokens)
    tokens.extend([0] * n_pad)
    segment_ids.extend([0] * n_pad)
    input_mask.extend([0] * n_pad)

    return (test_ques[i], imag_name, tokens, segment_ids, input_mask)

# Main execution
if __name__ == "__main__":

    config = Config()
