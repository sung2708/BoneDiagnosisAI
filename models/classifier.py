import torch
import torch.nn as nn
from Configs.config import Config

class Classifier(nn.Module):
    def __init__(self, num_question_types):
        super().__init__()
        # Create separate classifiers for each question type
        self.classifiers = nn.ModuleDict({
            q_type: nn.Sequential(
                nn.Linear(Config.dim, Config.dim // 2),
                nn.ReLU(),
                nn.Dropout(Config.drop),
                nn.Linear(Config.dim // 2, len(Config.QUESTION_TYPES[q_type]))
            ) for q_type in Config.QUESTION_TYPES
        })

    def forward(self, x, question_types):
        # Process each sample with its corresponding classifier
        all_logits = []
        for features, q_type in zip(x, question_types):
            all_logits.append(self.classifiers[q_type](features))

        # Return as list instead of stacking since sizes differ
        return all_logits
