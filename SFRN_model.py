import torch
import torch.nn as nn
from transformers import AutoModel
from constant import *

class SFRNModel(nn.Module):
    def __init__(self):
        super(SFRNModel, self).__init__()
        # load the pre-trained BERT model for feature extraction
        self.bert = AutoModel.from_pretrained(hyperparameters['model_name'])

        # dropout layer for regularization to reduce overfitting
        self.dropout = torch.nn.Dropout(hyperparameters['hidden_dropout_prob'])

        # define the number of labels in the classification task
        num_labels = hyperparameters['num_labels']

        # define a multi-layer perceptron (MLP) for intermediate transformations
        mlp_hidden = hyperparameters['mlp_hidden']
        self.g = nn.Sequential(
            nn.Linear(hyperparameters['hidden_dim'], mlp_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )

        # another MLP for generating the final output logits
        self.f = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden, num_labels),
        )

        # additional modules for feature-wise transformations
        self.alpha = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.beta = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(),
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # pass input through BERT to get contextualized embeddings
        outputs = self.bert(input_ids.squeeze(), attention_mask=attention_mask.squeeze())

        # take the pooled output from BERT (represents the entire input sequence)
        pooled_output = outputs[0]

        # apply dropout to pooled output
        pooled_output = self.dropout(pooled_output)

        # pass through the g network for intermediate feature transformations
        g_t = self.g(pooled_output)

        # apply feature-wise transformations with alpha and beta
        g_t = self.alpha(g_t) * g_t + self.beta(g_t)

        # sum across all features to create a condensed representation
        g = g_t.sum(1)

        # pass through the final MLP (f) to get the output logits
        output = self.f(g)

        # apply softmax to get probability distributions over the classes
        logits = torch.softmax(output, dim=1)

        return logits
