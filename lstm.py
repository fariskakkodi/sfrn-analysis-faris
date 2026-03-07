import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size, embedding_matrix):
        super(LSTMClassifier, self).__init__()

        # store model parameters for future reference
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # use pretrained embedding matrix to initialize embeddings
        # nn.Embedding.from_pretrained automatically sets up embeddings from a given tensor
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))

        # initialize LSTM layer with a single layer, bidirectionality not specified (default unidirectional)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)

        # linear layer to project the hidden state to the output size
        self.hidden2out = nn.Linear(hidden_dim, output_size)

        # use log softmax for stable probability calculations
        self.softmax = nn.LogSoftmax(dim=1)

        # dropout layer to reduce overfitting
        self.dropout_layer = nn.Dropout(p=0.2)

    def init_hidden(self, batch_size):
        # initializes the hidden and cell states for LSTM
        # dimensions are (num_layers, batch_size, hidden_dim)
        return (
            autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
            autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
        )

    def forward(self, batch, lengths):
        # reset the hidden state at the start of each forward pass
        self.hidden = self.init_hidden(batch.size(-1))

        # convert input word indices to their embedding vectors
        embeds = self.embedding(batch)

        # pack the sequence of embeddings for efficient LSTM computation on variable-length data
        packed_input = pack_padded_sequence(embeds, lengths, enforce_sorted=False)
        outputs, (ht, ct) = self.lstm(packed_input, self.hidden)

        # ht is the last hidden state from the LSTM for all sequences in the batch
        # shape: (1, batch_size, hidden_dim)
        # ht[-1] gives (batch_size, hidden_dim)

        # apply dropout to the last hidden state
        output = self.dropout_layer(ht[-1])

        # pass through the linear layer to map to output classes
        output = self.hidden2out(output)

        # apply log softmax to get stable log probabilities for each class
        output = self.softmax(output)

        return output
