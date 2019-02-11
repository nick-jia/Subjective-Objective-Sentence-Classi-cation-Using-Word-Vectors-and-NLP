import torch.nn as nn
from torch import squeeze, sigmoid, relu, transpose, unsqueeze, cat


class Baseline(nn.Module):
    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        x = x.mean(0, keepdim=True)
        x = relu(self.fc1(x))
        x = sigmoid(self.fc2(x)).squeeze()
        return x


class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=False)
        x = self.gru(x)
        if lengths is not None:
            (_, x) = x
        x = relu(self.fc1(x))
        x = sigmoid(self.fc2(x)).squeeze()
        return x


class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.conv1 = nn.Conv2d(1, n_filters, (filter_sizes[0], embedding_dim))
        self.conv2 = nn.Conv2d(1, n_filters, (filter_sizes[1], embedding_dim))
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        x = transpose(x, 0, 1).unsqueeze(1)
        x1 = relu(self.conv1(x).squeeze(3))
        x2 = relu(self.conv2(x).squeeze(3))
        pool1 = nn.MaxPool1d(x1.shape[2])
        pool2 = nn.MaxPool1d(x2.shape[2])
        x1 = pool1(x1).squeeze(2)
        x2 = pool2(x2).squeeze(2)
        x = cat([x1, x2], dim=1)
        x = relu(self.fc1(x))
        x = sigmoid(self.fc2(x)).squeeze()
        return x



