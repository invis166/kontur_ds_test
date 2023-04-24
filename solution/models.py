import torch
from torch import nn
from navec import Navec
from slovnet.model.emb import NavecEmbedding


class LSTMLabelEndClassifier(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            hidden_size: int,
            num_layers=1,
            embedding_dim=256,
            dropout=0,
            bidirectional=False
            ):
        super().__init__()

        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        proj_hidden_size = hidden_size * 2 if self.bidirectional else hidden_size
        self.project = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(proj_hidden_size, 1)
        )
        self.softmax = nn.Softmax(1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        output, _ = (self.lstm(self.embedding(batch)))

        return self.softmax(self.project(output)).flatten(1)


class LSTMLabelEndClassifierNavec(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_layers=1,
            embedding_dim=256,
            dropout_e=0,
            dropout_p=0,
            bidirectional=False
            ):
        super().__init__()

        self.bidirectional = bidirectional

        navec = Navec('navec_hudlit_v1_12B_500K_300d_100q.tar')
        NAVEC_EMB_DIM = 300
        self.embedding = nn.Sequential(
            NavecEmbedding(navec),
            nn.Dropout(dropout_e),
            nn.Linear(NAVEC_EMB_DIM, embedding_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        proj_hidden_size = hidden_size * 2 if self.bidirectional else hidden_size
        self.project = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(proj_hidden_size, 1)
        )
        self.softmax = nn.Softmax(1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        output, _ = (self.lstm(self.embedding(batch)))

        return self.softmax(self.project(output)).flatten(1)
