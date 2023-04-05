import typing as tp
from collections import defaultdict

import torch
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import wordpunct_tokenize
from torch.utils.data import Dataset


class DocumentDataset(Dataset):
    def __init__(self, documents: dict, word_idx: dict, max_document_length: int):
        self.max_document_length = max_document_length
        self._documents = documents
        self._word_idx = word_idx
        self._idx_word = dict(zip(self._word_idx.values(), self._word_idx.keys()))
        self.tokens_count = len(self._word_idx)

        self._cache = {}

        self._pad_idx = word_idx['PAD']

    def __getitem__(self, index: int) -> tuple:
        if index in self._cache:
            return self._cache[index]

        document = self._documents[index]

        item = (
            self._encode_text(document['text']),
            self._encode_label(document['label']),
        )
        self._cache[index] = item

        return item

    def _tokenize(self, text: str) -> tp.List[str]:
        return wordpunct_tokenize(text)

    def _encode_text(self, text: str) -> torch.Tensor:
        encoded = []
        stemmer = PorterStemmer()
        for word in self._tokenize(text):
            word = stemmer.stem(word)
            if word not in self._word_idx:
                word = 'UNK'
            encoded.append(self._word_idx[word])

        return torch.nn.functional.pad(
            torch.tensor(encoded, dtype=torch.long),
            (self._pad_idx, self.max_document_length - len(encoded))
        )

    def _encode_label(self, label: str) -> int:
        if label == 'обеспечение исполнения контракта':
            return 0
        if label == 'обеспечение гарантийных обязательств':
            return 1

        raise ValueError(f'can not encode label "{label}"')

    def __len__(self) -> int:
        return len(self._documents)


class LabeledDocumentDataset(DocumentDataset):
    def __getitem__(self, index: int) -> tuple:
        if index in self._cache:
            return self._cache[index]

        document = self._documents[index]
        extracted_part = document['extracted_part']

        item = (
            self._encode_text(document['text']),
            self._encode_label(document['label']),
            self._encode_position(document['text'], extracted_part['answer_start'][0]),
            self._encode_position(document['text'], extracted_part['answer_end'][0])
        )
        self._cache[index] = item

        return item

    def _encode_position(self, text: str, position: int) -> torch.Tensor:
        idx = len(self._tokenize(text[:position]))
        vec = torch.zeros((self.max_document_length,))
        vec[idx] = 1

        return vec
