import typing as tp

import torch
from nltk.tokenize import wordpunct_tokenize
from torch.utils.data import Dataset


class DocumentDataset(Dataset):
    def __init__(
            self,
            documents: dict,
            word_idx: dict,
            max_document_length: int,
            stemmer,
            add_end_token=False
            ):
        self.max_document_length = max_document_length
        self.add_end_token = add_end_token
        self.document_length = max_document_length + add_end_token
        self.stemmer = stemmer

        self._documents = documents
        self._word_idx = word_idx

        self._cache = {}

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
        for word in self._tokenize(text):
            word = self.stemmer(word)
            if word not in self._word_idx:
                word = '<unk>'
            encoded.append(self._word_idx[word])

        padded = torch.nn.functional.pad(
            torch.tensor(encoded, dtype=torch.long),
            (0, self.max_document_length - len(encoded)),
            value=self._word_idx['<pad>']
        )

        if self.add_end_token:
            return torch.concat((padded, torch.tensor([self._word_idx['<end>']])))

        return padded

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
        vec = torch.zeros((self.document_length,))
        if position == 0 and self.add_end_token:
            vec[-1] = 1
        else:
            idx = len(self._tokenize(text[:position]))
            vec[idx] = 1

        return vec
