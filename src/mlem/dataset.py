import csv
import pprint

from typing import Optional, Iterable
from math import log2

from . import utils


class Attribute:
    data: list[str]
    values: set[str]

    def __init__(self, data: Optional[list[str]]):
        self.data = data or []
        self.values = set(self.data)

    def __bool__(self):
        return bool(self.data)

    def __contains__(self, value: str) -> bool:
        return value in self.values

    def __eq__(self, other: "Attribute") -> bool:
        return self.data == other.data

    def __getitem__(self, idx: int) -> str:
        return self.data[idx]

    def __setitem__(self, idx: int, value: str):
        self.data[idx] = value
        self.values.add(value)
    
    def __iter__(self) -> Iterable[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def add(self, value: str):
        self.data.append(value)
        self.values.add(value)

    def pop(self, idx: int = -1):
        popped = self.data.pop(idx)
        if popped not in self.data:
            self.values.remove(popped)
    
    def remove_all(self, value: str):
        self.values.remove(value)
        for i, d in enumerate(self.data):
            if d == value:
                self.data.pop(i)
    
    def find(self, value: str) -> int:
        return self.data.index(value)
    
    def insert(self, idx: int, value: str):
        self.data.insert(idx, value)
        self.values.add(value)

    def mode(self) -> str:
        return max(self.values, key=self.data.count)
    
    def portion(self, value: str) -> float:
        return len([d for d in self.data if d == value]) / len(self)

    def entropy(self) -> float:
        return -sum(
            self.portion(value) * log2(self.portion(value))
            for value in self.values
        )

    def info_gain(self, target: "Attribute") -> float:
        assert len(self) == len(target)
        split_entropy = 0
        for value in self.values:
            v_portion = Attribute([
                t_val for s_val, t_val in zip(self, target)
                if s_val == value
            ])
            split_entropy += len(v_portion) / len(self) * v_portion.entropy()

        return target.entropy() - split_entropy

    def __str__(self) -> str:
        return str(self.data)
    
    def __repr__(self) -> str:
        return f"Attribute({repr(self.data)})"


class Dataset:
    data: dict[str, Attribute]
    target: Attribute

    def __init__(self, data: dict[str, list[str]], target: list[str]):
        self.data = {name: Attribute(ds) for name, ds in data.items()}
        self.target = Attribute(target)

    @classmethod
    def from_csv(cls, filename: str):
        with open(filename) as f:
            lines = list(csv.reader(f.readlines()))

        attrs = lines.pop(0)[:-1]
        dataset = cls({attr: [] for attr in attrs}, [])
        for line in lines:
            line_tgt = line.pop()
            dataset.add(dict(zip(attrs, line)), line_tgt)

        return dataset

    def __getitem__(self, idx: int) -> tuple[dict[str, str], str]:
        return utils.row_of_doa(self.data, idx), self.target[idx]

    def __iter__(self) -> Iterable[tuple[dict[str, str], bool]]:
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return len(self.target)
    
    def __bool__(self) -> bool:
        return bool(self.data)

    def col(self, attr: str) -> Attribute:
        return self.data[attr]

    def values_of(self, attr: str) -> set[str]:
        return self.data[attr].values

    def attrs(self) -> Iterable[str]:
        return self.data.keys()

    def values(self) -> Iterable[Attribute]:
        return self.data.values()
    
    def add(self, sample_data: dict[str, str], sample_tgt: str):
        for attr in self.attrs():
            self.data[attr].add(sample_data[attr])
        self.target.add(sample_tgt)

    def pop(self, idx: int):
        for attr in self.attrs():
            self.data[attr].pop(idx)
        self.target.pop(idx)

    def remove_attr(self, attr: str):
        self.data.pop(attr)
        
    def split(self, attr: str) -> dict[str, "Dataset"]:
        splits = {}
        for value in self.values_of(attr):
            splits[value] = Dataset({a: [] for a in self.attrs()}, [])
            # no longer needed as it would be constant among each split
            splits[value].remove_attr(attr)
            for sample_data, sample_tgt in self:
                if sample_data[attr] == value:
                    splits[value].add(sample_data, sample_tgt)
        return splits

    def most_informative_attr(self) -> str:
        return max(
            self.attrs(),
            key=lambda attr: self.col(attr).info_gain(self.target)
        )

    def __str__(self) -> str:
        return pprint.pformat(self.data)
