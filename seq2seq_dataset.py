import linecache
import math
import os
import pickle
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler
from transformers import BartTokenizer
from transformers.file_utils import cached_property

try:
    from fairseq.data.data_utils import batch_by_size

    FAIRSEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRSEQ_AVAILABLE = False


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)


class DistributedSortishSampler(Sampler):
    """Copied from torch DistributedSampler"""

    def __init__(
        self,
        dataset,
        batch_size,
        num_replicas=None,
        rank=None,
        add_extra_examples=True,
        shuffle=True,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if add_extra_examples:
            self.num_samples = int(
                math.ceil(len(self.dataset) * 1.0 / self.num_replicas)
            )
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(dataset)
            self.num_samples = len(self.available_indices)
        self.batch_size = batch_size
        self.add_extra_examples = add_extra_examples
        self.shuffle = shuffle


class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size, shuffle=True):
        self.data, self.bs, self.shuffle = data, batch_size, shuffle

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(sortish_sampler_indices(self.data, self.bs, shuffle=self.shuffle))


class AbstractSeq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
        **dataset_kwargs,
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.len_file = Path(data_dir).joinpath(type_path + ".len")
        self.graph_file = Path(data_dir).joinpath(type_path + ".graph.tok")
        if os.path.exists(self.len_file):
            self.src_lens = pickle_load(self.len_file)
            self.used_char_len = False
        else:
            self.src_lens = self.get_char_lens(self.src_file)
            self.used_char_len = True
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.dataset_kwargs = dataset_kwargs
        dataset_kwargs.update(
            {"add_prefix_space": True}
            if isinstance(self.tokenizer, BartTokenizer)
            else {}
        )

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt_file)

    def make_sortish_sampler(
        self, batch_size, distributed=False, shuffle=True, **kwargs
    ):
        if distributed:
            return DistributedSortishSampler(
                self, batch_size, shuffle=shuffle, **kwargs
            )
        else:
            return SortishSampler(self.src_lens, batch_size, shuffle=shuffle)

    def make_dynamic_sampler(self, max_tokens_per_batch=1024, **kwargs):
        assert FAIRSEQ_AVAILABLE, "Dynamic batch size requires `pip install fairseq`"
        assert (
            not self.used_char_len
        ), "You must call  python make_len_file.py before calling make_dynamic_sampler"
        sorted_indices = list(self.make_sortish_sampler(1024, shuffle=False))

        def num_tokens_in_example(i):
            return min(self.src_lens[i], self.max_target_length)

        # call fairseq cython function
        batch_sampler: List[List[int]] = batch_by_size(
            sorted_indices,
            num_tokens_fn=num_tokens_in_example,
            max_tokens=max_tokens_per_batch,
            required_batch_size_multiple=64,
        )
        shuffled_batches = [
            batch_sampler[i] for i in np.random.permutation(range(len(batch_sampler)))
        ]
        # move the largest batch to the front to OOM quickly (uses an approximation for padding)
        approximate_toks_per_batch = [
            max(self.src_lens[i] for i in batch) * len(batch)
            for batch in shuffled_batches
        ]
        largest_batch_idx = np.argmax(approximate_toks_per_batch)
        shuffled_batches[0], shuffled_batches[largest_batch_idx] = (
            shuffled_batches[largest_batch_idx],
            shuffled_batches[0],
        )
        return shuffled_batches

    def __getitem__(self, item):
        raise NotImplementedError("You must implement this")

    def collate_fn(self, batch):
        raise NotImplementedError("You must implement this")


class Seq2SeqDataset(AbstractSeq2SeqDataset):
    """A dataset that calls prepare_seq2seq_batch."""

    def __getitem__(self, index) -> Dict[str, str]:

        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip(
            "\n"
        )
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        graph_line = linecache.getline(str(self.graph_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        return {
            "tgt_texts": tgt_line,
            "src_texts": source_line,
            "src_graphs": graph_line,
            "id": index - 1,
        }

    def collate_fn(self, batch):
        """Call prepare_seq2seq_batch."""
        batch_encoding: Dict[str, torch.Tensor] = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.max_source_length,
            max_target_length=self.max_target_length,
            return_tensors="pt",
            **self.dataset_kwargs,
        ).data
        # lens = (batch_encoding['attention_mask'] == 1.).sum(dim=1).tolist()
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
        batch_encoding["graphs"] = self.generate_edge_tensors(batch)
        batch_encoding["tgt_text"] = [x["tgt_texts"] for x in batch]
        return batch_encoding

    def generate_edge_tensors(self, batch):

        graphs_edges = []
        set_edges = {"d": 0, "r": 1, "seq": 2}
        for g in batch:

            g = g["src_graphs"]

            edge_index_1 = []
            edge_index_2 = []
            edges_types = []

            x = g.split()
            for e in x:
                e = e.replace("(", "").replace(")", "")
                e = e.split(",")
                e2, e1, l = e

                if int(e1) >= self.max_source_length:
                    continue
                if int(e2) >= self.max_source_length:
                    continue

                edge_index_1.append(int(e1))
                edge_index_2.append(int(e2))
                edges_types.append(set_edges[l])

            edges_index = torch.tensor([edge_index_1, edge_index_2], dtype=torch.long)
            edges_types = torch.tensor(edges_types, dtype=torch.long)

            graphs_edges.append((edges_index, edges_types))

        return graphs_edges
