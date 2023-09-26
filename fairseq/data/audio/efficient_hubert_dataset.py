# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging

import numpy as np

from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset

logger = logging.getLogger(__name__)

# import IPython

def load_audio(manifest_path, max_keep, min_keep):
    n_long, n_short = 0, 0
    names, inds, sizes = [], [], []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) == 2, line
            sz = int(items[1])
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                names.append(items[0])
                inds.append(ind)
                sizes.append(sz)
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, inds, tot, sizes


def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


def verify_label_lengths(
    audio_sizes,
    audio_rate,
    label_path,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    with open(label_path) as f:
        lengths = [len(line.rstrip().split()) for line in f]
        assert len(lengths) == tot
        lengths = [lengths[i] for i in inds]
    num_invalid = 0
    for i, ind in enumerate(inds):
        dur_from_audio = audio_sizes[i] / audio_rate
        dur_from_label = lengths[i] / label_rate
        if abs(dur_from_audio - dur_from_label) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"in line {ind+1} of {label_path}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_sizes[i]}; "
                    f"label length = {lengths[i]}"
                )
            )
            num_invalid += 1
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )

class EfficientHubertDataset(FairseqDataset):
    """
    Within each batch,
    randomly substitute certain portion of unlabeled audio to labeled audio
    """

    def __init__(
        self,
        unlabeled_dataset: FairseqDataset,
        labeled_dataset: FairseqDataset,
        supervision_ratio: float,
        seed: int,
    ):
        assert supervision_ratio >= 0
        assert supervision_ratio <= 1
        self.unlabeled_dataset = unlabeled_dataset
        self.labeled_dataset = labeled_dataset
        self.supervision_ratio = supervision_ratio
        self.seed = seed

    def __len__(self):
        """
        Length of this dataset is the length of unlabeled dataset
        """
        return len(self.unlabeled_dataset)
    
    def __getitem__(self, index):
        return self.unlabeled_dataset.__getitem__(index)
    
    def collater(self, samples):
        if len(samples) == 0:
            return None
        
        n_supervision = max(int(len(samples) * self.supervision_ratio), 1)     # at least 1 labeled sample within the batch
        with data_utils.numpy_seed(self.seed, self.epoch):
            substituted_indices = np.random.choice(len(samples), n_supervision)
            substituted_sizes = [self.unlabeled_dataset.size(x['id']) for x in np.array(samples)[substituted_indices]]
            labeled_indices = np.abs(np.subtract(self.labeled_dataset.sizes, np.expand_dims(substituted_sizes, axis=1))).argmin(axis=1)

            assert len(labeled_indices) == n_supervision

            for (index, new_index) in zip(substituted_indices, labeled_indices):
                samples[index] = self.labeled_dataset.__getitem__(new_index)
            try:
                batch = self.unlabeled_dataset.collater(samples)
            except Exception:
                print(f"Collating failed!!", flush=True)
                raise
            return batch
        
    def num_tokens(self, index: int):
        return self.unlabeled_dataset.num_tokens(index)

    def size(self, index):
        return self.unlabeled_dataset.size(index)
    
    def ordered_indices(self):
        return self.unlabeled_dataset.ordered_indices()

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        logger.info(f"setting epoch of multi_corpus_dataset to {epoch}")
        self.epoch = epoch

    @property
    def supports_prefetch(self):
        return False
    
    @property
    def supports_fetch_outside_dataloader(self):
        return self.unlabeled_dataset.supports_fetch_outside_dataloader