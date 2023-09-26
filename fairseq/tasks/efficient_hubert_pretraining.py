# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging

from typing import List, Optional

from dataclasses import dataclass, field
from fairseq.data import Dictionary, HubertDataset
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from omegaconf import MISSING

##### MT IMPORT #####
from collections import OrderedDict
from fairseq.data import EfficientHubertDataset

from fairseq.tasks.hubert_pretraining import (
    HubertPretrainingTask,
    HubertPretrainingConfig
)
# from fairseq.data import MultiCorpusDataset

logger = logging.getLogger(__name__)

# import IPython

class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        return self.dictionary.encode_line(
            label,
            append_eos=False,
            add_if_not_exist=False,
        )
    
@dataclass
class EfficientHubertPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    fine_tuning: bool = field(
        default=False, metadata={"help": "set to true if fine-tuning Hubert"}
    )
    labels: List[str] = field(
        default_factory=lambda: ["ltr"],
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )
    label_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, looks for labels in this directory instead",
        },
    )
    label_rate: float = field(
        default=-1.0,
        metadata={"help": "label frame rate. -1.0 for sequence label"},
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_keep_size: Optional[int] = field(
        default=None,
        metadata={"help": "exclude sample longer than this"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    single_target: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, AddTargetDatasets outputs same keys " "as AddTargetDataset"
        },
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )

    ##### ADD SUPERVISION RATIO #####
    supervision_ratio: float = field(
        default=0.15,
        metadata={"help": "ratio of labeled audio within a batch"},
    )

@register_task("efficient_hubert_pretraining", dataclass=EfficientHubertPretrainingConfig)
class EfficientHubertPretrainingTask(HubertPretrainingTask):
    cfg: EfficientHubertPretrainingConfig
    def __init__(
        self,
        cfg: EfficientHubertPretrainingConfig,
    ) -> None:
        super().__init__(cfg)
        self.cfg = cfg


    def load_dataset(self, split: str, **kwargs) -> None:
        ##### TODO - CHANGE THE DATATSET INTO MultiCorpusDataset #####
        if split == "train":
            ##### BUILD LABELED AUDIO DATASET #####
            sub_splits = ['labeled', 'unlabeled']
            train_datasets = OrderedDict()

            for sub_split in sub_splits:
                manifest = f"{self.cfg.data}/{split}_{sub_split}.tsv"
                dicts = [self.target_dictionary] if self.cfg.fine_tuning else self.dictionaries
                pad_list = [dict.pad() for dict in dicts]
                eos_list = [dict.eos() for dict in dicts]
                procs = [LabelEncoder(dict) for dict in dicts]
                paths = [f"{self.get_label_dir()}/{split}_{sub_split}.{l}" for l in self.cfg.labels]

                train_datasets[sub_split] = HubertDataset(
                    manifest,
                    sample_rate=self.cfg.sample_rate,
                    label_paths=paths,
                    label_rates=self.cfg.label_rate,
                    pad_list=pad_list,
                    eos_list=eos_list,
                    label_processors=procs,
                    max_keep_sample_size=self.cfg.max_keep_size,
                    min_keep_sample_size=self.cfg.min_sample_size,
                    max_sample_size=self.cfg.max_sample_size,
                    pad_audio=self.cfg.pad_audio,
                    normalize=self.cfg.normalize,
                    store_labels=False,
                    random_crop=self.cfg.random_crop,
                    single_target=self.cfg.single_target,
                )
            self.datasets[split] = EfficientHubertDataset(
                unlabeled_dataset = train_datasets['unlabeled'],
                labeled_dataset = train_datasets['labeled'],
                supervision_ratio = self.cfg.supervision_ratio,
                seed = 610,
            )
        else:
            manifest = f"{self.cfg.data}/{split}.tsv"
            dicts = [self.target_dictionary] if self.cfg.fine_tuning else self.dictionaries
            pad_list = [dict.pad() for dict in dicts]
            eos_list = [dict.eos() for dict in dicts]
            procs = [LabelEncoder(dict) for dict in dicts]
            paths = [f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels]

            # hubert v1: pad_audio=True, random_crop=False;
            self.datasets[split] = HubertDataset(
                manifest,
                sample_rate=self.cfg.sample_rate,
                label_paths=paths,
                label_rates=self.cfg.label_rate,
                pad_list=pad_list,
                eos_list=eos_list,
                label_processors=procs,
                max_keep_sample_size=self.cfg.max_keep_size,
                min_keep_sample_size=self.cfg.min_sample_size,
                max_sample_size=self.cfg.max_sample_size,
                pad_audio=self.cfg.pad_audio,
                normalize=self.cfg.normalize,
                store_labels=False,
                random_crop=self.cfg.random_crop,
                single_target=self.cfg.single_target,
            )