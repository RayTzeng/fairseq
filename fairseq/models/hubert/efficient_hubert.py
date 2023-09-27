# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import II

from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import (
    EXTRACTOR_MODE_CHOICES,
    MASKING_DISTRIBUTION_CHOICES,
    LAYER_TYPE_CHOICES,
    ConvFeatureExtractionModel,
    TransformerEncoder,
)
from fairseq.modules import GradMultiply, LayerNorm
from fairseq.tasks.efficient_hubert_pretraining import (
    EfficientHubertPretrainingConfig,
    EfficientHubertPretrainingTask,
)

logger = logging.getLogger(__name__)

from fairseq.models.hubert.hubert import HubertModel, HubertConfig

@register_model("efficient_hubert", dataclass=HubertConfig)
class EfficientHubertModel(BaseFairseqModel):
    def __init__(
        self,
        cfg: HubertConfig,
        task_cfg: EfficientHubertPretrainingConfig,
        dictionaries: List[Dictionary],
    ) -> None:
        super().__init__()
        logger.info(f"EfficientHubertModel Config: {cfg}")

        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )
        feature_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
        self.feat2tar_ratio = cfg.label_rate * feature_ds_rate / task_cfg.sample_rate

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.untie_final_proj = cfg.untie_final_proj
        if self.untie_final_proj:
            self.final_proj = nn.Linear(
                cfg.encoder_embed_dim, final_dim * len(dictionaries)
            )
        else:
            self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

        # modules below are not needed during fine-tuning
        if any([d is None for d in dictionaries]):
            logger.info("cannot find dictionary. assume will be used for fine-tuning")
        else:
            self.num_classes = [len(d) for d in dictionaries]
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(sum(self.num_classes), final_dim)
            )
            nn.init.uniform_(self.label_embs_concat)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: HubertConfig, task: EfficientHubertPretrainingTask):
        """Build a new model instance."""

        model = EfficientHubertModel(cfg, task.cfg, task.dictionaries)
        return model

    def apply_mask(self, x, padding_mask, target_list):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits

    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        return features

    def forward_targets(
        self,
        features: torch.Tensor,
        target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, target_list

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def forward(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        features = self.forward_features(source)
        if target_list is not None:
            features, target_list = self.forward_targets(features, target_list)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask, target_list)
        else:
            x = features
            mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )

        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features}
        
        ##### GET UNLABELED AUDIO INDICES #####
        # indicator = torch.all(target_list[0] == 4, 1)
        # unlabeled_indices = torch.where(indicator)
        # labeled_indices = torch.where(~indicator)

        ###### TODO - assign label for unlabled data #####
        # cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        # labeled_features = x[labeled_indices]
        # ref_labels = target_list[0][labeled_indices]
        # for index in unlabeled_indices:
        #     for i, frame_features in enumerate(x[index]):
        #         # TODO - assign label
        #         frame_features = frame_features.view(1,1,-1)
        #         dist = cos(frame_features, labeled_features)
        #         _, i = torch.topk(dist.flatten(), 1)
        #         matched_index = unravel_index(i, dist.shape)
        #         target_list[0][index][i] = ref_labels[matched_index]

        
        def compute_pred(proj_x, target, label_embs):
            # compute logits for the i-th label set
            y = torch.index_select(label_embs, 0, target.long())
            negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)
            # proj_x: (S, D)
            # y: (S, D)
            # negs: (Neg, S, D)
            return self.compute_nce(proj_x, y, negs)

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)

        ##### For Testing #####
        indicator = torch.all(target_list[0] == 4, 1)
        ##### For Testing #####
        

        ##### TODO - write case for valid set #####

        if not self.skip_masked:
            masked_indices = torch.logical_and(~padding_mask, mask_indices)
            #####
            # print("finding NN for unlabeled MASKED frames....")
            nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
            labeled_indices = torch.logical_and(nomask_indices, ~indicator.view(-1,1))
            labeled_features = x[labeled_indices].view(-1, x.shape[-1])
            labeled_phone_label = target_list[0][labeled_indices].view(-1)
            # print(f"{labeled_features.size(0)} labeled frames is used as reference")

            cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
            
            unlabeled_masked_indices = torch.logical_and(masked_indices, indicator.view(-1,1))
            unlabeled_masked_features = x[unlabeled_masked_indices]
            # print(f"{unlabeled_masked_features.size(0)} unlabeled frames needs to find Nearest labeled frame")

            if unlabeled_masked_features.size(0) != 0:
                dist = cos(unlabeled_masked_features.unsqueeze(1), labeled_features.unsqueeze(0))
                dist = dist.view(dist.size(0), -1)
                # print(f"minimum distance: {dist.min()}")
                # print(f"maximum distance: {dist.max()}")

                _, nn_index = torch.topk(dist, 1)  # Choose the frame with largest Cosine similarity
                nn_index = nn_index.view(-1)
                # nn_index = unravel_indices(nn_index.view(-1), labeled_features.shape[:2])
                target_list[0][unlabeled_masked_indices] = labeled_phone_label[nn_index]
                
                # del dist

                # print("Finish!")
                #####

            proj_x_m = self.final_proj(x[masked_indices])
            if self.untie_final_proj:
                proj_x_m_list = proj_x_m.chunk(len(target_list), dim=-1)
            else:
                proj_x_m_list = [proj_x_m for _ in range(len(target_list))]
            logit_m_list = [
                compute_pred(proj_x_m, t[masked_indices], label_embs_list[i])
                for i, (proj_x_m, t) in enumerate(zip(proj_x_m_list, target_list))
            ]
        else:
            logit_m_list = [None for _ in target_list]

        if not self.skip_nomask:
            nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)

            #####
            # print("finding NN for unlabeled UNMASKED frames....")
            labeled_indices = torch.logical_and(nomask_indices, ~indicator.view(-1,1))
            labeled_features = x[labeled_indices].view(-1, x.shape[-1])
            labeled_phone_label = target_list[0][labeled_indices].view(-1)
            # print(f"{labeled_features.size(0)} labeled frames is used as reference")

            cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)

            unlabeled_nomask_indices = torch.logical_and(nomask_indices, indicator.view(-1,1))
            unlabeled_nomask_features = x[unlabeled_nomask_indices]
            # print(f"{unlabeled_nomask_features.size(0)} unlabeled frames needs to find Nearest labeled frame")

            if unlabeled_nomask_features.size(0) != 0:
                dist = cos(unlabeled_nomask_features.unsqueeze(1), labeled_features.unsqueeze(0))
                dist = dist.view(dist.size(0), -1)
                # print(f"minimum similarity: {dist.min()}")
                # print(f"maximum similarity: {dist.max()}")

                _, nn_index = torch.topk(dist, 1)  # Choose the frame with largest Cosine similarity
                nn_index = nn_index.view(-1)
                # nn_index = unravel_indices(nn_index.view(-1), labeled_features.shape[:2])
                target_list[0][unlabeled_nomask_indices] = labeled_phone_label[nn_index]

                # del dist
                # print("Finish!")
            ######

            proj_x_u = self.final_proj(x[nomask_indices])
            if self.untie_final_proj:
                proj_x_u_list = proj_x_u.chunk(len(target_list), dim=-1)
            else:
                proj_x_u_list = [proj_x_u for _ in range(len(target_list))]

            logit_u_list = [
                compute_pred(proj_x_u, t[nomask_indices], label_embs_list[i])
                for i, (proj_x_u, t) in enumerate(zip(proj_x_u_list, target_list))
            ]
        else:
            logit_u_list = [None for _ in target_list]

        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }
        return result

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        return feature, res["padding_mask"]

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, net_output, is_masked=True):
        logits_list = self.get_logits(net_output, is_masked)
        targets_list = [x.new_zeros(x.size(0), dtype=torch.long) for x in logits_list]
        return targets_list

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None

def unravel_indices(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord

def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> Tuple[torch.LongTensor, ...]:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (N,).
        shape: The targeted shape, (D,).

    Returns:
        A tuple of unraveled coordinate tensors of shape (D,).
    """

    coord = unravel_indices(indices, shape)
    return tuple(coord)
