from dataclasses import dataclass
from typing import List, Optional, Any

import torch
import torchmetrics
import torchvision
from torch import nn
from torchmetrics import Metric

from DNAnet.data.data_models.dna_models import Panel
from DNAnet.data.data_models.extracted_peak import ExtractedPeak
from DNAnet.data.data_models.structs import PeakPrediction, ClassAnnotation
from DNAnet.data.strategies.strategy_registry import StrategyRegistry
from DNAnet.models.base_model import BaseModel, TransformData
from DNAnet.models.classification.peak_classification_torch import PeakClassificationModel


@dataclass
class PeakClassifierTransformData(TransformData):
    include_marker: bool = True
    label_to_idx: dict = None

    def __call__(self, data: dict) -> dict[str, Any]:
        peak: ExtractedPeak = data['peak']
        annotation: Optional[ClassAnnotation] = data['annotation']
        adjusted_panel: Optional[Panel] = data['adjusted_panel']

        input_data = torch.tensor(peak.data, dtype=torch.float32)
        annotation_idx = self.label_to_idx[annotation.data] if annotation else -1
        target = torch.tensor(annotation_idx, dtype=torch.long) if annotation else None

        if input_data.dim() == 1:
            input_data = input_data.unsqueeze(0)  # (1, W)
        elif input_data.dim() == 2:
            pass  # already (C, W)
        elif input_data.dim() == 3 and input_data.shape[-1] == 1:
            input_data = input_data.squeeze(-1)  # (C, W, 1) -> (C, W)
        else:
            raise ValueError(f"peak.data must be 1D or 2D, got shape {tuple(input_data.shape)}")

        if self.include_marker:
            marker_idx = StrategyRegistry.get_scaling_strategy().marker_to_idx[peak.get_marker_name()]
            marker_tensor = torch.tensor([marker_idx], dtype=torch.long)
        else:
            marker_tensor = torch.full((1,), -1, dtype=torch.long)

        return {
            'peak': input_data,
            'target': target,
            'adjusted_panel': adjusted_panel,
            'marker_idx': marker_tensor,
        }

class PeakClassification(BaseModel):
    def get_transform(self) -> TransformData:
        return PeakClassifierTransformData(include_marker=self.include_marker, label_to_idx=self.label_to_idx)

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        peaks = [item['peak'] for item in batch]
        targets = torch.tensor([item['target'] for item in batch], dtype=torch.long)
        adjusted_panels = [item['adjusted_panel'] for item in batch]
        marker_idxs = torch.tensor([item['marker_idx'] for item in batch], dtype=torch.long)

        return {
            'peak': peaks,
            'target': targets,
            'adjusted_panel': adjusted_panels,
            'marker_idx': marker_idxs,
        }


    def __init__(self,
                 labels: List[str],
                 device: Optional[str] = None,
                 window_size: int = 120,
                 include_marker: bool = True,
                 channels: List[int] = [32, 64],
                 kernel_size: int = 3,
                 pooling: str = 'avg',     # 'avg' or 'attn'
                 include_max_pool_dyes: bool = False,
                 loss_fn: str = "cross_entropy",
                 use_batchnorm: bool = False,
                 bn_momentum: float = 0.1,
                 conv_dropout_p: float = 0.0,
                 head_dropout_p: float = 0.0,
                 downsample: str = "maxpool",           # "maxpool" (original) or "conv",
                 activation: str = "relu",
                 ):

        self.num_classes = len(labels)
        embedding_dim = 8 if include_marker else 0

        model = PeakClassificationModel(
            num_classes=self.num_classes,
            width=window_size,
            n_markers=len(StrategyRegistry.get_scaling_strategy().marker_to_idx) + 1,
            embedding_dim=embedding_dim,
            include_max_pool_dyes=include_max_pool_dyes,
            hidden_channels=channels,
            kernel_size=kernel_size,
            pooling=pooling,
            use_batchnorm=use_batchnorm,
            bn_momentum=bn_momentum,
            conv_dropout_p=conv_dropout_p,
            head_dropout_p=head_dropout_p,
            downsample=downsample,
            activation=activation
        )


        self.window_size = window_size
        self.label_to_idx = dict(zip(labels, range(self.num_classes)))
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()} # invert the mapping
        self.labels = labels
        self.include_marker = include_marker
        self.include_max_pool_dyes = include_max_pool_dyes
        if loss_fn == "cross_entropy":
            loss = nn.CrossEntropyLoss()
        elif loss_fn == "focal":
            loss = torchvision.ops.sigmoid_focal_loss
        elif loss_fn == "KL_divergence":
            loss = self._kl_div_loss
            self._kl_div = nn.KLDivLoss(reduction="batchmean")
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

        super().__init__(model, loss, device, apply_allele_caller=False)

    def _kl_div_loss(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(logits, dim=1)
        targets = nn.functional.one_hot(y_true, num_classes=self.num_classes).to(dtype=log_probs.dtype)
        return self._kl_div(log_probs, targets)


    def predict_class(self, probs: torch.Tensor) -> str:
        """
        From the predicted probabilities of a single image, find the class index with the highest
        probability and translate this to the actual class label.
        """
        return self.idx_to_label[torch.argmax(probs).item()]

    def set_up_metrics(self, use_evaluation_metric: bool) -> List[Optional[torchmetrics.Metric]]:
        """
        Use multiclass accuracy as evaluation metric if desired.
        """
        if not use_evaluation_metric:
            return []
        else:
            metrics = [torchmetrics.classification.Accuracy(task="multiclass",
                                                            num_classes=len(self.labels),
                                                            average="micro"),
                       torchmetrics.classification.Precision(task="multiclass",
                                                             num_classes=len(self.labels),
                                                             average="macro",),
                       torchmetrics.classification.Recall(task="multiclass",
                                                          num_classes=len(self.labels),
                                                          average="macro",),
                       torchmetrics.classification.F1Score(task="multiclass",
                                                           num_classes=len(self.labels),
                                                           average="macro",)
                       ]
        return [metric.to(self._device) for metric in metrics]

    def update_metric(self,
                      metric: Metric,
                      logits: torch.Tensor,
                      y_true: torch.Tensor) -> None:
        probs = self.compute_probabilities(logits)
        y_pred = torch.argmax(probs, dim=1)
        metric.update(y_pred, y_true)

    @staticmethod
    def compute_probabilities(logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the probabilities from the logits by taking a softmax
        """
        return torch.softmax(logits, dim=1)

    def create_predictions(self, logits: torch.Tensor, batch: dict) -> List[PeakPrediction]:
        predictions = []
        for probs in self.compute_probabilities(logits):
            pred_dict = dict(zip(self.labels, map(float, probs)))
            predictions.append(PeakPrediction(data=pred_dict))
        return predictions

