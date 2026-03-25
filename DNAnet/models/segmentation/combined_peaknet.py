import logging
import os
from dataclasses import dataclass
from typing import Optional, List, Mapping, Any, Sequence

import numpy as np
import torch
import torchmetrics
from torchmetrics import Metric

from DNAnet.data.data_models.dna_models import Panel
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.data.data_models.structs import ScanpointAnnotation, ScanpointPrediction
from DNAnet.data.preprocessing.peak_extraction import extract_peaks_torch
from DNAnet.models.base_model import BaseModel, TransformData
from DNAnet.models.classification.peak_classification import PeakClassification
from DNAnet.models.reconstruction.autoencoder import Autoencoder
from DNAnet.models.segmentation.peaknet_architecture import CombinedClassifier, PeakOnlyClassifier
from DNAnet.typing import PathLike
from config_io import load_model

LOGGER = logging.getLogger('dnanet')


@dataclass
class CombinedPeakNetTransformData(TransformData):
    threshold: int = 40,
    window_size: int = 120,
    include_max_pool_dyes: bool = True,

    def __call__(self, data: dict) -> dict:
        image: HIDImage = data['image']
        scaler: np.ndarray = data['scaler']
        annotation: Optional[ScanpointAnnotation] = data['annotation']
        adjusted_panel: Optional[Panel] = data['adjusted_panel']

        peak_tensors, marker_idx, peak_centers = extract_peaks_torch(image,
                                                                     adjusted_panel,
                                                                     self.threshold,
                                                                     self.window_size,
                                                                     self.include_max_pool_dyes)

        scaler_data = torch.tensor(scaler, dtype=torch.float32)
        image_data = torch.tensor(image.data, dtype=torch.float32)
        target = torch.tensor(annotation.data, dtype=torch.long) if annotation else None

        return {
            'image': image_data, # (C, L)
            'target': target, # (C, L)
            'peak_windows': peak_tensors, # (N_p, C_p, W)
            'marker_idxs': marker_idx, # (N_p,)
            'peak_centers': peak_centers, # (N_p, 2)
            'adjusted_panel': adjusted_panel, # (B,)
            'scaler': scaler_data, # (L,)
        }



class CombinedPeakNet(BaseModel):


    def __init__(self,
                 peak_classifier_checkpoint: Optional[PathLike] = None,
                 autoencoder_checkpoint: Optional[PathLike] = None,
                 threshold: int = 40,
                 hidden_dims: List[int] = [256, 128],
                 default_label: str = "noise",
                 freeze_autoencoder: bool = True,
                 combiner_strategy: str = "mlp",
                 autoencoder: Mapping[str, Any] = None,
                 peak_classifier: Mapping[str, Any] = None,
                 ):

        torch.set_float32_matmul_precision("high")

        self.hidden_dims = hidden_dims
        self.threshold = threshold

        self.freeze_autoencoder = freeze_autoencoder
        self.combiner_strategy = combiner_strategy

        print(f" peak classfier config: {peak_classifier}")

        # self.peak_classifier: DNANet_PeakClassification = load_model(peak_classifier_config) if not context \
        #     else load_model(peak_classifier_config, builder=context.MODELS)

        if peak_classifier is not None:
            self.peak_classifier: PeakClassification = load_model(peak_classifier)
        if peak_classifier_checkpoint is not None:
            self.peak_classifier.load(peak_classifier_checkpoint)
        if peak_classifier_checkpoint is None and peak_classifier is None:
            raise ValueError("Either peak_classifier or peak_classifier_checkpoint must be provided.")


        self.window_size = self.peak_classifier.window_size
        self.noise_label_idx = self.peak_classifier.label_to_idx[default_label]
        self.num_classes = self.peak_classifier.num_classes

        if autoencoder is not None or autoencoder_checkpoint is not None:
            if autoencoder is not None and autoencoder_checkpoint is not None:
                self.autoencoder: Autoencoder = load_model(autoencoder)
                self.autoencoder.load(autoencoder_checkpoint)
                LOGGER.info(f"Loaded autoencoder from {autoencoder_checkpoint}")
            elif autoencoder is not None:
                self.autoencoder: Autoencoder = load_model(autoencoder)
                LOGGER.info(f"Loaded autoencoder from config")
            elif autoencoder_checkpoint is not None:
                self.autoencoder: Autoencoder = load_model(autoencoder_checkpoint)
                LOGGER.info(f"Loaded autoencoder from {autoencoder_checkpoint}")

            autoencoder_out_shape = self.autoencoder.encoded_shape

            model = CombinedClassifier(
                autoencoder=self.autoencoder.model,
                autoencoder_out_shape=autoencoder_out_shape,
                peak_classifier=self.peak_classifier.model,
                peak_classifier_out_shape=self.peak_classifier.model.backbone_out_shape(),
                hidden_dims=self.hidden_dims,
                num_classes=self.peak_classifier.num_classes,
                default_class=self.noise_label_idx,
                freeze_autoencoder=self.freeze_autoencoder,
                combiner_strategy=self.combiner_strategy,
            )
        else:
            self.autoencoder = None
            model = PeakOnlyClassifier(
                peak_classifier=self.peak_classifier.model,
                num_classes=self.peak_classifier.num_classes,
                default_class=self.noise_label_idx
            )


        loss = self.peak_classifier.loss_fn

        super().__init__(model, loss)

    def get_transform(self) -> TransformData:
        return CombinedPeakNetTransformData()


    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        ## could be optimized by only iterating over batch once
        images = torch.tensor([item['image'] for item in batch]) # (B, C, L)
        targets = torch.tensor([item['target'] for item in batch]) # (B, C, L)
        adjusted_panel = [item['adjusted_panel'] for item in batch] # (B,)
        scaler = torch.tensor([item['scaler'] for item in batch]) # (B, L)
        peak_tensors = torch.nested.nested_tensor([item['peak_windows'] for item in batch], layout=torch.jagged) # (N, N_p, C, W)
        marker_idx = torch.nested.nested_tensor([item['marker_idxs'] for item in batch], layout=torch.jagged) # (N, N_p)
        peak_centers = torch.nested.nested_tensor([item['peak_centers'] for item in batch], layout=torch.jagged) # (N, N_p, 2)

        return {
            'image': images,
            'target': targets,
            'adjusted_panel': adjusted_panel,
            'scaler': scaler,
            'peak_windows': peak_tensors,
            'marker_idxs': marker_idx,
            'peak_centers': peak_centers,
        }


    def set_up_metrics(self, use_evaluation_metric: bool) -> List[Optional[Metric]]:
        if not use_evaluation_metric:
            return []

        acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, average="micro").to(self._device)
        f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average="macro").to(self._device)

        return [f1, acc]

    def update_metric(self,
                      metric: Metric,
                      logits: torch.Tensor,
                      y_true: torch.Tensor):

        # logits: (N, K, C, L)
        # y_true: (N, C, L)
        if logits.ndim != 4:
            raise ValueError(f"logits must be (N, K, C, L), got shape {tuple(logits.shape)}")
        if y_true.ndim != 3:
            raise ValueError(f"y_true must be (N, C, L), got shape {tuple(y_true.shape)}")
        if logits.shape[0] != y_true.shape[0] or logits.shape[2:] != y_true.shape[1:]:
            raise ValueError(f"Batch/space dims mismatch: logits {tuple(logits.shape)} vs targets {tuple(y_true.shape)}")

        # ----- Argmax over class dimension to get predicted labels -----
        # preds: (N, C, L) integer class indices
        preds = logits.argmax(dim=1)  # (N, K, C, L) -> (N, C, L)

        # ----- Flatten for torchmetrics -----
        # preds_flat, targets_flat: (N*C*L,)
        preds_flat = preds.reshape(-1)
        targets_flat = y_true.reshape(-1)

        metric.update(preds_flat, targets_flat)


    def create_predictions(self, logits: torch.Tensor, batch: Sequence[HIDImage]) -> List[ScanpointPrediction]:
        ## TODO: support for multiclass classification
        probs = torch.softmax(logits, dim=1) # (N, num_classes, C, 4096)
        y_pred = probs[:, 1, :, :]  # Take the probabilities for the 'allele' class (index 1) (N, C, 4096)
        y_pred = y_pred.unsqueeze(-1)  # (N, C, 4096, 1)

        predictions = []
        for pred_image in y_pred:
            pred_image_np = pred_image.cpu().numpy()
            predictions.append(ScanpointPrediction(data=pred_image_np))

        return predictions

    def save(self, model_dir: PathLike):
        os.makedirs(model_dir, exist_ok=True)

        if self.autoencoder is not None:
            # 2) Save submodels
            ae_dir = os.path.join(model_dir, "autoencoder")
            self.autoencoder.save(ae_dir)

            combiner_path = os.path.join(model_dir, "combiner.pt")
            torch.save(self._model.combiner.state_dict(), combiner_path)

        pc_dir = os.path.join(model_dir, "peak_classifier")
        self.peak_classifier.save(pc_dir)


    def load(self, model_dir: PathLike):
        # 1) Load submodels first (so their modules are in place)
        if self.autoencoder is not None:
            ae_dir = os.path.join(model_dir, "autoencoder")
            self.autoencoder.load(ae_dir)


            combiner_path = os.path.join(model_dir, "combiner.pt")
            state = torch.load(combiner_path, map_location=self._device)
            self._model.combiner.load_state_dict(state)

        pc_dir = os.path.join(model_dir, "peak_classifier")

        if os.path.isdir(pc_dir):
            self.peak_classifier.load(pc_dir)




