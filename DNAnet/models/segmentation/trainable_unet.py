from dataclasses import dataclass
from typing import Sequence, Tuple, Optional, List, Union

import torch
from torch.nn import CrossEntropyLoss
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

from DNAnet.data.data_models.dna_models import Panel
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.data.data_models.structs import ScanpointAnnotation, ScanpointPrediction
from DNAnet.data.utils import process_image
from DNAnet.models.base_model import BaseModel, TransformData
from DNAnet.models.loss import DiceLoss
from DNAnet.models.segmentation.unet_architecture import UNet


@dataclass
class UNetTransformData(TransformData):
    def __call__(self, data: dict[str, Union[HIDImage, ScanpointAnnotation, Panel, None]]) -> dict[str, torch.Tensor]:
        image = data['image']
        input_data = torch.tensor(
            data=process_image(image.data, channels_first=True),
            dtype=torch.float32)

        annotation = data['annotation']
        target = torch.tensor(annotation).long() if annotation is not None else None
        return {
            'input': input_data,
            'target': target,
            'adjusted_panel': data['adjusted_panel'],
            'scaler': data['scaler'],
        }

class DNANet_UNet(BaseModel):
    """
    A setup for a U-Net model geared towards analysing dna profiles using PyTorch.
    """

    def get_transform(self) -> TransformData:
        return UNetTransformData()

    def __init__(self,
                 depth: int,
                 kernel_size: Tuple[int, int],
                 num_filters: int = 64,
                 device: Optional[str] = None,
                 apply_allele_caller: Optional[bool] = True,
                 num_classes: int = 2):
        """
        Initialize the DNAnet UNet.

        :param depth: depth the unet should have
        :param kernel_size: (height, width) of the kernel to use in the convolutional layers
        :param num_filters: the number of initial filters in the first conv layer
        :param device: The device on which the model should run. Should be either "cpu" or
        "cuda" for CPU or GPU respectively.
        :param apply_allele_caller: Whether to call actual alleles from the predicted segmentation
        """
        model = UNet(depth, kernel_size, num_filters, device, num_classes=num_classes)
        loss = CrossEntropyLoss() if num_classes > 2 else DiceLoss()
        super().__init__(model, loss, device, apply_allele_caller)
        self.num_classes = num_classes


    def update_metric(self,
                      metric: Metric,
                      logits: torch.Tensor,
                      y_true: torch.Tensor):

        best_class = torch.argmax(logits, dim=1)
        metric.update(torch.flatten(best_class), torch.flatten(y_true))


    def set_up_metrics(self, use_evaluation_metric: bool) -> List[Optional[Metric]]:
        """
        Use binary accuracy as evaluation metric if desired.
        """
        if not use_evaluation_metric:
            return []
        else:
            metrics = [MulticlassAccuracy(num_classes=self.num_classes, average='none'),
                       MulticlassJaccardIndex(num_classes=self.num_classes, average='macro')]
        return [metric.to(self._device) for metric in metrics]


    def create_predictions(self, logits: torch.Tensor, batch: Sequence[HIDImage]) -> List[ScanpointPrediction]:
        predictions = []
        for pred_im in logits:
            segmentation = torch.sigmoid(pred_im).cpu().detach().numpy()
            prediction = ScanpointPrediction(data=segmentation)
            predictions.append(prediction)
        return predictions
