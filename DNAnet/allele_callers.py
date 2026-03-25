import logging
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from typing import List, Tuple, Any

import numpy as np
import torch

from DNAnet.data.data_models.dna_models import Panel, Marker, Allele
from DNAnet.data.data_models.structs import ScanpointPrediction, AllelePrediction

LOGGER = logging.getLogger('dnanet')

NON_AUTOSOMAL_MARKERS = ['AMEL', 'DYS391', 'DYS576', 'DYS570']
scan_point_annotation = namedtuple('ScanPointAnnotation',
                                   ['dye_index', 'start', 'end', 'label'])


class AlleleCaller(ABC):
    """
    Base class for an object that can call alleles from the predicted segmentation image.
    """

    @classmethod
    @abstractmethod
    def call_alleles_batch(cls,
                           data: dict[str, Any],
                           predictions: List[ScanpointPrediction]) -> List[AllelePrediction]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def translate_pixels_to_alleles(
            cls,
            scaler: torch.Tensor,
            prediction: ScanpointPrediction,
            image: torch.Tensor,
            panel: Panel
    ) -> List[Marker]:
        raise NotImplementedError


class NearestBasePairCaller(AlleleCaller):
    """
    Object that calls alleles from predicted segmentation image, by comparing the mean base
    pair location of a predicted bin with the mean base pair of an allele from the panel.
    """

    @classmethod
    def call_alleles_batch(cls,
                           data: dict[str, Any],
                           predictions: List[ScanpointPrediction]) -> List[AllelePrediction]:
        adjusted_panels: List[Panel] = data['adjusted_panel'] # (B)
        scalers: torch.Tensor = data['scaler'] # (B, N)
        images: torch.Tensor = data['input'] # (B, C, N)

        allele_predictions = []
        for prediction, adjusted_panel, scaler, image in zip(predictions, adjusted_panels, scalers, images):
            allele_predictions.append(cls.translate_pixels_to_alleles(scaler=scaler, prediction=prediction, image=image, panel=adjusted_panel))

        return allele_predictions




    @classmethod
    def translate_pixels_to_alleles(
            cls,
            scaler: torch.Tensor,
            prediction: ScanpointPrediction,
            image: torch.Tensor,
            panel: Panel
    ) -> List[Marker]:
        """
        Translate the predictions in the `prediction_image` to actual marker
        and allele names. First search for the mean base pair of a prediction
        group using the `scaler`. Then find the allele name corresponding to
        the base pair that is closest to the mean predicted base pair via the panel.
        """
        scaler = scaler.numpy()
        prediction_image = prediction.data
        image = image.numpy()

        loci_dict = defaultdict(set)
        rfus = defaultdict(int)
        for dye_index, dye in enumerate(prediction_image):
            # find indices of groups of positive predictions (where logits are greater than .5)
            positives, _ = np.where(dye >= 0.5)
            if positives.size == 0:  # no predictions present in this dye
                LOGGER.warning(f"No predictions present in dye row {dye_index}")
                continue

            # split the positives in separate arrays by splitting on where the
            # indices are not consecutive
            predicted_bins = np.split(positives, np.where(np.diff(positives) != 1)[0] + 1)
            for prediction_bin in predicted_bins:
                # get the mean basepair of the bin via its pixel values and the scaler
                allele_name, marker_name = cls.get_marker_and_allele_from_bin(dye_index, panel,
                                                                              prediction_bin,
                                                                              scaler)
                loci_dict[(dye_index, marker_name)].add(allele_name)
                # save highest rfu found for this allele (alleles may be found several times)
                max_rfu = max(image[dye_index, prediction_bin])
                rfus[(marker_name, allele_name)] = int(max(
                    rfus[(marker_name, allele_name)],
                    max_rfu
                ))

        return [
            Marker(
                dye_index,
                marker_name,
                alleles=[
                    Allele(name=allele_name, height=rfus[(marker_name, allele_name)])
                    for allele_name in alleles
                ],
            )
            for (dye_index, marker_name), alleles in loci_dict.items()
        ]

    @classmethod
    def get_marker_and_allele_from_bin(cls, dye_index, panel, prediction_bin, scaler):
        mean_bp = np.mean(scaler[:, prediction_bin])
        marker_name, allele_name = cls.get_allele_by_nearest_bp(dye_index, mean_bp, panel)
        return allele_name, marker_name

    @staticmethod
    def get_allele_by_nearest_bp(
            dye_index: int,
            base_pair: float,
            panel: Panel
    ) -> Tuple[str, str]:
        """
        Retrieve the marker and allele name that is, according to the panel,
        closest to the provided `base_pair` located on the dye with index `dye_index`.
        """
        all_basepairs_on_dye = panel.dye_bp_to_allele_mapping[dye_index].keys()
        nearest_bp = min(all_basepairs_on_dye, key=lambda k: abs(k - base_pair))
        marker_name, allele_name = panel.get_allele_name_by_dye_and_bp(dye_index, nearest_bp)
        return marker_name, allele_name
