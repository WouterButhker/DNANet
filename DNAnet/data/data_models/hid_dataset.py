from __future__ import annotations

import csv
import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from DNAnet.data.caching import _load_cached_hf_data
from DNAnet.data.data_models.dna_models import Panel
from DNAnet.data.data_models.hid_image import HIDImage, Ladder
from DNAnet.data.data_models.structs import AlleleAnnotation, ScanpointAnnotation
from DNAnet.data.strategies.dataset_strategies import DatasetStrategy
from DNAnet.data.strategies.kit_strategies.scaling_strategy import ScalingStrategy
from DNAnet.data.strategies.strategy_registry import StrategyRegistry
from DNAnet.data.utils import find_peak_idx_near_or_in_range, find_peak_boundary
from DNAnet.typing import PathLike, ProfileTuple
from DNAnet.utils import (
    is_no_control, is_rd_hid_filename,
)

if TYPE_CHECKING:
    from DNAnet.models.base_model import TransformData

LOGGER = logging.getLogger('dnanet')

FILENAME_FILTERS = {'is_rd_hid_filename': is_rd_hid_filename,
           'is_no_control': is_no_control,}

CATEGORIES = ["", "Allele", "Stutter", "PullUp", "BleedThrough", "Spike", "DyeBlob", "Artefact",
              "Unclear", "Shoulder", "ForeignDNA", "OverloadingArtifact"]


class HIDDataset(Dataset):
    """
    A class that can load the HID images from the 2p-5p R&D dataset with their annotations.

    :param root: root folder containing the (subdirectories containing) HID files.
    :param annotations_path: location of the folder containing annotations.
    :param panel_path: path to read the panel from.
    :param hid_to_annotations_path: path of the file that maps hid files to annotations.
    :param best_ladder_paths_csv: path of the file that maps the path of a hid image to the file
        path of its best ladder (produced by scripts/select_ladder_for_images.py).
    :param limit: number of hid files to read.
    :param load_in_memory: whether to read data from the cache.
    :param cache_path: location of the cache to read files from (if use_cache is true), or to
    write the files to (if use_cache is false).
    :param adjustment_of_annotations: the adjustment to be applied to the annotations, either
    'complete' to label the entire peak, or 'top' to only label the top of the peak.
    :param shuffle: whether the dataset should be shuffled when iterating.
    :param skip_if_invalid_ladder: whether to skip images that have invalid ladders.
    :param analysis_threshold_type: the analysis threshold type to use for annotations (either
        `DTH` (high) or `DTL` (low).
    :param ground_truth_as_annotations: whether to load the ground truth donor alleles as
        annotations.
    :param include_size_standard: whether to include the size standard in the HIDImage.
    :param data_loading_strategy: the strategy to load the data, either "raw", "analyzed" or "superior".
        "raw" means loading the raw data, "analyzed" means loading the analyzed data,
        and "superior" means loading the raw data and applying baseline subtraction
    :param group_replicas_in_split: whether to put measurements from the same profile (replicas)
    in the same set when splitting, and balance the number of profiles per noc, if false all
    replicas will be mixed.
    :param include_size_standard: whether to keep the internal lane standard (ILS) data in the
    underlying profiles. Useful for QC and labelling.
    :param data_loading_strategy: strategy to load the HID data. One of
    "raw", "analyzed", "superior"
    :param skip_if_invalid_internal_standard: if True, drops the file if the internal standard
    cannot be parsed. If false, uses no internal scaling (use with care!).
    """

    def __init__(self,
                 root: PathLike,
                 panel_path: Optional[PathLike] = None,
                 limit: Optional[int] = None,
                 load_in_memory: bool = False,
                 use_cache: bool = False,
                 cache_path: Optional[PathLike] = None,
                 adjustment_of_annotations: Optional[str] = None,
                 skip_if_invalid_ladder: Optional[bool] = False,
                 analysis_threshold_type: Optional[str] = 'DTL', # TODO add to annotation
                 ground_truth_as_annotations: Optional[bool] = False,
                 include_size_standard: bool = False,
                 split_by_replicates: bool = False,
                 dataset_strategy: str | DatasetStrategy = "NFI_RND",
                 scaling_strategy: str | ScalingStrategy = "POWER_PLEX_FUSION_6C",
                 skip_if_invalid_internal_standard: bool = True,
                 transform_func: Optional[TransformData] = None):




        self.transform = transform_func

        StrategyRegistry.configure_dataset(dataset_strategy, split_by_replicates=split_by_replicates)
        dataset_strategy = StrategyRegistry.get_dataset()
        StrategyRegistry.configure_kit(scaling_strategy)




        # If cache path is given and use_cache is set to true, load cached data.
        # TODO check cache
        if cache_path and use_cache:
            LOGGER.info(f"Loading data from arrow cache: {cache_path}")
            if ground_truth_as_annotations:
                if panel_path is None:
                    raise ValueError("Need panel when loading ground truth annotations")
                adjusted_panel = Panel(panel_path=panel_path)
            self._data = _load_cached_hf_data(cache_path, limit, include_size_standard)
            return
        # Otherwise, read data and cache (optional)
        else:
            LOGGER.info(f"Loading raw data from {root}")
            self._validate_dataset_args(panel_path, adjustment_of_annotations)

            adjusted_panel = Panel(panel_path=panel_path)
            # Map the hid file names to the annotation (optional for non-RD datasets)

            # if annotations_path and hid_to_annotations_path:
            #     self.annotation_dict = self._create_annotation_mapping_rd(
            #         analysis_threshold_type=analysis_threshold_type,
            #         hid_to_annotations_path=hid_to_annotations_path,
            #         annotations_path=annotations_path,
            #     )
            # else:
            #     self.annotation_dict = {}


            # Create a list of .hid files, with their ladder paths and annotations
            # self._files = self._collect_and_filter_file_paths()
            profile_path_tuples = dataset_strategy.collect_dataset_files(root)

            # if limit: # TODO add limit
            #     # Make sure to shuffle files before limiting, otherwise we would always take the
            #     # first `limit` when loading images.
            #     if shuffle:
            #         profile_path_tuples = random.Random().sample(profile_path_tuples, limit)
            #     else:
            #         profile_path_tuples = profile_path_tuples[:limit]
            #     LOGGER.info(f"Limiting dataset to {limit}")

            # Create all HIDImages and filter out invalid ones
            # TODO add filter step

            assert len(profile_path_tuples) > 0, "No files found for dataset! Please check the provided root path and dataset strategy."

            profile_paths, annotations, ladder_paths = map(list, zip(*profile_path_tuples))
            profile_paths: List[Path]
            annotations: List[Optional[AlleleAnnotation]]
            ladder_paths: List[Optional[Path]]

            ## parse ladders
            adjusted_panels: List[Optional[Panel]] = []
            for ladder_path in tqdm(ladder_paths, "Parsing ladders"):
                adjusted_panels.append(self._parse_ladder_to_adjusted_panel(ladder_path,
                                                                            adjusted_panel,
                                                                            skip_if_invalid_internal_standard))

            # init HIDImages
            scalers: List[np.ndarray] = []
            profiles: List[HIDImage] = []
            for profile_path in tqdm(profile_paths, "Initializing HIDImages"):
                profile = HIDImage(profile_path,
                                   include_size_standard=include_size_standard,
                                   load_in_memory=load_in_memory,
                                   skip_if_invalid_internal_standard=skip_if_invalid_internal_standard)
                if load_in_memory:
                    profile.load_from_disk() # force data to be loaded into memory
                    scalers.append(profile.scaler)
                profiles.append(profile)


            # parse annotations when necessary and possible.
            # We cannot parse annotations beforehand if we do not load the data into memory,
            # because we need the image rescaling. If we do not load the data into memory,
            # we parse the annotations during iteration
            if not load_in_memory:
                raise NotImplementedError() # TODO implement read on the go iterator

            scanpoint_annotations: List[Optional[ScanpointAnnotation]] = []

            if load_in_memory and isinstance(annotations[0], AlleleAnnotation):
                # convert allele annotation to scanpoint annotation
                for annotation, adjusted_panel, scaler in tqdm(zip(annotations, adjusted_panels, scalers), "Parsing annotations"):
                    annotation: Optional[AlleleAnnotation]
                    adjusted_panel: Optional[Panel]
                    scaler: np.ndarray

                    # if the annotation or panel is None, we cannot parse it, so we just skip it
                    if annotation is None or adjusted_panel is None:
                        scanpoint_annotations.append(None)
                    else:
                        scanpoint_annotations.append(self._translate_allele_to_scanpoint_annotation(annotation, adjusted_panel, scaler))

                annotations = scanpoint_annotations

            # we can now assume that all annotations are ScanpointAnnotations
            annotations: List[Optional[ScanpointAnnotation]]

            assert len(profile_paths) == len(annotations) == len(ladder_paths) == len(adjusted_panels) == len(profiles)

            # TODO filter out images with missing annotations?

            # TODO filter out images with invalid ladders?
            if skip_if_invalid_ladder:
                pass

            if adjustment_of_annotations:
                annotations = self._adjust_annotations(profiles, annotations, adjustment_of_annotations)

            profile_tuples: List[ProfileTuple] = list(zip(profiles, annotations, adjusted_panels, scalers))

            self._data = profile_tuples


            # TODO caching
            # if self.cache_path:
            #     LOGGER.info(f"Writing data to arrow cache: {self.cache_path}")
            #     write_to_hf_cache(self.cache_path, self._data, include_size_standard)


    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, Union[HIDImage, ScanpointAnnotation, Panel, np.ndarray, None]]:
        image, annotation, adjusted_panel, scaler = self._data[index]
        image: HIDImage
        annotation: Optional[ScanpointAnnotation]
        adjusted_panel: Optional[Panel]
        scaler: np.ndarray

        image.load_from_disk()
        sample = {
            "image": image,
            "annotation": annotation,
            "adjusted_panel": adjusted_panel,
            "scaler": scaler,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


    @staticmethod
    def _validate_dataset_args(adjustment_of_annotations: Optional[str],
                               panel_path: Optional[PathLike]):
        """
        Check the presence and validity of dataset arguments.
        """
        if adjustment_of_annotations is not None and \
                adjustment_of_annotations not in ["top", "complete"]:
            raise ValueError(
                "Unknown adjustment type for annotations found: "
                f"{adjustment_of_annotations}. Provide `top` or `complete`."
            )

        if not panel_path:
            raise ValueError("Panel path missing.")

    @staticmethod
    @lru_cache(maxsize=128)
    def _parse_ladder_to_adjusted_panel(ladder_path: Optional[Path],
                                        default_panel: Panel,
                                        use_cache: bool = False
                                        ) -> Optional[Panel]:
        if ladder_path is None:
            return None

        ladder = Ladder(ladder_path, default_panel, use_cache, skip_if_invalid_internal_standard=True)
        panel = ladder.panel
        return panel



    @staticmethod
    def _load_best_ladder_paths(best_ladders_csv_path: Optional[PathLike]) -> Dict[str, str]:
        """
        Loads a csv file containing for every HID image path the path of the best corresponding
        ladder. If not provided, an empty dictionary is returned, meaning that no ladders will
        be considered when loading the data.
        """
        if not best_ladders_csv_path:
            LOGGER.info("No csv path for best ladders is provided.")
            return dict()

        LOGGER.info(f"Loading best ladder paths from {best_ladders_csv_path}.")
        with open(best_ladders_csv_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            next(reader)
            data = {k: v if v != '' else None for (k, v) in reader}
        return data


    @staticmethod
    def _translate_allele_to_scanpoint_annotation(allele_annotation: AlleleAnnotation, adjusted_panel: Panel, scaler: np.ndarray) -> ScanpointAnnotation:
        """
        Translates allele annotation to scanpoint annotation by finding the closest scanpoint indices for the left and right bins
        of each allele using the provided scaler and adjusted panel. The entire allelic bin is annotated as 1.

        All non-annotated scanpoints are labeled as 0, while annotated scanpoints are labeled as 1.
        The resulting scanpoint annotation is a binary matrix of shape (num_dyes, num_scanpoints)

            :param allele_annotation: AlleleAnnotation object containing the allele annotations to be translated.
            :param adjusted_panel: Panel object containing the adjusted panel information to be used for translation.
            :param scaler: numpy array containing the scaler values to be used for finding the closest scanpoint indices.
            :return: ScanpointAnnotation object containing the translated scanpoint annotation.
        """
        scanpoint_annotation = np.zeros((StrategyRegistry.get_scaling_strategy().kit.num_dyes,
                                         StrategyRegistry.get_scaling_strategy().scanpoint_resolution), dtype=np.int8)
        for locus in allele_annotation.data:
            for allele in locus.alleles:
                # for each allele, find the left and right bin of the allele using the panel that has been adjusted by the corresponding ladder.
                _, left_bin, right_bin = adjusted_panel.get_allele_basepair_and_bins(locus.name, allele.name)

                # use the scaler to find the closest scanpoint indices for the left and right bins of the allele
                left_scanpoint = np.argmin(np.abs(scaler - left_bin))
                right_scanpoint = np.argmin(np.abs(scaler - right_bin))

                scanpoint_annotation[locus.dye_row, left_scanpoint : right_scanpoint] = 1

        return ScanpointAnnotation(data=scanpoint_annotation)

    @staticmethod
    def _adjust_annotations(profiles: List[HIDImage],
                            annotations: List[Optional[ScanpointAnnotation]],
                            adjustment_type: str = 'top',
                            threshold: int = 40
                            ) -> List[Optional[ScanpointAnnotation]]:
        """
        Adjust the annotation of the image.
        If `adjustment_type` is 'top', (by default) we label the top of the peak, instead of the
        entire bin. If the type is 'complete', we find the entire peak and label this.
        Note that the original image annotations are overwritten in place.
        """

        assert len(profiles) == len(annotations)

        for profile, annotation in zip(profiles, annotations):
            if annotation is None:
                continue


            for dye_idx, dye_data in enumerate(profile.data):
                # find indices of groups of positive annotations
                _annotations = np.where(annotation.data[dye_idx] == 1)
                if _annotations.size == 0:  # no annotation present in this dye
                    continue
                annotation_groups = np.split(_annotations, np.where(np.diff(_annotations) != 1)[0] + 1)
                for ann_group in annotation_groups:
                    annotation.data[dye_idx, ann_group] = 0.
                    peak_idx = find_peak_idx_near_or_in_range(dye_data, ann_group, threshold)

                    if peak_idx.size == 0:
                        LOGGER.warning(f"No peak found above {threshold}rfu. "
                                       f"Original annotation is removed "
                                       "and no adjustment is applied "
                                       f"(dye {dye_idx}, bin {ann_group}, "
                                       f"rfus {dye_data[ann_group].flatten()}).")
                    else:
                        if adjustment_type == 'complete':
                            # find the boundary of the peak and annotate the range
                            start, end = find_peak_boundary(dye_data, int(peak_idx),
                                                            threshold)
                            annotation.data[dye_idx, np.arange(start, end + 1)] = 1.
                        elif adjustment_type == 'top':
                            # label only the top of the peak
                            annotation.data[dye_idx, peak_idx] = 1.
                        else:
                            raise ValueError("Unknown adjustment type found: "
                                             f"{adjustment_type}. Please provide"
                                             " either `top` or `complete`.")
        return annotations
