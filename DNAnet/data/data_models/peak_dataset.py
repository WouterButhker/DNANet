import logging
from typing import Optional, List, Iterator, Any

from datasets import IterableDataset

from DNAnet.data.data_models.hid_dataset import HIDDataset
from DNAnet.data.preprocessing.peak_extraction import extract_peak_windows
from DNAnet.data.preprocessing.preprocess_item import RFU_MAX_VALUE, preprocess_peak

LOGGER = logging.getLogger('dnanet')


class PeakWindowDataset(IterableDataset):
    """
    A dataset that extracts peak windows from HIDImages, with optional preprocessing and caching.
    Each item is an ExtractedPeak object containing the data around the peak center and its annotation.

     Caching:
      - If use_cache_peaks=True and cache_path_peaks is provided, peaks will be loaded from cache if it
      exists, otherwise they will be extracted and written to cache for future use.
      - The cache stores only the necessary fields to reconstruct ExtractedPeak objects, and loads them
      as memory-mapped numpy arrays for efficient access without loading everything into RAM at once.
    """
    def __init__(self,
                 hid_dataset: HIDDataset,
                 threshold: float,
                 window_size: int,
                 filter_peaks: bool,
                 preprocess: bool = True,
                 smooth_keep_factor: Optional[float] = 0.4,
                 log_scale: bool = True,
                 max_rfu_value: Optional[int] = RFU_MAX_VALUE,
                 labels: List[str] = ["allele", "noise"],
                 include_max_pool_dyes: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.hid_dataset = hid_dataset
        self.threshold = threshold
        self.window_size = window_size
        self.filter_peaks = filter_peaks
        self.preprocess = preprocess
        self.smooth_keep_factor = smooth_keep_factor
        self.log_scale = log_scale
        self.max_rfu_value = max_rfu_value

        self.include_max_pool_dyes = include_max_pool_dyes
        self.label_to_idx = dict(zip(labels, range(len(labels))))
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()} # invert the mapping



        LOGGER.info(f"Initialized PeakWindowDataset with threshold={threshold}, window_size={window_size}")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.hid_dataset._data is None:
            raise ValueError("HIDDataset must be loaded before using PeakWindowDataset")


        for profile_dict in self.hid_dataset:
            image = profile_dict["image"]
            annotation = profile_dict["annotation"]
            adjusted_panel = profile_dict["adjusted_panel"]

            if annotation is None or adjusted_panel is None:
                continue

            peaks = extract_peak_windows(image, annotation, adjusted_panel, self.threshold, self.window_size,
                                         include_max_pool_dyes=self.include_max_pool_dyes)

            # TODO: filter peaks based on threshold
            # if self.filter_peaks:
            #     threshold_type = self.analysis_threshold_type if self.analysis_threshold_type else "AT"
            #     peaks = filter_peaks_AT_LT(peaks, threshold_type)
            for peak in peaks:
                if self.preprocess:

                    peak = preprocess_peak(peak,
                                          smooth_keep_factor=self.smooth_keep_factor,
                                          log_scale=self.log_scale,
                                          max_rfu_value=self.max_rfu_value)

                sample = {
                    'peak': peak,
                    'annotation': peak.annotation,
                    'adjusted_panel': adjusted_panel,
                }
                yield sample
