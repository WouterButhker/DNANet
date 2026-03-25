import csv
import logging
from binascii import crc32
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks

from DNAnet.data.data_models.base import Image
from DNAnet.data.data_models.dna_models import Allele, Marker, Panel
from DNAnet.data.parsing.parse_raw_hid import get_peak_data
from DNAnet.data.strategies.strategy_registry import StrategyRegistry
from DNAnet.data.utils import (
    assert_image_data_valid_format
)
from DNAnet.typing import PathLike

LOGGER = logging.getLogger("dnanet")


class HIDImage(Image):
    """
    Image representation of the raw peaks from a HID file serving as a DNA profile

    :param path: location of HID file
    :param include_size_standard: include size standard in the data attribute.
        if `true` all six dyes are included. For inspection of the HID file.
        if `false` only the first five dyes are included. For training + testing models.
    :param load_in_memory: whether retrieved peaks should be cached
    :param skip_if_invalid_internal_standard: if True, drops the file if the internal standard
       cannot be parsed. If false, uses no internal scaling (use with care!).
    """

    def __init__(self,
                 path: PathLike,
                 include_size_standard: bool = False,
                 load_in_memory: bool = True,
                 skip_if_invalid_internal_standard: bool = True):
        self.path = path if isinstance(path, Path) else Path(path)
        self.include_size_standard = include_size_standard
        self.load_in_memory = load_in_memory
        self.skip_if_invalid_internal_standard = skip_if_invalid_internal_standard
        self.root = self.path.parent
        self._data: Optional[np.ndarray] = None


    @property
    def data(self) -> np.ndarray:
        if self.load_in_memory:
            if self._data is None:
                self._data = self._read()
            return self._data
        return self._read()

    def load_from_disk(self) -> None:
        """
        Loads data from disk into memory if the `load_in_memory` attribute is True.

        Raises
        ------
        ValueError
            If `load_in_memory` is False, indicating that loading data into memory
            is not permitted.
        """
        if not self.load_in_memory:
            raise ValueError("Cannot load from disk to memory if `load_in_memory` is False.")
        if self._data is None:
            self._data = self._read()

    @cached_property
    def dimensions(self) -> Tuple[int, int]:
        """
        Returns a `(height, width)` tuple of the dimensions of the image.
        """
        return self.data.shape[0], self.data.shape[1]


    def _read(self) -> Optional[np.ndarray]:
        """
        Parse the raw hid image, validate the size standard and parse called alleles into a
        segmentation, if annotations are present.
        """
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))

        # Parse the raw hid image into a numpy array.
        if (profile := get_peak_data(self.path)) is None:
            return None


        size_standard_dye_lane = np.array(profile[-1])
        try:
            ss = StrategyRegistry.get_scaling_strategy().parse_size_standard(size_standard_dye_lane)
        except ValueError as e:
            LOGGER.warning(f"Size standard invalid for {self.path.name}: {e}")
            return None

        data = self._rescale_profile(
            profile,
            ss.rescaled_indices,
            self.include_size_standard,
        )
        self._scaler = ss.scaler



        if data is None:
            raise ValueError(f'Reading {self.path} resulted in None')
        try:
            assert_image_data_valid_format(data, n_color_channels=1)
        except ValueError as e:
            # TODO: Convert to uint8 successfully (strange behavior)
            if 'dtype of `data` must be' in str(e):
                pass
            else:
                raise

        # Cache the dimensions in case `use_cache` is False, so that we don't
        # have to reload the entire image when the dimensions are requested separately.
        self._dimensions = data.shape[:2]
        return data

    @property
    def hash(self) -> int:
        return crc32("/".join(self.path.relative_to(self.root).parts).encode())

    @property
    def scaler(self) -> np.ndarray:
        """
        Array in which the value are the base pairs and
        the index represents its position within the
        (scaled) array/data of the profile, e.g.:

        [65, 66, 67, ...,  474, 474.5, 475]
        in this example base-pair 67 should be placed on
        index 3 of an array. The size of the scalar depends
        on the `utils.RESCALE_SIZE` constant

        TODO: INCLUDE LOGIC
        TODO: | np.argmin(np.abs(self.scaler - allele.bin)
        """
        if self._scaler is None:
            # to avoid missing the scaler when we have not yet read the file.
            self._read()
        return self._scaler[np.newaxis, :]


    @staticmethod
    def _rescale_profile(
        profile: np.ndarray,
        rescaled_indices: np.ndarray,
        include_standard: bool,
    ) -> np.ndarray:
        """Rescale profile based on precomputed rescale indices.

        :param profile: array of dyes in chronological order
        :param rescale_indices: indices of the original profile corresponding to
            each pixel in the rescaled profile
        :param include_standard: if the size standard should be included in the
            final profile/data
        :return: parsed profile as array
        """
        # Select profile based on include_standard flag
        selected_profile = profile if include_standard else profile[:-1]
        data = selected_profile[:, rescaled_indices]
        return data[..., np.newaxis]

    @classmethod
    def _get_segmentation(
        cls,
        scaler,
        called_alleles: Sequence[Marker],
        shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Creates a binary mask based on the locations of called alleles in the annotation. Use
        the scaler to determine for an allele bin (a single base pair), the pixel location in
        the segmentation array.
        """
        image = np.zeros(shape, dtype=np.int8)
        for marker in called_alleles:
            for allele in marker.alleles:
                image[
                    marker.dye_row,
                    slice(*tuple(np.argmin(np.abs(scaler - allele.bin), axis=1))),
                    0
                ] = 1
        return image



    def __repr__(self):
        return f"HIDImage({self.path.name})"


class Ladder(HIDImage):
    """
    Base class for a ladder .hid image. The ladder contains (almost) all alleles present on
    every dye (these can be found in the `alleles_in_ladder` csv file) and by finding the exact
    locations of these peaks in every dye, we can rescale the .hid images of actual DNA profiles.
    This is necessary since sometimes alleles are not located at the base pairs we expect them to
    be, i.e. the base pair locations of the default panel (loaded from the .xml file).
    """
    _alleles_in_ladder = None

    def __init__(self,
                 path: PathLike,
                 default_panel: Panel,
                 load_in_memory: bool = True,
                 skip_if_invalid_internal_standard: bool = True):
        super().__init__(path=path,
                         include_size_standard=True,
                         load_in_memory=load_in_memory,
                         skip_if_invalid_internal_standard=skip_if_invalid_internal_standard)
        if Ladder._alleles_in_ladder is None:
            # load the alleles that should be present in every ladder
            Ladder._alleles_in_ladder = self.read_alleles_in_ladder()

        # we can try to find peaks if the ladder has valid data
        self.peak_indices = self.get_peak_indices() if self.data is not None else None
        # we can adjust the panel and find the corrected base pair locations of alleles if the
        # ladder has the correct number of peaks on every dye
        self._panel = self.adjust_panel(default_panel) if self.peak_indices else None

    @property
    def alleles_in_ladder(self) -> Dict:
        return self._alleles_in_ladder

    @property
    def panel(self) -> Panel | None:
        return self._panel

    @staticmethod
    def read_alleles_in_ladder() -> Dict[int, List[Tuple[str, str]]]:
        """
        Read 'ladder_alleles.csv' (with columns 'Marker', 'Allele' and 'Dye') into
        a dictionary with the dye rows as keys and Marker/Allele names as values. This file
        contains allele alleles that should be present in the ladder. Note that this file, and
        therefore neither the ladder, do not contain all alleles possible alleles.
        """
        lines = defaultdict(list)
        # FIXME: remove hardcoded path
        with open('resources/data/ladder_alleles.csv',
                  mode='r', newline='', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                lines[int(row['Dye'])].append((row['Marker'], row['Allele']))
        return lines

    def get_peak_indices(self) -> Optional[List[np.array]]:
        """
        Retrieve all peak indices per dye (using a simple peak finding algorithm) and check per
        dye whether the number of peaks in the ladder correspond to the expected number of peaks
        provided in `alleles_in_ladder`. If this is not the case, return None. The peak indices
        are pixel indices on the array.
        """
        all_peaks = []
        for dye_row in range(self.data.shape[0] - 1):
            dye = self.data[dye_row, :, 0]
            dynamic_threshold = np.max(dye) * 0.6

            # To be able to detect peaks at index 0, we pad the data with a 0
            padded_data = np.concatenate((np.zeros(1), dye))
            peaks, _ = find_peaks(padded_data, height=dynamic_threshold)
            peaks = peaks - 1  # remove padding

            if len(peaks) == len(self.alleles_in_ladder[dye_row]):
                all_peaks.append(peaks)
            else:
                LOGGER.warning(f"Expected {len(self.alleles_in_ladder[dye_row])} peaks on "
                               f"dye row {dye_row}, but found {len(peaks)} for ladder {self.path}.")
                return None
        return all_peaks

    def adjust_panel(self, original_panel: Panel) -> Panel:
        """
        Create an adjusted panel (meaning we adjust the base pairs of the alleles) using the
        found peak indices of the peaks in the ladder, using inter- and extrapolation.

        For more information on panel adjustment see:
        https://softgenetics.com/PDF/CalibratingPanelautoadjust_icon.pdf

        :param original_panel: The original panel, loaded from
        `resources/data/SGPanel_PPF6C.xml` from which we know all possible alleles.
        :return: Panel with the same alleles as the original panel, but adjusted base pairs based
        on the peaks in the ladder.
        """
        adjusted_panel = []
        for dye_row in range(self.data.shape[0] - 1):
            alleles_ladder_dye = self.alleles_in_ladder[dye_row]
            panel_markers = [marker for marker in original_panel._panel
                             if marker.dye_row == dye_row]

            # map marker/allele names to the base pair locations in the ladder
            marker_allele_to_bp = {
                (marker, allele): self.scaler[0, peak_idx]
                for (marker, allele), peak_idx in
                zip(alleles_ladder_dye, self.peak_indices[dye_row])
            }

            for marker in panel_markers:
                adjusted_alleles = []
                for idx, allele in enumerate(marker.alleles):
                    if allele.name in [a.name for a in adjusted_alleles]:
                        # we have already adjusted this allele
                        continue
                    ladder_base_pair = marker_allele_to_bp.get((marker.name, allele.name))
                    if ladder_base_pair:
                        # this allele was present in the ladder, therefore we can directly adjust
                        # the location with the ladder base pair location.
                        adjusted_alleles.append(
                            Allele(name=allele.name,
                                   base_pair=float(ladder_base_pair),
                                   left_bin=allele.left_bin,
                                   right_bin=allele.right_bin))
                    else:
                        # otherwise we should inter- or extrapolate using the other alleles
                        # present on the marker
                        result_allele = self._handle_missing_bp_in_ladder(
                            marker, allele, idx, marker_allele_to_bp)
                        adjusted_alleles.extend(result_allele)

                adjusted_panel.append(Marker(dye_row=dye_row,
                                             name=marker.name,
                                             alleles=adjusted_alleles))

        return Panel(panel_contents=adjusted_panel)

    def _handle_missing_bp_in_ladder(self,
                                     marker: Marker,
                                     allele: Allele,
                                     index: int,
                                     marker_allele_to_bp: Dict[Tuple[str, str], float]) -> \
            Sequence[Allele]:
        """
        Inter-/extrapolate to find the base pair from an allele in the panel that is not present
        in the ladder.

        :param marker: The marker to inter-/extrapolate basepair for.
        :param allele: The allele to inter-/extrapolate basepair for.
        :param index: The index of the allele in the marker.
        :param marker_allele_to_bp: mapping from marker/alleles to base pairs from the ladder.
        :return: A list of Alleles containing the inter-/extrapolated basepair.
        """
        # find all alleles after the provided alleles we can possibly use for inter-/extrapolation
        following_alleles = [
            allele for allele in marker.alleles[index + 1:]
            if marker_allele_to_bp.get((marker.name, allele.name))
        ]

        adjusted_alleles = []
        if index == 0:
            # We are at the beginning of the profile, therefore extrapolate with the next two
            # alleles with known ladder base pairs, that should always be present
            bp_in_panel = [allele.base_pair for allele in following_alleles[:2]]
            bp_in_ladder = [marker_allele_to_bp.get((marker.name, allele.name))
                        for allele in following_alleles[:2]]

            adjusted_alleles.append(self._extrapolate_base_pair(
                bp_in_panel, bp_in_ladder, marker, allele, marker_allele_to_bp))
        else:
            if not following_alleles:
                # We are at the end of the profile, therefore extrapolate with two previous
                # alleles with known ladder base pairs
                previous_alleles = marker.alleles[index - 2: index]
                bp_in_panel = [allele.base_pair for allele in previous_alleles]
                bp_in_ladder = [marker_allele_to_bp.get((marker.name, allele.name))
                            for allele in previous_alleles]

                adjusted_alleles.append(self._extrapolate_base_pair(
                    bp_in_panel, bp_in_ladder, marker, allele, marker_allele_to_bp))
            else:
                # Interpolate between previous and next allele with known ladder base pair. There
                # is always a previous allele since we move from left to right. This interpolation
                # may result in the adjustment of multiple alleles that have no ladder base pair,
                # in between the `previous` and `next` allele.
                prev_allele = marker.alleles[index - 1]
                prev_bp_in_ladder = marker_allele_to_bp[(marker.name, prev_allele.name)]
                prev_bp_in_panel = prev_allele.base_pair

                next_allele = following_alleles[0]
                next_bp_in_ladder = marker_allele_to_bp[(marker.name, next_allele.name)]
                next_bp_in_panel = next_allele.base_pair
                next_idx = marker.alleles.index(next_allele)

                # Interpolate all alleles in between the previous and next
                interp = StrategyRegistry.get_scaling_strategy().basepair_interpolator(
                    indices=[prev_bp_in_panel, next_bp_in_panel],
                    original_x_values=[prev_bp_in_ladder, next_bp_in_ladder]
                )
                for following_allele in marker.alleles[index:next_idx]:
                    interpolated_bp = interp(following_allele.base_pair)
                    marker_allele_to_bp[(marker.name, following_allele.name)] = \
                        float(interpolated_bp)

                    adjusted_alleles.append(Allele(
                        name=following_allele.name,
                        base_pair=float(interpolated_bp),
                        left_bin=following_allele.left_bin,
                        right_bin=following_allele.right_bin))

        return adjusted_alleles

    @staticmethod
    def _extrapolate_base_pair(indices: List[float],
                               x_values: List[float],
                               marker: Marker,
                               allele: Allele,
                               marker_allele_to_bp: Dict[
                                   Tuple[str, str], float]) -> Allele:
        """
        Extrapolates base pair for specified allele and returns an `Allele` object with the
        newly computed basepair location.
        """
        interp = StrategyRegistry.get_scaling_strategy().basepair_interpolator(indices=indices,
                                       original_x_values=x_values,
                                       extrapolate=True)
        extrapolated_bp = interp(allele.base_pair)
        # update the marker/allele to basepair mapping
        marker_allele_to_bp[(marker.name, allele.name)] = float(extrapolated_bp)

        return Allele(name=allele.name,
                      base_pair=float(extrapolated_bp),
                      left_bin=allele.left_bin,
                      right_bin=allele.right_bin)

    def __eq__(self, other):
        if not isinstance(other, Ladder):
            return False

        return self.path == other.path and np.array_equal(self.data, other.data) and all(
            np.array_equal(s, o) for s, o in zip(self.peak_indices, other.peak_indices))

    def __repr__(self):
        return f"Ladder({self.path.name})"
