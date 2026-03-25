import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Literal, Iterable, Optional, Sequence, Set, Union

import torch
from torch.utils.data import Dataset, Subset

from DNAnet.data.data_models.dna_models import Allele, Marker
from DNAnet.data.data_models.structs import AlleleAnnotation, Annotation
from DNAnet.utils import get_prefix_from_filename

FileCategory = Literal['sample', 'ladder', 'control', 'unknown']

LOGGER = logging.getLogger('dnanet')

class DataLoadingStrategy(Enum):
    SUPERIOR = 'superior' # loads the raw data, and applies baseline correction
    ANALYZED = 'analyzed' # loads the data preprocessed by the electropherogram equipment
    RAW = 'raw' # loads the raw data without any preprocessing


class DatasetStrategy(ABC):
    """Unified strategy interface for dataset-specific behavior.

    This can include file categorization, contributor parsing and allele loading.
    """

    def __init__(self, split_by_replicates: bool = False, **kwargs):
        self.split_by_replicates = split_by_replicates

    @property
    def data_loading_strategy(self) -> DataLoadingStrategy:
        return DataLoadingStrategy.SUPERIOR

    @classmethod
    @abstractmethod
    def collect_dataset_files(
        cls, path: str | Path, **kwargs
    ) -> List[Tuple[Path, Optional[Annotation], Optional[Path]]]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def categorize_file(cls, file_name: str) -> FileCategory:
        """Return the category (sample/ladder/control/unknown) for a given file name."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_contributors(cls, file_name: str) -> List[str]:
        """Derive contributor file stems from the HID filename."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def parse_annotation_file(cls, path: str | Path) -> Dict[str, List[Marker]] | None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create_annotation_for_sample(
        cls, annotation_mapping: Dict[str, List[Marker]], sample_name: str
    ) -> AlleleAnnotation:
        raise NotImplementedError

    # TODO: test splitting methods
    def split_dataset(self,
                      dataset: Dataset,
                      k_folds: Optional[int] = None,
                      split_frac: Optional[Union[int, float]] = None,
                      seed: Optional[int] = None,
                      ) -> List[Tuple[Subset, Subset]]:
        """
        Splits a dataset into multiple subsets based on specified parameters. This method supports several
        splitting strategies, including k-fold splitting and custom length-based splitting. Optionally, the
        splitting can account for replicates within the dataset.

        Arguments:
            dataset (Dataset): The dataset to be split into subsets.
            k_folds (Optional[int]): Number of folds to split the dataset into for k-fold cross-validation.
                This parameter cannot be specified simultaneously with `lengths`.
            split_frac (Optional[Sequence[int | float]]): A sequence indicating the proportions or counts for
                each subset in the split. Cannot be used simultaneously with `k_folds`.
            seed (Optional[int]): A seed for the random number generator to ensure reproducibility of the split.
            split_by_replicates (bool): If True, splits the dataset by considering replicates. Defaults to False.

        Returns:
            List[Subset] | List[Tuple[Subset, Subset]]: A list containing the resulting subsets after the split.
                When `split_by_replicates` is True, each subset is represented as a tuple of (subset with
                replicates, subset without replicates).

        Raises:
            ValueError: If both `k_folds` and `lengths` are specified, or if `split_by_replicates` is True while
                `lengths` does not have exactly two elements, or if neither `k_folds` nor `lengths` is provided.
        """
        assert not (k_folds and split_frac), "Cannot specify both k_folds and lengths."

        if split_frac is not None and self.split_by_replicates:
            return [self._split_by_replicas_and_noc(dataset, split_frac, seed=seed)]
        elif split_frac is not None and not self.split_by_replicates:
            lengths = [split_frac, 1 - split_frac]
            datasets = self._split_dataset(dataset, lengths, seed=seed)
            return [(datasets[0], datasets[1])]
        elif k_folds is not None and self.split_by_replicates:
            return self._split_k_fold_by_replicas_and_noc(dataset, k_folds, seed=seed)
        elif k_folds is not None and not self.split_by_replicates:
            return self._split_dataset_k_fold(dataset, k_folds, seed=seed)
        else:
            raise ValueError("Either lengths or k_folds must be specified.")



    @staticmethod
    def _split_dataset(dataset: Dataset | Sequence,
                       lengths: Sequence[int | float],
                       seed: Optional[int] = None
                       ) -> List[Subset]:
        """Split a dataset into subsets with the given lengths."""

        class _SequenceWrapper(Dataset):
            def __init__(self, seq: Sequence):
                self.data = seq

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        if isinstance(dataset, Sequence):
            dataset = _SequenceWrapper(dataset)

        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        return torch.utils.data.random_split(dataset, lengths, generator=generator)

    @classmethod
    def _split_dataset_k_fold(cls, dataset: Dataset, k_folds: int, seed: Optional[int] = None) -> List[Tuple[Subset, Subset]]:
        """Split a dataset into 'k_folds' folds."""
        # TODO test method
        dataset_size = len(dataset)

        if k_folds < 2:
            raise ValueError("k_folds must be at least 2.")
        if k_folds > dataset_size:
            raise ValueError(
                f"k_folds ({k_folds}) cannot exceed dataset size ({dataset_size})."
            )

        base_fold_size = dataset_size // k_folds
        remainder = dataset_size % k_folds
        fold_lengths = [
            base_fold_size + (1 if i < remainder else 0) for i in range(k_folds)
        ]

        folds = cls._split_dataset(dataset, fold_lengths, seed=seed)

        kfold_subsets: List[Tuple[Subset, Subset]] = []
        for val_idx, val_subset in enumerate(folds):
            train_indices: List[int] = []
            for fold_idx, fold_subset in enumerate(folds):
                if fold_idx != val_idx:
                    train_indices.extend(fold_subset.indices)

            train_subset = Subset(dataset, train_indices)
            kfold_subsets.append((train_subset, val_subset))

        return kfold_subsets

    @classmethod
    def split_by_genotype(cls, dataset: Dataset, genotypes: Set[str]) -> Tuple[Subset, Subset]:
        """
        Split images into two datasets based on contributor IDs. Images with contributors that
        are a subset of `genotypes` go to A; images disjoint from `genotypes` go to B; mixed or
        unparseable contributors are discarded.
        """
        # TODO method is unused?
        community_A_images: List[int] = []
        community_B_images: List[int] = []
        ambiguous_images: List[int] = []

        for i, img in enumerate(dataset):
            try:
                contribs = set(cls.get_contributors(img.path.name))
            except Exception as e:
                LOGGER.warning("Could not extract contributors for %s: %s", img.path, e)
                ambiguous_images.append(i)
                continue

            if contribs.issubset(genotypes):
                community_A_images.append(i)
            elif contribs.isdisjoint(genotypes):
                community_B_images.append(i)
            else:
                ambiguous_images.append(i)

        LOGGER.info("Split by genotypes: A=%d, B=%d, ambiguous=%d",
                    len(community_A_images), len(community_B_images), len(ambiguous_images))

        community_A_dataset = Subset(dataset, community_A_images)
        community_B_dataset = Subset(dataset, community_B_images)

        return community_A_dataset, community_B_dataset

    @staticmethod
    def _group_indices_by_noc_and_prefix(dataset: Dataset) -> Dict[str, Dict[str, List[int]]]:
        """Group dataset indices by NoC and profile prefix so replicas stay together."""
        grouped: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        for idx, img in enumerate(dataset):
            prefix = get_prefix_from_filename(img.path.stem)
            noc = prefix[-1]
            grouped[noc][prefix].append(idx)
        return grouped

    @classmethod
    def _split_by_replicas_and_noc(
            cls, dataset: Dataset, fraction: float, seed: Optional[int] = None
    ) -> Tuple[Subset, Subset]:
        """
        Split dataset while keeping replicas (same prefix) together and balancing NoC.
        Returns two torch Subset objects.
        """
        if not 0 < fraction < 1:
            raise ValueError(f"fraction must be between 0 and 1, got {fraction}.")

        grouped = cls._group_indices_by_noc_and_prefix(dataset)
        indices_1: List[int] = []
        indices_2: List[int] = []

        for prefix_to_indices in grouped.values():
            all_prefixes = list(prefix_to_indices.keys())
            split_idx = int(len(all_prefixes) * fraction)
            lengths = [split_idx, len(all_prefixes) - split_idx]

            prefix_subsets = cls._split_dataset(all_prefixes, lengths, seed=seed)
            for prefix in prefix_subsets[0]:
                indices_1.extend(prefix_to_indices[prefix.data])
            for prefix in prefix_subsets[1]:
                indices_2.extend(prefix_to_indices[prefix.data])

        return Subset(dataset, indices_1), Subset(dataset, indices_2)

    @classmethod
    def _split_k_fold_by_replicas_and_noc(
            cls, dataset: Dataset, n_folds: int, seed: Optional[int] = None
    ) -> List[Tuple[Subset, Subset]]:
        """
        K-fold split while keeping replicas together and balancing NoC across folds.
        Returns list of (train_subset, val_subset).
        """
        dataset_size = len(dataset)
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2.")
        if n_folds > dataset_size:
            raise ValueError(
                f"n_folds ({n_folds}) cannot exceed dataset size ({dataset_size})."
            )

        grouped = cls._group_indices_by_noc_and_prefix(dataset)
        fold_indices: List[List[int]] = [[] for _ in range(n_folds)]

        for prefix_to_indices in grouped.values():
            all_prefixes = list(prefix_to_indices.keys())

            base_fold_size = len(all_prefixes) // n_folds
            remainder = len(all_prefixes) % n_folds
            fold_lengths = [
                base_fold_size + (1 if i < remainder else 0) for i in range(n_folds)
            ]

            prefix_folds = cls._split_dataset(all_prefixes, fold_lengths, seed=seed)
            for fold_idx, prefix_fold in enumerate(prefix_folds):
                for prefix in prefix_fold:
                    fold_indices[fold_idx].extend(prefix_to_indices[prefix.data])

        kfold_subsets: List[Tuple[Subset, Subset]] = []
        for val_idx, val_indices in enumerate(fold_indices):
            train_indices: List[int] = []
            for fold_idx, indices in enumerate(fold_indices):
                if fold_idx != val_idx:
                    train_indices.extend(indices)

            kfold_subsets.append(
                (Subset(dataset, train_indices), Subset(dataset, val_indices))
            )

        return kfold_subsets

    @classmethod
    def build_marker(
        cls,
        marker_name: str,
        allele_names: Iterable[str],
        allele_heights: Optional[Iterable[float]] = None,
    ) -> Marker:
        from DNAnet.data.strategies.strategy_registry import StrategyRegistry

        _scaling_strategy = StrategyRegistry.get_scaling_strategy()
        dye_row = _scaling_strategy.panel.get_dye_row(marker_name)
        if dye_row is None:
            raise RuntimeError(
                f'Marker {marker_name} not found in panel {_scaling_strategy.panel}. '
                'Please check the panel or the marker name.'
            )

        allele_names = list(allele_names)
        allele_heights_checked: Iterable[float | None]
        if allele_heights is None:
            allele_heights_checked = [None] * len(allele_names)
        else:
            allele_heights_checked = allele_heights

        return Marker(
            dye_row=dye_row,
            name=marker_name,
            alleles=[
                Allele(name=allele_name, height=allele_height)
                for allele_name, allele_height in zip(
                    allele_names, allele_heights_checked, strict=True
                )
            ],
        )
