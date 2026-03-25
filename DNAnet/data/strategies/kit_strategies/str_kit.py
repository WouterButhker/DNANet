"""
Kit definitions: capture the user-facing choice of kit (name) and all
kit-specific settings such as size standard and panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, List

from DNAnet.data.data_models.dna_models import Panel
from DNAnet.data.strategies.kit_strategies.internal_size_standard import SizeStandard, WEN_ILS_BPS


@dataclass(frozen=True)
class STRKit:
    """
    Describes a DNA profiling kit and its key configuration.

    Attributes:
        name: Human-readable kit name/identifier (e.g. "provedit", "nfi", "globalfiler").
        raw_data_columns: Columns in the HID file containing raw data.
        analyzed_data_columns: Columns in the HID file containing analyzed data (preprocessed).
        size_standard: Internal size standard used by this kit.
        panel_path: Optional path to the panel file describing markers/alleles.
        panel: Panel object describing markers/alleles.
        markers: Optional list of marker names used by this kit (for quick checks or validation).
        description: Optional free-text description.
    """

    name: str
    size_standard: SizeStandard
    raw_data_columns: Optional[List[str]] = None
    analyzed_data_columns: Optional[List[str]] = None
    num_dyes: int = 6
    panel_path: Optional[Path] = None
    panel: Optional[Panel] = None
    markers: Optional[Sequence[str]] = None
    description: Optional[str] = None




POWERPLEX_Y23 = STRKit(
    name="POWERPLEX_Y23",
    size_standard=WEN_ILS_BPS,
    panel_path=None,
    panel=None,
    markers=None,
    description="POWERPLEX_Y23 kit using WEN_ILS size standard.",
    raw_data_columns=["DATA_1","DATA_2", "DATA_3","DATA_4", "DATA_105"],
    analyzed_data_columns=["DATA_9","DATA_10", "DATA_11","DATA_12", "DATA_205"]
)
