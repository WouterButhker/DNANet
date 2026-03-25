from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from DNAnet.data.data_models.dna_models import Panel
    from DNAnet.data.data_models.hid_image import HIDImage
    from DNAnet.data.data_models.structs import ScanpointAnnotation

PathLike = Union[str, Path]

ProfileTuple = Tuple["HIDImage", Optional["ScanpointAnnotation"], Optional["Panel"], Optional[np.ndarray]]
