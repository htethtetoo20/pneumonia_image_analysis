"""
Parse pneumonia sub-type and patient identity out of Kermany CXR filenames.

    person1003_bacteria_2934.jpeg   -> BACTERIAL, patient "person1003"
    person1003_virus_1685.jpeg      -> VIRAL,     patient "person1003"
    IM-0115-0001.jpeg               -> NORMAL,    patient "IM-0115"
    NORMAL2-IM-0135-0001.jpeg       -> NORMAL,    patient "NORMAL2-IM-0135"

Patient IDs matter: several studies share one child, and splitting by image
instead of by patient leaks anatomy across train/val and inflates validation
scores (worst for bacterial-vs-viral, where memorising a patient is a viable
shortcut). Two caveats, both resolved conservatively toward over- rather than
under-grouping: `personN` numbering restarts per sub-type and again in the
official test folder, so a patient key is only comparable within one source
folder -- never pool train and test IDs.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

NORMAL = "NORMAL"
BACTERIAL = "BACTERIAL"
VIRAL = "VIRAL"

#: Class order used for the 3-class task. Matches ImageFolder's alphabetical
#: sort, so `datasets.ImageFolder` assigns these exact indices.
SUBTYPE_CLASSES = [BACTERIAL, NORMAL, VIRAL]

_PNEUMONIA_RE = re.compile(r"^(person\d+)_(bacteria|virus)_", re.IGNORECASE)
_NORMAL_RE = re.compile(r"^(?:(NORMAL\d+)-)?IM-(\d+)", re.IGNORECASE)


class UnparseableFilename(ValueError):
    """Raised when a filename matches neither the pneumonia nor normal pattern."""


def subtype_of(path: str | Path) -> str:
    """Return BACTERIAL, VIRAL or NORMAL for one image path."""
    name = Path(path).name
    match = _PNEUMONIA_RE.match(name)
    if match:
        return BACTERIAL if match.group(2).lower() == "bacteria" else VIRAL
    if _NORMAL_RE.match(name):
        return NORMAL
    raise UnparseableFilename(f"Cannot infer sub-type from filename: {name}")


def patient_id(path: str | Path) -> str:
    """Return a patient key for one image path (only comparable within one source folder)."""
    name = Path(path).name
    match = _PNEUMONIA_RE.match(name)
    if match:
        return match.group(1).lower()

    match = _NORMAL_RE.match(name)
    if match:
        prefix, study = match.groups()
        return f"{prefix.lower()}-im-{study}" if prefix else f"im-{study}"

    raise UnparseableFilename(f"Cannot infer patient ID from filename: {name}")


def hold_out_patients(
    paths: Sequence[Path],
    val_fraction: float,
    seed: int,
    stratum_of: Callable[[Path], str],
) -> set[Path]:
    """
    Choose validation images by holding out whole patients, stratified by `stratum_of`.

    All `paths` must come from one source folder -- patient IDs are only unique within one.
    """
    by_patient: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_patient[patient_id(path)].append(path)

    by_stratum: dict[str, list[str]] = defaultdict(list)
    for pid, images in by_patient.items():
        majority = Counter(stratum_of(p) for p in images).most_common(1)[0][0]
        by_stratum[majority].append(pid)

    rng = random.Random(seed)
    val_paths: set[Path] = set()
    for stratum in sorted(by_stratum):
        patients = sorted(by_stratum[stratum])
        rng.shuffle(patients)
        target = sum(len(by_patient[pid]) for pid in patients) * val_fraction
        held = 0
        for pid in patients:
            if held >= target:
                break
            val_paths.update(by_patient[pid])
            held += len(by_patient[pid])

    return val_paths
