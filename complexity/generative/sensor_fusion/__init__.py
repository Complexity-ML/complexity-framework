"""TR-Hash multimodal sensor-fusion models."""

from .config import SENSOR_MODALITIES, TRHashSensorFusionConfig
from .cuhkx_records import (
    CUHKX_DIRECTORY_MODALITIES,
    CUHKX_TRAIN_USERS,
    CUHKXRecord,
    CUHKXTestRecord,
    build_cuhkx_records,
    build_cuhkx_test_records,
    load_cuhkx_manifest,
    save_cuhkx_manifest,
    subject_disjoint_records,
)
from .data import (
    CUHKXSmallTrackDataset,
    CUHKXSmallTrackTestDataset,
    collate_cuhkx,
    collate_cuhkx_test,
)
from .folds import (
    CUHKX_CROSS_SUBJECT_FOLDS,
    CrossSubjectFold,
    resolve_cross_subject_fold,
    validate_cross_subject_folds,
)
from .model import TRHashSensorFusionBlock, TRHashSensorFusionClassifier
from .preprocessing import CUHKXPreprocessingConfig
from .transfer import load_pretrained_visual_tower

__all__ = [
    "SENSOR_MODALITIES",
    "CUHKX_DIRECTORY_MODALITIES",
    "CUHKX_TRAIN_USERS",
    "CUHKXRecord",
    "CUHKXTestRecord",
    "CUHKXPreprocessingConfig",
    "CUHKXSmallTrackDataset",
    "CUHKXSmallTrackTestDataset",
    "build_cuhkx_records",
    "build_cuhkx_test_records",
    "load_cuhkx_manifest",
    "save_cuhkx_manifest",
    "subject_disjoint_records",
    "collate_cuhkx",
    "collate_cuhkx_test",
    "CUHKX_CROSS_SUBJECT_FOLDS",
    "CrossSubjectFold",
    "resolve_cross_subject_fold",
    "validate_cross_subject_folds",
    "TRHashSensorFusionConfig",
    "TRHashSensorFusionBlock",
    "TRHashSensorFusionClassifier",
    "load_pretrained_visual_tower",
]
