"""Defining self-explainable datastructures"""

from dataclasses import dataclass
from functools import total_ordering
from typing import Optional

from profun.models.ifaces import BaseConfig


@dataclass
class HmmConfig(BaseConfig):
    """
    A config class for profile HMM
    """

    search_e_threshold: float
    zero_conf_level: float
    group_column_name: Optional[str] = None
    n_jobs: Optional[int] = 56
    pred_batch_size: Optional[int] = 100


@total_ordering
@dataclass
class HmmPrediction:
    """
    A data class to store and post-process profile HMM predictions
    """

    e_value: float
    score: float
    id: str
    prediction_label: str

    def __eq__(self, other):
        return self.score == other.score

    def __lt__(self, other):
        return self.score < other.score

