"""This script contains routines for working with project info"""

import datetime
from dataclasses import dataclass
from pathlib import Path

from dataclasses_json import dataclass_json  # type: ignore
from typing import Optional


@dataclass_json
@dataclass
class ExperimentInfo:
    """A dataclass to store information about a particular experiment scenario"""

    validation_schema: Optional[str] = None
    model_type: Optional[str] = None
    model_version: Optional[str] = None

    def __post_init__(self):
        """Setting up an experiment timestamp and fold info"""
        self.timestamp = datetime.datetime.now()
        self._fold = "all_folds"

    @property
    def fold(self):
        return self._fold

    @fold.setter
    def fold(self, value: str):
        self._fold = value

    @property
    def model(self):
        return self.model_type

    @model.setter
    def model(self, value: str):
        self.model_type = value

    def get_experiment_name(self):
        """Detailed experiment name getter"""
        experiment_name = (
            f"validation_{self.validation_schema}__model_{self.model_type}_{self.model_version}_"
            f'{self.timestamp.strftime("%Y%m%d-%H%M%S")}'
        )
        return experiment_name


def get_project_root() -> Path:
    """
    Returns: absolute path to the project root directory
    """
    return Path.home() / "profun_outputs"


def get_output_root() -> Path:
    """
    Returns: absolute path to the output directory
    """
    return get_project_root() / "outputs"


def get_experiments_output() -> Path:
    """
    Returns: absolute path to the experiments directory
    """
    return get_output_root() / "experiment_results"


def get_config_root() -> Path:
    """
    Returns: absolute path to the config directory
    """
    return Path(get_project_root()) / "configs"
