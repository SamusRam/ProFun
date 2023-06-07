"""This module defines an abstract class for models"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional

import yaml  # type: ignore

from utils.project_info import ExperimentInfo


@dataclass
class BaseConfig:
    """
    A data class to store model attributes
    """

    experiment_info: ExperimentInfo
    id_col_name: str
    target_col_name: str
    seq_col_name: str
    class_names: list[str]
    optimize_hyperparams: bool
    n_calls_hyperparams_opt: int
    hyperparam_dimensions: dict[
        str,
    ]
    per_class_optimization: bool
    class_weights: dict[str, float]

    @classmethod
    def load(cls, path_to_config: Union[str, Path]) -> dict:
        """
        This class function loads config from a configs folder
        :param path_to_config:
        :return: a dictionary loaded from the config yaml
        """
        with open(path_to_config, encoding="utf-8") as file:
            configs_dict = yaml.load(file, Loader=yaml.FullLoader)
        return configs_dict



