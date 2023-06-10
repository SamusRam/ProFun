"""This module defines an abstract class for models"""
import inspect
import json
import os
import pickle
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Type

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold  # type: ignore
from skopt import gp_minimize  # type: ignore
from skopt.space import Categorical, Integer, Real  # type: ignore
from skopt.utils import use_named_args  # type: ignore

from profun.evaluation.metrics import eval_model_mean_average_precision
from profun.models.ifaces.config_baseclasses import BaseConfig
from profun.utils.project_info import get_output_root
import logging

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logger = logging.getLogger(__name__)


class BaseModel(ABC, BaseEstimator):
    """Base model class with an abstract method train_and_predict"""

    def __init__(
        self,
        config: BaseConfig,
    ):
        self.config = config
        assert isinstance(self.config.experiment_info.timestamp, datetime)
        if self.config.experiment_info.model_type is None:
            self.config.experiment_info.model_type = self.__class__.__name__
        self.output_root = (
            get_output_root()
            / config.experiment_info.model_type
            / config.experiment_info.model_version
            / config.experiment_info.validation_schema
            / config.experiment_info.fold
            / config.experiment_info.timestamp.strftime("%Y%m%d-%H%M%S")
        )
        self.output_root.mkdir(exist_ok=True, parents=True)
        self.classifier_class = None
        try:
            self.per_class_optimization = self.config.per_class_optimization
            if self.per_class_optimization is None:
                self.per_class_optimization = False
        except AttributeError:
            self.per_class_optimization = False
        if self.per_class_optimization:
            self.class_2_classifier = dict()

    @abstractmethod
    def fit_core(self, train_df: pd.DataFrame, class_name: str = None):
        """
        Function for training model instance
        :param train_df: pandas dataframe containing training data
        :param class_name: name of a class for the separate model fitting for the class
        """
        raise NotImplementedError

    def fit(self, train_df: pd.DataFrame):
        """
        Fit function
        :param train_df: pandas dataframe containing training data
        """
        try:
            per_class_optimization = self.config.per_class_optimization
        except AttributeError:
            per_class_optimization = False
        if self.config.optimize_hyperparams:
            try:
                n_fold_splits = self.config.n_fold_splits
            except AttributeError:
                n_fold_splits = 5
            try:
                use_cross_validation = self.config.use_cross_validation
            except AttributeError:
                use_cross_validation = True
            try:
                reuse_existing_partial_results = (
                    self.config.reuse_existing_partial_results
                )
            except AttributeError:
                reuse_existing_partial_results = False

            self.optimize_hyperparameters(
                train_df,
                n_calls=self.config.n_calls_hyperparams_opt,
                per_class_optimization=per_class_optimization,
                n_fold_splits=n_fold_splits,
                use_cross_validation=use_cross_validation,
                reuse_existing_partial_results=reuse_existing_partial_results,
                **self.config.hyperparam_dimensions,
            )
        try:
            load_per_class_params_from = self.config.load_per_class_params_from
            if load_per_class_params_from is None:
                load_per_class_params_from = False
        except AttributeError:
            load_per_class_params_from = False
        if load_per_class_params_from:
            load_per_class_params_from = Path(load_per_class_params_from)
            previous_results = list(
                load_per_class_params_from.glob(
                    f"*/hyperparameters_optimization/best_params_*.json"
                )
            )
            assert len(
                previous_results
            ), f"Requested to load per-class parameters from {load_per_class_params_from}, but no parameters in json are found"
            logger.info(f"Loading hyper parameters from: {previous_results[0]}")
            with open(previous_results[0], "r") as file:
                best_params = json.load(file)
            self.set_params(**best_params)

        if (
            self.config.optimize_hyperparams or load_per_class_params_from
        ) and per_class_optimization:
            for class_name in self.config.class_names:
                self.fit_core(train_df, class_name=class_name)
        else:
            self.fit_core(train_df)

    @abstractmethod
    def predict_proba(self, val_df: pd.DataFrame) -> np.ndarray:
        """
        Model predict method
        :param val_df: pandas dataframe containing instances to score the model on
        :returns numpy array containing predicted class probabilities
        """
        raise NotImplementedError

    def set_params(self, **kwargs):
        """
        It's a generic function setting values of all parameters in the kwargs
        """
        for attribute_name, value in kwargs.items():
            if attribute_name not in {"class_name", "per_class"}:
                self.__setattr__(attribute_name, value if value != "None" else None)

    def optimize_hyperparameters(
        self,
        train_df: pd.DataFrame,
        n_calls: int,
        per_class_optimization: bool,
        n_fold_splits: int,
        use_cross_validation: bool,
        reuse_existing_partial_results: bool,
        **dimension_params,
    ):
        logger.info("Starting hyperparameter optimization...")
        if per_class_optimization:
            class_names = self.config.class_names
        else:
            class_names = ["all_classes"]
        for class_name in class_names:
            prefix = "" if class_name == "all_classes" else f"{class_name}_"
            if reuse_existing_partial_results:
                previous_results = list(
                    self.output_root.glob(
                        f"../*/hyperparameters_optimization/optimization_results_detailed_{prefix}*.pkl"
                    )
                )
                if len(previous_results):
                    logger.info(
                        f"found previous results for class {class_name}: {previous_results}"
                    )
                    with open(previous_results[0], "rb") as file:
                        best_params, _, _ = pickle.load(file)
                    self.set_params(**best_params)
                    logger.info("Restored previous results")
                    continue

            type_2_skopt_class = {
                "categorical": Categorical,
                "float": Real,
                "int": Integer,
            }
            dimensions = []
            # x0 -> to enforce evaluation of the default parameters
            initial_instance_parameters = self.__dict__
            if hasattr(self, "classifier_class"):
                classifier_attributes = inspect.getfullargspec(
                    self.classifier_class
                ).kwonlydefaults
                if classifier_attributes is not None:
                    classifier_attributes.update(initial_instance_parameters)
                    initial_instance_parameters = classifier_attributes
            x0 = []
            for name, characteristics in dimension_params.items():
                if characteristics["type"] != "categorical":
                    next_dims = type_2_skopt_class[characteristics["type"]](
                        *characteristics["args"], name=name
                    )
                else:
                    next_dims = type_2_skopt_class[characteristics["type"]](
                        characteristics["args"], name=name
                    )
                dimensions.append(next_dims)
                assert (
                    name in initial_instance_parameters
                ), f"Hyperparameter {name} does not seem to be a model attribute"
                x0.append(initial_instance_parameters[name])
            run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

            # optimization results will be stored in
            if not (self.output_root / "hyperparameters_optimization").exists():
                (self.output_root / "hyperparameters_optimization").mkdir(exist_ok=True)

            # The objective function to be minimized
            def make_objective(train_df, space, cross_validation):
                # This decorator converts your objective function with named arguments into one that
                # accepts a list as argument, while doing the conversion automatically
                @use_named_args(space)
                def objective_value(**params):
                    if class_name == "all_classes":
                        prefix = ""
                    else:
                        prefix = f"{class_name}_"
                    params = {
                        f"{prefix}{param_name}": param_value
                        for param_name, param_value in params.items()
                    }
                    logger.info("setting params", params)
                    self.set_params(**params)
                    map_scores = []
                    available_ids = train_df[self.config.id_col_name].drop_duplicates()
                    id_2_classes = train_df.groupby(self.config.id_col_name)[
                        self.config.target_col_name
                    ].agg(set)
                    try:
                        for trn_idx, val_idx in cross_validation.split(
                            available_ids,
                            available_ids.map(
                                lambda uniprot_id: tuple(
                                    sorted(id_2_classes.loc[uniprot_id])
                                )
                            )
                            if not per_class_optimization
                            else available_ids.map(
                                lambda uniprot_id: class_name
                                in id_2_classes.loc[uniprot_id]
                            ).astype(int),
                        ):
                            trn_ids = set(available_ids.iloc[trn_idx].values)
                            val_ids = set(available_ids.iloc[val_idx].values)
                            trn_df = train_df[train_df[self.config.id_col_name].isin(trn_ids)]
                            vl_df = train_df[train_df[self.config.id_col_name].isin(val_ids)]
                            self.fit_core(
                                trn_df,
                                class_name=class_name
                                if per_class_optimization
                                else None,
                            )
                            map_scores.append(
                                eval_model_mean_average_precision(
                                    self,
                                    vl_df,
                                    selected_class_name=None
                                    if not per_class_optimization
                                    else class_name,
                                )
                            )
                            if not use_cross_validation:
                                break
                        score = np.mean(map_scores)
                    except ValueError as e:
                        print(e)
                        score = 1.0

                    ckpts = list(
                        (self.output_root / "hyperparameters_optimization").glob(
                            "*_params.json"
                        )
                    )
                    if len(ckpts) > 0:
                        past_performances = sorted(
                            [
                                float(str(ckpt_name.stem).split("_")[0])
                                for ckpt_name in ckpts
                            ]
                        )
                    if len(ckpts) == 0 or past_performances[-1] > score:
                        for ckpt in ckpts:
                            os.remove(ckpt)
                        with open(
                            self.output_root
                            / "hyperparameters_optimization"
                            / f"{score:.5f}_params_{run_timestamp}.json",
                            "w",
                            encoding="utf8",
                        ) as file:
                            json.dump(
                                {
                                    key: (
                                        val
                                        if not isinstance(val, np.integer)
                                        else int(val)
                                    )
                                    for key, val in params.items()
                                },
                                file,
                            )

                    return score

                return objective_value

            k_fold = StratifiedKFold(
                n_splits=n_fold_splits, shuffle=True, random_state=42
            )

            objective = make_objective(
                train_df, space=dimensions, cross_validation=k_fold
            )

            gp_round = gp_minimize(
                func=objective,
                dimensions=dimensions,
                acq_func="gp_hedge",
                n_calls=n_calls,
                n_initial_points=min(10, n_calls // 5),
                random_state=42,
                verbose=True,
                x0=x0,
            )
            best_params = {
                f"{prefix}{dimensions[i].name}": param_value
                for i, param_value in enumerate(gp_round.x)
            }
            self.set_params(**best_params)

            with open(
                self.output_root
                / "hyperparameters_optimization"
                / f"optimization_results_detailed_{prefix}{run_timestamp}.pkl",
                "wb",
            ) as file:
                pickle.dump((best_params, gp_round.x_iters, gp_round.func_vals), file)

        def _jsonify_value(value):
            if isinstance(value, np.int64):
                return int(value)
            if isinstance(value, np.float):
                return float(value)
            return value

        # just in case all hyperparameters for all classes were already pre-computed
        if not (self.output_root / "hyperparameters_optimization").exists():
            (self.output_root / "hyperparameters_optimization").mkdir(exist_ok=True)
        with open(
            self.output_root
            / "hyperparameters_optimization"
            / f"best_params_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json",
            "w",
        ) as file:
            json.dump(
                {
                    param: _jsonify_value(val)
                    for param, val in self.get_model_specific_params().items()
                },
                file,
            )

    @classmethod
    @abstractmethod
    def config_class(cls) -> Type[BaseConfig]:
        """
        A getter of a config class
        """
        raise NotImplementedError

    def get_params(self, deep: bool = True):
        return {
            "config": (
                deepcopy(self.__dict__["config"]) if deep else self.__dict__["config"]
            )
        }

    def get_model_specific_params(self, class_name: str = None):
        initial_instance_parameters = self.__dict__
        try:
            classifier_args = inspect.getfullargspec(self.classifier_class)
            classifier_attributes = set(classifier_args.kwonlydefaults.keys())
            classifier_attributes.update(
                {arg for arg in classifier_args.args if arg != "self"}
            )
        except AttributeError:  # sometimes everything is being hidden in **kwargs
            # (e.g. it's the case for sklearn wrapper of xgboost),
            # then fallback to default constructor
            instance = self.classifier_class()
            classifier_attributes = instance.__dict__
        return {
            key: val
            for key, val in initial_instance_parameters.items()
            if key in classifier_attributes
            or (
                "_".join(key.split("_")[1:]) in classifier_attributes
                and (class_name is None or key.split("_")[0] == class_name)
            )
        }
