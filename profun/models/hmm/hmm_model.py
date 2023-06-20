"""This class implements profile Hidden Markov model"""
from __future__ import annotations

import logging
import os
import pickle
import uuid
from itertools import groupby
from shutil import rmtree
from typing import Dict, List, Type

import numpy as np
import pandas as pd  # type: ignore
from tqdm.auto import tqdm

from profun.models.ifaces import BaseModel
from profun.utils.msa import get_fasta_seqs, generate_msa_mafft
from .hmm_dataclasses import HmmConfig, HmmPrediction

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def read_predictions_from_file(
        file_path: str, prediction_label: str
) -> List[HmmPrediction]:
    """
    The function gathers predictions from outputs on the disk
    :param file_path: a path to file with raw prediction
    :param prediction_label: label of interest
    :return: list of predictions
    """
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf8") as file:
        lines_file = file.readlines()
    lines = [line.split() for line in lines_file[3:]]
    predictions = []
    i = 0
    if len(lines) == 0:
        return []
    while len(lines[i]) > 1:
        line = lines[i]
        predictions.append(
            HmmPrediction(float(line[4]), float(line[5]), line[0], prediction_label)
        )
        i += 1
    return predictions


class ProfileHMM(BaseModel):
    """Class with the profile HMM algorithm implementation"""

    def __init__(self, config: HmmConfig):
        super().__init__(
            config=config,
        )
        self.config = config
        self.working_directory = self.output_root / "_out"
        if os.path.exists(self.working_directory):
            rmtree(self.working_directory)
        os.makedirs(self.working_directory)
        self.class_name_2_path_to_model_path: Dict[str, str]
        self.class_2_groups = None
        self.class_name_2_path_to_model_paths = None

    def prep_fasta_seqs(
            self, df: pd.DataFrame, type_name: str = None, group_name: str = None
    ) -> str:
        """
        This function prepares inputs for a particular class as a fasta format
        :param df: input dataframe
        :param type_name: class name
        :param group_name: name of a group (e.g. a clade)
        :return: string representation in a fasta format
        """
        if type_name is not None and group_name is not None:
            df_subset = df.loc[
                (df[self.config.target_col_name] == type_name)
                & (df[self.config.group_column_name] == group_name)
                ]
            seqs = df_subset[self.config.seq_col_name].values
            ids = df_subset[self.config.id_col_name].values
        else:
            seqs = df[self.config.seq_col_name].values
            ids = df[self.config.id_col_name].values
        return get_fasta_seqs(seqs, ids)

    def _train_for_class_group(
            self, df: pd.DataFrame, class_name: str, group_name: str
    ) -> str:
        """
        This function trains a HMM predictor for a given class
        :param df: input dataframe
        :param class_name: class name
        :return: path to a stored HMM
        """
        fasta_str = self.prep_fasta_seqs(df, class_name, group_name)

        logger.info(
            "Training for class %s, group %s, fasta size: %d",
            class_name,
            group_name,
            len(fasta_str.split(">")),
        )

        model_id = str(uuid.uuid4())
        generate_msa_mafft(fasta_str=fasta_str,
                           output_name=f"{self.working_directory}/{model_id}_msa.out",
                           n_jobs=self.config.n_jobs,
                           clustal_output_format=False)
        # check number of lines in the msa file
        with open(f"{self.working_directory}/{model_id}_msa.out", "r") as file:
            msa_lines = file.readlines()
        if len(msa_lines) == 0:
            raise ValueError("Empty MSA file")
        os.system(
            f"hmmbuild {self.working_directory}/{model_id}.hmm {self.working_directory}/{model_id}_msa.out"
        )
        return f"{self.working_directory}/{model_id}.hmm"

    def predict_for_class_group(
            self, df: pd.DataFrame, class_name: str, group: str
    ) -> str:
        """
        Prediction of the specified class
        :param df: input dataframe
        :param class_name: class to predict
        :return: path to a table with predictions
        """
        test_fasta = self.prep_fasta_seqs(df)
        with open(
                f"{self.working_directory}/_test.fasta",
                "w",
                encoding="utf8",
        ) as file:
            file.writelines(test_fasta.replace("'", "").replace('"', ""))

        logger.info(f'Predicting for class {class_name}, fasta size: {len(test_fasta.split(">"))}')

        result_id = str(uuid.uuid4())
        assert (
                self.class_name_2_path_to_model_paths is not None
        ), "Predicting class before training the profile HMM model"
        os.system(
            f"hmmsearch -E {self.config.search_e_threshold} --tblout {self.working_directory}/_{result_id}.tbl {self.class_name_2_path_to_model_paths[(class_name, group)]} {self.working_directory}/_test.fasta > {self.working_directory}/_{result_id}.out"
        )
        return f"{self.working_directory}/_{result_id}.tbl"

    def aggregate_predictions(self, class_name_2_pred_path: Dict[tuple[str, str], str], do_major_class_agg: bool = False
                              ) -> pd.DataFrame:
        """
        The function aggregates prediction analogously to Terzyme algorithm https://link.springer.com/article/10.1186/s13007-017-0269-0 if do_major_class_agg == True
        :return: a dataframe with predictions
        """
        class_name_2_pred_list = {}
        for (class_name, kingdom), prediction_path in class_name_2_pred_path.items():
            class_name_2_pred_list[(class_name, kingdom)] = read_predictions_from_file(
                prediction_path, class_name
            )
            os.remove(prediction_path)
            os.remove(prediction_path.replace(".tbl", ".out"))
        if do_major_class_agg:
            all_predictions_sorted = sorted(
                sum(class_name_2_pred_list.values(), []), key=lambda x: x.id
            )
            predictions_dict = {
                uniprot_id: max(predictions)
                for uniprot_id, predictions in groupby(
                    all_predictions_sorted, key=lambda x: x.id
                )
            }
            predictions_to_output = predictions_dict.values()
        else:
            predictions_to_output = sum(class_name_2_pred_list.values(), [])
        ids_list = []
        e_val_list = []
        y_pred_list = []
        for prediction in predictions_to_output:
            ids_list.append(prediction.id)
            e_val_list.append(prediction.e_value)
            y_pred_list.append(prediction.prediction_label)

        predictions_df = pd.DataFrame(
            {self.config.id_col_name: ids_list,
             self.config.target_col_name: y_pred_list,
             "E": e_val_list}
        )
        return predictions_df

    def fit_core(self, train_df: pd.DataFrame, class_name: str = None):
        assert isinstance(
            self.config, HmmConfig
        ), "HHM config instance is expected to be of type HmmConfig"
        if self.config.group_column_name is None:
            train_df[self.config.group_column_name] = "all"
        train_df.drop_duplicates(
            subset=[self.config.id_col_name, self.config.target_col_name], inplace=True
        )

        logger.info("Train size: %d", len(train_df))
        if self.config.class_names is None:
            self.config.class_names = [
                x
                for x in train_df[self.config.target_col_name].unique()
                if not pd.isnull(x)
            ]

        self.class_2_groups = {
            class_name: train_df.loc[
                train_df[self.config.target_col_name] == class_name,
                self.config.group_column_name,
            ].unique()
            for class_name in self.config.class_names
        }
        self.class_name_2_path_to_model_paths = dict()
        for class_name in self.config.class_names:
            for kingdom in self.class_2_groups[class_name]:
                if sum(
                        (train_df[self.config.target_col_name] == class_name)
                        & (train_df[self.config.group_column_name] == kingdom)
                ) >= 2:
                    try:
                        self.class_name_2_path_to_model_paths[(class_name, kingdom)] = self._train_for_class_group(
                            train_df, class_name=class_name, group_name=kingdom)
                    except ValueError:
                        continue

        with open(
                f"{self.working_directory}/class_name_2_path_to_model_paths.pkl",
                "wb",
        ) as file:
            pickle.dump(self.class_name_2_path_to_model_paths, file)

    def predict_proba(self, val_df: pd.DataFrame, return_long_df: bool = False) -> np.ndarray | pd.DataFrame:
        if self.config.group_column_name is None:
            val_df[self.config.group_column_name] = "all"
        assert val_df[self.config.id_col_name].nunique() == len(
            val_df
        ), "Expected input to predict_proba without duplicated ids"
        logger.info("Val size: %d", len(val_df))
        assert (
                self.class_name_2_path_to_model_paths is not None
        ), "Predicting before training the HMM model"
        assert (
                self.config.class_names is not None
        ), "Class names were not derived and stored during training"
        batch_results = []
        for batch_i in tqdm(range(len(val_df) // self.config.pred_batch_size + 1),
                            desc='Predicting with BLASTp-matching..'):
            val_df_batch = val_df.iloc[
                           batch_i * self.config.pred_batch_size: (batch_i + 1) * self.config.pred_batch_size]

            class_name_2_pred_path = {
                (class_name, kingdom): self.predict_for_class_group(
                    val_df_batch, class_name, kingdom
                )
                for class_name in self.config.class_names
                for kingdom in self.class_2_groups[class_name]
                if (class_name, kingdom) in self.class_name_2_path_to_model_paths
            }
            pred_df = self.aggregate_predictions(class_name_2_pred_path)
            pred_df = pred_df.merge(val_df, on=self.config.id_col_name, how="right").set_index(
                self.config.id_col_name
            )
            pred_df = pred_df.loc[val_df[self.config.id_col_name]].reset_index()
            pred_df["probability"] = pred_df["E"]
            pred_df.loc[
                pred_df["probability"] > self.config.zero_conf_level, "probability"
            ] = self.config.zero_conf_level
            pred_df["probability"] /= self.config.zero_conf_level
            pred_df.loc[pred_df["probability"].isnull(), "probability"] = 1
            pred_df["probability"] = 1 - pred_df["probability"]

            if return_long_df:
                batch_results.append(pred_df[[self.config.id_col_name,
                                              self.config.target_col_name,
                                              "probability"]])
            else:
                val_proba_np = np.zeros((len(val_df), len(self.config.class_names)))
                for class_i, class_name in enumerate(self.config.class_names):
                    bool_idx = (pred_df[self.config.target_col_name] == class_name).values
                    val_proba_np[bool_idx, class_i] = pred_df.loc[bool_idx, "probability"]
                batch_results.append(val_proba_np)
        if return_long_df:
            return pd.concat(batch_results)
        return np.concatenate(batch_results)

    @classmethod
    def config_class(cls) -> Type[HmmConfig]:
        return HmmConfig
