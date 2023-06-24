from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree, copyfile
from typing import Type, Optional, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from uuid import uuid4

from profun.models.ifaces import BaseModel
from profun.models.blast_model import BlastConfig

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logger = logging.getLogger(__name__)


@dataclass
class FoldseekConfig(BlastConfig):
    """
    A data class to store Foldseek-model attributes
    """
    local_pdb_storage_path: Optional[str | Path] = None


class FoldseekMatching(BaseModel):
    def __init__(
            self,
            config: FoldseekConfig,
    ):
        super().__init__(
            config=config,
        )
        self.config = config
        self.working_directory = self.output_root / "_working_directory"
        if os.path.exists(self.working_directory):
            rmtree(self.working_directory)
        os.makedirs(self.working_directory)
        if self.config.local_pdb_storage_path is None:
            self.local_pdb_storage_path = self.working_directory / "_trn_db"
        else:
            self.local_pdb_storage_path = Path(self.config.local_pdb_storage_path)
        self.local_pdb_storage_path.mkdir(exist_ok=True, parents=True)
        self.trn_db_path = None
        self.train_df = None

    def organize_db_folder(self, list_of_required_ids: List[str]):
        available_ids = {file.stem for file in self.local_pdb_storage_path.glob('*.pdb')}
        ids_to_download = [uniprot_id for uniprot_id in list_of_required_ids if uniprot_id not in available_ids]
        path_to_id_file = self.working_directory / "_temp_ids_list"
        with open(path_to_id_file, "w") as file:
            file.writelines('\n'.join(ids_to_download))
        subprocess.check_output(
            f"python -m profun.utils.alphafold_struct_downloader --structures-output-path {self.local_pdb_storage_path} --path-to-file-with-ids {path_to_id_file}".split(),
            stderr=sys.stdout,
        )
        os.remove(path_to_id_file)
        # moving only the required ids; a possible alternative for the future: --tar-exclude option of foldseek createdb
        selection_path = self.working_directory / f"_{uuid4()}"
        for uniprot_id in list_of_required_ids:
            filename = f"{uniprot_id}.pdb"
            copyfile(self.local_pdb_storage_path/filename, selection_path/filename)
        return selection_path

    def _train(self, df: pd.DataFrame) -> str:
        list_of_required_trn_ids = list(set(df[self.config.id_col_name].values))
        trn_structs_path = self.organize_db_folder(list_of_required_trn_ids, self.trn_db_path)
        logger.info(
            f"Prepared Foldseek trn folder"
        )
        createdb_out = subprocess.check_output(
            f"foldseek createdb {trn_structs_path} trn_db --threads {self.config.n_jobs}".split(),
            stderr=sys.stdout,
        )
        logger.info(f"Trn DB, foldseek createdb output: {createdb_out}")
        # moving back the additional ids
        rmtree(trn_structs_path)
        return "trn_db"

    def _predict(self, df: pd.DataFrame, trn_db_name: str) -> str:
        list_of_required_trn_ids = list(set(df[self.config.id_col_name].values))
        query_structs_path = self.organize_db_folder(list_of_required_trn_ids)
        logger.info(
            f"Prepared Foldseek query folder"
        )
        createdb_out = subprocess.check_output(
            f'foldseek createdb {query_structs_path} query_db --threads {self.config.n_jobs}'.split(),
            stderr=sys.stdout,
        )
        logger.info(f"Query DB, foldseek createdb output: {createdb_out}")
        rmtree(query_structs_path)
        search_out = subprocess.check_output(f"foldseek search query_db {self.trn_db_path} {self.working_directory}/resultDB tmp -e {self.config.e_threshold} --max-seqs {self.config.n_neighbours}".split(),
                                             stderr=sys.stdout)
        logger.info(f"Foldseek search output: {search_out}")
        result_conversion_out = subprocess.check_output(f'foldseek convertalis query_db {self.trn_db_path} {self.working_directory}/resultDB {self.working_directory}/result.tsv --format-output query,target,evalue'.split(),
                                             stderr=sys.stdout)
        logger.info(f"Result conversion output: {result_conversion_out}")
        return f"{self.working_directory}/result.tsv"

    def fit_core(self, train_df: pd.DataFrame, class_name: str = None):
        train_df.drop_duplicates(
            subset=[self.config.id_col_name, self.config.target_col_name], inplace=True
        )
        if (self.trn_db_path is None or
                np.any(self.train_df[[self.config.id_col_name, self.config.target_col_name]] != train_df[
                    [self.config.id_col_name, self.config.target_col_name]])):
            self.train_df = train_df.copy()
            self.trn_db_path = self._train(train_df.drop_duplicates(subset=[self.config.id_col_name]))

    def predict_proba(self, val_df: pd.DataFrame, return_long_df: bool = False) -> [np.ndarray | pd.DataFrame]:
        assert val_df[self.config.id_col_name].nunique() == len(
            val_df
        ), "Expected input to predict_proba without duplicated ids"
        if return_long_df:
            predicted_ids, predicted_classes, predicted_probs = [], [], []
        else:
            all_predicted_batches = []
        for batch_i in tqdm(range(len(val_df) // self.config.pred_batch_size + 1),
                            desc='Predicting with Foldseek-matching..'):
            val_df_batch = val_df.iloc[
                           batch_i * self.config.pred_batch_size: (batch_i + 1) * self.config.pred_batch_size]
            if len(val_df_batch):
                output_path = self._predict(val_df_batch, self.trn_db_path)
                results_df = pd.read_csv(
                    output_path, sep='\t', header=None, names=[f"{self.config.id_col_name}_queried", f"{self.config.id_col_name}_matched", "evalue"],
                )
                for colname in [f"{self.config.id_col_name}_queried", f"{self.config.id_col_name}_matched"]:
                    results_df[colname] = results_df[colname].map(lambda x: x.replace(".pdb", ""))
                results_merged_with_train_df = results_df.merge(
                    self.train_df[[self.config.id_col_name, self.config.target_col_name]],
                    left_on=f"{self.config.id_col_name}_matched",
                    right_on=self.config.id_col_name,
                    copy=False,
                )
                label_and_nn_counts = (results_merged_with_train_df
                                       .groupby(f"{self.config.id_col_name}_queried")[
                                           [self.config.target_col_name, f"{self.config.id_col_name}_matched"]]
                                       .agg(
                    {self.config.target_col_name: lambda x: [Counter(x)], f"{self.config.id_col_name}_matched": lambda x: [len(set(x))]})
                                       .reset_index()
                                       )
                label_and_nn_counts['prediction_dict'] = (
                        label_and_nn_counts[self.config.target_col_name] + label_and_nn_counts['Matched ID']).map(
                    lambda x: {class_name: class_count / x[1] for class_name, class_count in x[0].items()})
                label_and_nn_counts = label_and_nn_counts.merge(
                    val_df_batch, left_on=f"{self.config.id_col_name}_blasted",
                    right_on=self.config.id_col_name, how="right"
                )
                if return_long_df:
                    for _, row in label_and_nn_counts.iterrows():
                        if isinstance(row['prediction_dict'], dict):
                            for class_name, class_prob in row['prediction_dict'].items():
                                predicted_ids.append(row[self.config.id_col_name])
                                predicted_classes.append(class_name)
                                predicted_probs.append(class_prob)
                else:
                    val_proba_np_batch = np.zeros((len(val_df_batch), len(self.config.class_names)))
                    for class_i, class_name in enumerate(self.config.class_names):
                        val_proba_np_batch[:, class_i] = label_and_nn_counts['prediction_dict'].map(
                            lambda x: x[class_name] if isinstance(x, dict) and class_name in x else 0
                        )
                    indices_batch = label_and_nn_counts[self.config.id_col_name].values
                    orig_val_2_ord = {value: i for i, value in enumerate(val_df_batch[self.config.id_col_name])}
                    order_of_predictions_in_orig_batch = sorted(range(len(indices_batch)),
                                                                key=lambda idx: orig_val_2_ord[indices_batch[idx]])
                    all_predicted_batches.append(val_proba_np_batch[order_of_predictions_in_orig_batch])
        if return_long_df:
            return pd.DataFrame({self.config.id_col_name: predicted_ids,
                                 self.config.target_col_name: predicted_classes,
                                 "probability": predicted_probs})
        val_proba_np = all_predicted_batches[0] if len(all_predicted_batches) == 1 else np.concatenate(
            all_predicted_batches)
        return val_proba_np

    @classmethod
    def config_class(cls) -> Type[FoldseekConfig]:
        return FoldseekConfig
