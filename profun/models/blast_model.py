import logging
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from shutil import rmtree
from typing import Type, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from profun.models.ifaces import BaseConfig, BaseModel

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logger = logging.getLogger(__name__)


@dataclass
class BlastConfig(BaseConfig):
    """
    A data class to store Blast-model attributes
    """

    n_neighbours: int
    e_threshold: float
    n_jobs: Optional[int] = 20
    pred_batch_size: Optional[int] = 10_000


class BlastMatching(BaseModel):
    def __init__(
            self,
            config: BlastConfig,
    ):
        super().__init__(
            config=config,
        )
        self.config = config
        self.working_directory = self.output_root / "_working_directory"
        if os.path.exists(self.working_directory):
            rmtree(self.working_directory)
        os.makedirs(self.working_directory)
        self.db_path = None
        self.train_df = None

    def get_fasta_seqs(self, df, type_name=None):
        df.drop_duplicates(subset=[self.config.id_col_name], inplace=True)
        if type_name is not None:
            seqs = df.loc[
                df[self.config.target_col_name] == type_name, self.config.seq_col_name
            ].values
            ids = df.loc[
                df[self.config.target_col_name] == type_name, self.config.id_col_name
            ].values
        else:
            seqs = df[self.config.seq_col_name].values
            ids = df[self.config.id_col_name].values
        full_entries = [f">{entry_id}\n{entry_seq}" for entry_id, entry_seq in zip(ids, seqs)]
        unique_ids = {el.replace("'", "").replace('"', "") for el in ids}
        logger.info(f"For class {type_name}, the number of duplicated ids is {len(ids) - len(unique_ids)}")
        fasta_str = "\n".join(full_entries)
        return fasta_str.replace("'", "").replace('"', "")

    def _train(self, tps_df: pd.DataFrame) -> str:
        fasta_str = self.get_fasta_seqs(tps_df)
        with open(f"{self.working_directory}/_temp.fasta", "w") as f:
            f.writelines(fasta_str)
        all_id_lines = [line for line in fasta_str.split() if ">" in line]
        logger.info(
            f"Written fasta file. Number of duplicated id lines: {len(all_id_lines) - len(set(all_id_lines))}"
        )

        x = subprocess.check_output(
            f"makeblastdb -in {self.working_directory}/_temp.fasta -dbtype prot -parse_seqids".split(),
            stderr=sys.stdout,
        )
        logger.info(f"makeblastdb output: {x}")

        return f"{self.working_directory}/_temp.fasta"

    def _predict(self, tps_df: pd.DataFrame, db_name: str) -> str:
        test_fasta = self.get_fasta_seqs(tps_df)
        with open(f"{self.working_directory}/_test.fasta", "w") as f:
            f.writelines(test_fasta.replace("'", "").replace('"', ""))
        if os.path.exists(f"{self.working_directory}/results_raw.csv"):
            os.remove(f"{self.working_directory}/results_raw.fasta")
        os.system(
            f"blastp -db {db_name} -evalue {self.config.e_threshold} -query {self.working_directory}/_test.fasta -out {self.working_directory}/results_raw.csv -max_target_seqs {self.config.n_neighbours} -outfmt 10 -num_threads {self.config.n_jobs}"
        )
        os.remove(f"{self.working_directory}/_test.fasta")
        return f"{self.working_directory}/results_raw.csv"

    def fit_core(self, train_df: pd.DataFrame, class_name: str = None):
        train_df.drop_duplicates(
            subset=[self.config.id_col_name, self.config.target_col_name], inplace=True
        )
        if (self.db_path is None or
                self.train_df[[self.config.id_col_name, self.config.target_col_name]] != train_df[
                    [self.config.id_col_name, self.config.target_col_name]]):
            self.train_df = train_df.copy()
            self.db_path = self._train(train_df.drop_duplicates(subset=[self.config.id_col_name]))

    def predict_proba(self, val_df: pd.DataFrame) -> np.ndarray:
        assert val_df[self.config.id_col_name].nunique() == len(
            val_df
        ), "Expected input to predict_proba without duplicated ids"
        all_predicted_batches = []
        for batch_i in tqdm(range(len(val_df) // self.config.pred_batch_size + 1),
                            desc='Predicting with BLASTp-matching..'):
            val_df_batch = val_df.iloc[
                           batch_i * self.config.pred_batch_size: (batch_i + 1) * self.config.pred_batch_size]
            output_path = self._predict(val_df_batch, self.db_path)
            blast_results_df = pd.read_csv(
                output_path, names=[f"{self.config.id_col_name}_blasted", "Matched ID"] + list(range(10))
            )
            blasted_merged_with_train_df = blast_results_df.merge(
                self.train_df[[self.config.id_col_name, self.config.target_col_name]],
                left_on="Matched ID",
                right_on=self.config.id_col_name,
                copy=False,
            )
            label_and_nn_counts = (blasted_merged_with_train_df
                                   .groupby(f"{self.config.id_col_name}_blasted")[
                                       [self.config.target_col_name, "Matched ID"]]
                                   .agg(
                {self.config.target_col_name: lambda x: [Counter(x)], "Matched ID": lambda x: [len(set(x))]})
                                   .reset_index()
                                   )
            label_and_nn_counts['prediction_dict'] = (
                    label_and_nn_counts[self.config.target_col_name] + label_and_nn_counts['Matched ID']).map(
                lambda x: {class_name: class_count / x[1] for class_name, class_count in x[0].items()})

            label_and_nn_counts = label_and_nn_counts.merge(
                val_df_batch, left_on=f"{self.config.id_col_name}_blasted",
                right_on=self.config.id_col_name, how="right"
            ).set_index(self.config.id_col_name)
            class_2_probs_series = label_and_nn_counts['prediction_dict']
            val_proba_np_batch = np.zeros((len(val_df_batch), len(self.config.class_names)))
            for class_i, class_name in enumerate(self.config.class_names):
                val_proba_np_batch[:, class_i] = class_2_probs_series.map(
                    lambda x: x[class_name] if isinstance(x, dict) and class_name in x else 0
                )
            print(label_and_nn_counts.columns)
            indices_batch = label_and_nn_counts[self.config.id_col_name].values
            orig_val_2_ord = {value: i for i, value in enumerate(val_df_batch[self.config.id_col_name])}
            order_of_predictions_in_orig_batch = sorted(range(len(indices_batch)),
                                                        key=lambda idx: orig_val_2_ord[indices_batch[idx]])
            all_predicted_batches.append(val_proba_np_batch[order_of_predictions_in_orig_batch])
        val_proba_np = all_predicted_batches[0] if len(all_predicted_batches) == 1 else np.concatenate(
            all_predicted_batches)
        return val_proba_np

    @classmethod
    def config_class(cls) -> Type[BlastConfig]:
        return BlastConfig
