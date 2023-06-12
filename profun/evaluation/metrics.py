"""This module implements metrics computation"""

import numpy as np  # type: ignore
import pandas as pd
from sklearn.metrics import average_precision_score, recall_score  # type: ignore
import logging

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logger = logging.getLogger(__name__)


def eval_model_mean_average_precision(
    model, val_df: pd.DataFrame, selected_class_name: str = None
):
    id_col_name = model.config.id_col_name
    val_df_unique = val_df.drop_duplicates(subset=id_col_name)
    gt_type = val_df[[id_col_name, model.config.target_col_name]].drop_duplicates()
    gt_type = (
        gt_type.groupby(id_col_name)[model.config.target_col_name]
        .agg(set)
        .reset_index()
    )
    gt_type.columns = [id_col_name, "target_set"]
    val_df_unique = val_df_unique.merge(gt_type, on=id_col_name)
    y_pred = model.predict_proba(val_df_unique)

    average_precisions = []
    try:
        class_weights = model.config.class_weights
        if class_weights is None:
            class_weights = 1.0
    except AttributeError:
        class_weights = 1.0
    weights_sum = 0
    for class_i, class_name in enumerate(model.config.class_names):
        if selected_class_name is None or class_name == selected_class_name:
            y_true = val_df_unique["target_set"].map(lambda x: class_name in x)
            ap = average_precision_score(y_true, y_pred[:, class_i])
            if isinstance(class_weights, float):
                class_weight = class_weights
            elif isinstance(class_weights, dict):
                class_weight = class_weights[class_name]
            else:
                raise NotImplementedError(f"Unexpected type {type(class_weights)} for the class_weight parameter.")
            ap_weighted = class_weight*ap
            average_precisions.append(ap_weighted)
            weights_sum += class_weight
            logger.info(f"{class_name}: ap = {ap:.3f}, weighted ap = {ap_weighted: .3f}")

    return 1 - np.sum(average_precisions)/weights_sum