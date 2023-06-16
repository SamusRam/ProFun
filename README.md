# ProFun
Library of models for **Pro**tein **Fun**ction prediction

# Installation
```
wget https://ftp.ncbi.nlm.nih.gov/blast/executables/LATEST/ncbi-blast-2.14.0+-x64-linux.tar.gz
tar zxvpf ncbi-blast-2.14.0+-x64-linux.tar.gz
# add ncbi-blast-2.14.0+/bin to PATH
pip install git+https://github.com/SamusRam/ProFun.git
```

# Basic usage
Please see [this notebook](https://www.kaggle.com/code/samusram/blastp-sprof-go) as a usage demo.

```
from profun.models import BlastMatching, BlastConfig
from profun.utils.project_info import ExperimentInfo

experiment_info = ExperimentInfo(validation_schema='public_lb', 
                                 model_type='blast', model_version='1nn')

config = BlastConfig(experiment_info=experiment_info, 
                     id_col_name='EntryID', 
                     target_col_name='term', 
                     seq_col_name='Seq', 
                     class_names=list(train_df_long['term'].unique()), 
                     optimize_hyperparams=False, 
                     n_calls_hyperparams_opt=None,
                    hyperparam_dimensions=None,
                    per_class_optimization=None,
                    class_weights=None,
                    n_neighbours=5,
                    e_threshold=0.0001,
                     n_jobs=100,
                     pred_batch_size=10
                    )

blast_model = BlastMatching(config)

# fit
blast_model.fit(train_df_long)

# predict
test_pred_df = blast_model.predict_proba(test_seqs_df.sample(42).drop_duplicates('EntryID'), return_long_df=True)
```
