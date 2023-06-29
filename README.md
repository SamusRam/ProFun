# ProFun
Library of models for **Pro**tein **Fun**ction prediction

# Installation
The majority of dependencies will be installed automatically via the command
`pip install git+https://github.com/SamusRam/ProFun.git`.

If you want to use the BLAST-based model, please run these commands:
```
wget https://ftp.ncbi.nlm.nih.gov/blast/executables/LATEST/ncbi-blast-2.14.0+-x64-linux.tar.gz
tar zxvpf ncbi-blast-2.14.0+-x64-linux.tar.gz
# add ncbi-blast-2.14.0+/bin to PATH
```
If you want to use profile Hidden Markov models, please run the following commands:
```
conda install -c bioconda mafft -y
conda install -c bioconda hmmer -y
```

If you want to use Foldseek-based model, please run the following command:
```
conda install -c conda-forge -c bioconda foldseek -y
```

# Basic usage
## BLAST
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

## Profile Hidden Markov model
```
from profun.models import ProfileHMM, HmmConfig
from profun.utils.project_info import ExperimentInfo

experiment_info = ExperimentInfo(validation_schema='public_lb', 
                                 model_type='profileHMM', model_version='24additional')

config = HmmConfig(experiment_info=experiment_info, 
                     id_col_name='EntryID', 
                     target_col_name='term', 
                     seq_col_name='Seq', 
                     class_names=list(additional_classes), 
                     optimize_hyperparams=False, 
                     n_calls_hyperparams_opt=None,
                     hyperparam_dimensions=None,
                     per_class_optimization=None,
                     class_weights=None,
                     search_e_threshold=0.000001,
                     zero_conf_level=0.00001,
                     group_column_name='taxonomyID',
                     n_jobs=56,
                     pred_batch_size=20000)

hmm_model = ProfileHMM(config)
hmm_model.fit(train_df_long)
test_pred_df = hmm_model.predict_proba(test_seqs_df.drop_duplicates('EntryID'), return_long_df=True)
```

## Foldseek-based classifier
Please see [this notebook](https://www.kaggle.com/code/samusram/leveraging-foldseek) as a usage demo.

```
from profun.models import FoldseekMatching, FoldseekConfig
from profun.utils.project_info import ExperimentInfo

experiment_info = ExperimentInfo(validation_schema='public_lb', 
                                 model_type='foldseek', model_version='5nn')

config = FoldseekConfig(experiment_info=experiment_info, 
                        id_col_name='EntryID', 
                        target_col_name='term',
                        seq_col_name='Seq',
                        class_names=list(train_df_long_sample['term'].unique()), 
                        optimize_hyperparams=False, 
                        n_calls_hyperparams_opt=None,
                        hyperparam_dimensions=None,
                        per_class_optimization=None,
                        class_weights=None,
                        n_neighbours=5,
                        e_threshold=0.0001,
                        n_jobs=56,
                        pred_batch_size=10,
                        local_pdb_storage_path=None #then it stores structures into the working dir
                    )

model = FoldseekMatching(config)
model.fit(train_df_long)
test_pred_df = model.predict_proba(test_seqs_df.drop_duplicates('EntryID'), return_long_df=True)
```

