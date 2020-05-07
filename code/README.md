# LIBERT - Lexically-Informed BERT

This repository contains the code associated with the following paper:


Specializing Unsupervised Pretraining Models for Word-Level Semantic Similarity

(Anne Lauscher, Ivan Vulić, Edoardo Maria Ponti, Anna Korhonen, Goran Glavaš)

https://arxiv.org/pdf/1909.02339.pdf

## Repository Description
### Model
The model is only different from the original BERT code in the way it shares the embeddings. For this, we've used a conditioned variable scope. This is implemented in 
- ```conditioned_scope.py```
- ```modeling.py``` line 179 with ```cond_scope(is_shared=shared_embeddings): ...```

### Pretraining Procedure
#### Data Generation
We trained both BERT and LIBERT on a dump of the English Wikipedia. For this, we've used 
- ```poc_pretraining_bert.sh```
- ```poc_create_pretraining_data.sh```

The code needed to preprocess the lexico-semantic constraints is given in ```preprocess_wn.py```.

#### Actual Pretraining
LIBERT is pretrained from scratch via two classes of objectives (1) BERT's "standard" objectives, MLM and NSP, and (2) Lexical Relation Classification. We therefore provide the pretraining script in two variants accordingly:
- ```run_pretraining_bert.py``` implements the standard objectives only (for comparison along the training process with BERT)
- ```run_pretraining_libert.py``` implements the standard objectives plus the LRC

This is demonstrated in 
- ```poc_pretraining_bert.sh```
- ```pos_pretraining_libert.sh```


### Downstream evaluation
#### GLUE
For running simple classification and regression tasks, for instance, for evaluation on GLUE, we refer to the following scripts:

- ```run_classifier_libert.py```
- ```run_regression_libert.py```

We adapted the original BERT tensorflow scripts such that the variable scope matches our models, so that the model is loaded correctly. This behavior is controlled via an additional parameter ```original_model```. Additionally, we added support for grid searching hyperparameter configurations via these scripts.

How to call the scripts is demonstrated in 
- ```poc_finetuning_bert.sh```
- ```pos_finetuning_libert.sh```

For the predictions, we refer to
- ```poc_predictions_bert.sh```
- ```pos_predictions_libert.sh```

#### Lexical Simplification
For the lexical simplification evaluation, we've used the BERT-LS code: https://github.com/qiang2100/BERT-LS . For this, we had to port the models to pytorch, which was done via the transformers libray (just make sure to adapt the code in a way that it loads LIBERT's variables correcly.

## Credit
The code is based on the original BERT Tensorflow code at: https://github.com/google-research/bert

## Other
Please cite the paper as follows:
```
@misc{lauscher2019specializing,
    title={Specializing Unsupervised Pretraining Models for Word-Level Semantic Similarity},
    author={Anne Lauscher and Ivan Vulić and Edoardo Maria Ponti and Anna Korhonen and Goran Glavaš},
    year={2019},
    eprint={1909.02339},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}```
