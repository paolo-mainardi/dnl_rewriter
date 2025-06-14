# DNL Rewriter

This repository contains data and code related to our paper "Fine-Tuning vs Prompting Techniques for Gender-Fair Rewriting of Machine Translations", submitted to the 6th Workshop on Gender Bias in NLP, taking place at ACL 2025. 

## Data
* The `data` folder contains the manual DNL reformulations that we used to create our task-specific training dataset. 
* The `dataset` folder contains our training dataset, split into training and test splits. 
* The `adapted_neogate` folder contains the version of Neo-GATE we adapted to our DNL grammar (as defined in `schwa_template.json`); `neomorphemes.txt` is needed to run the evaluation. 
Please refer to [the original dataset](https://huggingface.co/datasets/FBK-MT/Neo-GATE) for details on the adaptation and evaluation process. 

## Code
The `scripts` we share are used for the following purposes:
* `create_my_dataset.py` creates a copy of our training dataset, using files from the `data` folder and external datasets, and saves it to the `dataset` folder
* `finetune_rewriter.py` fine-tunes a model on our task using our training dataset
* `generate_predictions.py` collects raw outputs on the evaluation dataset from a fine-tuned model
* `prompt_rewriter.py` prompts a model to perform our task and collects its raw outputs on the evaluation dataset; prompts are created following our prompt templates and using our training dataset when in a few-shot setting

## Results
The `experiments` folder contains the following for each of the experiments we carried out:
* post-processed (cleaned) model predictions that can be evaluated directly
* detailed reports generated by Neo-GATE as part of the evaluation

## Cite
If you wish to use and/or refer to any material from this repository, please use the following citation:

Paolo Mainardi, Federico Garcea, Alberto Barrón-Cedeño. 2025. 'Fine-Tuning vs Prompting Techniques for Gender-Fair Rewriting of Machine Translations'. In _Proceedings of the 6th Workshop on Gender Bias in Natural Language Processing (GeBNLP)_. Vienna, Austria. Association for Computational Linguistics. 