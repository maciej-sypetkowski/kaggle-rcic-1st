# 1st Place Solution for [Kaggle Recursion Cellular Image Classification](https://www.kaggle.com/c/recursion-cellular-image-classification/) Challenge

For the description of the solution, refer to [this post](https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110543).

## Environment
This repository contains `Dockerfile` that covers all required dependencies.

By default mixed precision training and inference is used (see `--fp16` flag in `main.py`) which is fully supported in Volta and Turing architectures.

## Training
```bash
python main.py --save <experiment name> <additional arguments>
```
will result in:
* saving checkpoint each epoch into `<experiment name>.<epoch>`
* saving best checkpoint (on local validation) into `<experiment name>`
* saving log from training into `<experiment name>.log`

For additional arguments and default values see `python main.py --help` or `main.py` file.

## Prediction
```bash
python main.py --mode predict --load <path to checkpoint> <additional arguments>
```
will result in:
* saving raw predictions into `<path to checkpoint>.output` or into `<path to checkpoint>.output<pred suffix>` if `--pred-suffix` is specified
* saving log into `<path to checkpoint>.output.log`

Raw predictions are pickled logits for each class and each test and validation image. To convert it into CSV submission or ensemble multiple raw predictions, run:
```bash
./make_submission.py -o <csv output file> <path to one or multiple raw predictions>
```
This script will also ensemble and print score on validation set if it's not empty (see `--cv-number` flag for `main.py`).


## Examples
These examples assume data (in the same format as in Kaggle with extracted files from archives) to be in `../data`.

```bash
python main.py -e 130 --pl-epoch 90 --lr cosine,1.5e-4,90,6e-5,150,0 --pl-size-func 0.6*x+0.4 --cv-number -1 --seed 0 --save /results/dn161_0
python main.py --mode predict --cv-number -1 --tta 8 --load /results/dn161_0.129
./make_submission.csv /results/dn161_0.129.output -o /results/submission_0.csv
```
results in submission with 0.99658 private score and 0.98826 public score.

```bash
python main.py -e 130 --pl-epoch 90 --lr cosine,1.5e-4,90,6e-5,150,0 --pl-size-func 0.6*x+0.4 --cv-number -1 --seed 1 --save results/dn161_1
python main.py --mode predict --cv-number -1 --tta 8 --load /results/dn161_1.129
./make_submission.csv /results/dn161_1.129.output -o /results/submission_1.csv
```
results in submission with 0.99623 private score and 0.98871 public score.

Ensembling both of them (`./make_submission.csv /results/dn161_0.129.output /results/dn161_1.129.output -o /results/submission.csv`) results in 0.99784 private score and 0.99187 public score.
