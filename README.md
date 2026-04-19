# SR-GNN

## Appendable Experiment Quick Start

Run all commands from the repository root.

Formal strict comparison requires:

- the same target cohort / selected targets
- the same requested victim set
- slice-aware run bundles regenerated under the new analysis flow

Do not feed old pre-slice-aware bundles such as `results/runs/diginetica__...__summary_*` into the new strict comparison flow. You do not need to delete them, but you should regenerate fresh slice-aware bundles from current raw run groups. If the raw source is still a legacy `eval_*` run root, migrate it or rerun it first under the current architecture.

### 1. Produce comparable raw run groups

```powershell
python attack/pipeline/runs/run_clean.py --config attack/configs/diginetica_clean_ratio1_formal.yaml
python attack/pipeline/runs/run_dp_sbr_baseline.py --config attack/configs/diginetica_attack_dpsbr_ratio1_formal.yaml
python attack/pipeline/runs/run_random_nonzero.py --config attack/configs/diginetica_attack_random_nonzero_when_possible_ratio1_formal.yaml
python attack/pipeline/runs/run_prefix_nonzero_when_possible.py --config attack/configs/diginetica_attack_prefix_nonzero_when_possible_ratio1_formal.yaml
python attack/pipeline/runs/run_position_opt_mvp.py --config attack/configs/diginetica_attack_position_optimization_reward.yaml
```

### 2. Generate fresh slice-aware run bundles

```powershell
python analysis/pipeline/long_csv_generator.py --summary (Get-ChildItem outputs/runs/diginetica/clean_run_no_attack_ratio1_formal/run_group_*/summary_current.json).FullName --output-name diginetica_clean_ratio1_formal_popular3x3 --slice-policy largest_complete_prefix --victim srgnn --victim miasrec --victim tron --target-count 3
python analysis/pipeline/long_csv_generator.py --summary (Get-ChildItem outputs/runs/diginetica/attack_dpsbr_ratio1_formal/run_group_*/summary_current.json).FullName --output-name diginetica_dpsbr_ratio1_formal_popular3x3 --slice-policy largest_complete_prefix --victim srgnn --victim miasrec --victim tron --target-count 3
python analysis/pipeline/long_csv_generator.py --summary (Get-ChildItem outputs/runs/diginetica/attack_random_nonzero_when_possible_ratio1_formal/run_group_*/summary_current.json).FullName --output-name diginetica_random_nonzero_ratio1_formal_popular3x3 --slice-policy largest_complete_prefix --victim srgnn --victim miasrec --victim tron --target-count 3
python analysis/pipeline/long_csv_generator.py --summary (Get-ChildItem outputs/runs/diginetica/attack_prefix_nonzero_when_possible_ratio1_formal/run_group_*/summary_current.json).FullName --output-name diginetica_prefix_nonzero_ratio1_formal_popular3x3 --slice-policy largest_complete_prefix --victim srgnn --victim miasrec --victim tron --target-count 3
python analysis/pipeline/long_csv_generator.py --summary (Get-ChildItem outputs/runs/diginetica/attack_position_optimization_reward_mvp_ratio1/run_group_*/summary_current.json).FullName --output-name diginetica_position_opt_ratio1_formal_popular3x3 --slice-policy largest_complete_prefix --victim srgnn --victim miasrec --victim tron --target-count 3
```

### 3. Build the strict comparison bundle

```powershell
python analysis/pipeline/compare_runs.py --config analysis/configs/comparisons/diginetica_ratio1_formal_popular3x3.yaml
```

### 4. Build views and render PNGs

```powershell
python analysis/pipeline/view_table_builder.py --config analysis/configs/views/diginetica_ratio1_formal_popular3x3.yaml
python analysis/pipeline/report_table_renderer.py --bundle-parent-dir results/comparisons/diginetica_ratio1_formal_popular3x3/attack_vs_victim_metrics_by_target --config analysis/configs/render/diginetica_ratio1_formal_popular3x3.yaml
```

## Appendable Experiment Docs

This repository now also contains an appendable experiment container workflow for targeted SBR poisoning experiments.

Operator-facing docs:

- [docs/appendable_experiment_architecture.md](docs/appendable_experiment_architecture.md)
- [docs/operator_workflow_guide.md](docs/operator_workflow_guide.md)
- [docs/migration_tool_usage.md](docs/migration_tool_usage.md)
- [analysis/README.md](analysis/README.md)

The original SR-GNN upstream model notes remain below for base-model reference.

## Paper data and code

This is the code for the AAAI 2019 Paper: [Session-based Recommendation with Graph Neural Networks](https://arxiv.org/abs/1811.00855). We have implemented our methods in both **Tensorflow** and **Pytorch**.

Here are two datasets we used in our paper. After downloaded the datasets, you can put them in the folder `datasets/`:

- YOOCHOOSE: <http://2015.recsyschallenge.com/challenge.html> or <https://www.kaggle.com/chadgostopp/recsys-challenge-2015>

- DIGINETICA: <http://cikm2016.cs.iupui.edu/cikm-cup> or <https://competitions.codalab.org/competitions/11161>

There is a small dataset `sample` included in the folder `datasets/`, which can be used to test the correctness of the code.

We have also written a [blog](https://sxkdz.github.io/research/SR-GNN) explaining the paper.

## Usage

You need to run the file  `datasets/preprocess.py` first to preprocess the data.

For example: `cd datasets; python preprocess.py --dataset=sample`

```bash
usage: preprocess.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  dataset name: diginetica/yoochoose/sample
```

Then you can run the file `pytorch_code/main.py` or `tensorflow_code/main.py` to train the model.

For example: `cd pytorch_code; python main.py --dataset=sample`

You can add the suffix `--nonhybrid` to use the global preference of a session graph to recommend instead of the hybrid preference.

You can also change other parameters according to the usage:

```bash
usage: main.py [-h] [--dataset DATASET] [--batchSize BATCHSIZE]
               [--hiddenSize HIDDENSIZE] [--epoch EPOCH] [--lr LR]
               [--lr_dc LR_DC] [--lr_dc_step LR_DC_STEP] [--l2 L2]
               [--step STEP] [--patience PATIENCE] [--nonhybrid]
               [--validation] [--valid_portion VALID_PORTION]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset name:
                        diginetica/yoochoose1_4/yoochoose1_64/sample
  --batchSize BATCHSIZE
                        input batch size
  --hiddenSize HIDDENSIZE
                        hidden state size
  --epoch EPOCH         the number of epochs to train for
  --lr LR               learning rate
  --lr_dc LR_DC         learning rate decay rate
  --lr_dc_step LR_DC_STEP
                        the number of epochs after which the learning rate
                        decay
  --l2 L2               l2 penalty
  --step STEP           gnn propogation steps
  --patience PATIENCE   the number of epoch to wait before early stop
  --nonhybrid           only use the global preference to predict
  --validation          validation
  --valid_portion VALID_PORTION
                        split the portion of training set as validation set
```

## Requirements

- Python 3
- PyTorch 0.4.0 or Tensorflow 1.9.0

## Other Implementation for Reference
There are other implementation available for reference:
- Implementation based on PaddlePaddle by Baidu [[Link]](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/gnn)
- Implementation based on PyTorch Geometric [[Link]](https://github.com/RuihongQiu/SR-GNN_PyTorch-Geometric)
- Another implementation based on Tensorflow [[Link]](https://github.com/jimanvlad/SR-GNN)
- Yet another implementation based on Tensorflow [[Link]](https://github.com/loserChen/TensorFlow-In-Practice/tree/master/SRGNN)

## Citation

Please cite our paper if you use the code:

```
@inproceedings{Wu:2019ke,
title = {{Session-based Recommendation with Graph Neural Networks}},
author = {Wu, Shu and Tang, Yuyuan and Zhu, Yanqiao and Wang, Liang and Xie, Xing and Tan, Tieniu},
year = 2019,
booktitle = {Proceedings of the Twenty-Third AAAI Conference on Artificial Intelligence},
location = {Honolulu, HI, USA},
month = jul,
volume = 33,
number = 1,
series = {AAAI '19},
pages = {346--353},
url = {https://aaai.org/ojs/index.php/AAAI/article/view/3804},
doi = {10.1609/aaai.v33i01.3301346},
editor = {Pascal Van Hentenryck and Zhi-Hua Zhou},
}
```

