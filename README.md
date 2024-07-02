# graph-scaling-laws

The offical implementation of paper "Neural Scaling Laws on Graphs". In this repo, we provided pipelines to test the data and model scaling behaviors of graph deep learning models.

## Install

```
pip install -r "requirement.txt"
```

## Test the Scaling Behaviors

To run the codes, please first create three folders: dataset/, figures/ and ./results under the main folder.

To test the data scaling scaling behaviors, run

```
bash scale_data.sh
```

The user could customize the training set sizes by changing the `data_array` in scale_data.sh.

To test the data scaling scaling behaviors, run

```
bash scale_model.sh
```

The user could customize the training set sizes by changing the `emb_array` in model_data.sh.

The dataset can be defined with the argument `--dataset` and will be downloaded automatically under the /dataset folder. The scaling results will recorded under the ./results folder.

## Visualization

To visualize the scaling results, run

```
python curve_draw.py --filename $name
```

Here `$name` is the target result file name. The command will generate a scaling curve under /figures folder and calculte the value of R-square of the fitting.

## Acknowledgement

We thank [OGB]([snap-stanford/ogb: Benchmark datasets, data loaders, and evaluators for graph machine learning (github.com)](https://github.com/snap-stanford/ogb))  for their codes and datasets shared.
