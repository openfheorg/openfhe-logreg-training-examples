# Goal

The goal of this folder is to track the differences in results as we modify the Sigmoid Approx across 
various parameters. We investigate the approximation differences as well as the approximation and how it impacts training

## Approximations

### Data

#### Generating this data

- Use the `cheb_analysis` file in the top-level directory

#### Using the data

The data we analyze over is stored in:

- `raw/data/sigmoidResults_X_Y.txt`

where `X` is the start and end. We approximate over the range `-X` and `X`. `Y` is the degree of the approximation.

The data is ingested from: `chebyshev_approx_analysis.ipynb` and we plot various things such as the error. Notice in the first cell:

```python
conf_list = [
    Config(name="Degree128", start=-64, end=64, degree=128),
    Config(name="Degree119", start=-64, end=64, degree=119),
]
```

and this specifies the range of values, degree and a name (for the plot legend). **Again** this merely ingests the data
from the appropriate files. It does not generate them on its own.

### Analyzing the Approximations

The results are stored in `approx_plots/`. You should just focus on the error plot (second plot) as that is what was generated in the report.

## Training

### Data

##### Generating the data

You generate the data by running the training process from lr_nag.cpp which will output the data into a `nag_interactive_loss.csv` file.
Your job is to then rename it to an appropriate form and add it to the analysis script. Note: you should be training against the `data/X_norm_1024` and `data/y_1024` data.

##### Using the data

After renaming the data to a form like `nag_X_loss.csv`, where `X` is the approximation degree, modify `train_analysis.ipynb` so that it incorporates your data.