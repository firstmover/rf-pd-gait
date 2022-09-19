## Monitoring Gait at Home with Radio Waves in Parkinson's Disease: a Marker of Severity, Progression, and Medication Response

This repo includes the code for the Science Translational Medicine paper [Liu et al. 2022]().

![Python 3.9](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)
![R](https://img.shields.io/badge/R-3.6.1-blue)
[![GitHub Repo Stars](https://img.shields.io/github/stars/firstmover/rf-pd-gait?label=Stars&logo=github&color=red)](https://github.com/firstmover/rf-pd-gait)
[![Project](https://img.shields.io/badge/project-rf--pd--gait-lightgrey)](https://rf-pd-gait.csail.mit.edu)
[![Tweet](https://img.shields.io/twitter/url?url=https%3A%2F%2Fshields.io)](https://twitter.com/intent/tweet?text=Wow%2C+a+cool+project+from+MIT%3A+https%3A%2F%2Frf-pd-gait.csail.mit.edu%2F)

<br>
<a href="https://rf-pd-gait.csail.mit.edu/"><img src="https://rf-pd-gait.csail.mit.edu/static/figs/overview.svg"></a>
<hr/>

### Environment and dependency

The code is developed and tested on `Ubuntu 18.04`, `Python 3.9`, `R 3.6.1`. We recommend using [conda](https://github.com/conda/conda) to handle environments and dependencies.

##### Python

Create a new conda environment and install dependencies:

```shell
conda create -y -n pd_gait_py39 python=3.9
conda activate pd_gait_py39
pip install numpy scipy matplotlib seaborn pandas streamlit pingouin plotly tqdm openpyxl
```

Add this repo to `PYTHONPATH`.

```shell
export PYTHONPATH={$PATH_TO_THIS_REPO}:$PYTHONPATH
```

For example, if you downloaded this repo to `$HOME/code/rf-pd-gait/`, run this:

```shell
export PYTHONPATH=$HOME/code/rf-pd-gait/:$PYTHONPATH
```

##### R

Install R and dependencies with conda:

```shell
conda install -y -c r r
conda install -y -c conda-forge r-nloptr r-rcppeigen
```

In R CLI, run following command to install packages:

```R
install.packages(c('ggeffects', 'lmerTest', 'argparse', 'crayon'))
```

### Data

Download and save data to `./data`. Once they are ready, the folder should look like this:

```bash
data
├── covariate_adjusted_lm                 // request and download: https://rf-pd-gait.csail.mit.edu/
│  └── data.csv
├── data.xlsx                             // download from supplementary material
├── longitudinal_data_linear_regression   // request and download: https://rf-pd-gait.csail.mit.edu/
│  ├── hc.csv
│  └── pd.csv
└── ppmi                                  // request and download: https://www.ppmi-info.org/
   ├── Consensus_Committee_Analytic_Datasets_28OCT21.xlsx
   ├── MDS-UPDRS_Part_I.csv
   ├── MDS-UPDRS_Part_I_Patient_Questionnaire.csv
   ├── MDS-UPDRS_Part_IV__Motor_Complications.csv
   ├── MDS_UPDRS_Part_II__Patient_Questionnaire.csv
   ├── MDS_UPDRS_Part_III.csv
   └── Participant_Status.csv
```

If you only have access to parts of the data, the code will still work. However, it only produces results that correspond to the available data.

### Steps to produce results

##### Main results (fig.2~6, fig.S2, and figS3, etc.)

This part of the results requires you to have `./data/data.xlsx`.

Initialize a visualization server:

```shell
streamlit run ./scripts/visualization.py
```

Go to the visualization webpage with a browser. The server will run statistical analysis and visualize the outcomes when you click on different pages.

Available pages are: `test_retest_reliability`, `baseline_cross_sectional`, `longitudinal_analysis`, `medication_response_motor_fluctuation`, `covid_lock_down`, and `hospitalization`.

##### Monte Carlo simulation: MDS-UPDRS PPMI

This part of the results requires you to have `./data/ppmi`.

Initialize the same visualization server:

```shell
streamlit run ./scripts/visualization.py
```

Results can be found on the page: `ppmi_longitudinal_analysis`.

##### Confounding variables analysis

This part of the results requires you to have `./data/covariate_adjusted_lm`.

Run R script:

```shell
Rscript ./scripts/cross_sectional_analysis_with_covariates.r
```

Results can be found in the stdout of this command.

##### Linear mixed-effects model for longitudinal gait decline

This part of the results requires you to have `./data/longitudianl_data_linear_regression`.

Run R script:

```shell
Rscript ./scripts/longitudinal_analysis.r
```

Results can be found in the stdout of this command.

### License

This repo is licensed under the MIT License and the copyright belongs to all rf-pd-gait project authors - see the [LICENSE](https://github.com/firstmover/rf-pd-gait/blob/master/LICENSE) file for details.

### Citation

to be added.

### Contact

Email: liuyc@mit.edu
