## ACCESS-ENSO-recipes

### Overview

Currently splitting the metrics into the groups: background climatology, basic ENSO characteristics, teleconnections, physical processes.
- Descriptions: https://github.com/CLIVAR-PRP/ENSO_metrics/wiki
- With separate recipe for more diagnostic levels.
- example `work_dir` with *csv* for metrics:
```
└── work
    ├── diag_collect
    │   └── matrix_collect
    └── diagnostic_metrics
        └── plot_script
            └── matrix.csv
```

### Recipes and diagnostics


Recipes 

- climatology_metrics.yml
- climatology_diaglevel2.yml
- enso_metrics.yml

Diagnostics are stored in *diagnostic_scripts/*

- **climatology_diagnostic1.py**: compute metrics for background climatology
- **climatology_diagnosticlevel2.py**: plot dive down level 2 for background climatology with maps
- **matrix.py**: reads metrics in work_dir from csv file written out in climatology_diagnostic1, use for other groups of metrics
- **enso_diag1metrics.py**: metrics for basic ENSO characteristics


### User settings in recipe


Script: **matrix.py**

   *Required settings for script*
   **diag_metrics**: diagnostic name and script name in *yml* of the diagnostic that computes all the metrics 
   so it can find the *csv* in the `work_dir`
   - eg. diagnostic_metrics/plot_script



### Variables

* pr, tos, tauu (monthly)
* areacello (fx)


### Observations and reformat scripts


* HadISST
* ERA-Interim
* GPCP-SG


### References


### Example plots


<p align="center"><img src="figures/plot_matrix.png" alt="portrait plot" width="60%"/></p>

<p align="center"><img src="figures/plot_matrix_enso.png" alt="portrait plot" width="60%"/></p>

