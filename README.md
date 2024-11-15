# ACCESS-ENSO-Recipes (El Niño-Southern Oscillation)

## Overview

The **ACCESS-ENSO-Recipes** package is a collection of Jupyter notebooks designed to reproduce the metrics described in the **CLIVAR 2020 ENSO Metrics Package**. These metrics are based on the methodology outlined by Planton et al. (2021) ([DOI:10.1175/BAMS-D-19-0337.1](https://doi.org/10.1175/BAMS-D-19-0337.1)) and implemented following the instructions provided by Yann Planton.

This package integrates with **ESMValTool**, utilising its preprocessors to streamline analysis. Users can explore the ENSO metrics interactively in Jupyter notebooks or execute them as an **ESMValTool recipe** for batch processing.

### Key Features:
- Reproduction of ENSO metrics from the CLIVAR 2020 ENSO Metrics Package.
- Simplified diagnostics for integration into ESMValTool.
- Support for interactive exploration in Jupyter notebooks or execution via ESMValTool recipes.
- Configured for use on **NCI Gadi** with the **ACCESS-NRI conda environment**.

---

## What is ESMValTool?

The **Earth System Model Evaluation Tool (ESMValTool)** is a community-developed software package for the evaluation of Earth System Models (ESMs).

For more information, visit the [official ESMValTool website](https://esmvaltool.org/).

---

## Requirements

To use the **ACCESS-ENSO-Recipes**, ensure the following:

### Environment:
- Access to **NCI Gadi**.
- The **ACCESS-NRI conda environment** pre-configured with the esmvaltool-workflow.

---

## How to Use on ARE (Australian Research Environment)

## 1. Open ARE on Gadi
Go to the [Australian Research Environment](https://are-auth.nci.org.au/) website and login with your **NCI username and password**. If you don't have an NCI account, you can sign up for one at the [MyNCI website](https://my.nci.org.au).

<p align="center"><img src="docs/assets_ARE/login.png" alt="drawing" width="60%"/></p>

## 2. Start JupyterLab App
Click on `JupyterLab` under *Featured Apps* to configure a new JupyterLab instance. This option is also available under the *All Apps* section at the bottom of the page and the *Interactive Apps* dropdown located in the top menu.

<p align="center"><img src="docs/assets_ARE/jupyter_select.png" alt="drawing" width="60%"/></p>

## 3. Configure JupyterLab session
You will now be presented with the main JupyterLab instance configuration form. Please complete **only** the fields below - leave all other fields blank or to their default values.

- *3.1* **Walltime**: The number of hours the JupyterLab instance will run. For the hackathon, please insert a walltime of `4` hours.

<p align="center"><img src="docs/assets_ARE/walltime.png" alt="drawing" width="60%"/></p>

- *3.2* **Compute Size**: Select `Medium (4 cpus, 18G mem)` from the dropdown menu.

<p align="center"><img src="docs/assets_ARE/compute.png" alt="drawing" width="60%"/></p>

- *3.3* **Project**: Please enter `nf33`. This will allocate SU usage to the workshop project.

<p align="center"><img src="docs/assets_ARE/project.png" alt="drawing" width="60%"/></p>

- *3.4* **Storage**: This is the list of project data storage locations required to complete the hackathon exercises. In ARE, storage locations need to be explicitly defined to access these data from within a JupyterLab instance. Please copy and paste the following string in its entirety into the storage input field:
```
scratch/nf33+gdata/nf33+gdata/xp65+gdata/fs38+gdata/oi10+gdata/al33+gdata/rr3+gdata/rt52+gdata/zz93+gdata/ct11+gdata/zv30
```

<p align="center"><img src="docs/assets_ARE/project.png" alt="drawing" width="60%"/></p>

- *3.5* Click `Advanced options ...`
  * Optional: You can check the box here to receive an email notification when your JupyterLab instance starts, but as we are running a relatively small instance, it will likely spin up quickly so this probably isn't necessary.</p>

- *3.6* **Module directories**: To load the required environment modules, please copy and paste the following. This is equivalent to using `module use` on the command line.
```
/g/data/xp65/public/modules
```

<p align="center"><img src="docs/assets_ARE/module_directories.png" alt="drawing" width="60%"/></p>

- *3.7* **Modules** To load the ESMValTool-workflow environment, please copy and paste the following. This is equivalent to using `module load` on the command line.
```
esmvaltool
```

<p align="center"><img src="docs/assets_ARE/modules.png" alt="drawing" width="60%"/></p>

- *3.7* Click `Launch` to start your JupyterLab instance.


<p align="center"><img src="docs/assets_ARE/launch.png" alt="drawing" width="60%"/></p>

## 4. Launch JupyterLab session
Once you have clicked `Launch` the browser will redirect to the 'interactive sessions' page where you will see your JupyterLab instance details and current status which will look something like this:

<p align="center"><img src="docs/assets_ARE/queue.png" alt="drawing" width="60%"/></p>

Once the JupyterLab instance has started (this usually takes around 30 seconds), this status window should update and look something like the following, reporting that the instance has started and the time remaining. More detailed information on the instance can be accessed by clicking the `Session ID` link.

<p align="center"><img src="docs/assets_ARE/running.png" alt="drawing" width="60%"/></p>

Click `Open JupyterLab`. This opens the instance in a new browser window where you can navigate to the location of the files.

---

## Feedback and Support

This package is maintained by the **ACCESS-NRI Model Evaluation and Diagnostics Team**. For issues, suggestions, or assistance, please contact [ACCESS-NRI support](mailto:support@access-nri.org).

We welcome contributions! Please follow the [contribution guidelines](CONTRIBUTING.md) to submit enhancements or bug fixes.

---

## References

- CLIVAR 2020 ENSO Metrics Package: [CLIVAR Website](https://www.clivar.org/news/clivar-2020-enso-metrics-package)
- Planton et al. (2021): [DOI:10.1175/BAMS-D-19-0337.1](https://doi.org/10.1175/BAMS-D-19-0337.1)
- ESMValTool: [ESMValTool Official Website](https://esmvaltool.org/)
