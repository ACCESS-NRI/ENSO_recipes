"""diagnostic script to plot ENSO metrics matrix

"""

import matplotlib.pyplot as plt

import os
import logging

import pandas as pd
from esmvaltool.diag_scripts.shared import (run_diagnostic, 
                                            save_figure)

# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))

def plot_matrix(input_data, cfg): #input data is 2 - model and obs

    return 



def main(cfg):
    """Compute sea ice area for each input dataset."""
    provenance_record = {
        'caption': "ENSO metrics",
        'authors': [
            'chun_felicity',
        ],
        'references': [''],
        'ancestors': list(cfg['input_data'].keys()),
    }
    # input_data = cfg['input_data'].values() 
    metrics = cfg['diag_metrics']
    diag_path = '/'.join(cfg['work_dir'].split('/')[:-2])
    diag_path = '/'.join(diag_path, metrics, 'matrix.csv')
    logger.info(diag_path)
    
    metric_df = pd.read_csv(diag_path, header=None)
    transformls = []
    for mod in metric_df[0].unique(): #iterate model, translate metrics
        df = metric_df.loc[metric_df[0]==mod,:]
        transformls.append(df[[1,2]].set_index(1).T.rename(index={2:mod}))

    matrixdf = pd.concat(transformls)
    figure = plt.figure(dpi=300)
    plt.imshow(matrixdf, cmap='coolwarm')
    plt.colorbar()
    plt.xticks(range(len(matrixdf.columns)),matrixdf.columns)
    plt.yticks(range(len(matrixdf.index)),matrixdf.index)

    save_figure('plot_matrix', provenance_record, cfg, figure=figure)

if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
