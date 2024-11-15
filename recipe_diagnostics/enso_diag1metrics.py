"""diagnostic script to plot ENSO metrics

"""

import matplotlib.pyplot as plt
import iris.quickplot as qplt

import iris
import os
import logging
from pprint import pformat
import numpy as np
import scipy
import sacpy as scp
import xarray as xr

from esmvaltool.diag_scripts.shared import (run_diagnostic, 
                                            save_figure, 
                                            get_diagnostic_filename,
                                            group_metadata,
                                            select_metadata,
                                            sorted_metadata)
from esmvalcore.preprocessor import extract_month, zonal_statistics, meridional_statistics


# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))

def plot_level1(input_data, metricval, y_label, title, dtls): #input data is 2 - model and obs

    figure = plt.figure(figsize=(10, 6), dpi=300)

    # model first 
    plt.plot(*input_data[0], label=dtls[0]) # or plt.plot(array?

    plt.plot(*input_data[1], label=f'ref: {dtls[1]}', color='black')

    plt.title(title) # metric name
    plt.legend()
    plt.grid(linestyle='--')
    plt.ylabel(y_label) #param?
    # metric type: RMSE or %
    plt.text(0.5, 0.95, f"RMSE: {metricval:.2f}", fontsize=12, ha='center', transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    arrayline = True
    if arrayline: # if array, not scatter
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_longitude))

    logger.info(f"{dtls[0]} : metric:{metricval}")

    return figure

def lin_regress(cube_ssta, cube_nino34): #1d 
    slope_ls = []
    for lon_slice in cube_ssta.slices(['time']): 
        res = scipy.stats.linregress(cube_nino34.data, lon_slice.data)
        slope_ls.append(res[0])

    return cube_ssta.coord('longitude').points, slope_ls

def pattern_09(input_pair, dt_ls): #['tos_patdiv1':, 'tos_pat2':] input_pair

    # obs first
    mod_ssta = input_pair[1]['tos_pat2']
    mod_nino34 = input_pair[1]['tos_patdiv1']
    reg_mod = lin_regress(mod_ssta, mod_nino34)
    reg_obs = lin_regress(input_pair[0]['tos_pat2'], input_pair[0]['tos_patdiv1'])

    rmse = np.sqrt(np.mean((np.array(reg_obs[1]) - np.array(reg_mod[1])) ** 2))
    #save data? reg_mod as cube?
    
    # plot functions? title
    fig = plot_level1([reg_mod,reg_obs], rmse, 'reg(ENSO SSTA, SSTA)', 'ENSO pattern', dt_ls)

    return rmse, fig

def sst_regressed(n34_cube):
    # params cubes, 
    n34_dec = extract_month(n34_cube, 12)
    n34_dec = xr.DataArray.from_iris(n34_dec)
    n34 = xr.DataArray.from_iris(n34_cube)
    leadlagyr = 3 #rolling window cut off, not include first year
    n34_dec_ct = n34_dec[leadlagyr:-leadlagyr]
    event_years = n34_dec_ct.time.dt.year #
    # Ensure that the selected years are not the last or second last year in the n34 dataset
    years_of_interest_array = []
    
    # Fill the array with the years of interest for each event year 
    for i, year in enumerate(event_years):# 2:-3 for dec        
        years_of_interest_array.append([year.values - 2, year.values - 1, year.values, year.values + 1, year.values + 2, year.values + 3])
    
    n34_selected = []
    for i in range(len(years_of_interest_array)): #creates sst_time_series
        # Select the data for the current year and append it to n34_selected #n34 is not dec month only
        n34_selected.append(n34.sel(time=n34['time.year'].isin(years_of_interest_array[i])))

    # 1) linear regression of sst_time_series on sst_enso
    slope = scp.LinReg(n34_dec_ct.values, n34_selected).slope
    return slope

def lifecycle_10(inputs, dtls):
    # inputs pairs of model and obs
    ## metric computation - rmse of slopes
    model = sst_regressed(inputs[0]) #n34_cube
    obs = sst_regressed(inputs[1])
    rmse = np.sqrt(np.mean((obs - model) ** 2))
    months = np.arange(1, 73) - 36 #build tuples?
    #save data? slope as cube?

    # plot function #need xticks
    fig = plot_level1([ (months,model),(months,obs)], rmse, 'Degree C / C','ENSO lifecycle', dtls)
    return rmse, fig


def format_longitude(x, pos):
    if x > 180:
        return f'{int(360 - x)}°W'
    elif x == 180:
        return f'{int(x)}°'
    else:
        return f'{int(x)}°E'

def get_provenance_record(caption, ancestor_files):
    """Create a provenance record describing the diagnostic data and plot."""

    record = {
        'caption': caption,
        'statistics': ['anomaly'],
        'domains': ['eq'],
        'plot_types': ['line'],
        'authors': [
            'chun_felicity',
            # 'sullivan_arnold',
        ],
        'references': [
            'access-nri',
        ],
        'ancestors': ancestor_files,
    }
    return record

def main(cfg):
    """Run ENSO metrics."""

    input_data = cfg['input_data'].values() 

    # iterate through each metric and get variable group, select_metadata, map to function call
    metrics = {'09pattern': ['tos_patdiv1', 'tos_pat2'],
                '10lifecycle':['tos_lifdur1'], #'tos_lifdurdiv2' for lev2
                '11amplitude': ['tos_amp'],
                '12seasonality': ['tos_seas_asym'],
                '13asymmetry':['tos_seas_asym'],
                '14duration':['tos_lifdur1','tos_lifdurdiv2'],
                '15diversity':['tos_patdiv1','tos_lifdurdiv2']}
    
    # select twice with project to get obs, iterate through model selection
    for metric, var_preproc in metrics.items(): #if empty or try
        
        obs = []
        models = []
        for var_prep in var_preproc: #enumerate 1 or 2 length?
            obs.append(select_metadata(input_data, variable_group=var_prep, project='OBS'))
            selection = select_metadata(input_data, variable_group=var_prep, project='CMIP6')
            models.append(sorted_metadata(selection, sort='dataset'))

        # log
        msg = "{} : observation datasets {}, models {}".format(metric, len(obs), pformat(models))
        logger.info(msg)
        
        # list dt_files
        dt_files = []
        for ds in models: #and obs?
            dt_files.append(ds['filename'])
        prov_record = get_provenance_record(f'ENSO metrics {metric}', dt_files)
        # obs datasets for each model
        obs_datasets = {dataset['variable_group']: iris.load_cube(dataset['filename']) for dataset in obs}
        
        # group models by dataset
        model_ds = group_metadata(models, 'dataset', sort='project')        
        # dataset name
        
        for dataset in model_ds:
            logger.info(f"{metric}, preprocessed cubes:{len(dataset)}, dataset:{dataset}")
            
            model_datasets = {attributes['variable_group']: iris.load_cube(attributes['filename']) 
                              for attributes in model_ds[dataset]}
            input_pair = [obs_datasets, model_datasets]

            # process function for each metric - obs first.. if, else
            if metric == '09pattern':
                # sort datasetfiles    
                value, fig = pattern_09(input_pair, [dataset, obs[0]['dataset']])
            elif metric =='10lifecycle':
                value, fig = lifecycle_10(input_pair, [dataset, obs[0]['dataset']])

            # save metric for each pair, check not none
            if value:
                metricfile = get_diagnostic_filename('matrix', cfg, extension='csv')
                with open(metricfile, 'a+') as f:
                    f.write(f"{dataset},{metric},{value}\n")

                save_figure(f'{dataset}_{metric}', prov_record, cfg, figure=fig, dpi=300)
            #clear value,fig
            value = None


if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
