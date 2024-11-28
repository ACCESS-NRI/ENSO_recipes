"""diagnostic script to plot ENSO metrics

"""

import matplotlib.pyplot as plt
import iris.quickplot as qplt

import iris
import os
import logging
from pprint import pformat
import numpy as np
from scipy.stats import skew, linregress
import sacpy as scp
import xarray as xr

from esmvaltool.diag_scripts.shared import (run_diagnostic, 
                                            save_figure, 
                                            get_diagnostic_filename,
                                            group_metadata,
                                            select_metadata,
                                            )
from esmvalcore.preprocessor import (extract_month, extract_season,
                                     mask_above_threshold, mask_below_threshold,
                                     climate_statistics, meridional_statistics)


# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))

def plot_level1(input_data, metricval, y_label, title, dtls): #input data is 2 - model and obs

    figure = plt.figure(figsize=(10, 6), dpi=300)

    if title in ['ENSO pattern','ENSO lifecycle']:
        # model first 
        plt.plot(*input_data[0], label=dtls[0])
        plt.plot(*input_data[1], label=f'ref: {dtls[1]}', color='black')
        plt.text(0.5, 0.95, f"RMSE: {metricval:.2f}", fontsize=12, ha='center', transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    else:
        plt.scatter(range(len(input_data)), input_data, c=['black','blue'], marker='D')
        # obs first
        plt.xlim(-0.5,2)#range(-1,3,1)) #['model','obs']
        plt.xticks([])
        plt.text(0.75,0.95, f'* {dtls[0]}', color='blue', transform=plt.gca().transAxes)
        plt.text(0.75,0.9, f'* ref: {dtls[1]}', color='black', transform=plt.gca().transAxes)
        plt.text(0.75, 0.8, f"metric(%): {metricval:.2f}", fontsize=12, transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.title(title) # metric name
    plt.legend()
    plt.grid(linestyle='--')
    plt.ylabel(y_label) #param
    
    if title == 'ENSO pattern': # if array, not scatter
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_longitude))
    elif title == 'ENSO lifecycle':
        plt.axhline(y=0, color='black', linewidth=2)
        xticks = np.arange(1, 73, 6) - 36  # Adjust for lead/lag months
        xtick_labels = ['Jan', 'Jul'] * (len(xticks) // 2)
        plt.xticks(xticks, xtick_labels)
        plt.yticks(np.arange(-2,2.5, step=1))

    logger.info(f"{dtls[0]} : metric:{metricval}")

    return figure

def lin_regress(cube_ssta, cube_nino34): #1d pattern
    slope_ls = []
    for lon_slice in cube_ssta.slices(['time']): 
        res = linregress(cube_nino34.data, lon_slice.data)
        slope_ls.append(res[0])

    return cube_ssta.coord('longitude').points, slope_ls

def sst_regressed(n34_cube): #for lifecycle
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

def compute_enso_metrics(input_pair, dt_ls, var_group, metric): #['tos_patdiv1':, 'tos_pat2':] 

    # input_pair: obs first
    if metric == '09pattern':
        model_ssta = input_pair[1][var_group[1]]
        model_nino34 = input_pair[1][var_group[0]]
        reg_mod = lin_regress(model_ssta, model_nino34)
        reg_obs = lin_regress(input_pair[0][var_group[1]], input_pair[0][var_group[0]])

        val = np.sqrt(np.mean((np.array(reg_obs[1]) - np.array(reg_mod[1])) ** 2))
        #save data? reg_mod as cube?
        # plot functions? ylabel, title, data labels
        fig = plot_level1([reg_mod,reg_obs], val, 'reg(ENSO SSTA, SSTA)', 'ENSO pattern', dt_ls)
    
    elif metric =='10lifecycle':
        model = sst_regressed(input_pair[1][var_group[0]]) #n34_cube
        obs = sst_regressed(input_pair[0][var_group[0]])
        val = np.sqrt(np.mean((obs - model) ** 2))
        months = np.arange(1, 73) - 36 #build tuples?
        # plot function #need xticks, labels as dict/ls
        fig = plot_level1([ (months,model),(months,obs)], val, 'Degree C / C', 'ENSO lifecycle', dt_ls)
    
    elif metric =='11amplitude':
        data_values = [input_pair[1][var_group[0]].data.item(),input_pair[0][var_group[0]].data.item()]
        val = compute(data_values[1], data_values[0])
        #plt.scatter(range(len(metric)), metric, c=['blue','black'], marker='D')
        fig = plot_level1(data_values, val, 'SSTA std (°C)', 'ENSO amplitude', dt_ls)

    elif metric =='12seasonality':
        data_values = []
        for ds in input_pair: #obs 0, mod 1
            preproc = {}
            for seas in ['NDJ','MAM']:
                cube = extract_season(ds[var_group[0]], seas)
                cube = climate_statistics(cube, operator="std_dev", period="full")
                preproc[seas] = cube.data

            data_values.append(preproc['NDJ']/preproc['MAM'])

        val = compute(data_values[1], data_values[0])
        fig = plot_level1(data_values, val, 'SSTA std (NDJ/MAM)(°C/°C)','ENSO seasonality', dt_ls)

    elif metric =='13asymmetry':
        model_skew = skew(input_pair[1][var_group[0]].data, axis=0)
        obs_skew = skew(input_pair[0][var_group[0]].data, axis=0)
        data_values = [model_skew, obs_skew]

        val = compute(data_values[1], data_values[0])
        fig = plot_level1(data_values, val, 'SSTA skewness(°C)','ENSO skewness', dt_ls)

    elif metric =='14duration':
        model = sst_regressed(input_pair[1][var_group[0]])
        obs = sst_regressed(input_pair[0][var_group[0]])

        months = np.arange(1, 73) - 36
        counts = []
        # Calculate the number of months where slope > 0.25 in the range -20 to 20
        within_range = (months >= -30) & (months <= 30)
        for slopes in [model, obs]:
            slope_above_025 = slopes[within_range] > 0.25
            counts.append(np.sum(slope_above_025))
        val = compute(counts[1], counts[0])

        fig = plot_level1(counts, val, 'Duration (reg > 0.25) (months)','ENSO duration', dt_ls)
    elif metric =='15diversity':
        data_values = []
        for ds in input_pair: #obs first
            events = enso_events(ds[var_group[0]])
            results_lon = diversity(ds[var_group[1]], events)
            results_lon['enso'] = results_lon['nino'] + results_lon['nina']
            logger.info(f"{dt_ls}, enso IQR: {iqr(results_lon['enso'])}") 
            data_values.append(iqr(results_lon['enso']))

        val = compute(data_values[1], data_values[0]) 
        fig = plot_level1(data_values, val, 'IQR of min/max SSTA(°long)','ENSO diversity', dt_ls)

    return val, fig 

def mask_to_years(events):    # build time with mask
    maskedTime = np.ma.masked_array(events.coord('time').points, mask=events.data.mask)
    # return years
    return [events.coord('time').units.num2date(time).year for time in maskedTime.compressed()]
def enso_events(cube):
    a_events = mask_to_years(mask_above_threshold(cube.copy(), -0.75))
    o_events = mask_to_years(mask_below_threshold(cube.copy(), 0.75))
    return {'nina':a_events, 'nino':o_events}

def diversity(ssta_cube, events_dict): #2 masks/events list
    # each enso year, max/min SSTA, get longitude
    res_lon = {}
    for enso, events in events_dict.items():
        year_enso = iris.Constraint(time=lambda cell: cell.point.year in events)
        cube = ssta_cube.extract(year_enso)
        if enso == 'nina':
            cube = cube * -1
        #iterate through cube, each time get max/min value and return lon
        loc_ls = []
        for yr_slice in cube.slices(['longitude']):
            indx = np.argmax(yr_slice.data) # if nina multiply by -1 or min
            loc_ls.append(cube.coord('longitude').points[indx])

        res_lon[enso] = loc_ls
    return res_lon # return data to plot 

def iqr(data):
    q3, q1 = np.percentile(data, [75 ,25])
    iqr = q3 - q1
    return iqr

def format_longitude(x, pos):
    if x > 180:
        return f'{int(360 - x)}°W'
    elif x == 180:
        return f'{int(x)}°'
    else:
        return f'{int(x)}°E'
def compute(obs, mod):
    return abs((mod-obs)/obs)*100

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
                '14duration':['tos_lifdur1'], #,'tos_lifdurdiv2'
                '15diversity':['tos_patdiv1','tos_lifdurdiv2']}
    
    # select twice with project to get obs, iterate through model selection
    for metric, var_preproc in metrics.items(): #if empty or try
        logger.info(f"{metric},{var_preproc}")
        obs, models = [], []
        for var_prep in var_preproc: #enumerate 1 or 2 length? if 2 append,
            obs += select_metadata(input_data, variable_group=var_prep, project='OBS')
            models += select_metadata(input_data, variable_group=var_prep, project='CMIP6')

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
            logger.info(f"{metric}, preprocessed cubes:{len(model_ds)}, dataset:{dataset}")
            
            model_datasets = {attributes['variable_group']: iris.load_cube(attributes['filename']) 
                              for attributes in model_ds[dataset]}
            input_pair = [obs_datasets, model_datasets]
            logger.info(pformat(model_datasets))
            # process function for each metric - obs first.. if, else
            ### make one function, with the switches - same params
            value, fig = compute_enso_metrics(input_pair, [dataset, obs[0]['dataset']], var_preproc, metric)

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
