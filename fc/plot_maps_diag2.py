"""diagnostic script to plot map comparison of ENSO metrics

"""

import matplotlib.pyplot as plt
import iris.plot as iplt
import cartopy.feature as cfeature
import cartopy.crs as ccrs

import iris
import os
import logging
from pprint import pformat
import numpy as np

from esmvaltool.diag_scripts.shared import (run_diagnostic, 
                                            save_figure, 
                                            group_metadata)
from esmvalcore.preprocessor import convert_units


# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))

def plotmaps_level2(input_data): #input data is pairs - model and obs
    
    var_units = {'tos': 'degC', 'pr': 'mm/day', 'tauu': '1e-3 N/m2'}
    fig = plt.figure(figsize=(18, 6))
    proj = ccrs.Orthographic(central_longitude=210.0)
    i =121

    for dataset in input_data:  
        # Load the data
        fp, sn, dt = (dataset['filename'], dataset['short_name'], dataset['dataset'])
  
        logger.info(f"dataset: {dt} - {dataset['long_name']}")
    
        cube = iris.load_cube(fp) 
        #convert units for different variables 
        cube = convert_units(cube, units=var_units[sn])
        cbar_label = sn.upper()
        if len(cube.coords('month_number')) == 1:
            cube = sea_cycle_stdev(cube)
            cbar_label = f'{sn.upper()} std'
        
        ax1 = plt.subplot(i,projection=proj)
        ax1.add_feature(cfeature.LAND, facecolor='gray')  # Add land feature with gray color
        ax1.coastlines()
        cf1 = iplt.contourf(cube, cmap='coolwarm') # levels=np.arange(0,5,0.2),

        ax1.set_extent([130, 290, -20, 20], crs=ccrs.PlateCarree())
        ax1.set_title(dt)

        # Add gridlines for latitude and longitude
        gl1 = ax1.gridlines(draw_labels=True, linestyle='--')
        gl1.top_labels = False
        gl1.right_labels = False

        i+=1

    # Add a single colorbar at the bottom
    cax = plt.axes([0.15,0.08,0.7,0.05])
    cbar = fig.colorbar(cf1, cax=cax, orientation='horizontal', extend='both') #, ticks=np.arange(0,6,1)
    cbar.set_label(cbar_label)

    return fig

def sea_cycle_stdev(cube):

    cube.coord('month_number').guess_bounds()
    cube = cube.collapsed('month_number', iris.analysis.STD_DEV)

    return cube

def main(cfg):
    """Compute sea ice area for each input dataset."""
    provenance_record = {
        'caption': "ENSO metrics comparison maps",
        'authors': [
            'chun_felicity',
        ],
        'references': [''],
        'ancestors': list(cfg['input_data'].keys()),
    }
    input_data = cfg['input_data'].values() 
    
    # group by variables
    variable_groups = group_metadata(input_data, 'variable_group', sort='project')
    # for each select obs and iterate others, obs last
    for grp in variable_groups:
        # create pairs
        msg = "{} : {}, {}".format(grp, len(variable_groups[grp]), pformat(variable_groups[grp]))
        logger.info(msg) 
        obs_data = variable_groups[grp][-1]
               
        for metadata in variable_groups[grp]:
            logger.info('iterate though datasets\n %s',pformat(metadata))
            pairs = [obs_data]
            if metadata['project'] == 'CMIP6':
                pairs.append(metadata)
                fig = plotmaps_level2(pairs)
                filename = '_'.join([metadata['dataset'], 
                                     metadata['short_name'], 
                                     metadata['preprocessor']])
                save_figure(filename, provenance_record, cfg, figure=fig, dpi=300)

if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
