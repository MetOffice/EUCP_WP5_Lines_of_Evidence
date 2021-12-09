# script for processing the files from KNMI into CMOR like ESMValTool compatible format.

import iris
from netCDF4 import Dataset

short_names = {
    'precipitation_flux': 'pr',
    'air_temperature': 'tas'
}


def get_model_from_path(path):
    # get model info from path
    if "/ECEARTH/" in path:
        model = "EC-EARTH"
        driving_model = None
    elif "/RACMO/" in path:
        model = "KNMI-RACMO23E"
        driving_model = "KNMI-EC-EARTH"

    return model, driving_model


def get_dates_ensemble(cube):
    # establish start and end dates
    # and ensemble number of driving model
    start = cube.coord('time').cell(0).point.strftime('%Y%m%d')
    end = cube.coord('time').cell(-1).point.strftime('%Y%m%d')

    # get ensemble member
    # 1996-2005 = r14
    # 2041-2050 = r13

    if int(start[:4]) < 2010:
        ensemble = 'r14i1p1'
        experiment = 'historical'
    else:
        ensemble = 'r13i1p1'
        experiment = 'rcp85'

    return start, end, ensemble, experiment


# IN_FILES = '/project/applied/Data/cordex-cpm/from_hylke/ECEARTH/ECEARTH-kh01+13/Daily_data/*.nc'
IN_FILES = '/project/applied/Data/cordex-cpm/from_hylke/RACMO/uCY33-v520-fECEARTH-Member13/Daily_data/*.nc'
OUT_FOLDER = '/project/applied/Data/cordex-cpm/from_hylke/processed/'

# standard name of variable to process
#var = 'precipitation_flux'
var = 'air_temperature'

# load data
cube_l = iris.load(IN_FILES, var)
iris.util.equalise_attributes(cube_l)

# remove unneeded attributes on time coord
for c in cube_l:
    c.coord('time').attributes = None

# concatenate to a single cube
cube = cube_l.concatenate_cube()

# remove single dimension
cube = iris.util.squeeze(cube)

# save with proper filename
# e.g. for GCM: tas_Amon_HadGEM2-ES_historical_r1i1p1_198412-200511.nc
# <short_name>_<mip>_<model>_<experiment>_<ensemble>_<startdate>-<enddate>.nc
# for RCM:
# <short_name>_<domain>_<driving-model>_<experiment>_<ensemble>_<model>_<version>_<frequency>_<start-date>-<end-date>.nc

# set metadata
short_name = short_names[cube.standard_name]
freq = 'day'
model, driving_model = get_model_from_path(IN_FILES)
start, end, ensemble, experiment = get_dates_ensemble(cube)

cube.var_name = short_name

if short_name == 'tas':
    cube.coord('height').points = 2.0

# model specific stuff
if model == "EC-EARTH":
    # GCM
    fname = f'{short_name}_{freq}_{model}_{experiment}_{ensemble}_{start}-{end}.nc'
else:
    # RCM
    # correct coord names
    cube.coord('longitude', dim_coords=True).standard_name = 'grid_longitude'
    cube.coord('latitude', dim_coords=True).standard_name = 'grid_latitude'
    fname = f'{short_name}_EUR-11_{driving_model}_{experiment}_{ensemble}_{model}_v1_{freq}_{start}-{end}.nc'
    vname = cube.var_name

# save data
iris.save(cube, f'{OUT_FOLDER}/{fname}')

if model == 'KNMI-RACMO23E':
    # need to sort out rotated pole attributes
    # load one of the original files as a netCDF dataset
    orig_nc = Dataset('/project/applied/Data/cordex-cpm/from_hylke/RACMO/uCY33-v520-fECEARTH-Member13/Daily_data/precip.KNMI-2040.CXFPS12.uCY33-v520-fECEARTH-Member13.3H.daymean.nc', 'r')
    nc_file = Dataset(f'{OUT_FOLDER}/{fname}', 'r+')

    # create variable
    orig_rp = orig_nc.variables['rotated_pole']
    rp = nc_file.createVariable('rotated_pole', orig_rp.dtype)
    # copy attributes
    for att in orig_rp.ncattrs():
        rp.setncattr(att, orig_rp.getncattr(att))

    # set grid mapping attribute
    nc_file.variables[vname].setncattr('grid_mapping', 'rotated_pole')

    orig_nc.close()
    nc_file.close()
