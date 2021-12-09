# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# workbook to pull coord detail from the 1hr file, and add to the processed monthly files

# %%
# imports
import netCDF4 as nc
import argparse
import os
import sys

# %%
# files to work with
parser = argparse.ArgumentParser()

parser.add_argument('hourly_file')
parser.add_argument('monthly_file')
args = parser.parse_args()

# hrly_file = '/project/applied/Data/cordex-fpsc/from_escience/CNRS-CNRM/ALP-3/tas_ALP-3_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-AROME41t1_fpsconv-x2yn2-v1_1hr_199601010030-199612312330.nc'
# monthly_file = '/project/applied/Data/cordex-fpsc/from_escience/CNRS-CNRM/ALP-3/rcp85/pr_ALP-3_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_CNRM-AROME41t1_fpsconv-x2yn2-v1_mon_204101-210001.nc'
# var = 'pr'
hrly_file = args.hourly_file
monthly_file = args.monthly_file
var = os.path.basename(monthly_file).split('_')[0]

# %%
# open hrly file
hrly_nc = nc.Dataset(hrly_file, 'r')


# %%
# grab coords and grid mapping info
x_coord = hrly_nc.variables['x']
y_coord = hrly_nc.variables['y']
lc = hrly_nc.variables['Lambert_Conformal']


# %%
# load the monthly file
# create a copy to work with
#copy_name = "/scratch/tcrocker/test_file.nc"
#shutil.copy(monthly_file, copy_name)
print(f"Fixing {monthly_file}")
mthly_nc = nc.Dataset(monthly_file, 'r+')

if 'x' in mthly_nc.variables.keys():
    print("Doesn't look like file needs fixing")
    mthly_nc.close()
    hrly_nc.close()
    sys.exit()

# %%
# Add x and y dimensions to the file
mthly_nc.renameDimension('dim1', 'y')
mthly_nc.renameDimension('dim2', 'x')


# %%
# this is not sufficient, maybe I should use the createVariable method
#mthly_nc.variables['x'] = x_coord
#mthly_nc.variables['y'] = y_coord

# create variables
new_y = mthly_nc.createVariable('y', y_coord.datatype, y_coord.dimensions)
new_x = mthly_nc.createVariable('x', x_coord.datatype, x_coord.dimensions)

# copy metadata
for propname in y_coord.ncattrs():
    if propname == '_FillValue':
        continue
    prop = getattr(y_coord, propname)
    setattr(new_y, propname, prop)

for propname in x_coord.ncattrs():
    if propname == '_FillValue':
        continue
    prop = getattr(x_coord, propname)
    setattr(new_x, propname, prop)

# copy data
new_y[:] = y_coord[:]
new_x[:] = x_coord[:]


# %%
# create fixed Lambert conformal variable
new_lc = mthly_nc.createVariable('Lambert_Conformal', 'i', ())
# set properties
for propname in lc.ncattrs():
    prop = getattr(lc, propname)
    setattr(new_lc, propname, prop)

# add grid mapping attribute to file
mthly_nc.variables[var].grid_mapping = 'Lambert_Conformal'


# %%
print("FIXED")


# %%
# tidy up
hrly_nc.close()
mthly_nc.close()


# %%



