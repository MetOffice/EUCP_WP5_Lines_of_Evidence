import iris
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import iris.quickplot as qplt
import cartopy


# global variables to control everything
RECIPE_RUN = 'recipe_GCM_and_RCM_Romania_20211103_153305'
BASE_PATH = f'/home/h02/tcrocker/code/EUCP_WP5_Lines_of_Evidence/esmvaltool/esmvaltool_output/{RECIPE_RUN}/work/maps/main/'
SEASON = 'JJA'

CPM_MODELS = [
    'cordex-cpm_ICTP-RegCM4-7 MOHC-HadGEM2-ES',
    'cordex-cpm_SMHI-HCLIM38-AROME ICHEC-EC-EARTH'
]

NICULITA_MODELS = [
    'CORDEX_RCA4 MPI-M-MPI-ESM-LR',
    'CORDEX_RCA4 MOHC-HadGEM2-ES',
    'CORDEX_RCA4 ICHEC-EC-EARTH',
    'CORDEX_RCA4 CNRM-CERFACS-CNRM-CM5',
    'CORDEX_REMO2009 MPI-M-MPI-ESM-LR',
    'CORDEX_RACMO22E MOHC-HadGEM2-ES',
    'CORDEX_RACMO22E ICHEC-EC-EARTH', 
    'CORDEX_HIRHAM5 ICHEC-EC-EARTH',
]

MODELS = NICULITA_MODELS
MODE = 'map'


def plot_hists(model_cubes):
    plt.figure()
    i = 0
    for c in model_cubes:
        if c.data.size < 10000:
            bins = 25
        else:
            bins = 100

        p = plt.hist(c.data.flatten(), density=True, histtype='step', bins=bins, label=MODELS[i])
        color = p[2][0].get_edgecolor()
        plt.axvline(np.ma.mean(c.data.flatten()), color=color, linestyle='--', alpha = 0.4)
        plt.axvline(np.ma.median(c.data.flatten()), color=color, linestyle='-', alpha = 0.4)
        
        i = i+1

    plt.legend()
    plt.show()


# load models into cubelist
model_cubes = iris.cube.CubeList()
for m in MODELS:
    cube = iris.load_cube(f'{BASE_PATH}{SEASON}/{m}_anom_{SEASON}.nc')
    model_cubes.append(cube)

# plot models
# histograms
if MODE == 'hist':
    plot_hists(model_cubes)
else:
    # maps
    # extent is +/- 2 of this:
    # start_longitude: 20
    # end_longitude: 30
    # start_latitude: 43.5
    # end_latitude: 48.5

    # compute multi-model mean of provided cubes
    all_models = model_cubes.merge_cube()
    mm_mean = all_models.collapsed('multi_model', iris.analysis.MEAN)

    # then create map plot
    qplt.pcolormesh(mm_mean, vmin=-50, vmax=50, cmap='RdBu')
    ax = plt.gca()
    ax.set_extent([18, 32, 41.5, 50.5])
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle=":")
    plt.show()
