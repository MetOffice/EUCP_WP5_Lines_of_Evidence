import iris
import iris.plot as iplt
from scipy.stats.mstats import linregress
import cartopy
import numpy as np
import pandas as pd
from iris.util import equalise_attributes

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from glob import glob
import os

RX_RECIPE = 'recipe_GCM_and_RCM_Romania_Rx1Day_20211112_113355'
MEAN_RECIPE = 'recipe_GCM_and_RCM_Romania_20211103_153305'
BASE_PATH = '/home/h02/tcrocker/code/EUCP_WP5_Lines_of_Evidence/esmvaltool/esmvaltool_output/'
SEASON = 'JJA'
RX_PATH = f'{BASE_PATH}/{RX_RECIPE}/work/maps/main/'
MEAN_PATH = f'{BASE_PATH}/{MEAN_RECIPE}/work/maps/main/{SEASON}/'
PLOT_PATH = f"{BASE_PATH}{RX_RECIPE}/plots/maps/main/"

SEASONS = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}


def plot_formatting(ax):
    ax.set_extent([18, 32, 41.5, 50.5])
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle=":")


def guide_lines(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    min = np.min([xlim, ylim])
    max = np.max([xlim, ylim])
    ax.plot([min, max], [min, max], linestyle=":", color="k", alpha = 0.75)

    ax.axhline(0, linestyle=":", color="k", alpha=0.75)
    ax.axvline(0, linestyle=":", color="k", alpha=0.75)


def plot_3panel(x, y, model):
    plt.figure(figsize=[19.2,  4.85])
    # for plotting a colorbar
    vmn = np.ma.min([x.data, y.data])
    vmx = np.ma.max([x.data, y.data])
    if vmx < 0:
        vmx = 0.001
    if vmn > 0:
        vmn = -0.001
    divnorm = mcolors.TwoSlopeNorm(vmin=vmn, vcenter=0, vmax=vmx)

    # x
    plt.subplot(1, 3, 1)
    pmesh = iplt.pcolormesh(x, norm=divnorm, cmap='RdBu')
    plot_formatting(plt.gca())
    plt.title('Mean pr')

    # y
    plt.subplot(1, 3, 2)
    iplt.pcolormesh(y, norm=divnorm, cmap='RdBu')
    plot_formatting(plt.gca())
    plt.title('Rx1Day')

    plt.subplot(1, 3, 3)
    plt.scatter(x.data, y.data, s=1)
    guide_lines(plt.gca())
    plt.xlabel('Mean pr')
    plt.ylabel('Rx1day')
    # compute simple linear regression
    fit = linregress(x.data.flatten(), y.data.flatten())
    plt.title(f"R: {fit.rvalue:.2f}. Slope: {fit.slope:.2f}")

    plt.suptitle(f"{model} %")

    # add colorbar for map plots
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    cbax = fig.add_axes([0.125, 0.1, 0.5, 0.075])
    fig.colorbar(pmesh, cax=cbax, orientation="horizontal")
    
    # save plot
    plt.savefig(f"{PLOT_PATH}{model}.png")
    plt.close()

def plot_means(cubes, type):
    # compute and plot means
    equalise_attributes(cubes["mean_pr"])
    equalise_attributes(cubes["rx1day"])
    # also clear up attributes on season number coord in rx1day
    for c in cubes["rx1day"]:
        c.coord("season_number").attributes = None
    means = {}
    means["mean"] = cubes["mean_pr"].merge_cube().collapsed('multi_model', iris.analysis.MEAN)
    means["rx1day"] = cubes["rx1day"].merge_cube().collapsed('multi_model', iris.analysis.MEAN)
    plot_3panel(means["mean"], means["rx1day"], f"{type} mean change")


def plot_regression(data_dict, type):
    fit = linregress(data_dict["mean"], data_dict["rx1day"])
    plt.plot(
        data_dict["mean"],
        fit.intercept + fit.slope*data_dict["mean"],
        linestyle="--",
        label=f'{type} least squares fit'
    )

    return fit


# load data
# loop through the files in the work folder
rx_cubes = {}
mean_cubes = {}

season_con = iris.Constraint(season_number=SEASONS[SEASON])
for f in glob(f"{RX_PATH}/*.nc"):
    dataset = os.path.basename(f)
    dataset = os.path.splitext(dataset)[0]

    # load rx1day data
    rx_cubes[dataset] = iris.load_cube(f).extract(season_con)

    # also need to load equivalent mean data
    fname = f"{MEAN_PATH}{dataset}_{SEASON}.nc"
    mean_cubes[dataset] = iris.load_cube(fname)

# now some plotting
cubes_dict = {
    "cmip5": {
        "mean_pr": iris.cube.CubeList(),
        "rx1day": iris.cube.CubeList()
        },
    "cordex": {
        "mean_pr": iris.cube.CubeList(),
        "rx1day": iris.cube.CubeList()
        },
    "cpm": {
        "mean_pr": iris.cube.CubeList(),
        "rx1day": iris.cube.CubeList()
        }
    }

for i, k in enumerate(rx_cubes.keys()):
    # fill dictionary of data for plotting means later
    if k[:7] == "CORDEX_":
        m_type = "cordex"
    elif k[:5] == "CMIP5":
        m_type = "cmip5"
    else:
        m_type = "cpm"
    cubes_dict[m_type]["mean_pr"].append(mean_cubes[k])
    # add multi model coord to rx1day cubes
    mm_coord = iris.coords.AuxCoord(i, long_name="multi_model")
    rx_cubes[k].add_aux_coord(mm_coord)
    cubes_dict[m_type]["rx1day"].append(rx_cubes[k])

    plot_3panel(mean_cubes[k], rx_cubes[k], k)

# compute and plot means
plot_means(cubes_dict["cordex"], "CORDEX")
plot_means(cubes_dict["cpm"], "CPM")
plot_means(cubes_dict["cmip5"], "CMIP5")

# also, calculate means and produce scatter of romania averages of mean and Rx1Day
mean_mean = []
mean_rx1day = []
labels = []

for k in mean_cubes.keys():
    grid_areas = iris.analysis.cartography.area_weights(mean_cubes[k])
    mean_mean.append(
        mean_cubes[k].collapsed(
            ['latitude', 'longitude'], iris.analysis.MEAN, weights=grid_areas
        ).data.item()
    )

    mean_rx1day.append(
        rx_cubes[k].collapsed(
            ['latitude', 'longitude'], iris.analysis.MEAN, weights=grid_areas
        ).data.item()
    )

    labels.append(k)

# put into a dataframe to help with selecting CORDEX values only
dict_for_df = {}
for i, l in enumerate(labels):
    dict_for_df[l] = {"mean": mean_mean[i], "rx1day": mean_rx1day[i]}
mean_df = pd.DataFrame(dict_for_df).T
cordex_rows = [x[:7] == "CORDEX_" for x in mean_df.index]
cmip5_rows = [x[:6] == "CMIP5_" for x in mean_df.index]
cordex_mean_df = mean_df[cordex_rows]
cmip5_mean_df = mean_df[cmip5_rows]
# sort by mean value for plotting purposes later
cordex_mean_df = cordex_mean_df.sort_values('mean')
cmip5_mean_df = cmip5_mean_df.sort_values('mean')

fig, ax = plt.subplots(figsize=(12.8, 9.6))
for k in range(len(labels)):
    if labels[k][:7] == "CORDEX_":
        marker = "o"
        size = 25
    elif labels[k][:6] == "CMIP5_":
        marker = "x"
        size = 25
    else:
        marker = "^"
        size = 100
    ax.scatter(mean_mean[k], mean_rx1day[k], label=labels[k], marker=marker, s=size)

plt.xlabel('Mean pr')
plt.ylabel('Rx1Day')
guide_lines(ax)

# compute regression for CORDEX
cordex_fit = plot_regression(cordex_mean_df, "CORDEX")
# and for CMIP5
cmip5_fit = plot_regression(cmip5_mean_df, "CMIP5")

title_str = f"CORDEX - R: {cordex_fit.rvalue:.2f} (p: {cordex_fit.pvalue:.4f}). Slope: {cordex_fit.slope:.2f}\n"
title_str = title_str + f"CMIP5 - R: {cmip5_fit.rvalue:.2f} (p: {cmip5_fit.pvalue:.4f}). Slope: {cmip5_fit.slope:.2f}"
plt.title(title_str)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

plt.savefig(f"{PLOT_PATH}scatter.png")
plt.close()
