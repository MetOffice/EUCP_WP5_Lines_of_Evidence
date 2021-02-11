# Calculate anomalies, and then plot our model groups in a boxplot

from esmvaltool.diag_scripts.shared import (
    run_diagnostic,
    group_metadata,
    select_metadata,
)

import iris

# TODO this needs to be shifted to the util version on upgrade to iris v3.
from iris.experimental.equalise_cubes import equalise_attributes

import os
import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(os.path.basename(__file__))


def get_decades(clim_start_yr):
    """Return tuple of end years for decades used in a 30 yr climatology period.

    Args:
        clim_start_yr (int): Start year of climatology period, e.g. 1961 for the 1961-1990 period

    Returns:
        tuple: start years of decades in 30 year climatology period
    """
    clim_start_yrs = (clim_start_yr, clim_start_yr + 10, clim_start_yr + 20)

    return clim_start_yrs


def combine_decades(files):
    """Combine the cubes contained in the provided files into a single cube that is the mean of them

    Args:
        files (list of str): filenames of netCDF data

    Returns:
        iris cube: mean of data loaded from files.
    """
    # load the files and mean them..
    cubes = iris.cube.CubeList()
    i = 1
    for f in files:
        c = iris.load_cube(f)

        dummy_coord = iris.coords.AuxCoord(i, long_name="dummy_coord", units=1)
        c.add_aux_coord(dummy_coord)
        i = i + 1
        cubes.append(c)

    # remove any extraneous attributes
    equalise_attributes(cubes)
    cube_mean = cubes.merge_cube().collapsed("dummy_coord", iris.analysis.MEAN)

    return cube_mean


def process_projections_dict(proj_dict, season, out_list):
    # recursive function to pull out data from dictionary
    for k, v in proj_dict.items():
        if isinstance(v, dict):
            process_projections_dict(v, season, out_list)
        else:
            # extract required season
            season_con = iris.Constraint(season_number=season)
            data = v.extract(season_con)
            # this should be a scalar cube..
            out_list.append(data.data.item())


def get_anomalies(ds_list, base_clim_start, fut_clim_start):
    # construct baseline
    base_file = select_metadata(ds_list, start_year=base_clim_start)[0]["filename"]
    base_cube = iris.load_cube(base_file)

    # get future
    fut_file = select_metadata(ds_list, start_year=fut_clim_start)[0]["filename"]
    fut_cube = iris.load_cube(fut_file)

    anomaly = fut_cube - base_cube

    return anomaly


def main(cfg):
    # The config object is a dict of all the metadata from the pre-processor
    logger.info(cfg)

    # these could come from recipe in future
    base_start = 1961
    fut_start = 2070

    # first group datasets by project..
    # this creates a dict of datasets keyed by project (CMIP5, CMIP6 etc.)
    projects = group_metadata(cfg["input_data"].values(), "project")
    # how to uniquely define a dataset varies by project, for CMIP it's simple, just dataset...
    # for CORDEX, combo of dataset and driver (and possibly also domain if we start adding those)
    # also gets more complex if we start adding in different ensembles..

    logger.info("Loading data")
    # empty dict to store results
    projections = dict.fromkeys(projects.keys())
    # loop over projects
    for proj in projects:
        # we now have a list of all the data entries..
        # for CMIPs we can just group metadata again by dataset then work with that..
        models = group_metadata(projects[proj], "dataset")

        # empty dict for results
        projections[proj] = dict.fromkeys(models.keys())
        # loop over the models
        for m in models:
            if proj == "CORDEX":
                # then we need to go one deeper in the dictionary to deal with driving models
                drivers = group_metadata(models[m], "driver")
                projections[proj][m] = dict.fromkeys(drivers.keys())
                for d in drivers:
                    projections[proj][m][d] = get_anomalies(
                        drivers[d], base_start, fut_start
                    )
            else:
                projections[proj][m] = get_anomalies(models[m], base_start, fut_start)

    # we now have all the projections in the projections dictionary
    # now lets plot them
    # first we need to process the dictionary, and move the data into a list of vectors
    logger.info("Processing for plotting")
    proj_plotting = dict.fromkeys(projections.keys())
    for p in projections:
        vals = []
        process_projections_dict(projections[p], 1, vals)
        proj_plotting[p] = vals

    logger.info("Plotting")
    # eventually plotting code etc. will go in a seperate module I think.
    plot_keys, plot_values = zip(*proj_plotting.items())
    plt.boxplot(plot_values)
    plt.gca().set_xticklabels(plot_keys)
    plt.savefig(f"{cfg['plot_dir']}/boxplot.png")
    print("all done")


if __name__ == "__main__":
    with run_diagnostic() as cfg:
        main(cfg)
