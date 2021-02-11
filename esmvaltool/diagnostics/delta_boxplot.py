# Calculate anomalies, and then plot our model groups in a boxplot

from esmvaltool.diag_scripts.shared import (
    run_diagnostic,
    group_metadata,
    select_metadata,
)
from esmvalcore.preprocessor import extract_region

import iris

# this needs to be shifted to the util version on upgrade to iris v3.
from iris.experimental.equalise_cubes import equalise_attributes

import os
import logging

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


def get_anomalies(ds_list, base_clim_start, fut_clim_start):
    base_yrs = get_decades(base_clim_start)
    fut_yrs = get_decades(fut_clim_start)

    # construct baseline
    base_files = [
        select_metadata(ds_list, start_year=base_yrs[0])[0]["filename"],
        select_metadata(ds_list, start_year=base_yrs[1])[0]["filename"],
        select_metadata(ds_list, start_year=base_yrs[2])[0]["filename"],
    ]
    # compute the mean
    base_mean = combine_decades(base_files)

    # get future
    fut_files = [
        select_metadata(ds_list, start_year=fut_yrs[0])[0]["filename"],
        select_metadata(ds_list, start_year=fut_yrs[1])[0]["filename"],
        select_metadata(ds_list, start_year=fut_yrs[2])[0]["filename"],
    ]
    # compute mean
    fut_mean = combine_decades(fut_files)

    anomaly = fut_mean - base_mean

    return anomaly


def main(cfg):
    # The config object is a dict of all the metadata from the pre-processor
    logger.info(cfg)

    # 1961-1990
    base_start = 1961
    # 2071-2100
    fut_start = 2071

    domain = PICHELLI_DOMAIN

    # first group datasets by project..
    # this creates a dict of datasets keyed by project (CMIP5, CMIP6 etc.)
    projects = group_metadata(cfg["input_data"].values(), "project")
    # how to uniquely define a dataset varies by project, for CMIP it's simple, just dataset...
    # for CORDEX, combo of dataset and driver (and possibly also domain if we start adding those)
    # also gets more complex if we start adding in different ensembles..

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
    # lets turn them into spatial means too..

    season = "JJA"

    # need to work on dictionary recursively since it may be nested
    def process_dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                process_dict(v)
            else:
                return mean_region_and_season(v, domain, season)

    # create dictionary just with project keys
    proj_means = dict.fromkeys(projections.keys())
    for p in proj_means.keys():
        models = projections[p].keys()
        proj_means[p] = process_dict(projections[p])

    print("all done")


if __name__ == "__main__":
    with run_diagnostic() as cfg:
        main(cfg)
