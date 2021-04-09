# Calculate anomalies, and then plot our model groups in a boxplot

from esmvaltool.diag_scripts.shared import (
    run_diagnostic,
    group_metadata,
    select_metadata,
    extract_variables,
)

import iris

# TODO this needs to be shifted to the util version on upgrade to iris v3.
from iris.experimental.equalise_cubes import equalise_attributes

import os
import logging
import re

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(os.path.basename(__file__))


def process_projections_dict(proj_dict, season):
    # recursive function to pull out data from dictionary
    out_data = {}
    for k, v in proj_dict.items():
        if isinstance(v, dict):
            vals = process_projections_dict(v, season)
            for k1, v1 in vals.items():
                out_data[f"{k} {k1}"] = v1
        else:
            if v is None:
                continue
            # extract required season
            season_con = iris.Constraint(season_number=season)
            data = v.extract(season_con)
            # this should be a scalar cube, add the value to a dictionary
            out_data[k] = data.data.item()
    return out_data


def get_anomalies(ds_list, base_clim_start, fut_clim_start, relative=False):
    # construct baseline
    base_metadata = select_metadata(ds_list, start_year=base_clim_start)
    if base_metadata == []:
        logging.warning(f"Base climatology (start {base_clim_start}) not found")
        return None
    base_file = base_metadata[0]["filename"]
    base_cube = iris.load_cube(base_file)

    # get future
    fut_metadata = select_metadata(ds_list, start_year=fut_clim_start)
    if fut_metadata == []:
        logging.warning(f"Future climatology (start {fut_clim_start}) not found")
        return None
    fut_file = fut_metadata[0]["filename"]
    fut_cube = iris.load_cube(fut_file)

    if relative:
        diff = fut_cube - base_cube
        anomaly = (diff / base_cube) * 100
        anomaly.units = "%"
    else:
        anomaly = fut_cube - base_cube

    return anomaly


def main(cfg):
    # The config object is a dict of all the metadata from the pre-processor

    # get variable processed
    var = list(extract_variables(cfg).keys())
    assert len(var) == 1
    var = var[0]

    if var == "pr":
        rel_change = True
    else:
        rel_change = False

    # these should come from recipe in future
    base_start = 1998
    fut_start = 2040

    # first group datasets by project..
    # this creates a dict of datasets keyed by project (CMIP5, CMIP6 etc.)
    projects = group_metadata(cfg["input_data"].values(), "project")
    # how to uniquely define a dataset varies by project, for CMIP it's simple, just dataset...
    # for CORDEX, combo of dataset and driver (and possibly also domain if we start adding those)
    # also gets more complex if we start adding in different ensembles..

    logger.info("Loading data")
    # empty dict to store results
    projections = dict.fromkeys(projects.keys())
    model_lists = dict.fromkeys(projects.keys())
    cordex_drivers = []
    UKCP_ensembles = []
    # loop over projects
    for proj in projects:
        # we now have a list of all the data entries..
        # for CMIPs we can just group metadata again by dataset then work with that..
        models = group_metadata(projects[proj], "dataset")

        # empty dict for results
        projections[proj] = dict.fromkeys(models.keys())
        model_lists[proj] = []
        # loop over the models
        for m in models:
            if proj[:6].upper() == "CORDEX":
                # then we need to go one deeper in the dictionary to deal with driving models
                drivers = group_metadata(models[m], "driver")
                projections[proj][m] = dict.fromkeys(drivers.keys())
                for d in drivers:
                    logging.info(f"Calculating anomalies for {proj} {m} {d}")
                    anoms = get_anomalies(drivers[d], base_start, fut_start, rel_change)
                    if anoms is None:
                        continue
                    projections[proj][m][d] = anoms
                    model_lists[proj].append(f"{m} {d}")
                    cordex_drivers.append(d)
            elif proj == "UKCP18":
                # go one deeper to deal with individual ensembles
                ensembles = group_metadata(models[m], "ensemble")
                projections[proj][m] = dict.fromkeys(ensembles.keys())
                for ens in ensembles:
                    logging.info(f"Calculating anomalies for {proj} {m} {ens}")
                    anoms = get_anomalies(
                        ensembles[ens], base_start, fut_start, rel_change
                    )
                    if anoms is None:
                        continue
                    projections[proj][m][ens] = anoms
                    model_lists[proj].append(f"{m} {ens}")
                    UKCP_ensembles.append(ens)
            else:
                logging.info(f"Calculating anomalies for {proj} {m}")
                anoms = get_anomalies(models[m], base_start, fut_start, rel_change)
                if anoms is None:
                    continue
                projections[proj][m] = anoms
                model_lists[proj].append(f"{m}")
    cordex_drivers = set(cordex_drivers)

    # we now have all the projections in the projections dictionary
    # check what driving models we have from CORDEX and decide on some fixed colours for them..
    # get default colours
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colours = prop_cycle.by_key()["color"]
    colours_it = enumerate(colours)
    driver_colours = {}
    for k in cordex_drivers:
        driver_colours[k] = next(colours_it)[1]

    # now lets plot them
    # first we need to process the dictionary, and move the data into a list of vectors
    seasons = {0: "DJF", 1: "MAM", 2: "JJA", 3: "OND"}
    logger.info("Plotting")
    for s in seasons.keys():
        proj_plotting = dict.fromkeys(projections.keys())
        for p in projections:
            proj_plotting[p] = process_projections_dict(projections[p], s)

        # eventually plotting code etc. will go in a seperate module I think.
        projects, plot_values = zip(*proj_plotting.items())
        # plots_values is a list of dictionaries of model names and associated values
        plt.figure(figsize=(12.8, 9.6))
        plt.boxplot([list(v.values()) for v in plot_values])
        plotted_drivers = set()
        for i, p in enumerate(proj_plotting.keys()):
            for m, v in proj_plotting[p].items():
                if p[:6].upper() == "CORDEX":
                    # extract the driving model from the string
                    driver = m.split(" ")[1]
                    color = driver_colours[driver]
                    alpha = 1
                    sz = 25
                    # Check if we have already plotted this driving model before..
                    # This means we only use the label once, and just end up with one legend entry per driver
                    if driver in plotted_drivers:
                        driver = None
                    else:
                        plotted_drivers.add(driver)
                else:
                    driver = [cd for cd in cordex_drivers if m in cd]
                    # check if model matches any of the CORDEX ones
                    if driver:
                        driver = driver[0]
                        color = driver_colours[driver]
                        alpha = 1
                        sz = 25
                        # Check if we have already plotted this driving model before..
                        # This means we only use the label once, and just end up with one legend entry per driver
                        if driver in plotted_drivers:
                            driver = None
                        else:
                            plotted_drivers.add(driver)
                    else:
                        color = "k"
                        alpha = 0.3
                        driver = None
                        sz = 10
                # Add some random "jitter" to the x-axis
                x = np.random.normal(i + 1, 0.05, size=1)
                plt.scatter(x, v, label=driver, color=color, alpha=alpha, s=sz)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.axhline(0, linestyle="dotted", color="k")
        plt.gca().set_xticklabels(projects)
        plt.title(f"{seasons[s]} {var} change")
        plt.tight_layout()
        plt.savefig(f"{cfg['plot_dir']}/boxplot_{seasons[s]}.png")
        plt.close()

    # print all datasets used
    print("Input models for plots:")
    for p in model_lists.keys():
        print(f"{p}: {len(model_lists[p])} models")
        print(model_lists[p])
        print("")


if __name__ == "__main__":
    with run_diagnostic() as cfg:
        main(cfg)
