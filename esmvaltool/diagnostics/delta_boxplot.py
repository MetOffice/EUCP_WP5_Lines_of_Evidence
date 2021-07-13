# Calculate anomalies, and then plot our model groups in a boxplot

from esmvaltool.diag_scripts.shared import (
    run_diagnostic,
    group_metadata,
    select_metadata,
    extract_variables,
)

import iris

import os
import logging
import re

import numpy as np
import matplotlib.pyplot as plt

from cycler import cycler

logger = logging.getLogger(os.path.basename(__file__))

# Institutes that appear in front of the driver string for CORDEX RCMS
INSTITUTES = [
    'IPSL',
    'NCC',
    'MPI-M',
    'CNRM-CERFACS',
    'ICHEC',
    'MOHC'
]

CPM_DRIVERS = {
    'CNRM-AROME41t1': 'ALADIN63 CNRM-CERFACS-CNRM-CM5',
    'CLMcom-CMCC-CCLM5-0-9': 'CCLM4-8-17 ICHEC-EC-EARTH',
    'HCLIMcom-HCLIM38-AROME': 'RACMO22E ICHEC-EC-EARTH',
    'GERICS-REMO2015': 'REMO2015 MPI-M-MPI-ESM-LR',
    'COSMO-pompa': 'CCLM4-8-17 MPI-M-MPI-ESM-LR',
    'ICTP-RegCM4-7-0': 'RegCM4-6 MOHC-HadGEM2-ES',
    'KNMI-HCLIM38h1-AROME': 'RACMO22E ICHEC-EC-EARTH',
}


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


def get_anomalies(ds_list, relative=False):
    # determine historic and future periods
    start_years = list(group_metadata(ds_list, "start_year"))
    base_clim_start = min(start_years)
    fut_clim_start = max(start_years)

    # construct baseline
    base_metadata = select_metadata(ds_list, start_year=base_clim_start)
    base_file = base_metadata[0]["filename"]
    base_cube = iris.load_cube(base_file)

    # get future
    fut_metadata = select_metadata(ds_list, start_year=fut_clim_start)
    fut_file = fut_metadata[0]["filename"]
    fut_cube = iris.load_cube(fut_file)

    if relative:
        diff = fut_cube - base_cube
        anomaly = (diff / base_cube) * 100
        anomaly.units = "%"
    else:
        anomaly = fut_cube - base_cube

    return anomaly


def save_anoms_txt(data, fname):
    # iterate over the supplied dictionary and write the data to a textfile
    # sort the data
    sorted_data = sorted(data.items(), key=lambda x: x[1])

    # open the file for writing
    with open(fname, mode="w") as f:
        for d in sorted_data:
            # write a line of data
            f.write(f"{d[0]}: {d[1]:.2f}\n")


def get_var(cfg):
    # get variable processed
    var = list(extract_variables(cfg).keys())
    assert len(var) == 1
    var = var[0]

    return var


def plot_boxplots(projections, legend_models, fname_suffix=None):
    # we now have all the projections in the projections dictionary
    # check what driving models we have from CORDEX and decide on some fixed colours for them..
    # get default colours
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colours = prop_cycle.by_key()["color"]
    p_cycler = (cycler(color=colours) * cycler(marker=["o", "D", "P", "X"]))
    enum_props = enumerate(p_cycler)
    legend_colours = {}
    legend_markers = {}
    for k in legend_models:
        k_props = next(enum_props)
        legend_colours[k] = k_props[1]['color']
        legend_markers[k] = k_props[1]['marker']

    # special models that have an extra large symbol
    # special_models = ["RACMO22E", "HadREM3-GA7-05"]
    special_models = []

    # get variable for title later
    var = get_var(cfg)

    # now lets plot them
    # first we need to process the dictionary, and move the data into a list of vectors
    # the projections object is the key one that contains all our data..
    seasons = {0: "DJF", 1: "MAM", 2: "JJA", 3: "OND"}
    p_keys = reorder_keys(list(projections.keys()))

    logger.info("Plotting")
    for s in seasons.keys():
        proj_plotting = dict.fromkeys(p_keys)
        for p in p_keys:
            proj_plotting[p] = process_projections_dict(projections[p], s)
            save_anoms_txt(proj_plotting[p],
                           f"{cfg['work_dir']}/{p}_{seasons[s]}_values.txt")

        # eventually plotting code etc. will go in a seperate module I think.
        projects, plot_values = zip(*proj_plotting.items())
        # plots_values is a list of dictionaries of model names and associated values
        plt.figure(figsize=(12.8, 9.6))
        plt.boxplot([list(v.values()) for v in plot_values])
        plotted_models = set()
        for i, p in enumerate(proj_plotting.keys()):
            for m, v in proj_plotting[p].items():
                if p[:6].upper() == "CORDEX":
                    # extract the driving model and RCM from the string
                    rcm, driver = m.split(" ")
                    # find the legend color
                    if driver in legend_colours.keys():
                        label = driver
                        color = legend_colours[driver]
                        marker = legend_markers[driver]
                    else:
                        label = rcm
                        color = legend_colours[rcm]
                        marker = legend_markers[rcm]
                    alpha = 1
                    if any(i in m for i in special_models):
                        sz = 100
                    else:
                        sz = 25
                    # Check if we have already plotted this model before..
                    # This means we only use the label once, and just end up with one legend entry per model
                    if any(i in m for i in plotted_models):
                        label = None
                    else:
                        plotted_models.add(label)
                elif "CMIP" in p:
                    model = [x for x in legend_models if m in x]
                    # check if model matches any of the CORDEX ones
                    if model:
                        model = model[0]
                        color = legend_colours[model]
                        alpha = 1
                        sz = 25
                        # Check if we have already plotted this model before..
                        # This means we only use the label once
                        if any(i in model for i in plotted_models):
                            label = None
                        else:
                            plotted_models.add(model)
                    else:
                        color = "k"
                        alpha = 0.3
                        label = None
                        marker = "."
                        sz = 10
                else:
                    color = "k"
                    alpha = 0.3
                    label = None
                    marker = "."
                    sz = 10
                # Add some random "jitter" to the x-axis
                x = np.random.normal(i + 1, 0.05, size=1)
                plt.scatter(x, v, label=label, color=color, alpha=alpha, s=sz, marker=marker)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.axhline(0, linestyle="dotted", color="k")
        plt.gca().set_xticklabels(projects)
        plt.title(f"{seasons[s]} {var} change")
        plt.tight_layout()
        plt.savefig(f"{cfg['plot_dir']}/boxplot_{seasons[s]}{fname_suffix}.png")
        plt.close()


def simple_dots_plot(projections, cordex_drivers, fname_suffix=''):
    # get variable for title later
    var = get_var(cfg)

    # now lets plot them
    # first we need to process the dictionary, and move the data into a list of vectors
    # the projections object is the key one that contains all our data..
    seasons = {0: "DJF", 1: "MAM", 2: "JJA", 3: "OND"}
    logger.info("Plotting")
    p_keys = reorder_keys(list(projections.keys()))
    for s in seasons.keys():
        proj_plotting = dict.fromkeys(p_keys)
        for p in p_keys:
            proj_plotting[p] = process_projections_dict(projections[p], s)
            save_anoms_txt(proj_plotting[p],
                           f"{cfg['work_dir']}/{p}_{seasons[s]}_values.txt")

        # plots_values is a list of dictionaries of model names and associated values
        plt.figure(figsize=(12.8, 9.6))
        projects, values = zip(*proj_plotting.items())
        for i, p in enumerate(p_keys):
            for m, v in proj_plotting[p].items():
                plt.plot(
                        i + 1,
                        v,
                        marker="o",
                        fillstyle="none",
                        color="k",
                    )
                if p == "CMIP5":
                    if any(m in d for d in cordex_drivers):
                        plt.plot(
                            i + 1,
                            v,
                            marker="o",
                            fillstyle="full",
                            color="k",
                            markersize=12,
                        )
                elif "CORDEX" in p:
                    if any(m == d for d in cordex_drivers):
                        plt.plot(
                            i + 1,
                            v,
                            marker="o",
                            fillstyle="full",
                            color="k",
                            markersize=12,
                        )
            if 'CMIP' in p:
                plt.violinplot(list(values[i].values()),
                               positions=[i + 1],
                               showmedians=True)
        plt.gca().set_xticks(range(1, len(projects) + 1))
        plt.gca().set_xticklabels(projects)
        plt.title(f"{seasons[s]} {var} change")
        plt.tight_layout()
        plt.savefig(f"{cfg['plot_dir']}/violin_{seasons[s]}{fname_suffix}.png")
        plt.close()


def prepare_scatter_data(x_data, y_data, project, full_y=None):
    # need to establish matching cmip value for each cordex value
    # cordex data keyed by RCM, then GCM
    x_vals = []
    y_vals = []
    labels = []

    if project == "CORDEX":
        for rcm in x_data:
            for driver in x_data[rcm]:
                x_vals.append(x_data[rcm][driver])

                # find corresponding cmip data
                actual_driver = remove_institute_from_driver(driver)
                y_vals.append(y_data[actual_driver])

                # construct label
                labels.append(f"{actual_driver} {rcm}")
        if full_y:
            full_y_vals = full_y.values()
    elif project == "UKCP18":
        for ensemble in x_data:
            x_vals.append(x_data[ensemble])
            y_vals.append(y_data[ensemble])

            labels.append(ensemble)
        if full_y:
            full_y_vals = full_y.values()
    elif project == "CPM":
        for cpm in x_data:
            for driver in x_data[cpm]:
                x_vals.append(x_data[cpm][driver])

                actual_driver = CPM_DRIVERS[cpm]
                rcm, sep, gcm = actual_driver.partition(' ')
                y_vals.append(y_data[rcm][gcm])

                # construct label
                labels.append(f"{actual_driver} {cpm}")
        if full_y:
            full_y_vals = []
            for rcm in full_y:
                for gcm in full_y[rcm]:
                    full_y_vals.append(full_y[rcm][gcm])
    else:
        raise ValueError(f"Unrecognised project {project}")

    if full_y:
        return x_vals, y_vals, labels, full_y_vals
    else:
        return x_vals, y_vals, labels


def scatter_response(x_data, y_data, labels, suffix='', full_x=None):
    seasons = {0: "DJF", 1: "MAM", 2: "JJA", 3: "OND"}
    for s in seasons.keys():
        # construct iris constraint
        season_con = iris.Constraint(season_number=s)
        plt.figure(figsize=(12.8, 9.6))
        max_val = 0
        min_val = 0

        if full_x:
            x_cubes = iris.cube.CubeList(full_x).extract(season_con)
            full_x_data = [y.data.item() for y in x_cubes]
            y_array = []

        # construct axes
        if full_x:
            ax_scatter = plt.subplot(232)
            ax_x = plt.subplot(231, sharey=ax_scatter)
            ax_y = plt.subplot(233, sharey=ax_scatter)
        else:
            ax_scatter = plt.axes()

        last_label = None
        if "cordex" in suffix:
            marker_props = enumerate((cycler(marker=['o', 'P']) * cycler(color=list('rgbmy'))))

        for i in range(len(x_data)):
            x_val = x_data[i].extract(season_con).data
            y_val = y_data[i].extract(season_con).data

            if full_x:
                y_array.append(y_val.item())

            # update max and min value encountered
            max_val = max(x_val, y_val, max_val)
            min_val = min(x_val, y_val, min_val)

            if "cordex" in suffix:
                if labels[i].split()[-1] != last_label:
                    props = next(marker_props)
                    last_label = labels[i].split()[-1]
                ax_scatter.scatter(
                    x_val, y_val, label=f"{i} - {labels[i]}",
                    color=props[1]['color'], marker=props[1]['marker']
                    )
            else:
                ax_scatter.scatter(x_val, y_val, label=f"{i} - {labels[i]}")

            if labels[i].isdigit():
                ax_scatter.text(x_val, y_val, labels[i])
            else:
                ax_scatter.text(x_val, y_val, i)

        if not labels[i].isdigit():
            h, ls = ax_scatter.get_legend_handles_labels()
            ax_legend = plt.subplot(236)
            ax_legend.axis('off')
            ax_legend.legend(h, ls, ncol=2)

        if suffix == "_cpm":
            ax_scatter.set_ylabel("CPM response")
            ax_scatter.set_xlabel("RCM response")
        else:
            ax_scatter.set_ylabel("RCM response")
            ax_scatter.set_xlabel("GCM response")
        ax_scatter.set_title(f"{get_var(cfg)} response")

        # plot a diagonal equivalence line
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

        if full_x:
            # violinplots
            ax_x.violinplot(full_x_data)
            for i in range(len(full_x_data)):
                ax_x.plot(
                        1,
                        full_x_data[i],
                        marker="o",
                        fillstyle="none",
                        color="k",
                    )
            ax_x.axis('off')
            if suffix == "_cpm":
                ax_x.set_title('Full RCM ensemble')
            else:
                ax_x.set_title('Full GCM ensemble')

            ax_y.violinplot(y_array)
            for i in range(len(y_array)):
                ax_y.plot(
                        1,
                        y_array[i],
                        marker="o",
                        fillstyle="none",
                        color="k",
                    )
            ax_y.axis('off')
            if suffix == "_cpm":
                ax_y.set_title('CPM ensemble')
            else:
                ax_y.set_title('RCM ensemble')

        # save
        plt.savefig(f"{cfg['plot_dir']}/scatter_{seasons[s]}{suffix}.png")
        plt.close()


def remove_institute_from_driver(driver_str):
    # remove the institute bit from the "driver" string

    new_str = driver_str
    # loop through the institutes and remove them if found
    for i in INSTITUTES:
        i = '^' + i + '-'
        new_str = re.sub(i, '', new_str)

    if new_str == driver_str:
        raise ValueError(f"No institute found to remove from {driver_str}")

    return new_str


def reorder_keys(keys):
    # order keys so that first 3 projects are CMIP6, CMIP5, CORDEX
    if 'CORDEX' in keys:
        # remove from present location and move to front
        i = keys.index('CORDEX')
        keys.insert(0, keys.pop(i))
    if 'CMIP5' in keys:
        # remove from present location and move to front
        i = keys.index('CMIP5')
        keys.insert(0, keys.pop(i))
    if 'CMIP6' in keys:
        # remove from present location and move to front
        i = keys.index('CMIP6')
        keys.insert(0, keys.pop(i))

    return keys


def main(cfg):
    # The config object is a dict of all the metadata from the pre-processor

    # get variable processed
    var = get_var(cfg)

    if var == "pr":
        rel_change = True
    else:
        rel_change = False

    # first group datasets by project..
    # this creates a dict of datasets keyed by project (CMIP5, CMIP6 etc.)
    projects = group_metadata(cfg["input_data"].values(), "project")
    # how to uniquely define a dataset varies by project, for CMIP it's simple, just dataset...
    # for CORDEX, combo of dataset and driver (and possibly also domain if we start adding those)
    # also gets more complex if we start adding in different ensembles..

    # get "special" RCMS if being used
    if 'special_rcms' in cfg:
        spec_rcms = cfg['special_rcms']
    else:
        spec_rcms = None

    # This section of the code loads and organises the data to be ready for plotting
    logger.info("Loading data")
    # empty dict to store results
    projections = {}
    model_lists = {}
    cordex_drivers = []
    cordex_rcms = []
    rcm_drivers = cfg['rcm_drivers']
    # loop over projects
    for proj in projects:
        # we now have a list of all the data entries..
        # for CMIPs we can just group metadata again by dataset then work with that..
        models = group_metadata(projects[proj], "dataset")

        # empty dict for results
        projections[proj] = {}
        # loop over the models
        for m in models:
            if proj[:6].upper() == "CORDEX":
                # then we need to go one deeper in the dictionary to deal with driving models
                drivers = group_metadata(models[m], "driver")
                projections[proj][m] = dict.fromkeys(drivers.keys())
                for d in drivers:
                    logging.info(f"Calculating anomalies for {proj} {m} {d}")
                    anoms = get_anomalies(drivers[d], rel_change)
                    if anoms is None:
                        continue
                    projections[proj][m][d] = anoms
                    if proj not in model_lists:
                        model_lists[proj] = []
                    model_lists[proj].append(f"{m} {d}")
                    cordex_drivers.append(d)
                    cordex_rcms.append(m)
            elif proj == "UKCP18":
                # go deeper to deal with ensembles and datasets
                # split UKCP into seperate GCM and RCM
                proj_key = f"UKCP18 {m}"
                ensembles = group_metadata(models[m], "ensemble")
                projections[proj_key] = dict.fromkeys(ensembles.keys())
                for ens in ensembles:
                    logging.info(f"Calculating anomalies for {proj_key} {ens}")
                    anoms = get_anomalies(ensembles[ens], rel_change)
                    if anoms is None:
                        continue
                    projections[proj_key][ens] = anoms
                    if proj_key not in model_lists:
                        model_lists[proj_key] = []
                    model_lists[proj_key].append(f"{proj_key} {ens}")
            else:
                logging.info(f"Calculating anomalies for {proj} {m}")
                anoms = get_anomalies(models[m], rel_change)
                if anoms is None:
                    continue
                projections[proj][m] = anoms
                if proj not in model_lists:
                    model_lists[proj] = []
                model_lists[proj].append(f"{m}")

        # seperate CORDEX RCMs into special and normal if needed
        if spec_rcms and proj[:6].upper() == "CORDEX":
            # create new dictionary entry if needed
            if 'CORDEX_aerosol' not in projections:
                projections['CORDEX_aerosol'] = {}
            for m in models:
                if m in spec_rcms:
                    data = projections[proj].pop(m)
                    projections['CORDEX_aerosol'][m] = data

        # remove any empty categories (i.e. UKCP18 which has been split into rcm and gcm)
        if projections[proj] == {}:
            del projections[proj]
    cordex_drivers = set(cordex_drivers)
    cordex_rcms = set(cordex_rcms)

    # this section of the code does all the plotting..
    plot_boxplots(projections, cordex_drivers, "_drivers")
    plot_boxplots(projections, cordex_rcms, "_rcms")
    simple_dots_plot(projections, list(cordex_drivers) + rcm_drivers)

    # scatter plots - regular cordex
    if all([p in projections for p in ["CORDEX", "CMIP5"]]):
        rcm_points, gcm_points, labels, cmip5 = prepare_scatter_data(
            projections["CORDEX"], projections["CMIP5"], "CORDEX", projections["CMIP5"]
            )
        scatter_response(gcm_points, rcm_points, labels, "_cordex_simple_aerosol", cmip5)

    # cordex with clever aerosol
    if all([p in projections for p in ["CORDEX_aerosol", "CMIP5"]]):
        rcm_points, gcm_points, labels, cmip5 = prepare_scatter_data(
            projections["CORDEX_aerosol"], projections["CMIP5"], "CORDEX", projections["CMIP5"]
            )
        scatter_response(gcm_points, rcm_points, labels, "_cordex_dynamic_aerosol", cmip5)

    # UKCP
    if all([p in projections for p in ["UKCP18 land-rcm", "UKCP18 land-gcm"]]):
        rcm_points, gcm_points, labels, ukcp_gcm = prepare_scatter_data(
            projections["UKCP18 land-rcm"], projections["UKCP18 land-gcm"], "UKCP18", projections["UKCP18 land-gcm"]
            )
        scatter_response(gcm_points, rcm_points, labels, "_UKCP", ukcp_gcm)

    # CPMs
    if all([p in projections for p in ["CORDEX", "CORDEX_aerosol", "cordex-cpm"]]):
        all_cordex = projections["CORDEX"].copy()
        all_cordex.update(projections["CORDEX_aerosol"])
        rcm_points, gcm_points, labels, all_gcm = prepare_scatter_data(
            projections["cordex-cpm"], all_cordex, "CPM", all_cordex
            )
        scatter_response(gcm_points, rcm_points, labels, "_cpm", all_gcm)

    # print all datasets used
    print("Input models for plots:")
    for p in model_lists.keys():
        print(f"{p}: {len(model_lists[p])} models")
        print(model_lists[p])
        print("")


if __name__ == "__main__":
    with run_diagnostic() as cfg:
        main(cfg)
