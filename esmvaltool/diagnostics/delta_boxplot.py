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
import pickle

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


def proj_dict_to_season_dict(proj_dict):
    # take a dictionary of data keyed by project, and reorganise to key by season
    seasons = {0: "DJF", 1: "MAM", 2: "JJA", 3: "OND"}

    season_dict = {}
    # output dict will be keyed by season, then project, then model
    for s in seasons:
        season_dict[seasons[s]] = {}
        for p in proj_dict:
            season_dict[seasons[s]][p] = process_projections_dict(proj_dict[p], s)

    return season_dict


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


def plot_boxplots(projections, legend_models, season, fname_suffix=None):
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
    p_keys = reorder_keys(list(projections.keys()))

    logger.info("Plotting")

    box_values = []
    for p in p_keys:
        box_values.append(list(projections[p].values()))

    plt.figure(figsize=(12.8, 9.6))
    # plot all projects as boxplots
    plt.boxplot(box_values)

    plotted_models = set()

    # now plot dots
    for i, p in enumerate(p_keys):
        for m, v in projections[p].items():
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
    plt.gca().set_xticklabels(p_keys)
    plt.title(f"{season} {var} change")
    plt.tight_layout()
    plt.savefig(f"{cfg['plot_dir']}/boxplot_{season}{fname_suffix}.png", bbox_inches='tight')
    plt.close()


def simple_dots_plot(projections, cordex_drivers, season, fname_suffix=''):
    # get variable for title later
    var = get_var(cfg)

    # now lets plot them
    logger.info("Plotting")
    p_keys = reorder_keys(list(projections.keys()))

    plt.figure(figsize=(12.8, 9.6))

    for i, p in enumerate(p_keys):
        for m, v in projections[p].items():
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
            plt.violinplot(list(projections[p].values()),
                           positions=[i + 1],
                           showmedians=True)
    plt.gca().set_xticks(range(1, len(p_keys) + 1))
    plt.gca().set_xticklabels(p_keys)
    plt.title(f"{season} {var} change")
    plt.tight_layout()
    plt.savefig(f"{cfg['plot_dir']}/violin_{season}{fname_suffix}.png", bbox_inches='tight')
    plt.close()


def prepare_scatter_data(x_data, y_data, project):
    # need to establish matching cmip value for each cordex value
    x_vals = []
    y_vals = []
    labels = []

    if project == "CORDEX":
        # expect rcm vals are y vals. GCM, x
        for rcm in y_data:
            y_vals.append(y_data[rcm])

            # find corresponding cmip data
            actual_rcm, driver = rcm.split(' ')
            actual_driver = remove_institute_from_driver(driver)

            x_vals.append(x_data[actual_driver])

            # construct label
            labels.append(f"{actual_driver} {actual_rcm}")
    elif project == "UKCP18":
        for ensemble in x_data:
            x_vals.append(x_data[ensemble])
            y_vals.append(y_data[ensemble])

            labels.append(ensemble)
    elif project == "CPM":
        # cpm on y axis, rcm on x axis
        for cpm in y_data:
            y_vals.append(y_data[cpm])

            driver = CPM_DRIVERS[cpm.split(' ')[0]]
            cpm = cpm.split(' ')[0]
            x_vals.append(x_data[driver])

            # construct label
            labels.append(f"{driver} {cpm}")
    else:
        raise ValueError(f"Unrecognised project {project}")

    return x_vals, y_vals, labels


def labelled_scatter(x_data, y_data, labels, ax, RCM_markers=False):
    if RCM_markers:
        label_props = {}
        marker_props = enumerate((cycler(marker=['o', 'P']) * cycler(color=list('bgrcmy'))))

    max_val = 0
    min_val = 0

    for i in range(len(x_data)):
        x_val = x_data[i]
        y_val = y_data[i]

        # update max and min value encountered
        max_val = max(x_val, y_val, max_val)
        min_val = min(x_val, y_val, min_val)

        if RCM_markers:
            rcm = labels[i].split()[-1]
            if rcm in label_props:
                props = label_props[rcm]
            else:
                props = next(marker_props)
                label_props[rcm] = props

            ax.scatter(
                x_val, y_val, label=f"{i} - {labels[i]}",
                color=props[1]['color'], marker=props[1]['marker']
                )
        else:
            ax.scatter(x_val, y_val, label=f"{i} - {labels[i]}")

        ax.text(x_val, y_val, i)

    # plot a diagonal equivalence line
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)


def scatter_response(x_data, y_data, labels, season, suffix='', full_x=None):
    plt.figure(figsize=(12.8, 9.6))
    max_val = 0
    min_val = 0

    full_x_data = [y.data.item() for y in full_x]
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
    plt.savefig(f"{cfg['plot_dir']}/scatter_{seasons[s]}{suffix}.png", bbox_inches='tight')
    plt.close()


def mega_scatter(GCM_sc, RCM_sc1, RCM_sc2, CPM, all_GCM, all_RCM, labels1, labels2, suffix=''):
    '''
    GCM_sc: GCMs for first scatter
    RCM_sc1: RCMs for first scatter
    RCM_sc2: RCMs for second scatter
    CPM: CPMs
    all_GCM: all GCMs for the violin
    all_RCM: all RCMs for a violin
    labels1: Legend labels for scatter1
    labels2: Legend labels for scatter2
    suffix: suffix to end to filename
    '''
    plt.figure(figsize=(19.2, 14.4))

    # construct axes
    ax_violins = plt.subplot(211)
    ax_scatter1 = plt.subplot(223)
    ax_scatter2 = plt.subplot(224)

    # Create GCM / RCM scatter
    labelled_scatter(GCM_sc, RCM_sc1, labels1, ax_scatter1, RCM_markers=True)
    ax_scatter1.set_xlabel('GCM')
    ax_scatter1.set_ylabel('RCM')
    if min(RCM_sc1) < 0 < max(RCM_sc1):
        ax_scatter1.axhline(ls=':', color='k', alpha=0.75)

    # create RCM / CPM scatter
    labelled_scatter(RCM_sc2, CPM, labels2, ax_scatter2)
    ax_scatter2.set_xlabel('RCM')
    ax_scatter2.set_ylabel('CPM')
    if min(CPM) < 0 < max(CPM):
        ax_scatter2.axhline(ls=':', color='k', alpha=0.75)

    # legend information
    h1, l1 = ax_scatter1.get_legend_handles_labels()
    h2, l2 = ax_scatter2.get_legend_handles_labels()
    ax_violins.legend(h1 + h2, l1 + l2, bbox_to_anchor=(1.05, 1.0), loc='upper left')

    # create GCM / RCM / CPM violins / dots
    # GCMs go in position 1, RCMs position 2, CPMs position 3
    coloured_violin(all_GCM, 1, ax_violins, 'lightgrey')
    coloured_violin(all_RCM, 2, ax_violins,'lightgrey')
    coloured_violin(CPM, 3, ax_violins,'lightgrey')
    
    # set x labels
    ax_violins.set_xticks(range(1, 4))
    ax_violins.set_xticklabels(['CMIP5', 'CORDEX', 'CPM'])

    # also plot individual dots for each model..
    plot_points(all_GCM, 0.8, ax_violins)
    plot_points(GCM_sc, 1.3, ax_violins, color='r')
    plot_points(RCM_sc1, 1.8, ax_violins, color='r')
    plot_points(RCM_sc2, 2.3, ax_violins, color='b')
    plot_points(CPM, 3, ax_violins, color='b')

    max_violin = max(max(all_GCM), max(RCM_sc1), max(CPM))
    min_violin = min(min(all_GCM), min(RCM_sc1), min(CPM))
    if min_violin < 0 < max_violin:
        ax_violins.axhline(ls=':', color='k', alpha=0.75)

    var = get_var(cfg)
    plt.suptitle(f"{suffix} {var} change")

    # save plot
    plt.savefig(f"{cfg['plot_dir']}/mega_scatter_{suffix}.png", bbox_inches='tight')
    plt.close()


def plot_points(points, x, ax, color='k'):
    for p in points:
         ax.plot(x, p, marker="o", fillstyle="none", color=color)


def coloured_violin(data, pos, ax, color=None):
    vparts = ax.violinplot(data, [pos])

    if color:
        for part in ['bodies', 'cbars', 'cmins', 'cmaxes']:
            if part == 'bodies':
                for c in vparts[part]:
                    c.set_color(color)
            else:
                vparts[part].set_color(color)


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

    # reorganise and extract data for plotting
    plotting_dict = proj_dict_to_season_dict(projections)

    for season in plotting_dict.keys():
        # this section of the code does all the plotting..
        plot_boxplots(plotting_dict[season], cordex_drivers, season, "_drivers")
        plot_boxplots(plotting_dict[season], cordex_rcms, season, "_rcms")
        simple_dots_plot(plotting_dict[season], list(cordex_drivers) + rcm_drivers, season)

        # # scatter plots - regular cordex
        # if all([p in projections for p in ["CORDEX", "CMIP5"]]):
        #     rcm_points, gcm_points, labels, cmip5 = prepare_scatter_data(
        #         projections["CORDEX"], projections["CMIP5"], "CORDEX", projections["CMIP5"]
        #         )
        #     scatter_response(gcm_points, rcm_points, labels, "_cordex_simple_aerosol", cmip5)

        # # cordex with clever aerosol
        # if all([p in projections for p in ["CORDEX_aerosol", "CMIP5"]]):
        #     rcm_points, gcm_points, labels, cmip5 = prepare_scatter_data(
        #         projections["CORDEX_aerosol"], projections["CMIP5"], "CORDEX", projections["CMIP5"]
        #         )
        #     scatter_response(gcm_points, rcm_points, labels, "_cordex_dynamic_aerosol", cmip5)

        # # UKCP
        # if all([p in projections for p in ["UKCP18 land-rcm", "UKCP18 land-gcm"]]):
        #     rcm_points, gcm_points, labels, ukcp_gcm = prepare_scatter_data(
        #         projections["UKCP18 land-rcm"], projections["UKCP18 land-gcm"], "UKCP18", projections["UKCP18 land-gcm"]
        #         )
        #     scatter_response(gcm_points, rcm_points, labels, "_UKCP", ukcp_gcm)

        # # CPMs
        # if all([p in projections for p in ["CORDEX", "CORDEX_aerosol", "cordex-cpm"]]):
        #     all_cordex = projections["CORDEX"].copy()
        #     all_cordex.update(projections["CORDEX_aerosol"])
        #     cpm_points, rcm_points, cpm_labels = prepare_scatter_data(
        #         projections["cordex-cpm"], all_cordex, "CPM"
        #         )
        #     scatter_response(gcm_points, rcm_points, cpm_labels, "_cpm", all_gcm)

        # mega scatter plot
        # need to prepare subsets of projects
        all_CORDEX = {}
        all_CORDEX.update(plotting_dict[season]['CORDEX'])
        all_CORDEX.update(plotting_dict[season]['CORDEX_aerosol'])
        gcm_sc, rcm_sc1, labels1 = prepare_scatter_data(plotting_dict[season]['CMIP5'], all_CORDEX, 'CORDEX')
        rcm_sc2, cpm_sc, labels2 = prepare_scatter_data(all_CORDEX, plotting_dict[season]['cordex-cpm'], 'CPM')

        mega_scatter(
            gcm_sc, rcm_sc1, rcm_sc2, cpm_sc,
            list(plotting_dict[season]['CMIP5'].values()), list(all_CORDEX.values()),
            labels1, labels2, f'{season}'
        )

        # save some plotting data for notebook experiments
        # create dictionary of all the required data for one particular season
        if season == 'JJA':
            pickle_dict = {}
            pickle_dict['CMIP5_sc'] = gcm_sc
            pickle_dict['RCM_sc1'] = rcm_sc1
            pickle_dict['RCM_sc2'] = rcm_sc2
            pickle_dict['labels1'] = labels1
            pickle_dict['labels2'] = labels2
            pickle_dict['cpm'] = cpm_sc
            pickle_dict['CMIP6'] = list(plotting_dict[season]['CMIP6'].values())
            pickle_dict['CMIP5'] = list(plotting_dict[season]['CMIP5'].values())
            pickle_dict['CORDEX'] = list(all_CORDEX.values())
            pickle_dict['UKCP18 land-gcm'] = plotting_dict[season]['UKCP18 land-gcm']
            pickle_dict['UKCP18 land-rcm'] = plotting_dict[season]['UKCP18 land-rcm']

            pickle.dump(pickle_dict, open(f'{cfg["work_dir"]}/sample_plotting_data.pkl', 'wb'))



    # print all datasets used
    print("Input models for plots:")
    for p in model_lists.keys():
        print(f"{p}: {len(model_lists[p])} models")
        print(model_lists[p])
        print("")


if __name__ == "__main__":
    with run_diagnostic() as cfg:
        main(cfg)
