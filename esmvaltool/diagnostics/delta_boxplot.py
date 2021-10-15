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
from iris.io import save

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
    'MOHC',
    'KNMI',
    'HCLIMcom',
    'SMHI'
]

# This dictionary maps CPM string to a RCM GCM string
CPM_DRIVERS = {
    'CNRM-AROME41t1': 'ALADIN63 CNRM-CERFACS-CNRM-CM5',
    'CLMcom-CMCC-CCLM5-0-9': 'CCLM4-8-17 ICHEC-EC-EARTH',
    'HCLIMcom-HCLIM38-AROME': 'HCLIMcom-HCLIM38-ALADIN ICHEC-EC-EARTH',
    'GERICS-REMO2015': 'REMO2015 MPI-M-MPI-ESM-LR',
    'COSMO-pompa': 'CCLM4-8-17 MPI-M-MPI-ESM-LR',
    'ICTP-RegCM4-7-0': 'ICTP-RegCM4-7-0 MOHC-HadGEM2-ES',
    'ICTP-RegCM4-7': 'ICTP-RegCM4-7-0 MOHC-HadGEM2-ES',
    'KNMI-HCLIM38h1-AROME': 'KNMI-RACMO23E KNMI-EC-EARTH',
    'SMHI-HCLIM38-AROME': 'SMHI-HCLIM38-ALADIN ICHEC-EC-EARTH',
    'HadREM3-RA-UM10.1': 'MOHC-HadGEM3-GC3.1-N512 MOHC-HadGEM2-ES'
}

PATH_TO_GLENS_CDFS = '/home/h02/tcrocker/code/EUCP_WP5_Lines_of_Evidence/weighting_data/glen/'
PATH_TO_CLIMWIP_DATA = '/home/h02/tcrocker/code/EUCP_WP5_Lines_of_Evidence/weighting_data/ClimWIP/'

SEASONS_DICT = {'DJF': 0, 'MAM': 1, 'JJA': 2, 'SON': 3}

DOMAIN_FOR_CDF = {
    'ALP-3': 'PALP',
    'CEE-3': 'CEEU',
    'NWE-3': 'NWEU',
    'NEU-3': 'NEU'
}


def get_glens_cdf(domain, season, var):
    # construct path to file
    nc_file = f'{PATH_TO_GLENS_CDFS}{var}Anom_rcp85_eu_{domain}-WP3_Wall-N600000-P21_cdf_b9605_10y_{season.lower()}_20401201-20501130.nc'

    # load data
    try:
        cube = iris.load_cube(nc_file)
    except OSError:
        logger.warning(f"Couldn't load: {nc_file}")
        return None

    # return data
    return cube.data


def get_ClimWIP_percentiles(domain, season, var):
    # construct path to files
    weighted_file = f'{PATH_TO_CLIMWIP_DATA}{var}_weighted_CMIP5_{domain}_1980-2014_2031-2060.nc'
    unweighted_file = f'{PATH_TO_CLIMWIP_DATA}{var}_unweighted_CMIP5_{domain}_1980-2014_2031-2060.nc'
    # weighted_file = f'{PATH_TO_CLIMWIP_DATA}{var}_weighted_CMIP5_2_{domain}_1996-2005_2041-2050.nc'
    # unweighted_file = f'{PATH_TO_CLIMWIP_DATA}{var}_unweighted_CMIP5_2_{domain}_1996-2005_2041-2050.nc'

    # load data
    try:
        weighted_cube = iris.load_cube(weighted_file)
        unweighted_cube = iris.load_cube(unweighted_file)
    except OSError:
        logger.warning(f"Couldn't load: {weighted_file} or {unweighted_file}")
        return None, None

    # extract season
    sea_con = iris.Constraint(season_number=SEASONS_DICT[season])
    weighted_cube = weighted_cube.extract(sea_con)
    unweighted_cube = unweighted_cube.extract(sea_con)

    # return data
    return weighted_cube.data, unweighted_cube.data


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
    seasons = {0: "DJF", 1: "MAM", 2: "JJA", 3: "SON"}

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
        # we expect y_data to be the RCM
        for ensemble in y_data:
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
        marker_props = enumerate((cycler(marker=['o', 'P', 'd']) * cycler(color=list('bgrcmy'))))

    max_val = 0
    min_val = 999999

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

        ax.text(x_val, y_val, i, fontsize=10)

    # plot a diagonal equivalence line
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

    # plot zero lines
    if min_val < 0:
        ax.axhline(ls=':', color='k', alpha=0.75)
        ax.axvline(ls=':', color='k', alpha=0.75)


def simpler_scatter(drive_data, downscale_data, labels, suffix=""):
    '''
    Simpler scatter that just plots distributions and scatter of a pair of simulations
    '''
    plt.figure(figsize=(19.2, 14.4))

    # construct axes
    ax_violins = plt.subplot2grid((1, 3), (0, 0))
    ax_scatter = plt.subplot2grid((1, 3), (0, 1), colspan=2, sharey=ax_violins)

    # make scatter
    labelled_scatter(drive_data, downscale_data, labels, ax_scatter)
    ax_scatter.set_xlabel('GCM')
    ax_scatter.set_ylabel('RCM')

    # make violins
    coloured_violin(drive_data, 1, ax_violins, 'lightgrey')
    coloured_violin(downscale_data, 2, ax_violins, 'lightgrey')

    # set x labels
    ax_violins.set_xticks(range(1, 3))
    ax_violins.set_xticklabels(['Global', 'Regional'])

    # also plot individual dots for each model..
    plot_points(drive_data, 1, ax_violins, color='r')
    plot_points(downscale_data, 2, ax_violins, color='r')

    var = get_var(cfg)
    plt.suptitle(f"{suffix} {var} change")

    # save plot
    plt.savefig(f"{cfg['plot_dir']}/simple_scatter_{suffix}.png", bbox_inches='tight')
    plt.close()


def mega_scatter(GCM_sc, RCM_sc1, RCM_sc2, CPM, all_GCM, all_RCM, labels1, labels2, suffix=''):
    '''
    A mega plot that shows distributions and scatter plots of GCM, RCM and CPM
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
    ax_violins.legend(
        h1 + h2, l1 + l2, bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=10)

    # create GCM / RCM / CPM violins / dots
    # GCMs go in position 1, RCMs position 2, CPMs position 3
    coloured_violin(all_GCM, 1, ax_violins, 'lightgrey')
    coloured_violin(all_RCM, 2, ax_violins, 'lightgrey')
    coloured_violin(CPM, 3, ax_violins, 'lightgrey')
    
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


def all_violins(datasets, labels, season):
    ''' 
    Plot violins and dots of the datasets
    datasets: list of datasets
    labels: list of labels
    '''
    # plot all of our data, plus Glen's method
    var = get_var(cfg)

    # if dataset has more than 10 points plot violin, otherwise don't
    # always plot individual points
    plt.figure(figsize=(19.2, 14.4))

    ax = plt.axes()

    for i, data in enumerate(datasets):
        if len(data) > 10:
            coloured_violin(data, i+1, ax)
        plot_points(data, i+1, ax)

    # now add on glen's data (if it exists)
    glens_data = get_glens_cdf(DOMAIN_FOR_CDF[cfg["reg_name"]["name"]], season, var)
    if glens_data is not None:
        i = i + 1
        labels.append("Glen's pdf")
        coloured_violin(glens_data, i+1, ax)

    # add ClimWIP data if we have it
    climWIP_weighted, climWIP_unweighted = get_ClimWIP_percentiles(cfg["reg_name"]["name"], season, var)
    if climWIP_weighted is not None:
        i = i + 1
        labels.append('CMIP5 ClimWIP w|u')
        coloured_violin(climWIP_weighted, i+0.75, ax)
        coloured_violin(climWIP_unweighted, i+1.25, ax)


    # set x labels
    plt.xticks(range(1, i+2), labels, rotation=45, ha="right")
    # ax.set_xticks(range(1, i+2))
    # ax.set_xticklabels(labels)

    # add zero line
    ax.axhline(ls=':', color='k', alpha=0.75)

    plt.suptitle(f"{season} {var} change")

    plt.savefig(f"{cfg['plot_dir']}/all_violins_{season}.png", bbox_inches='tight')
    plt.close()


def plot_points(points, x, ax, color='k'):
    for p in points:
        ax.plot(x, p, marker="o", fillstyle="none", color=color)


def coloured_violin(data, pos, ax, color=None):
    vparts = ax.violinplot(data, [pos], showmedians=True, quantiles=[0.05, 0.95])

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
    # set global plotting settings
    plt.rcParams.update({'font.size': 18})

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

    # This section of the code loads and organises the data to be ready for plotting
    logger.info("Loading data")
    # empty dict to store results
    projections = {}
    model_lists = {}
    cordex_drivers = []
    cordex_rcms = []
    # loop over projects
    for proj in projects:
        # we now have a list of all the data entries..
        # for CMIPs we can just group metadata again by dataset then work with that..
        models = group_metadata(projects[proj], "dataset")

        # empty dict for results
        if proj == 'non-cordex-rcm':
            proj = 'CORDEX'

        if proj == 'non-cmip5-gcm':
            proj = 'CMIP5'
            
        if proj not in projections.keys():
            projections[proj] = {}

        # loop over the models
        for m in models:
            if "CORDEX" in proj.upper():
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

        # remove any empty categories (i.e. UKCP18 which has been split into rcm and gcm)
        if projections[proj] == {}:
            del projections[proj]
    cordex_drivers = set(cordex_drivers)
    cordex_rcms = set(cordex_rcms)

    # reorganise and extract data for plotting
    plotting_dict = proj_dict_to_season_dict(projections)

    for season in plotting_dict.keys():
        # this section of the code does all the plotting..
        
        # mega scatter plot
        # need to prepare subsets of projects
        gcm_sc, rcm_sc1, labels1 = prepare_scatter_data(plotting_dict[season]['CMIP5'], plotting_dict[season]['CORDEX'], 'CORDEX')
        rcm_sc2, cpm_sc, labels2 = prepare_scatter_data(plotting_dict[season]['CORDEX'], plotting_dict[season]['cordex-cpm'], 'CPM')

        mega_scatter(
            gcm_sc, rcm_sc1, rcm_sc2, cpm_sc,
            list(plotting_dict[season]['CMIP5'].values()), list(plotting_dict[season]['CORDEX'].values()),
            labels1, labels2, f'{season}'
        )

        # simpler scatter for UKCP
        UKCP_g, UKCP_r, UKCP_labels = prepare_scatter_data(
            plotting_dict[season]['UKCP18 land-gcm'], plotting_dict[season]['UKCP18 land-rcm'], "UKCP18")
        simpler_scatter(UKCP_g, UKCP_r, UKCP_labels, f'UKCP_{season}')

        # side by side violins / dots for all models plus Glen's method...
        data_for_violins = [
            plotting_dict[season]['CMIP6'].values(),
            plotting_dict[season]['CMIP5'].values(),
            rcm_sc1, cpm_sc, UKCP_g, UKCP_r
            ]
        labels_for_violins = [
            'CMIP6', 'CMIP5', 'CORDEX', 'CPM', 'UKCP_g', 'UKCP_r'
            ]
        all_violins(data_for_violins, labels_for_violins, season)

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
            pickle_dict['CORDEX'] = list(plotting_dict[season]['CORDEX'].values())
            pickle_dict['UKCP18 land-gcm'] = plotting_dict[season]['UKCP18 land-gcm']
            pickle_dict['UKCP18 land-rcm'] = plotting_dict[season]['UKCP18 land-rcm']

            # save details of values used for plotting the boxplots
            save_anoms_txt(plotting_dict[season]['CMIP5'], f'{cfg["work_dir"]}/CMIP5_{season}.txt')
            save_anoms_txt(plotting_dict[season]['CORDEX'], f'{cfg["work_dir"]}/CORDEX_{season}.txt')
            save_anoms_txt(plotting_dict[season]['cordex-cpm'], f'{cfg["work_dir"]}/CPM_{season}.txt')

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
