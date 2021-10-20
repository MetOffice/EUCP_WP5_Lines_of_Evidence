# Calculate anomalies, and then plot our model groups in a boxplot

from esmvaltool.diag_scripts.shared import (
    run_diagnostic,
    group_metadata,
    select_metadata,
    extract_variables,
)
from esmvalcore.preprocessor import regrid

import iris
import iris.quickplot as qplt
import iris.plot as iplt
import cartopy
import cartopy.crs as ccrs

import os
import logging
import re

import matplotlib.pyplot as plt

logger = logging.getLogger(os.path.basename(__file__))

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


# recursive retrieval of all values in a dictionary
def NestedDictValues(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from NestedDictValues(v)
        else:
            yield v


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
            # if the result is a scalar cube, just store the value
            # else store the whole cube
            if data.ndim == 0:
                out_data[k] = data.data.item()
            else:
                out_data[k] = data
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

    # ensure longitude coord is on -180 to 180 range
    try:
        anomaly = anomaly.intersection(longitude=(-180.0, 180.0))
    except ValueError:
        # remove and re add bounds to attempt to fix
        anomaly.coord('longitude').bounds = None
        anomaly.coord('longitude').guess_bounds()
        anomaly = anomaly.intersection(longitude=(-180.0, 180.0))

    return anomaly


def compute_multi_model_stats(cl, agg):
    # given cubes, apply the supplied iris aggregator to it and return the result

    cl = iris.cube.CubeList(cl)

    # sort attributes
    iris.util.equalise_attributes(cl)

    # add a dimension to merge along
    for i, c in enumerate(cl):
        if c.coords('multi-model') == []:
            concat_dim = iris.coords.AuxCoord(i, var_name='multi-model')

            c.add_aux_coord(concat_dim)

        # other things that can prevent merging
        for coord in c.coords():
            coord.long_name = None
            coord.attributes = None

        if c.coords('height'):
            c.remove_coord('height')

    merged_cube = cl.merge_cube()

    # now compute the stats
    stats = merged_cube.collapsed('multi-model', agg)

    return stats


def plot_map(pdata, extent, var, ax, legend=False):
    ax.set_extent(extent)
    # set scales
    if var in ("pr", "pr_diff"):
        vmn = -30
        vmx = 30
        # cmap = "brewer_RdYlBu_11"
        cmap = "RdBu"
    elif var == "tas_diff":
        vmn = -1
        vmx = 1
        cmap = "bwr"
    else:
        vmn = 0.5
        vmx = 5
        cmap = "brewer_YlOrRd_09"
        # cmap = "magma_r"
    # ensure longitude coordinates straddle the meridian for GCM origin data
    if pdata.coord("longitude").ndim == 1:
        # TODO This will probably cause issues if it's ever run with data
        # that straddles the dateline, so a check should be added.
        try:
            plot_cube = pdata.intersection(longitude=(-180.0, 180.0))
            plot_cube.coord("longitude").circular = False
        except ValueError:
            plot_cube = pdata
            plot_cube.coord('longitude').bounds = None
            plot_cube = plot_cube.intersection(longitude=(-180.0, 180.0))
            plot_cube.coord("longitude").circular = False
            plot_cube.coord('longitude').guess_bounds()
    else:
        plot_cube = pdata
    if legend:
        cmesh = qplt.pcolormesh(plot_cube, vmin=vmn, vmax=vmx, cmap=cmap)
    else:
        cmesh = iplt.pcolormesh(plot_cube, vmin=vmn, vmax=vmx, cmap=cmap)
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle=":")

    return cmesh


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
        
        proj_key = proj
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
                    
                    # fix shorthand driver names
                    if d == 'HadGEM':
                        d = 'MOHC-HadGEM2-ES'
                    elif d == 'MPI':
                        d = 'MPI-M-MPI-ESM-LR'
                    
                    if proj == "CORDEX":
                        cordex_drivers.append(d)
            elif proj == "UKCP18":
                # go deeper to deal with ensembles and datasets
                # split UKCP into seperate GCM and RCM
                proj_key = f"UKCP18 {m}"
                ensembles = group_metadata(models[m], "ensemble")
                projections[proj_key] = dict.fromkeys(ensembles.keys())
                for ens in ensembles:
                    logging.info(f"Calculating anomalies for {proj_key} {ens}")
                    anoms = get_anomalies(
                        ensembles[ens], rel_change
                    )
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

    # create two extra subsets containing CORDEX drivers, and CPM drivers
    projections['CORDEX_drivers'] = {}
    cmip5_driving_models = []
    for m in cordex_drivers:
        cmip5_driving_models.append(remove_institute_from_driver(m))
    
    for m in projections['CMIP5']:
        if m in cmip5_driving_models:
            projections['CORDEX_drivers'][m] = projections['CMIP5'][m]

    projections['CPM_drivers'] = {}
    for rcm in projections['CORDEX']:
        for d in projections['CORDEX'][rcm]:
            if f'{rcm} {d}' in list(CPM_DRIVERS.values()):
                projections['CPM_drivers'][f'{rcm} {d}'] = projections['CORDEX'][rcm][d]

    # compute multi model means
    for p in projections:
        mm_mean = compute_multi_model_stats(
            list(NestedDictValues(projections[p])), iris.analysis.MEAN
        )
        projections[p]['mean'] = mm_mean

    # compute regridded versions for CORDEX and CPMs
    for p in projections:
        grid = None
        if p == 'CORDEX':
            grid = projections['CORDEX_drivers']['mean']
            scheme = 'area_weighted'
        elif p == 'cordex-cpm':
            grid = projections['CPM_drivers']['mean']
            scheme = 'area_weighted'
        
        if grid:
            src = projections[p]['mean']
            regrid_mean = regrid(src, grid, scheme)
            projections[p]['mean_rg'] = regrid_mean

    # compute regrid diffs
    for p in projections:
        if p == 'CORDEX':
            diff = projections[p]['mean_rg'] - projections['CORDEX_drivers']['mean']
            projections[p]['diff_rg'] = diff
        elif p == 'cordex-cpm':
            diff = projections[p]['mean_rg'] - projections['CPM_drivers']['mean']
            projections[p]['diff_rg'] = diff

    # this section of the code does the plotting..
    # we now have all the projections in the projections dictionary

    # now lets plot them
    # first we need to process the dictionary, and move the data into a list of vectors
    # the projections object is the key one that contains all our data..
    seasons = {0: "DJF", 1: "MAM", 2: "JJA", 3: "OND"}
    logger.info("Plotting")
    extent = (
        cfg["domain"]["start_longitude"] - 2,
        cfg["domain"]["end_longitude"] + 2,
        cfg["domain"]["start_latitude"] - 2,
        cfg["domain"]["end_latitude"] + 2,
    )
    for s in seasons.keys():
        # make directory
        try:
            os.mkdir(f"{cfg['plot_dir']}/{seasons[s]}")
        except FileExistsError:
            pass
        for p in projections:
            pdata = process_projections_dict(projections[p], s)

            for m in pdata:
                # dont plot driving model data twice.
                if '_drivers' in p:
                    if m != 'mean':
                        continue

                title = f"{p} {m} {seasons[s]} {var} change"
                plt.figure(figsize=(12.8, 9.6))
                ax = plt.axes(projection=ccrs.PlateCarree())
                plot_map(pdata[m], extent, var, ax, True)
                plt.title(title)
                logging.info(f'Saving plot for {p} {m} {s}')
                plt.savefig(
                    f"{cfg['plot_dir']}/{seasons[s]}/{p}_{m}_map_{seasons[s]}.png"
                )
                plt.close()

        # now make panel plots for the mean data
        scon = iris.Constraint(season_number=s)
        logging.info(f'Making {seasons[s]} panel plot')
        plt.figure(figsize=(12.8, 9.6))
        # plots should include. All CMIP5, CORDEX drivers, CORDEX, CPM drivers, CPM.
        ax = plt.subplot(331, projection=ccrs.PlateCarree())
        cmesh = plot_map(projections['CMIP5']['mean'].extract(scon), extent, var, ax)
        plt.title('CMIP5')

        ax = plt.subplot(334, projection=ccrs.PlateCarree())
        plot_map(projections['CORDEX_drivers']['mean'].extract(scon), extent, var, ax)
        plt.title('CORDEX driving models')

        ax = plt.subplot(335, projection=ccrs.PlateCarree())
        plot_map(projections['CORDEX']['mean'].extract(scon), extent, var, ax)
        plt.title('CORDEX')

        # plot diff of CORDEX to CMIP
        ax = plt.subplot(336, projection=ccrs.PlateCarree())
        cmesh_diff = plot_map(projections['CORDEX']['diff_rg'].extract(scon), extent, f'{var}_diff', ax)
        plt.title('CORDEX - CMIP5 diff')

        ax = plt.subplot(337, projection=ccrs.PlateCarree())
        plot_map(projections['CPM_drivers']['mean'].extract(scon), extent, var, ax)
        plt.title('CPM driving models')

        ax = plt.subplot(338, projection=ccrs.PlateCarree())
        plot_map(projections['cordex-cpm']['mean'].extract(scon), extent, var, ax)
        plt.title('CPM')

        # plot diff of CPM to CORDEX
        ax = plt.subplot(339, projection=ccrs.PlateCarree())
        plot_map(projections['cordex-cpm']['diff_rg'].extract(scon), extent, f'{var}_diff', ax)
        plt.title('CPM - CORDEX diff')

        # add legends
        ax = plt.subplot(332)
        ax.axis("off")
        plt.colorbar(cmesh, orientation="horizontal")

        ax = plt.subplot(333)
        ax.axis("off")
        plt.colorbar(cmesh_diff, orientation="horizontal")

        plt.suptitle(f'{seasons[s]} {var} change')
        plt.savefig(
            f"{cfg['plot_dir']}/{seasons[s]}/all_means_map_{seasons[s]}.png"
        )

    # print all datasets used
    print("Input models for plots:")
    for p in model_lists.keys():
        print(f"{p}: {len(model_lists[p])} models")
        print(model_lists[p])
        print("")


if __name__ == "__main__":
    with run_diagnostic() as cfg:
        main(cfg)
