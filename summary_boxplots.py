import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import iris
from ascend import shape
from glob import glob as glob
import os
import argparse

from esmvaltool.diagnostics import plotting
# works in SCITOOLS Default/next (2021-03-18)


def mask_wp2_atlas_data(cube, shp):
    # mask wp2 data using shape file

    # approach varies depending on if cube is downloaded from WP2 atlas
    # or direct from Glen's folders
    if cube.ndim == 4:
        # first get lat / lon mask over 2 dimensions
        xy_mask = np.logical_not(shp.cube_intersection_mask(cube[0,:,:,0]))
        # broadcast to 3d
        xyp_mask = np.broadcast_to(xy_mask[:,:,np.newaxis], cube.shape[1:])
        # broadcast to 4d
        cube_mask = np.broadcast_to(xyp_mask, cube.shape)
    else:
        # 3 dimensional (lat, lon, percentile)
        # get 2d mask
        xy_mask = np.logical_not(shp.cube_intersection_mask(cube[:,:,0]))
        # broadcast to 3d
        cube_mask = np.broadcast_to(xy_mask[:,:,np.newaxis], cube.shape)

    # apply to cube
    # combine with existing mask
    cube_mask = np.logical_or(cube_mask, cube.data.mask)
    cube.data.mask = cube_mask

    return cube


def load_wp2_atlas(method, var, area, season):
    # load netCDF file
    base_path = "/net/home/h02/tcrocker/code/EUCP_WP5_Lines_of_Evidence/weighting_data/WP2_atlas"

    # define region constraint if lat and lon supplied
    if type(area) == list:
        region = iris.Constraint(
            longitude=lambda x: area[0] <= x <= area[1],
            latitude=lambda x: area[2] <= x <= area[3]
        )

    bxp_obs = []
    for data in ["cons", "uncons"]:
        fname = f"{base_path}/atlas_EUCP_{method}_{data}_{var}.nc"
        cube = iris.load_cube(fname)

        # extract shape / region
        if type(area) == list:
            cube = cube.extract(region)
        else:
            cube = mask_wp2_atlas_data(cube, area)

        if season == "JJA":
            # use first time point (JJA)
            cube = cube[0]
        elif season == "DJF":
            cube = cube[1]
        else:
            raise ValueError("Only JJA and DJF available.")

        # area average
        cube.coord("latitude").units = "degrees"
        cube.coord("latitude").guess_bounds()
        cube.coord("longitude").units = "degrees"
        cube.coord("longitude").guess_bounds()
        grid_areas = iris.analysis.cartography.area_weights(cube)
        cube_mean = cube.collapsed(["latitude", "longitude"], iris.analysis.MEAN, weights=grid_areas)

        # create boxplot stats object
        if data == "cons":
            label = method
        else:
            label = None

        bxp_stats = {
            "whislo": cube_mean.extract(iris.Constraint(percentile=10)).data.item(),
            "q1": cube_mean.extract(iris.Constraint(percentile=25)).data.item(),
            "med": cube_mean.extract(iris.Constraint(percentile=50)).data.item(),
            "q3": cube_mean.extract(iris.Constraint(percentile=75)).data.item(),
            "whishi": cube_mean.extract(iris.Constraint(percentile=90)).data.item(),
            "label": label
        }

        bxp_obs.append(bxp_stats)

    return bxp_obs


def load_wp2_glen(var, area, season):
    # Load WP2 constraint data from files in Glen's user space.
    # define constraint if using a rectangle
    if type(area) == list:
        region = iris.Constraint(
            longitude=lambda x: area[0] <= x <= area[1],
            latitude=lambda x: area[2] <= x <= area[3]
        )

    results = []
    season = season.lower()
    for d_type in ["all", "prior"]:
        file_name = f"/data/users/hadgh/eucp/data/v13/d23map/{var}Anom/{season}/{var}Anom_rcp85_eu_300km_W{d_type}-N600000-P21_cdf_b9514_20y_{season}_20401201-20601130.nc"

        cube = iris.load_cube(file_name)

        # extract shape / region
        if type(area) == list:
            cube = cube.extract(region)
        else:
            cube = mask_wp2_atlas_data(cube, area)

        grid_areas = iris.analysis.cartography.area_weights(cube)
        cube_mean = cube.collapsed(["latitude", "longitude"], iris.analysis.MEAN, weights=grid_areas)

        # create boxplot stats object
        if d_type == "all":
            label = "UKCP constraint"
        else:
            label = None

        bxp_stats = {
            "whislo": cube_mean.extract(iris.Constraint(percentile=10)).data.item(),
            "q1": cube_mean.extract(iris.Constraint(percentile=25)).data.item(),
            "med": cube_mean.extract(iris.Constraint(percentile=50)).data.item(),
            "q3": cube_mean.extract(iris.Constraint(percentile=75)).data.item(),
            "whishi": cube_mean.extract(iris.Constraint(percentile=90)).data.item(),
            "label": label
        }

        results.append(bxp_stats)

    return results


def load_esmval_gridded_data(recipe, type, area, season):
    # load gridded anomaly data that has been produced by the ESMValTool recipe
    # recipe: name of recipe run that contains data, e.g. recipe_GCM_and_RCM_pan_EU_20211214_170431
    # type: type of data to load, e.g. cmip5, cmip6, cordex, cpm, UKCP18 land-gcm, UKCP18 land-rcm
    # area: area to compute area averages for. As a shape file for now..
    # season: season to load data for, djf, mam, jja or son
    # return an array of the computed area means for each data file (model) found

    # first get path to where all the files will be
    season = season.upper()
    input_path = f"/net/home/h02/tcrocker/code/EUCP_WP5_Lines_of_Evidence/esmvaltool/esmvaltool_output/{recipe}/work/gridded_anoms/main/{season}/"

    # setup land mask shape
    lsm = shape.load_shp(
        '/home/h02/tcrocker/code/EUCP_WP5_Lines_of_Evidence/shape_files/ne_110m_land/ne_110m_land.shp'
        ).unary_union()
    # need to reduce size of lsm to avoid bug in ascend
    # see https://github.com/MetOffice/ascend/issues/8
    corners = [(-30, 20), (-30, 75), (50, 75), (50, 20)]
    rectangle = shape.create(corners, {'shape': 'rectangle'}, 'Polygon')
    lsm = lsm.intersection(rectangle)

    # process each file for the required datatype
    fnames = glob(f"{input_path}/{type}_*.nc")

    values = {}

    for fname in fnames:
        # ignore diff or mean files
        if any([s in fname for s in ['diff', 'mean']]):
            continue

        # load data
        cube = iris.load_cube(fname)

        # check proportion of area covered by valid cube data
        # ignore data if this is less than 1
        # i.e. there is not data for the whole of the shape
        area_cov = check_data_shape_intersection(cube, area)
        if area_cov < 0.9:
            print(f"WARNING: Data in {os.path.basename(fname)} only covers {area_cov * 100}% of supplied area")
            continue
        if area_cov < 1.0:
            print(f"WARNING: Data in {os.path.basename(fname)} only covers {area_cov * 100}% of supplied area")

        # now mask data
        # TODO - maybe turn whether / how to do this into an argument..
        # could also achieve maskng via preprocessor functions from esmvaltool 
        # if it is necesary to run this outside the met office where ascend is not available
        mask = lsm.intersection(area)
        mask.mask_cube_inplace(cube)
        # need some sort of logic to test if the cube contains data for all of the supplied area,
        # if not we should reject it...

        # and compute weighted area average
        # this is using weighted area weights from Ascend
        awts = mask.cube_2d_weights(cube, False)
        nwts = iris.analysis.cartography.area_weights(cube)

        wts = awts.data * nwts
        area_mean = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=wts)

        # get model name etc. from filename
        # want everything between the type and anom i.e.
        # type_<this bit>_anom_season.nc
        mname = os.path.basename(fname).split(f"{type}_")[1]
        mname = mname.split("_anom")[0]

        values[mname] = area_mean.data.item()

    return values


def check_data_shape_intersection(cube, shp):
    # return proportion of grid boxes in shp (when put on same grid as cube)
    # that have valid corresponding data in cube

    # check cube has a coord system, if not add one
    if cube.coord('longitude').coord_system is None:
        cube.coord('longitude').coord_system = iris.coord_systems.GeogCS(6371229.0)
        cube.coord('latitude').coord_system = iris.coord_systems.GeogCS(6371229.0)

    # first check intersection of cube bounding box with shape
    # i.e. if the shape doesn't lie entirely inside the cube
    # bounds we can return false immediately
    int = shape.cube_bbox_shape_intersection(cube, shp)
    diff = shp.difference(int)
    if diff.data.area > 0.0:
        return 0.0

    # now check intersection of cube data mask with shape
    shape_mask = shp.cube_2d_weights(cube).data
    data_mask = np.invert(cube.data.mask)

    data_shape_intersection = shape_mask * data_mask

    valid_boxes = np.sum(data_shape_intersection) / np.sum(shape_mask)

    return valid_boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe")
    parser.add_argument("variable")
    parser.add_argument("season")
    parser.add_argument("area")
    parser.add_argument("gcm_anoms_recipe")
    args = parser.parse_args()

    recipe = args.recipe
    var = args.variable
    season = args.season
    area = args.area
    anoms_files = f"/net/home/h02/tcrocker/code/EUCP_WP5_Lines_of_Evidence/esmvaltool/esmvaltool_output/{args.gcm_anoms_recipe}/work/global_tas_anomalies/anomalies/"

    if area == "boe":
        area = shape.create([(-5,42), (30,42), (30,52), (-5,52)], {'shape': 'rectangle', 'NAME': 'boe'}, 'Polygon')
    else:
        print(f"Loading {area} shape")
        area = shape.load_shp('/net/home/h02/tcrocker/code/EUCP_WP5_Lines_of_Evidence/shape_files/EUCP_WP3_domains/EUCP_WP3_domains.shp', name=area)[0]

    # read data and place in a dataframe
    plot_df = pd.DataFrame(columns=["model", "value", "project", "data type"])

    # easiest way seems to be to append a row at a time
    for proj in ["CMIP5", "CMIP6", "CORDEX", "cordex-cpm", "UKCP18 land-gcm", "UKCP18 land-rcm"]:
        data_dict = load_esmval_gridded_data(recipe, proj, area, season)
        for k, v in data_dict.items():
            row = [k, v, proj, "standard"]
            plot_df.loc[len(plot_df)] = row

    # load gcm temp anoms
    anoms = {}
    for p in ["CMIP5", "CMIP6", "UKCP18"]:
        csv_file = f"{anoms_files}{p}_global_tas_anom.csv"
        anoms[p] = pd.read_csv(csv_file, header=None, dtype={0:'string'})

        # now loop over values loaded and construct the temp weighted anomaly
        for row in anoms[p].iterrows():
            # get the appropriate model value
            if p == "UKCP18":
                proj = "UKCP18 land-gcm"
            else:
                proj = p
            m_val = plot_df[(plot_df["model"] == row[1][0]) & (plot_df["project"] == proj)]["value"].values
            if len(m_val) > 1:
                raise ValueError(f"Found multiple entries for {row[1][0]}")

            m_val = m_val[0]

            # compute the weighted anomaly
            # i.e. we divide by the global anomaly to get degrees of warming
            # per 1 degree of global warming
            weighted_anom = m_val / row[1][1]

            # now add new row in the datframe with the weighted info
            new_row = [row[1][0], weighted_anom, proj, "weighted"]
            plot_df.loc[len(plot_df)] = new_row


    # List of models for Romania case study
    # niculita_model_list = [
    #     'RCA4 MPI-M-MPI-ESM-LR',
    #     'RCA4 MOHC-HadGEM2-ES',
    #     'RCA4 ICHEC-EC-EARTH',
    #     'RCA4 CNRM-CERFACS-CNRM-CM5',
    #     'REMO2009 MPI-M-MPI-ESM-LR',
    #     'RACMO22E MOHC-HadGEM2-ES',
    #     'RACMO22E ICHEC-EC-EARTH', 
    #     'HIRHAM5 ICHEC-EC-EARTH',
    #     ]

    # list of models with evolving aerosols. See table B2 from:
    # Gutiérrez, C., Somot, S., Nabat, P., Mallet, M., Corre, L., Van Meijgaard, E., et al. (2020). Future evolution of surface solar radiation and photovoltaic potential in Europe: investigating the role of aerosols. Environmental Research Letters, 15(3). https://doi.org/10.1088/1748-9326/ab6666
    aerosol_model_list = []
    for c in plot_df[plot_df["project"] == "CORDEX"]["model"]:
        if any([s in c for s in ['RACMO22E', 'ALADIN', 'HadREM3']]):
            aerosol_model_list.append(c)

    # case_study_model_list = aerosol_model_list
    # case_study_model_list = niculita_model_list
    case_study_model_list = []

    # This dictionary maps CPM string to a RCM GCM string
    cpm_drivers = {
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

    # List of CPM drivers from CORDEX to know which to plot as triangles
    cpm_driver_list = []
    for n in plot_df[plot_df["project"] == "cordex-cpm"]["model"]:
        cpm_driver_list.append(cpm_drivers[n.split()[0]])

    # CMIP5 CORDEX drivers can be inferred directly from CORDEX model names
    cordex_driver_list = list(
        set(
            [plotting.remove_institute_from_driver(n.split(' ')[1]) for n in plot_df[plot_df["project"] == "CORDEX"]["model"]]
        )
    )

    driving_models = {
        "CORDEX": cordex_driver_list,
        "CPM": cpm_driver_list,
        "UKCP": list(plot_df[plot_df["project"] == "UKCP18 land-rcm"]["model"]),
        "case study": case_study_model_list
    }

    # load WP2 atlas constraint data
    constraint_data = {}
    for m in plotting.WP2_METHODS.keys():
        constraint_data[m] = load_wp2_atlas(m, var, area, season)

    # also load Glen's UKCP data
    constraint_data["UKMO_CMIP6_UKCP"] = load_wp2_glen(var, area, season)

    # now plot
    # panel plot
    plotting.panel_boxplot(plot_df, constraint_data, driving_models, area, season, var)

    # scatter plot
    # need to prepare data
    x = plot_df[(plot_df["model"].isin(driving_models["CORDEX"])) & (plot_df["data type"] == "standard")][["model", "value"]]
    x = pd.Series(x.value.values, index=x.model).to_dict()
    y = plot_df[(plot_df["project"] == "CORDEX") & (plot_df["data type"] == "standard")][["model", "value"]]
    y = pd.Series(y.value.values, index=y.model).to_dict()
    cmip_x, cordex_y, cordex_labels = plotting.prepare_scatter_data(x, y, "CORDEX")

    if len(driving_models["CPM"]) > 0:
        x = plot_df[(plot_df["model"].isin(driving_models["CPM"])) & (plot_df["data type"] == "standard")][["model", "value"]]
        x = pd.Series(x.value.values, index=x.model).to_dict()
        y = plot_df[(plot_df["project"] == "cordex-cpm") & (plot_df["data type"] == "standard")][["model", "value"]]
        y = pd.Series(y.value.values, index=y.model).to_dict()
        cordex_x, cpm_y, cpm_labels = plotting.prepare_scatter_data(x, y, "CPM")
        plotting.mega_scatter(
            cmip_x, cordex_y, cordex_x, cpm_y,
            plot_df[(plot_df["project"] == "CMIP5") & (plot_df["data type"] == "standard")]["value"].to_list(),
            plot_df[(plot_df["project"] == "CORDEX") & (plot_df["data type"] == "standard")]["value"].to_list(),
            cordex_labels, cpm_labels, f"{area.attributes['NAME']} {season} {var}",
            "/home/h02/tcrocker/code/EUCP_WP5_Lines_of_Evidence/esmvaltool/plots"
        )
    else:
        plotting.simpler_scatter(
            cmip_x, cordex_y, cordex_labels,
            f"{area.attributes['NAME']} {season} {var}",
            "/home/h02/tcrocker/code/EUCP_WP5_Lines_of_Evidence/esmvaltool/plots"
        )


if __name__ == '__main__':
    main()
