# quick diagnostic to just calculate and save anomalies
from esmvaltool.diag_scripts.shared import (
    run_diagnostic,
    group_metadata,
    select_metadata,
    get_diagnostic_filename,
    extract_variables
)

import iris
import os
import logging

from iris.exceptions import CoordinateNotFoundError

logger = logging.getLogger(os.path.basename(__file__))


def get_anomalies(ds_list, relative=False):
    # determine historic and future periods
    start_years = list(group_metadata(ds_list, "start_year"))
    base_clim_start = min(start_years)
    fut_clim_start = max(start_years)

    # construct baseline
    base_metadata = select_metadata(ds_list, start_year=base_clim_start)
    base_file = base_metadata[0]["filename"]
    base_cube = iris.load_cube(base_file)
    try:
        base_cube.remove_coord('clim_season')
    except CoordinateNotFoundError:
        pass
    try:
        base_cube.remove_coord('season_year')
    except CoordinateNotFoundError:
        pass

    # get future
    fut_metadata = select_metadata(ds_list, start_year=fut_clim_start)
    fut_file = fut_metadata[0]["filename"]
    fut_cube = iris.load_cube(fut_file)
    try:
        fut_cube.remove_coord('clim_season')
    except CoordinateNotFoundError:
        pass
    try:
        fut_cube.remove_coord('season_year')
    except CoordinateNotFoundError:
        pass

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


def main(cfg):
    # The config object is a dict of all the metadata from the pre-processor
    logger.debug(cfg)

    var = list(extract_variables(cfg).keys())
    assert len(var) == 1
    var = var[0]

    if var == "pr":
        rel_change = True
    else:
        rel_change = False

    projects = group_metadata(cfg["input_data"].values(), "project")

    for p in projects:
        # then group over the datasets, 
        # if we are dealing with RCM / CPM data also need to worry about drivers
        datasets = group_metadata(projects[p], "dataset")

        if p == 'non-cordex-rcm':
            p = 'CORDEX'
            
        if p == 'non-cmip5-gcm':
            p = 'CMIP5'

        for m in datasets:
            # now subset by driver if it exists
            if "driver" in datasets[m][0].keys():
                drivers = group_metadata(datasets[m], "driver")

                # now loop over these and get anomalies
                for d in drivers:
                    if len(drivers[d]) != 2:
                        logger.warning(f'Expect exactly 2 data sources for {p}_{m}_{d} found {len(drivers[d])}')
                        continue
                    anom = get_anomalies(drivers[d], rel_change)
                    iris.save(
                        anom, get_diagnostic_filename(f"{p}_{m}_{d}_anom", cfg)
                    )
            else:
                if len(datasets[m]) != 2:
                    logger.warning(f'Expect exactly 2 data sources for {p}_{m} found {len(datasets[m])}')
                    continue
                anom = get_anomalies(datasets[m], rel_change)
                iris.save(
                    anom, get_diagnostic_filename(f"{p}_{m}_anom", cfg)
                )


if __name__ == "__main__":
    with run_diagnostic() as cfg:
        main(cfg)
