#!/usr/bin/env python

# Script to create sftlf fx files for UKCP data

import iris
import numpy as np

from catnip.preparation import add_aux_unrotated_coords, rim_remove

TYPE = "GCM"

if TYPE == "RCM":
    # load original RCM landfrac ancil
    UKCP_rcm_lf = iris.load_cube(
        "/project/ciid/projects/EUCP/wp5/UKCP18/land-rcm/eur_12km/land_sea_mask/igbp/qrparm.landfrac"
    )

    # add regular coords, rim remove and fix units
    UKCP_rcm_lf = add_aux_unrotated_coords(UKCP_rcm_lf)
    UKCP_rcm_lf = rim_remove(UKCP_rcm_lf, 13)
    UKCP_rcm_lf.convert_units("%")

    # save
    iris.save(
        UKCP_rcm_lf,
        "/project/ciid/projects/EUCP/wp5/UKCP18/land-rcm/eur_12km/rcp85/00/sftlf/fx/latest/sftlf_rcp85_land-rcm_eur_12km_00_fx.nc",
    )
    # then need to run these NCO commands to fix last bits of metadata. This could be done with iris.
    # ncrename -v land_area_fraction,sftlf /project/ciid/projects/EUCP/wp5/UKCP18/land-rcm/eur_12km/rcp85/00/sftlf/fx/latest/sftlf_rcp85_land-rcm_eur_12km_00_fx.nc
    #
else:
    orog = iris.load_cube(
        "/project/ciid/projects/EUCP/wp5/UKCP18/land-gcm/qrparm.orog.nc"
    )

    orog_sq = iris.util.squeeze(orog)
    orog_sq.remove_coord("Surface")
    orog_sq.remove_coord("t")

    lsm = orog_sq.copy()
    lsm.data = np.ma.where(lsm.data > 0, 1, 0)
    lsm.units = 1

    lsm.standard_name = "land_binary_mask"
    lsm.convert_units("%")

    lsm.attributes = None

    iris.save(
        lsm,
        "/project/ciid/projects/EUCP/wp5/UKCP18/land-gcm/global_60km/rcp85/00/sftlf/fx/latest/sftlf_rcp85_land-gcm_global_60km_00_fx.nc",
    )

    # Also need to run some nco commands to fix attributes since we've converted an orog field to sftlf these could be moved to iris commands
    # ncrename -v ht,sftlf /project/ciid/projects/EUCP/wp5/UKCP18/land-gcm/global_60km/rcp85/00/sftlf/fx/latest/sftlf_rcp85_land-gcm_global_60km_00_fx.nc
    # ncatted -a standard_name,sftlf,o,c,land_area_fraction /project/ciid/projects/EUCP/wp5/UKCP18/land-gcm/global_60km/rcp85/00/sftlf/fx/latest/sftlf_rcp85_land-gcm_global_60km_00_fx.nc
    # ncatted -a long_name,sftlf,o,c,"Percentage of the Grid Cell Occupied by Land (Including Lakes)" /project/ciid/projects/EUCP/wp5/UKCP18/land-gcm/global_60km/rcp85/00/sftlf/fx/latest/sftlf_rcp85_land-gcm_global_60km_00_fx.nc
