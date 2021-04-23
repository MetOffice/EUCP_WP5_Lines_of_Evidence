#!/usr/bin/env python
# Script to take UKCP land-gcm files,
# and apply fixes to make them CMOR compliant for ESMValTool
# 1st argument, input files path e.g.
# /project/spice/ukcp18/GA7/COUPLED/ukcp18_monthly/land-gcm/global/60km/
# 2nd argument, output files path e.g.
# /project/ciid/projects/EUCP/wp5/UKCP/land-gcm/

# rsync to JASMIN. Something like this:
# rsync -anv /project/ciid/projects/EUCP/wp5/UKCP18/ tcrocker@xfer1.jasmin.ac.uk:~/EUCP/UKCP18/
import iris
from catnip.preparation import add_aux_unrotated_coords

import os
import argparse
import re
import logging


def remove_unneeded_coords(cube):
    for c in ["month_number", "year", "yyyymm"]:
        coord = cube.coords(c)
        if coord:
            cube.remove_coord(c)

    return cube


def fix_lon_lat_names(cube):
    cube.coord("longitude").var_name = "lon"
    cube.coord("latitude").var_name = "lat"

    return cube


def fix_gcm_file(f):
    # Fix the netCDF file located at f
    # return a fixed cube
    # load
    c = iris.load_cube(f)

    # squeeze out ensemble coord
    new_c = iris.util.squeeze(c)

    # fix lon and lat names
    new_c = fix_lon_lat_names(new_c)

    # remove unneeded coords
    new_c = remove_unneeded_coords(new_c)

    # variable specific fixes
    if new_c.var_name == "tas":
        height_c = iris.coords.Coord(1.5, "height", units="m")
        new_c.add_aux_coord(height_c)
        new_c.convert_units("K")
    elif new_c.var_name == "pr":
        # check units and convert if necessary
        if new_c.units == "mm/day":
            new_c.data = new_c.data / 86400
            new_c.units = "kg m-2 s-1"

    return new_c


def fix_rcm_file(f):
    # Fix the UKCP rcm netCDF file located at f
    # return a fixed cube

    # load
    c = iris.load_cube(f)

    # squeeze out ensemble coord
    new_c = iris.util.squeeze(c)

    # add lat and lon coords
    new_c = add_aux_unrotated_coords(new_c)
    new_c.coord("latitude").standard_name = "latitude"
    new_c.coord("longitude").standard_name = "longitude"
    new_c = fix_lon_lat_names(new_c)

    # remove unneeded coords
    new_c = remove_unneeded_coords(new_c)

    # variable specific fixes
    if new_c.var_name == "tas":
        height_c = iris.coords.Coord(1.5, "height", units="m")
        new_c.add_aux_coord(height_c)
        new_c.convert_units("K")
    elif new_c.var_name == "pr":
        # check units and convert if necessary
        if new_c.units == "mm/day":
            new_c.data = new_c.data / 86400
            new_c.units = "kg m-2 s-1"

    return new_c


logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("in_path", help="Input file to take files from")
parser.add_argument("out_path", help="Output path to save to")
parser.add_argument("type", help="UKCP file type to fix", choices=["gcm", "rcm"])
parser.add_argument("frequency", help="Frequency type to fix", choices=["day", "mon"])
parser.add_argument(
    "-n",
    "--dry_run",
    help="Dry run, show files processed but don't do anything",
    action="store_true",
)

args = parser.parse_args()

for dpath, dnames, fnames in os.walk(args.in_path):
    logging.debug(f"{dpath} {dnames} {fnames}")
    if fnames:
        # only work with tas and pr files
        for f in fnames:
            regex = rf"^(tas|pr).*{args.frequency}.*\.nc$"
            if re.search(regex, f):
                save_fname = os.path.join(
                    args.out_path, os.path.relpath(dpath, args.in_path), f
                )
                logging.info(
                    f"Fixing file: {os.path.join(dpath, f)} and saving to: {save_fname}"
                )
                if not args.dry_run:
                    if args.type == "gcm":
                        fixed_cube = fix_gcm_file(os.path.join(dpath, f))
                    else:
                        fixed_cube = fix_rcm_file(os.path.join(dpath, f))
                    # save
                    os.makedirs(os.path.dirname(save_fname), exist_ok=True)
                    iris.save(fixed_cube, save_fname)

if args.dry_run:
    logging.info("Dry run. Didn't do anything")
