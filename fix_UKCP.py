# Script to take UKCP land-gcm files,
# and apply fixes to make them CMOR compliant for ESMValTool
# 1st argument, input files path e.g.
# /project/spice/ukcp18/GA7/COUPLED/ukcp18_monthly/land-gcm/global/60km/
# 2nd argument, output files path e.g.
# /project/ciid/projects/EUCP/wp5/UKCP/land-gcm/
import iris

import os
import argparse
import re
import logging


def fix_gcm_file(f):
    # Fix the netCDF file located at f
    # return a fixed cube
    # load
    c = iris.load_cube(f)

    # squeeze out ensemble coord
    new_c = iris.util.squeeze(c)

    # fix lon and lat names
    new_c.coord("longitude").var_name = "lon"
    new_c.coord("latitude").var_name = "lat"

    # remove unneeded coords
    new_c.remove_coord("month_number")
    new_c.remove_coord("year")
    new_c.remove_coord("yyyymm")

    # add height coord for near surface temp
    if new_c.var_name == "tas":
        height_c = iris.coords.Coord(1.5, "height", units="m")
        new_c.add_aux_coord(height_c)

    return new_c


logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("in_path", help="Input file to take files from")
parser.add_argument("out_path", help="Output path to save to")
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
            if re.search(r"^(tas|pr).*\.nc$", f):
                save_fname = os.path.join(
                    args.out_path, os.path.relpath(dpath, args.in_path), f
                )
                logging.info(
                    f"Fixing file: {os.path.join(dpath, f)} and saving to: {save_fname}"
                )
                if not args.dry_run:
                    fixed_cube = fix_gcm_file(os.path.join(dpath, f))
                    # save
                    iris.save(fixed_cube, save_fname)

if args.dry_run:
    logging.info("Dry run. Didn't do anything")