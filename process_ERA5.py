# script to process ERA5 files downloaded from Copernicus CDS

import iris


# Mapping of long name to appropriate short name and units
NAME_DICT = {
    "Total precipitation": ("pr", "kg m-2 s-1"),
    "Surface thermal radiation downwards": ("rlds", "W m-2"),
    "2 metre temperature": ("tas", "K"),
    "Surface solar radiation downwards": ("rsds", "W m-2"),
    "Surface net solar radiation": ("rsns", "W m-2"),
    "Mean sea level pressure": ("psl", "Pa"),
}

# original file downloaded from COPERNICUS
SOURCE = "download.nc"
OUT_FOLDER = "observations/ERA5/"


# load the file
cubes = iris.load(SOURCE)

vercon = iris.Constraint(expver=1)
cubes = cubes.extract(vercon)

# process each of the variables found
for cube in cubes:
    # fix names
    var_name = NAME_DICT[cube.long_name][0]
    cube.var_name = var_name

    # # fix units
    # Fixing units is not necessary as ESMValTool will do it for us
    # new_unit = NAME_DICT[cube.long_name][1]
    # try:
    #     cube.convert_units(new_unit)
    # except ValueError as e:
    #     # Convert Joules to Watts by dividing by seconds in a day as per:
    #     # https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview
    #     if cube.units == "J m**-2" and new_unit == "W m-2":
    #         cube.data = cube.data / 86400
    #         cube.units = new_unit
    #     elif cube.units == "m" and new_unit == "kg m-2 s-1":
    #         # convert m to mm, and then scale by seconds in a day
    #         cube.data = (cube.data * 1000) / 86400
    #         cube.units = new_unit
    #     else:
    #         raise e

    # save
    # construct filename
    fname = f"ERA5_{var_name}_mon.nc"

    iris.save(cube, f"{OUT_FOLDER}{fname}")
