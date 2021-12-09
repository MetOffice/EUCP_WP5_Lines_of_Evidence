#!/usr/bin/env python3



# imports
import iris
import iris.coord_categorisation as iccat

import glob
import re
import os
import argparse

# function to fix any rounding issues with coord systems
# needed for _HCLIMcom/ICHEC-EC-EARTH/rcp85
def fix_coord_metadata(cube):
    for c in ['projection_x_coordinate', 'projection_y_coordinate']:
        cube.coord(c).coord_system.false_easting = round(cube.coord(c).coord_system.false_easting, 9)
        cube.coord(c).coord_system.false_northing = round(cube.coord(c).coord_system.false_northing, 9)


parser = argparse.ArgumentParser()
parser.add_argument('input_path', help='input files path')
parser.add_argument('frequency', help='Compute daily (day) or monthly (mon) frequency data')
parser.add_argument('output_folder', help='Parent output folder to save to')
args = parser.parse_args()

# folder of files to process
file_path = args.input_path
freq = args.frequency

# glob the file_path and process each file individually
print(f'globbing {file_path}')
files = glob.glob(file_path)

for f in files:
    # load data
    print(f'Loading {f}')

    cubes = iris.load(f'{f}', ['precipitation_flux', 'air_temperature'])

    print('Adding coords and doing any fixes....')
    # check and fix rounding issue in metadata for lambert coords
    if len(cubes[0].coords("projection_x_coordinate")) > 0:
        for c in cubes:
            fix_coord_metadata(c)

    # concatenate and add required coords for processing
    iris.util.equalise_attributes(cubes)
    cube = cubes.concatenate_cube()
    
    # check last two time points, if from a different month, subtract half an hour to fix
    if cube.coord('time').cell(-2).point.month != cube.coord('time').cell(-1).point.month:
        # check time units is what we expect
        unit = cube.coord('time').units
        if unit.name[:4] == 'hour':
            thirty_mins = 0.5
        elif unit.name[:6] == 'second':
            thirty_mins = 60 * 30
        elif unit.name[:3] == 'day':
            thirty_mins = 0.5 / 24
        else:
            raise ValueError(f"Don't know how to deal with: '{unit.name}'")
        
        print('Subtracting half an hour from time coord first')
        new_points = cube.coord('time').points - thirty_mins
        cube.coord('time').points = new_points
    
    iccat.add_year(cube, 'time')
    if freq == 'mon':
        iccat.add_month_number(cube, 'time')
        agg_by = ['month_number', 'year']
        remove_later = 'month_number'
    elif freq == 'day':
        iccat.add_day_of_year(cube, 'time')
        agg_by = ['day_of_year', 'year']
        remove_later = 'day_of_year'
    else:
        raise ValueError('Unrecognised frequency')


    # compute averages
    print(f'Computing {freq} average')
    means = cube.aggregated_by(agg_by, iris.analysis.MEAN)


    # remove no longer needed aux coords
    means.remove_coord(remove_later)
    means.remove_coord('year')

    # make sure standard names are being used
    if means.var_name == 'pr':
        means.standard_name = 'precipitation_flux'
    if means.var_name == 'tas':
        means.standard_name = 'air_temperature'

    # save
    # construct new filename
    # need to compute start and end month of file
    if args.frequency == 'mon':
        file_start = f'{means.coord("time").cell(0).point.year}{means.coord("time").cell(0).point.month:02}'
        file_end = f'{means.coord("time").cell(-1).point.year}{means.coord("time").cell(-1).point.month:02}'
    else:
        file_start = f'{means.coord("time").cell(0).point.year}{means.coord("time").cell(0).point.month:02}{means.coord("time").cell(0).point.day:02}'
        file_end = f'{means.coord("time").cell(-1).point.year}{means.coord("time").cell(-1).point.month:02}{means.coord("time").cell(-1).point.day:02}'

    file_template = f
    # do appropriate replacements
    if '1hr' in f:
        file_template = re.sub(r'\/1hr\/', f'/{freq}/', file_template)
        file_template = re.sub(r'_1hr_', f'_{freq}_', file_template)
    else:
        file_template = re.sub(r'_day_', f'_{freq}_', file_template)
    if re.search(r'\d{12}-\d{12}.nc', file_template):
        file_template = re.sub(r'\d{12}-\d{12}.nc', f'{file_start}-{file_end}.nc', file_template)
    elif re.search(r'\d{10}-\d{10}.nc', file_template):
        file_template = re.sub(r'\d{10}-\d{10}.nc', f'{file_start}-{file_end}.nc', file_template)
    elif re.search(r'\d{8}-\d{8}.nc', file_template):
        file_template = re.sub(r'\d{8}-\d{8}.nc', f'{file_start}-{file_end}.nc', file_template)
    elif re.search(r'\d{4}_\d{4}.nc', file_template):
        file_template = re.sub(r'\d{4}_\d{4}.nc', f'{file_start}-{file_end}.nc', file_template)
    else:
        raise ValueError("Couldn't match date strings in filename")

    out_file = file_template

    # make output folder if necessary
    out_dir, fname = os.path.split(out_file)
    out_dir = f"{args.output_folder}/{out_dir.split('/', 1)[1]}"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, fname)

    # save
    print(f'Saving to: {out_path}')
    iris.save(means, out_path)
