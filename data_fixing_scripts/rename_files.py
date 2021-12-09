import argparse
import os

# rename REU-25 domain files to be consistent with other RCM data

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--dry_run', help="Dry run, just print out new names but don't rename", action="store_true")
parser.add_argument('input_path', help='Top folder, will process all files below.')
args = parser.parse_args()


def rename_file(root, dir, file, dry_run):
    full_path = os.path.join(root, dir, file)
    
    # replace fist occurence of 'HadGEM3-GC3.1-N512' with 'MOHC-HadGEM2-ES'
    # replace second occurence of 'HadGEM3-GC3.1-N512' with 'MOHC-HadGEM3-GC3.1-N512'
    new_path = full_path.replace('HadGEM3-GC3.1-N512', 'MOHC-HadGEM2-ES', 1)
    new_path = new_path.replace('HadGEM3-GC3.1-N512', 'MOHC-HadGEM3-GC3.1-N512', 1)
    if dry_run:
        print(f"New name: {new_path}")
    else:
        os.rename(full_path, new_path)


# traverse directory tree and rename nc files as needed
for root, dirs, files in os.walk(args.input_path):
    # deal with files
    for f in files:
        if f[-3:] == '.nc':
            rename_file(root, '', f, args.dry_run)
