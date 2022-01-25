#!/usr/bin bash
# needs to run from a scitools environment.
# Currently working with: /net/project/ukmo/scitools/opt_scitools/environments/default/2021_03_18-1/bin/python

doms=('ALP-3' 'CEE-3' 'CEU-3' 'NEU-3' 'NWE-3' 'SEE-3' 'SWE-3' 'boe')
seas=('JJA' 'DJF')

# temperature
for dom in "${doms[@]}"; do
    for sea in "${seas[@]}"; do
        sbatch -t 5 -n 2 --mem=4G --wrap="python summary_boxplots.py recipe_GCM_and_RCM_pan_EU_20220113_120432 tas ${sea} ${dom} recipe_GCM_global_tas_20220121_172015"
    done
done

# precip
for dom in "${doms[@]}"; do
    for sea in "${seas[@]}"; do
        sbatch -t 5 -n 2 --mem=4G --wrap="python summary_boxplots.py recipe_GCM_and_RCM_pan_EU_20220113_120253 pr ${sea} ${dom} recipe_GCM_global_tas_20220121_172015"
    done
done
