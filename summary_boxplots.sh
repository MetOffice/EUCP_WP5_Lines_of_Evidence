#!/usr/bin bash

doms=('ALP-3' 'CEE-3' 'CEU-3' 'NEU-3' 'NWE-3' 'SEE-3' 'SWE-3')
seas=('JJA' 'DJF')

# temperature
for dom in "${doms[@]}"; do
    for sea in "${seas[@]}"; do
        sbatch -t 5 -n 2 --mem=4G --wrap="python summary_boxplots.py recipe_GCM_and_RCM_pan_EU_20220113_120432 tas ${sea} ${dom}"
    done
done

# precip
for dom in "${doms[@]}"; do
    for sea in "${seas[@]}"; do
        sbatch -t 5 -n 2 --mem=4G --wrap="python summary_boxplots.py recipe_GCM_and_RCM_pan_EU_20220113_120432 tas ${sea} ${dom}"
    done
done
