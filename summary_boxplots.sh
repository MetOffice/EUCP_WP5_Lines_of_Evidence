#!/usr/bin bash
# needs to run from a scitools environment.
# Currently working with: /net/project/ukmo/scitools/opt_scitools/environments/default/2021_03_18-1/bin/python

# doms=('ALP-3' 'CEE-3' 'CEU-3' 'NEU-3' 'NWE-3' 'SEE-3' 'SWE-3' 'boe' 'Scotland' '"United Kingdom"' 'Romania')
doms=('Romania')
# doms=('berthou')
seas=('JJA' 'DJF')
vars=('pr' 'tas')

# 2041-2060 data
for dom in "${doms[@]}"; do
    if [[ ${dom} =~ ^[A-Z]{3}-3$ ]]; then
        # EUCP WP3 domain
        SFILE=/home/h02/tcrocker/code/EUCP_WP5_Lines_of_Evidence/shape_files/EUCP_WP3_domains/EUCP_WP3_domains.shp
    elif [[ ${dom} == 'Scotland' ]]; then
        # region
        SFILE=/home/h02/tcrocker/code/EUCP_WP5_Lines_of_Evidence/shape_files/ne_50m_admin_0_map_units/ne_50m_admin_0_map_units.shp
    else
        # country
        SFILE=/home/h02/tcrocker/code/EUCP_WP5_Lines_of_Evidence/shape_files/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp
    fi
    for sea in "${seas[@]}"; do
        for var in "${vars[@]}"; do
            sbatch -t 5 -n 2 --mem=4G --wrap="python summary_boxplots.py recipe_GCM_and_RCM_pan_EU_20220207_101326 ${var} ${sea} recipe_GCM_global_tas_20220121_172015 ${dom} --shape_file=${SFILE}"
        done
    done
done


# # 2090-2099 data
# # temperature
# for dom in "${doms[@]}"; do
#     for sea in "${seas[@]}"; do
#         sbatch -t 5 -n 2 --mem=4G --wrap="python summary_boxplots.py recipe_GCM_and_RCM_pan_EU_20220131_160949 tas ${sea} ${dom} recipe_GCM_global_tas_20220121_172015"
#     done
# done

# # precip
# for dom in "${doms[@]}"; do
#     for sea in "${seas[@]}"; do
#         sbatch -t 5 -n 2 --mem=4G --wrap="python summary_boxplots.py recipe_GCM_and_RCM_pan_EU_20220131_172738 pr ${sea} ${dom} recipe_GCM_global_tas_20220121_172015"
#     done
# done
