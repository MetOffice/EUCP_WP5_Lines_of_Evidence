import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# considering results from this paper:
# Niculiţă, M. (2020). Landslide Hazard Induced by Climate Changes in North-Eastern Romania. 
# Climate Change Management, (May), 245–265. https://doi.org/10.1007/978-3-030-37425-9_13

# global variables to control everything
RECIPE_RUN = 'recipe_GCM_and_RCM_Romania_20211103_153305'
BASE_PATH = f'/home/h02/tcrocker/code/EUCP_WP5_Lines_of_Evidence/esmvaltool/esmvaltool_output/{RECIPE_RUN}/work/boxplots/main/'
SEASON = 'JJA'

INSTITUTES = [
    'IPSL',
    'NCC',
    'MPI-M',
    'CNRM-CERFACS',
    'ICHEC',
    'MOHC',
    'KNMI',
    'HCLIMcom',
    'SMHI'
]


def remove_institute_from_driver(driver_str):
    # remove the institute bit from the "driver" string
    new_str = driver_str
    # loop through the institutes and remove them if found
    for i in INSTITUTES:
        i = '^' + i + '-'
        new_str = re.sub(i, '', new_str)

    if new_str == driver_str:
        raise ValueError(f"No institute found to remove from {driver_str}")

    return new_str


# read data
cmip5 = pd.read_csv(f'{BASE_PATH}CMIP5_{SEASON}.txt', sep=':', header=None)
cmip6 = pd.read_csv(f'{BASE_PATH}CMIP6_{SEASON}.txt', sep=':', header=None)
cordex = pd.read_csv(f'{BASE_PATH}CORDEX_{SEASON}.txt', sep=':', header=None)
cpm = pd.read_csv(f'{BASE_PATH}CPM_{SEASON}.txt', sep=':', header=None)

niculita_model_list = [
    'RCA4 MPI-M-MPI-ESM-LR',
    'RCA4 MOHC-HadGEM2-ES',
    'RCA4 ICHEC-EC-EARTH',
    'RCA4 CNRM-CERFACS-CNRM-CM5',
    'REMO2009 MPI-M-MPI-ESM-LR',
    'RACMO22E MOHC-HadGEM2-ES',
    'RACMO22E ICHEC-EC-EARTH', 
    'HIRHAM5 ICHEC-EC-EARTH',
    ]

cpm_driver_list = [
    'ICTP-RegCM4-7-0 MOHC-HadGEM2-ES',
    'SMHI-HCLIM38-ALADIN ICHEC-EC-EARTH',
]

# create subset of models used in paper
nic_df = cordex[cordex[0].isin(niculita_model_list)]
cpm_driver_df = cordex[cordex[0].isin(cpm_driver_list)]

# calculate CMIP5 CORDEX drivers from CORDEX model names
cordex_driver_list = list(
    set(
        [remove_institute_from_driver(n.split(' ')[1]) for n in cordex[0]]
    )
)
cordex_driver_df = cmip5[cmip5[0].isin(cordex_driver_list)]

# chuck everything in a dataframe for plotting
plot_df = pd.DataFrame(
    {
        "CMIP6": cmip6[1],
        "CMIP5": cmip5[1],
        "CORDEX Drivers": cordex_driver_df[1],
        "CORDEX": cordex[1],
        "Niclulită models": nic_df[1],
        "CPM Drivers": cpm_driver_df[1],
        "CPM": cpm[1]
    }
)

# plot
plt.switch_backend('TKagg')
plt.rcParams.update({'font.size': 12})

pal = {}
for col in plot_df.columns:
    if col == 'Niclulită models':
        pal[col] = "tab:orange"
    elif "Drivers" in col:
        pal[col] = "tab:green"
    else:
        pal[col] = "tab:blue"

# only plot boxes for cmip5/6 and cordex
plt.boxplot(
    x=[cmip6[1], cmip5[1], cordex[1]],
    whis=[10, 90],
    showfliers=False,
    positions=[0, 1, 3],
    patch_artist=True,
    boxprops=dict(facecolor="lightgrey"),
    medianprops=dict(color="black")
    )
ax = plt.gca()
sns.swarmplot(data=plot_df, ax=ax, palette=pal)
ax.axhline(0, linestyle=':')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.title("Romania mean JJA Precipitation changes. 1996-2005 to 2041-2050. RCP8.5")
plt.ylabel('%')
plt.tight_layout()
plt.show()
