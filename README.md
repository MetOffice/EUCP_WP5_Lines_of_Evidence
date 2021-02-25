# EUCP_WP5_Lines_of_Evidence

This is a repo for work on the EUCP WP5 lines of evidence work

## Contents:
* esmvaltool/ - ESMValtool related stuff
    * recipes/ - ESMValtool recipes
    * diagnostics/ - diagnostics

Config files for use with these ESMValtool recipes:
* config-developer.yml
* config-user.yml

## recipes/recipe_calc_temporal_stats.yml
A recipe to calculate temporally averaged statistics of CMIP5/6, CORDEX and EUCP CMP data.

I have defined some extra modules beyond the ones expected by ESMValtool to create "base" versions of e.g. variables and preprocessors that the actual ones used for the recipe can then inherit from. There are also long lists of datasets defined seperately that are then intended to be used via the `additional_datasets:` key in the recipe, this is because e.g. we need to process both historical and future versions of the datasets, but for CMIP5 / CORDEX the future experiment type is "rcp85" and for CMIP6 it is "ssp585". Since there is no "if" type logic in YAML I've taken the approach of having 0 global datasets in the recipe, and then explicitly specifying the datasets for each variable in the diagnostic section.. On the other hand, it might be possible to go back to the "normal" approach of having some global datasets and using the "--skip_nonexistent" option when running the recipe.

There are two sets of variables inclucded. The "clim_periods" ones and the "decades" ones.
The "clim_periods" ones are used for the normal way of running the recipe where all the processing is done in one go, and the preprocessor does as much as it can before the diagnostic takes over.
The "decades" ones are designed for the use case of where we just want to run some basic pre-processing of the files into decadal chunks, with the intention they can be used later when combined with other data that has been processed elsewhere. E.g. MO processes some files for CMIP5/6 EUR-11, and SMHI provides files from the CPM runs. Then another diagnostic runs to combine all this data together and do the remaining processing and plotting that the user requires.

## diagnostics/delta_boxplot.py
A diagnostic to take the data from the processor. Calculate anomalies and then create a boxplot. This could do with some refactoring, and it probably makes sense to move bits (if not all) of the plotting code into a seperate plotting module of some sort.
I have also made a start on code to work on dealing with the decadal chunks of data produced by the "decades" variables from the recipe but this is not finished yet.

## diagnostics/print_diag.py
A simple diagnostic that simply prints some info about the files that have been produced by the pre-processor.