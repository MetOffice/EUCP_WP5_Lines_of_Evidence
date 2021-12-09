#!/usr/bin bash

# HCLIMcom
python3 -u mean_process.py '_HCLIMcom/ICHEC-EC-EARTH/historical/r12i1p1/HCLIMcom-HCLIM38-AROME/fpsconv-x2yn2-v1/1hr/pr/pr*.nc' day
python3 -u mean_process.py '_HCLIMcom/ICHEC-EC-EARTH/historical/r12i1p1/HCLIMcom-HCLIM38-AROME/fpsconv-x2yn2-v1/1hr/tas/tas*.nc' day

python3 -u mean_process.py '_HCLIMcom/ICHEC-EC-EARTH/rcp85/r12i1p1/HCLIMcom-HCLIM38-AROME/fpsconv-x2yn2-v1/1hr/pr/pr*.nc' day
python3 -u mean_process.py '_HCLIMcom/ICHEC-EC-EARTH/rcp85/r12i1p1/HCLIMcom-HCLIM38-AROME/fpsconv-x2yn2-v1/1hr/tas/tas*.nc' day


# HCLIM KNMI
python3 -u mean_process.py '_hclim_knmi/KNMI-EC-EARTH/historical/r14i1p1/HCLIM38h1-AROME/fpsconv-x2yn2-v1/1hr/pr/pr*.nc' day
python3 -u mean_process.py '_hclim_knmi/KNMI-EC-EARTH/historical/r14i1p1/HCLIM38h1-AROME/fpsconv-x2yn2-v1/1hr/tas/tas*.nc' day

python3 -u mean_process.py '_hclim_knmi/KNMI-EC-EARTH/rcp85/r13i1p1/HCLIM38h1-AROME/fpsconv-x2yn2-v1/1hr/pr/pr*.nc' day
python3 -u mean_process.py '_hclim_knmi/KNMI-EC-EARTH/rcp85/r13i1p1/HCLIM38h1-AROME/fpsconv-x2yn2-v1/1hr/tas/tas*.nc' day


# CMCC-CCLM
python3 -u mean_process.py '_cp-rcm/CMCC-CCLM/ALP-3/CLMCom-CMCC/ICHEC-EC-EARTH/historical/r12i1p1/CLMcom-CMCC-CCLM5-0-9/x2yn2v1/1hr/pr/pr*.nc' day
python3 -u mean_process.py '_cp-rcm/CMCC-CCLM/ALP-3/CLMCom-CMCC/ICHEC-EC-EARTH/historical/r12i1p1/CLMcom-CMCC-CCLM5-0-9/x2yn2v1/1hr/tas/tas*.nc' day

python3 -u mean_process.py '_cp-rcm/CMCC-CCLM/ALP-3/CLMCom-CMCC/ICHEC-EC-EARTH/rcp85/r12i1p1/CLMcom-CMCC-CCLM5-0-9/x2yn2v1/1hr/pr/pr*.nc' day
python3 -u mean_process.py '_cp-rcm/CMCC-CCLM/ALP-3/CLMCom-CMCC/ICHEC-EC-EARTH/rcp85/r12i1p1/CLMcom-CMCC-CCLM5-0-9/x2yn2v1/1hr/tas/tas*.nc' day


# GERICS REMO - Fix should work - days for time units
python3 -u mean_process.py '_cp-rcm/REMO-NH/historical/1hr/pr/pr*.nc' day
python3 -u mean_process.py '_cp-rcm/REMO-NH/historical/1hr/tas/tas*.nc' day

python3 -u mean_process.py '_cp-rcm/REMO-NH/rcp85/1hr/pr/pr*.nc' day
python3 -u mean_process.py '_cp-rcm/REMO-NH/rcp85/1hr/tas/tas*.nc' day


# ICTP RegCM - Fix should work
python3 -u mean_process.py '_cp-rcm/ICTP-RegCM/ALP-3/HadGEM/historical/r1i1p1/ICTP-RegCM4-7-0/v0/1hr/pr/pr*.nc' day
python3 -u mean_process.py '_cp-rcm/ICTP-RegCM/ALP-3/HadGEM/historical/r1i1p1/ICTP-RegCM4-7-0/v0/1hr/tas/tas*.nc' day

python3 -u mean_process.py '_cp-rcm/ICTP-RegCM/ALP-3/HadGEM/rcp85/r1i1p1/ICTP-RegCM4-7-0/v0/1hr/pr/pr*.nc' day
python3 -u mean_process.py '_cp-rcm/ICTP-RegCM/ALP-3/HadGEM/rcp85/r1i1p1/ICTP-RegCM4-7-0/v0/1hr/tas/tas*.nc' day


# ETHZ COSMO - fix should work tas files use seconds for units rather than hours..
python3 -u mean_process.py '_cp-rcm/ETHZ-COSMO-ALP/ALP-3/ETHZ-2/MPI/historical/r1i1p1/COSMO-pompa/5.0_2019.1/1hr/pr/pr*.nc' day
python3 -u mean_process.py '_cp-rcm/ETHZ-COSMO-ALP/ALP-3/ETHZ-2/MPI/historical/r1i1p1/COSMO-pompa/5.0_2019.1/1hr/tas/tas*.nc' day

python3 -u mean_process.py '_cp-rcm/ETHZ-COSMO-ALP/ALP-3/ETHZ-2/MPI/rcp85/r1i1p1/COSMO-pompa/5.0_2019.1/1hr/pr/pr*.nc' day
python3 -u mean_process.py '_cp-rcm/ETHZ-COSMO-ALP/ALP-3/ETHZ-2/MPI/rcp85/r1i1p1/COSMO-pompa/5.0_2019.1/1hr/tas/tas*.nc' day


# CNRS-CNRM - Fix Should work - Dont forget manual coord fix
python3 -u mean_process.py '_cp-rcm/CNRS-CNRM/ALP-3/historical/pr*.nc' day
python3 -u mean_process.py '_cp-rcm/CNRS-CNRM/ALP-3/historical/tas*.nc' day

python3 -u mean_process.py '_cp-rcm/CNRS-CNRM/ALP-3/rcp85/pr*.nc' day
python3 -u mean_process.py '_cp-rcm/CNRS-CNRM/ALP-3/rcp85/tas*.nc' day

