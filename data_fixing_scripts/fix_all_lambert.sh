#!/usr/bin bash
# If all data needs fixing
# this bash script will refix them all

# for f in $(ls ALP-3_day_means/CNRS-CNRM/ALP-3/historical/)
# do
# 	python3 fix_lambert.py tas_ALP-3_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-AROME41t1_fpsconv-x2yn2-v1_1hr_199601010030-199612312330.nc ALP-3_day_means/CNRS-CNRM/ALP-3/historical/${f}
# done

# for f in $(ls ALP-3_day_means/CNRS-CNRM/ALP-3/rcp85/)
# do
# 	python3 fix_lambert.py tas_ALP-3_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-AROME41t1_fpsconv-x2yn2-v1_1hr_199601010030-199612312330.nc ALP-3_day_means/CNRS-CNRM/ALP-3/rcp85/${f}
# done

ROOT=/project/applied/Data/cordex-cpm/NWE-3/CNRM-CERFACS-CNRM-CM5
HR_FILE=/project/applied/Data/cordex-cpm/ensemblesrt3.dmi.dk/data/EUCP/output/NWE-3/CNRM/CNRM-CERFACS-CNRM-CM5/historical/r1i1p1/CNRM-AROME41t1/fpsconv-x2yn2-v1/1hr/tas/tas_NWE-3_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-AROME41t1_fpsconv-x2yn2-v1_1hr_199601010030-199612312330.nc

# Historical
for f in $(ls ${ROOT}/historical/mon/tas)
do
	python fix_lambert.py ${HR_FILE} ${ROOT}/historical/mon/tas/${f}
done

for f in $(ls ${ROOT}/historical/mon/pr)
do
	python fix_lambert.py ${HR_FILE} ${ROOT}/historical/mon/pr/${f}
done

# RCP85
for f in $(ls ${ROOT}/rcp85/mon/tas)
do
	python fix_lambert.py ${HR_FILE} ${ROOT}/rcp85/mon/tas/${f}
done

for f in $(ls ${ROOT}/rcp85/mon/pr)
do
	python fix_lambert.py ${HR_FILE} ${ROOT}/rcp85/mon/pr/${f}
done
