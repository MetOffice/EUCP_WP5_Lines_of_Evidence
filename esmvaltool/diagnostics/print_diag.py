# quick diagnostic for printing output from pre-processor
from esmvaltool.diag_scripts.shared import run_diagnostic

import iris

import os
import logging

logger = logging.getLogger(os.path.basename(__file__))


def main(cfg):
    # The config object is a dict of all the metadata from the pre-processor
    logger.info(cfg)

    for f in cfg["input_data"].keys():
        cube = iris.load_cube(f)
        logger.info(f)
        logger.info(cube)


if __name__ == "__main__":
    with run_diagnostic() as cfg:
        main(cfg)
