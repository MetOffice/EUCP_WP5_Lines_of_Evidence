from esmvaltool.diag_scripts.shared import (
    run_diagnostic,
    group_metadata,
    get_plot_filename,
    get_diagnostic_filename
)

import iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(cfg):
    # just load the pre-processed anomlies, and plot them
    
    # first read them in, pop into dictionaries keyed by model name
    # group by project first (CMIP5, CMIP6, UKCP)
    projects = group_metadata(cfg["input_data"].values(), "project")

    results = {}
    for p in projects:
        results[p] = {}
        if p == "UKCP18":
            # loop over ensembles
            models = group_metadata(projects[p], "ensemble")
        else:
            # loop over datasets
            models = group_metadata(projects[p], "dataset")

        for m in models:
            if len(models[m]) > 1:
                raise ValueError("Too many bits of data")
            fname = models[m][0]["filename"]
            data = iris.load_cube(fname)
            results[p][m] = data.data.item()
    
    # plot and save the results
    for p in projects:
        # use pandas to create data for a csv file
        results_df = pd.DataFrame.from_dict(results[p], orient='index')
        # save data as csv
        results_df.to_csv(
            get_diagnostic_filename(f"{p}_global_tas_anom", cfg, "csv"),
            header=False
        )

        # get list of models
        models = results[p].keys()
        # and corresponding values
        vals = [results[p][m] for m in models]

        fig, ax = plt.subplots(figsize=(12.8, 9.6))

        # plot bar chart
        y_pos = np.arange(len(models))
        colors = np.empty(len(models,), dtype=str)
        colors[::2] = 'r'
        colors[1::2] = 'b'
        ax.barh(y_pos, vals, color=colors)
        ax.set_yticks(y_pos, labels=models)

        plot_fname = get_plot_filename(f'{p}_global_anomaly', cfg)
        fig.savefig(plot_fname)
        plt.tight_layout()
        plt.close(fig)


if __name__ == "__main__":
    with run_diagnostic() as cfg:
        main(cfg)
