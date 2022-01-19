import matplotlib.pyplot as plt
import re
from cycler import cycler


CPM_DRIVERS = {
    'CNRM-AROME41t1': 'ALADIN63 CNRM-CERFACS-CNRM-CM5',
    'CLMcom-CMCC-CCLM5-0-9': 'CCLM4-8-17 ICHEC-EC-EARTH',
    'HCLIMcom-HCLIM38-AROME': 'HCLIMcom-HCLIM38-ALADIN ICHEC-EC-EARTH',
    'GERICS-REMO2015': 'REMO2015 MPI-M-MPI-ESM-LR',
    'COSMO-pompa': 'CCLM4-8-17 MPI-M-MPI-ESM-LR',
    'ICTP-RegCM4-7-0': 'ICTP-RegCM4-7-0 MOHC-HadGEM2-ES',
    'ICTP-RegCM4-7': 'ICTP-RegCM4-7-0 MOHC-HadGEM2-ES',
    'KNMI-HCLIM38h1-AROME': 'KNMI-RACMO23E KNMI-EC-EARTH',
    'SMHI-HCLIM38-AROME': 'SMHI-HCLIM38-ALADIN ICHEC-EC-EARTH',
    'HadREM3-RA-UM10.1': 'MOHC-HadGEM3-GC3.1-N512 MOHC-HadGEM2-ES'
}


def remove_institute_from_driver(driver_str):
    # function to remove superfluous institute information from 
    # driving model supplied in CORDEX descriptions
    institutes = [
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
    # remove the institute bit from the "driver" string
    new_str = driver_str
    # loop through the institutes and remove them if found
    for i in institutes:
        i = '^' + i + '-'
        new_str = re.sub(i, '', new_str)

    if new_str == driver_str:
        raise ValueError(f"No institute found to remove from {driver_str}")

    return new_str


def coloured_violin(data, pos, ax, color=None):
    vparts = ax.violinplot(data, [pos], showmedians=True, quantiles=[0.1, 0.9])

    if color:
        for part in ['bodies', 'cbars', 'cmins', 'cmaxes']:
            if part == 'bodies':
                for c in vparts[part]:
                    c.set_color(color)
            else:
                vparts[part].set_color(color)


def boxplot(data, pos, ax, color=None):
    # plot colooured box plot of data
    box_parts = ax.boxplot(data, whis=(10, 90), positions=[pos], showmeans=True)

    if color:
        for parts in box_parts:
            for part in box_parts[parts]:
                part.set_color(color)


PLOT_FN = boxplot


def plot_points(points, x, ax, color='k'):
    for p in points:
        ax.plot(x, p, marker="o", fillstyle="none", color=color)


def prepare_scatter_data(x_data, y_data, project):
    # need to establish matching cmip value for each cordex value
    x_vals = []
    y_vals = []
    labels = []

    if project == "CORDEX":
        # expect rcm vals are y vals. GCM, x
        for rcm in y_data:
            y_vals.append(y_data[rcm])

            # find corresponding cmip data
            actual_rcm, driver = rcm.split(' ')
            actual_driver = remove_institute_from_driver(driver)

            x_vals.append(x_data[actual_driver])

            # construct label
            labels.append(f"{actual_driver} {actual_rcm}")
    elif project == "UKCP18":
        # we expect y_data to be the RCM
        for ensemble in y_data:
            x_vals.append(x_data[ensemble])
            y_vals.append(y_data[ensemble])

            labels.append(ensemble)
    elif project == "CPM":
        # cpm on y axis, rcm on x axis
        for cpm in y_data:
            y_vals.append(y_data[cpm])

            driver = CPM_DRIVERS[cpm.split(' ')[0]]
            cpm = cpm.split(' ')[0]
            x_vals.append(x_data[driver])

            # construct label
            labels.append(f"{driver} {cpm}")
    else:
        raise ValueError(f"Unrecognised project {project}")

    return x_vals, y_vals, labels


def labelled_scatter(x_data, y_data, labels, ax, RCM_markers=False, plot_text=True):
    if RCM_markers:
        label_props = {}
        marker_props = enumerate((cycler(marker=['o', 'P', 'd']) * cycler(color=list('bgrcmy'))))

    max_val = 0
    min_val = 999999

    for i in range(len(x_data)):
        x_val = x_data[i]
        y_val = y_data[i]

        # update max and min value encountered
        max_val = max(x_val, y_val, max_val)
        min_val = min(x_val, y_val, min_val)

        if RCM_markers:
            rcm = labels[i].split()[-1]
            if rcm in label_props:
                props = label_props[rcm]
            else:
                props = next(marker_props)
                label_props[rcm] = props

            ax.scatter(
                x_val, y_val, label=f"{i} - {labels[i]}",
                color=props[1]['color'], marker=props[1]['marker']
                )
        else:
            ax.scatter(x_val, y_val, label=f"{i} - {labels[i]}")

        if plot_text:
            ax.text(x_val, y_val, i, fontsize=10)

    # plot a diagonal equivalence line
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

    # plot zero lines
    if min_val < 0:
        ax.axhline(ls=':', color='k', alpha=0.75)
        ax.axvline(ls=':', color='k', alpha=0.75)


def mega_scatter(GCM_sc, RCM_sc1, RCM_sc2, CPM, all_GCM, all_RCM, labels1, labels2, title, plot_path):
    '''
    A mega plot that shows distributions and scatter plots of GCM, RCM and CPM
    GCM_sc: GCMs for first scatter
    RCM_sc1: RCMs for first scatter
    RCM_sc2: RCMs for second scatter
    CPM: CPMs
    all_GCM: all GCMs for the violin / box
    all_RCM: all RCMs for a violin / box
    labels1: Legend labels for scatter1
    labels2: Legend labels for scatter2
    suffix: suffix to end to filename
    '''
    plt.figure(figsize=(19.2, 14.4))

    # construct axes
    ax_datasets = plt.subplot(211)
    ax_scatter1 = plt.subplot(223)
    ax_scatter2 = plt.subplot(224)

    # Create GCM / RCM scatter
    labelled_scatter(GCM_sc, RCM_sc1, labels1, ax_scatter1, RCM_markers=True, plot_text=False)
    ax_scatter1.set_xlabel('GCM')
    ax_scatter1.set_ylabel('RCM')
    if min(RCM_sc1) < 0 < max(RCM_sc1):
        ax_scatter1.axhline(ls=':', color='k', alpha=0.75)

    # create RCM / CPM scatter
    labelled_scatter(RCM_sc2, CPM, labels2, ax_scatter2, RCM_markers=False, plot_text=False)
    ax_scatter2.set_xlabel('RCM')
    ax_scatter2.set_ylabel('CPM')
    if min(CPM) < 0 < max(CPM):
        ax_scatter2.axhline(ls=':', color='k', alpha=0.75)

    # legend information
    h1, l1 = ax_scatter1.get_legend_handles_labels()
    h2, l2 = ax_scatter2.get_legend_handles_labels()
    ax_datasets.legend(
        h1 + h2, l1 + l2, bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=10)

    # create GCM / RCM / CPM violins or boxes / dots
    # GCMs go in position 1, RCMs position 2, CPMs position 3
    PLOT_FN(all_GCM, 1, ax_datasets, 'lightgrey')
    PLOT_FN(all_RCM, 2, ax_datasets, 'lightgrey')
    PLOT_FN(CPM, 3, ax_datasets, 'lightgrey')

    # set x labels
    ax_datasets.set_xticks(range(1, 4))
    ax_datasets.set_xticklabels(['CMIP5', 'CORDEX', 'CPM'])

    # also plot individual dots for each model..
    plot_points(all_GCM, 0.8, ax_datasets)
    plot_points(GCM_sc, 1.3, ax_datasets, color='r')
    plot_points(RCM_sc1, 1.8, ax_datasets, color='r')
    plot_points(RCM_sc2, 2.3, ax_datasets, color='b')
    plot_points(CPM, 3, ax_datasets, color='b')

    max_ds = max(max(all_GCM), max(RCM_sc1), max(CPM))
    min_ds = min(min(all_GCM), min(RCM_sc1), min(CPM))
    if min_ds < 0 < max_ds:
        ax_datasets.axhline(ls=':', color='k', alpha=0.75)

    plt.suptitle(f"{title} change")

    # save plot
    plt.savefig(f"{plot_path}/mega_scatter_{title}.png", bbox_inches='tight')
    plt.close()
