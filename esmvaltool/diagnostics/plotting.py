import matplotlib.pyplot as plt
import re
import seaborn as sns
from cycler import cycler


# map WP2 data to GCM group
WP2_METHODS = {
    "ETHZ_CMIP6_ClimWIP": "CMIP6",
    "ICTP_CMIP6_REA": "CMIP6",
    "ICTP_CMIP5_REA": "CMIP5",
    "UKMO_CMIP6_UKCP": "UKCP_GCM"
}


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


def simpler_scatter(drive_data, downscale_data, labels, title, plot_path):
    '''
    Simpler scatter that just plots distributions and scatter of a pair of simulations
    '''
    plt.figure(figsize=(19.2, 14.4))

    # construct axes
    ax_datasets = plt.subplot2grid((1, 3), (0, 0))
    ax_scatter = plt.subplot2grid((1, 3), (0, 1), colspan=2, sharey=ax_datasets)

    # make scatter
    labelled_scatter(drive_data, downscale_data, labels, ax_scatter)
    ax_scatter.set_xlabel('GCM')
    ax_scatter.set_ylabel('RCM')

    # make boxes / violins
    PLOT_FN(drive_data, 1, ax_datasets, 'lightgrey')
    PLOT_FN(downscale_data, 2, ax_datasets, 'lightgrey')

    # set x labels
    ax_datasets.set_xticks(range(1, 3))
    ax_datasets.set_xticklabels(['Global', 'Regional'])

    # also plot individual dots for each model..
    if PLOT_FN == coloured_violin:
        plot_points(drive_data, 1, ax_datasets, color='r')
        plot_points(downscale_data, 2, ax_datasets, color='r')

    plt.suptitle(f"{title} change")

    # save plot
    plt.savefig(f"{plot_path}/simple_scatter_{title}.png", bbox_inches='tight')
    plt.close()


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
    title: Plot title and filename prefix
    plot_path: Path to save plot into
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


def box_plot(data, ax, edge_color, fill_color, positions, widths):
    bp = ax.boxplot(data, patch_artist=True, positions = positions, widths=widths, showfliers=False, whis=[10,90])

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        if element == 'medians':
            col = 'black'
        else:
            col = edge_color
        plt.setp(bp[element], color=col)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)

    return bp


def bxp(data, ax, colour, alpha, position, width, **kwargs):
    bp = ax.bxp(data, patch_artist=True, positions=position, widths=width, showfliers=False, **kwargs)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        if element == 'medians':
            col = 'black'
        else:
            col = colour
        plt.setp(bp[element], color=col)
        plt.setp(bp[element], alpha=alpha)

    for patch in bp['boxes']:
        patch.set(facecolor=colour)
        patch.set(alpha = alpha)

    return bp


def create_x_points(ys, basex, offset):
    xs = []
    for i, v in enumerate(ys):
        if i == 0:
            xs.append(basex)
            vm1 = v
        else:
            if abs(v - vm1) <= 1:
                if xs[i-1] < basex:
                    xs.append(basex + offset)
                elif xs[i-1] == basex:
                    xs[i-1] = basex - offset
                    xs.append(basex + offset)
                else:
                    # previous version has been offset positively
                    xs.append(basex - offset)
            else:
                xs.append(basex)
            vm1 = v

    return xs


def panel_boxplot(plot_df, constraint_data, area, season, var, case_study=False):
    # set colours
    colour_map = {
        "CMIP6": "tab:blue",
        "CMIP5": "tab:orange",
        "CORDEX": "tab:green",
        "CPM": "tab:red",
        "UKCP_GCM": "tab:purple",
    }

    plt.rcParams.update({'font.size': 14})
    # create figure and axes
    if case_study:
        f, axs = plt.subplots(1,4, sharey=True, figsize=[19.2 ,  9.77], gridspec_kw={'width_ratios': [3, 2, 4, 1]})
    else:
        f, axs = plt.subplots(1,3, sharey=True, figsize=[19.2 ,  9.77], gridspec_kw={'width_ratios': [3, 2, 4]})
    # f.suptitle("Projected % change in summer (JJA) rainfall for Romania. 2041-2060 vs 1995-2014. RCP8.5/ssp585")
    # size of dots in swarm plots
    swarm_size = 7

    # First panel
    # plot GCM boxes
    axs[0].clear()

    # plot boxes with matplotlib
    box_plot([plot_df["CMIP6"], plot_df["CMIP5"], plot_df["UKCP_GCM"]], axs[0], "black", "None", [0, 1, 2], [0.5, 0.5, 0.5])

    # plot dots
    sns.swarmplot(
        data=plot_df[["CMIP6", "CMIP5", "UKCP_GCM"]], ax=axs[0],
        size=swarm_size, palette=["tab:blue", "tab:orange", "tab:purple"],
        alpha=0.75
        )

    # last bits of formatting
    axs[0].axhline(linestyle=":", color="k", alpha=0.5)
    plt.setp(axs[0].get_xticklabels(), rotation=45, ha="right")
    axs[0].set_title("GCMs")

    # plot constrained ranges. 2nd panel
    axs[1].clear()
    for i, k in enumerate(constraint_data.keys()):
        colour = colour_map[WP2_METHODS[k]]
        # constrained
        bxp([constraint_data[k][0]], axs[1], colour, 0.75, [i], 0.375)
        # unconstrained
        bxp([constraint_data[k][1]], axs[1], colour, 0.25, [i], 0.5)

    axs[1].axhline(linestyle=":", color="k", alpha=0.5)
    plt.setp(axs[1].get_xticklabels(), rotation=45, ha="right")
    axs[1].set_title("Uncertainty estimates\nfrom GCMs and observations")

    # third panel downscaled information
    axs[2].clear()
    if 'cpm' in plot_df:
        data = plot_df[["CMIP5", "CORDEX", "CPM", "UKCP_GCM", "UKCP_RCM"]]
        palette = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:purple"]
    else:
        data = plot_df[["CMIP5", "CORDEX", "UKCP_GCM", "UKCP_RCM"]]
        palette = ["tab:orange", "tab:green", "tab:purple", "tab:purple"]

    sns.swarmplot(
        data=data,
        ax=axs[2],
        size=swarm_size,
        palette=palette
    )
    # CORDEX drivers
    y = plot_df["CORDEX Drivers"].dropna()
    x = create_x_points(y, 0.5, 0.05)
    axs[2].scatter(x, y, color="tab:orange", marker=">", s=50)

    if 'cpm' in plot_df:
        # CPM drivers
        y = plot_df["CPM Drivers"].dropna()
        x = create_x_points(y, 1.5, 0.05)
        axs[2].scatter(x, y, color="tab:green", marker=">", s=50)
        ukcp_div = 2.5
    else:
        ukcp_div = 1.5

    # Divider line for UKCP
    axs[2].axvline(ukcp_div, color="lightgrey")

    # UKCP drivers
    y = plot_df["UKCP Drivers"].dropna()
    x = create_x_points(y, ukcp_div+1, 0.05)
    axs[2].scatter(x, y, color="tab:purple", marker=">", s=50)

    # Final formatting etc.
    axs[2].axhline(linestyle=":", color="k", alpha=0.5)
    plt.setp(axs[2].get_xticklabels(), rotation=45, ha="right")
    axs[2].set_title("Downscaled Projections")

    # extra panel if a case study
    if len(plot_df["Case study models"].dropna()) > 0:
        axs[3].clear()
        sns.swarmplot(
            data=plot_df["Case study models"],
            ax=axs[3],
            size=swarm_size,
            palette=["tab:green"]
            )
        axs[3].set_xticklabels(["Case study models"])
        axs[3].axhline(linestyle=":", color="k", alpha=0.5)
        plt.setp(axs[3].get_xticklabels(), rotation=45, ha="right")
        axs[3].set_title("From study")

    # Final figure spacing etc.
    # axs[0].set_ylabel("%")
    plt.suptitle(f"{area.attributes['NAME']} {season} {var}")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, wspace=0.06)

    # save plot
    save_path = "/home/h02/tcrocker/code/EUCP_WP5_Lines_of_Evidence/esmvaltool/plots"
    plt.savefig(f"{save_path}/{area.attributes['NAME']}_{season}_{var}.png")
    plt.close()