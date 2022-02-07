import matplotlib.pyplot as plt
import re
import seaborn as sns
from cycler import cycler


# map WP2 data to GCM group
WP2_METHODS = {
    "ETHZ_CMIP6_ClimWIP": "CMIP6",
    "ICTP_CMIP6_REA": "CMIP6",
    "CNRM_CMIP6_KCC": "CMIP6",
    "UEdin_CMIP6_ASK": "CMIP6",
    "ICTP_CMIP5_REA": "CMIP5",
    "UOxf_CMIP5_CALL": "CMIP5",
    "UKMO_CMIP6_UKCP": "UKCP_GCM",
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

    # legend
    h, l = ax_scatter.get_legend_handles_labels()
    ax_scatter.legend(h, l, bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=10)

    # also plot individual dots for each model..
    if PLOT_FN == coloured_violin:
        plot_points(drive_data, 1, ax_datasets, color='r')
        plot_points(downscale_data, 2, ax_datasets, color='r')

    plt.suptitle(f"{title} change")

    # save plot
    plt.savefig(f"{plot_path}/simple_scatter_{title}.png", bbox_inches='tight')
    plt.close()


def mega_scatter(GCM_sc, RCM_sc1, RCM_sc2, CPM, labels1, labels2, title, plot_path):
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
    ax_scatter1 = plt.subplot(121)
    ax_scatter2 = plt.subplot(122)

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
    ax_scatter2.legend(
        h1 + h2, l1 + l2, bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=10)

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


def panel_boxplot(plot_df, constraint_data, driving_models, area, season, var, case_study=False):
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
        f, axs = plt.subplots(1,4, sharey=True, figsize=[19.2 ,  9.77], gridspec_kw={'width_ratios': [2, 3, 4, 1]})
    elif constraint_data == None:
        f, axs = plt.subplots(1,2, sharey=True, figsize=[19.2 ,  9.77], gridspec_kw={'width_ratios': [3, 3]})
    else:
        f, axs = plt.subplots(1,3, sharey=True, figsize=[19.2 ,  9.77], gridspec_kw={'width_ratios': [2, 3, 4]})
    # f.suptitle("Projected % change in summer (JJA) rainfall for Romania. 2041-2060 vs 1995-2014. RCP8.5/ssp585")
    # size of dots in swarm plots
    swarm_size = 7

    # First panel
    # plot GCM boxes
    axs[0].clear()

    # plot boxes with matplotlib
    cmip6 = plot_df[(plot_df["project"] == "CMIP6") & (plot_df["data type"] == "standard")]["value"]
    cmip5 = plot_df[(plot_df["project"] == "CMIP5") & (plot_df["data type"] == "standard")]["value"]
    UKCP = plot_df[(plot_df["project"] == "UKCP18 land-gcm") & (plot_df["data type"] == "standard")]["value"]
    plot_data = [cmip6, cmip5]
    positions = [0, 1]
    widths = [0.25, 0.25]
    order = ["CMIP6", "CMIP5"]
    if len(UKCP) > 0:
        plot_data.append(UKCP)
        positions.append(2)
        widths.append(0.25)
        order.append("UKCP18 land-gcm")
    box_plot(plot_data, axs[0], "black", "None", positions, widths)

    # plot dots
    gcm_projects = ["CMIP6", "CMIP5", "UKCP18 land-gcm"]
    plot_data = plot_df.query("(project in @gcm_projects) &  (`data type` == 'standard')")
    sns.swarmplot(
        x="project",
        y="value",
        data=plot_data, ax=axs[0],
        size=swarm_size, palette=["tab:blue", "tab:orange", "tab:purple"],
        alpha=0.75,
        dodge=True,
        order=order
        )

    # last bits of formatting
    if var=="pr":
        axs[0].axhline(linestyle=":", color="k", alpha=0.5)
    plt.setp(axs[0].get_xticklabels(), rotation=45, ha="right")
    axs[0].set_title("GCMs")
    if var == 'pr':
        y_lab = '%'
    elif var == 'tas':
        y_lab = 'K'
    else:
        raise ValueError(f"Unsupported variable {var}")
    axs[0].set_ylabel(y_lab)
    axs[0].set_xlabel(None)

    # plot constrained ranges. 2nd panel
    if constraint_data is not None:
        for i, k in enumerate(constraint_data.keys()):
            colour = colour_map[WP2_METHODS[k]]
            # constrained
            bxp([constraint_data[k][0]], axs[1], colour, 0.75, [i], 0.375)
            # unconstrained
            if len(constraint_data[k]) == 2:
                bxp([constraint_data[k][1]], axs[1], colour, 0.25, [i], 0.5)

        if var=="pr":
            axs[1].axhline(linestyle=":", color="k", alpha=0.5)
        plt.setp(axs[1].get_xticklabels(), rotation=45, ha="right")
        axs[1].set_title("Uncertainty estimates\nfrom GCMs and observations")

        # set y axes limits
        axs[1].set_ylim(auto=True)
        ax_num = 2
    else:
        ax_num = 1

    # next panel downscaled information
    if len(driving_models['CPM']) > 0:
        projects = ["CMIP5", "CORDEX", "cordex-cpm", "UKCP18 land-gcm", "UKCP18 land-rcm"]
        palette = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:purple"]
    else:
        projects = ["CMIP5", "CORDEX", "UKCP18 land-gcm", "UKCP18 land-rcm"]
        palette = ["tab:orange", "tab:green", "tab:purple", "tab:purple"]
    data = plot_df[(plot_df["project"].isin(projects)) & (plot_df["data type"] == "standard")]
    sns.swarmplot(
        x="project",
        y="value",
        data=data,
        ax=axs[ax_num],
        size=swarm_size,
        palette=palette
    )
    # CORDEX drivers
    y = plot_df[(plot_df["model"].isin(driving_models["CORDEX"])) & (plot_df["data type"] == "standard")]["value"]
    x = create_x_points(y, 0.5, 0.05)
    axs[ax_num].scatter(x, y, color="tab:orange", marker=">", s=50)

    if len(driving_models['CPM']) > 0:
        # CPM drivers
        y = plot_df[(plot_df["model"].isin(driving_models["CPM"])) & (plot_df["data type"] == "standard")]["value"]
        x = create_x_points(y, 1.5, 0.05)
        axs[ax_num].scatter(x, y, color="tab:green", marker=">", s=50)
        ukcp_div = 2.5
    else:
        ukcp_div = 1.5

    # Divider line for UKCP
    axs[ax_num].axvline(ukcp_div, color="lightgrey")

    # UKCP drivers
    y = plot_df[
            (plot_df["model"].isin(driving_models["UKCP"])) &
            (plot_df["data type"] == "standard") &
            (plot_df["project"] == "UKCP18 land-gcm")
        ]["value"]
    x = create_x_points(y, ukcp_div+1, 0.05)
    axs[ax_num].scatter(x, y, color="tab:purple", marker=">", s=50)

    # Final formatting etc.
    if var=="pr":
        axs[ax_num].axhline(linestyle=":", color="k", alpha=0.5)
    plt.setp(axs[ax_num].get_xticklabels(), rotation=45, ha="right")
    axs[ax_num].set_title("Downscaled Projections")
    axs[ax_num].set_xlabel(None)
    axs[ax_num].set_ylabel(None)

    ax_num = ax_num + 1

    # extra panel if a case study
    if len(driving_models['case study']) > 0:
        cs_models = driving_models["case study"]
        other_model_data = plot_df.query("(model not in @cs_models) & (project == 'CORDEX')")["value"]
        cs_model_data = plot_df.query("(model in @cs_models) & (project == 'CORDEX')")["value"]
        # do as from study1 and from study2.. i.e. seperate cordex to aerosol and non aerosol and then plot both in this panel
        sns.swarmplot(
            data=[other_model_data, cs_model_data],
            ax=axs[ax_num],
            size=swarm_size,
            palette=["tab:green"]
            )
        axs[ax_num].set_xticklabels(["Non case study models", "Case study models"])
        if var=="pr":
            axs[ax_num].axhline(linestyle=":", color="k", alpha=0.5)
        plt.setp(axs[ax_num].get_xticklabels(), rotation=45, ha="right")
        axs[ax_num].set_title("From study")
        axs[ax_num].set_xlabel(None)
        axs[ax_num].set_ylabel(None)

    # Final figure spacing etc.
    # axs[0].set_ylabel("%")
    plt.suptitle(f"{area.attributes['NAME']} {season} {var}")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, wspace=0.06)

    # save plot
    save_path = "/home/h02/tcrocker/code/EUCP_WP5_Lines_of_Evidence/esmvaltool/plots"
    plt.savefig(f"{save_path}/{area.attributes['NAME']}_{season}_{var}.png")
    plt.close()

def relative_to_global_plot(plot_df, area, season, var):
    # plot changes relative to model global warming
    f, ax = plt.subplots(1, 1, figsize=[19.2 ,  9.77])

    cmip6 = plot_df[(plot_df["project"] == "CMIP6") & (plot_df["data type"] == "weighted")]["value"]
    cmip5 = plot_df[(plot_df["project"] == "CMIP5") & (plot_df["data type"] == "weighted")]["value"]
    UKCP = plot_df[(plot_df["project"] == "UKCP18 land-gcm") & (plot_df["data type"] == "weighted")]["value"]
    plot_data = [cmip6, cmip5]
    positions = [0, 1]
    widths = [0.25, 0.25]
    order = ["CMIP6", "CMIP5"]

    if len(UKCP) > 0:
        plot_data.append(UKCP)
        positions.append(2)
        widths.append(0.25)
        order.append("UKCP18 land-gcm")
    box_plot(plot_data, ax, "black", "None", positions, widths)

    # size of dots in swarm plots
    swarm_size = 7

    # plot dots
    plot_data = plot_df.query("project in @order & `data type` == 'weighted'")
    sns.swarmplot(
        x="project",
        y="value",
        data=plot_data, ax=ax,
        size=swarm_size, palette=["tab:blue", "tab:orange", "tab:purple"],
        alpha=0.75,
        dodge=True,
        order=order
        )

    if var=="pr":
        ax.axhline(ls=':', color='k', alpha=0.75)

    # save plot
    save_path = "/home/h02/tcrocker/code/EUCP_WP5_Lines_of_Evidence/esmvaltool/plots"
    plt.suptitle(f"{area.attributes['NAME']} {season} {var}")
    plt.savefig(f"{save_path}/relative_to_global_{area.attributes['NAME']}_{season}_{var}.png")
    plt.close()
