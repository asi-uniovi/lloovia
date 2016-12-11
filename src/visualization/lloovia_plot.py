"""
This module implements helper functions to simplify the creation
of plots and legends as the ones shown in the paper and in the
companion notebooks.
"""

import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import re


def prettify_names(instance_classes, amazon_or_azure=False):
    """Generate readable instance names from instance attributes"""
    colnames=[]
    regions = set()
    cont_regions = 0
    n_provider=""
    for col in instance_classes:
        if col.reserved:
            n_res = " [res]"
            n_zone = re.search(r'AZ(\d+)',col.cloud.name)
            if n_zone:
                n_zone = "-Z{}".format(n_zone.groups()[0])
            else:
                n_zone = ""
        else:
            n_res = " [dem]"
            n_zone = ""
        if "(" in col.cloud.name:
            m = re.search(r'(.*?) \((.*?)\)', col.cloud.name)
            n_region = "{}-{}".format(m.group(1), m.group(2))
            if amazon_or_azure:
                n_provider = "Amazon, "
        else:
            n_region = col.cloud.name
            if amazon_or_azure:
                n_provider = "Azure, "

        n_cloud = " ({}{}{})".format(n_provider, n_region, n_zone)
        name = "{}{}{}".format(col.name, n_cloud, n_res)
        colnames.append(name)
    return colnames

def plot_solution_slots(solution, ax=None, xlim=None, colors=None, figsize=None, 
        linewidth=None, kind="steps", colormaps=None, legend=True,
        pos_legend=None, amazon_or_azure=True):
    """
    Plots allocations as a stacked area plot. It can be used both for load-level 
    and for time-slot representations. Several customizations are possible
    through arguments.

    solution: the solution to plot, as provided by lloovia classes
    
    ax: the matplotlib axis in which to plot, useful for subplots. If None,
        a new figure is created
    
    xlim: the range to zoom. If None, the whole solution is plot
    
    colors: the colors to use for each instance class. This is a list of 
       matplotlib colors. If there are more ICs than colors, those are recycled.
       If none, a sensible colormap is used, by using matplotlib palettes.
    
    figsize: a tuple containing the width and height (inches) for the figure
    
    linewidth: width (in points) of the line around areas. If None, 0.5 is used.

    kind: the kind of area. "steps" by default

    colormaps: it is a dictionary whose keys are regexps and the values are
       matplotlib palettes. The colors in the palette are used for all ICs
       whose name matches the regexp. This allows to assign similar colors
       to sets of ICs who share something in their names (eg: the region)

    legend: Boolean indicating if a legend should be shown

    pos_legend: a pair indicating where to put the legend. The first element
       is passed to matplotlib's "loc" parameter and the second one to
       matplotlib "bbox_to_anchor" parameter. If None, the default position
       would be centered below (outside) the plot.

    amazon_or_azure: is a boolean indicating whether the solution to plot
       correspond to instance classes from amazon or azure. If True, some
       heuristics are applied to guess from the instance attributes if 
       it is an Amazon VM or an Azure VM, and this is used as part of
       the name of the instance class and for the colors of the legend
       (unless different colors were enforced through colors or colormaps
       parameters)
    """

    # Make a copy, since we will alter the data to plot it stepwise
    df = solution.copy()
    # Clip-out unwanted xlim
    if xlim is not None:
        df=df[slice(*xlim)]
    # Remove unused Instance Classes
    df = df[df.columns[(df.fillna(0) != 0).any()]]  

    # Reorder columns: first by provider, then by reserved vs. on-demand
    # then by efficiency (perf/price) with greatest efficiency first
    def order(s):
        efficiency = s.price/s.performance  # Use the inverse
        if "US" in s.cloud.name:
            efficiency -= 10
        if s.reserved:
            efficiency -= 10
        return efficiency
    def order(s):
        efficiency = s.price/s.performance  # Use the inverse
        r = "00" if s.reserved else "10"
        return "{}{}{}".format(r,s.cloud.name, efficiency)
    df = df[sorted(df.columns, key=order)]

    df.columns = prettify_names(df.columns, amazon_or_azure=amazon_or_azure)

    # Create the colormaps
    if colors is None:
        if colormaps is None:
            colormaps = {"Azure": cm.cool, "Amazon": cm.autumn}
        colors = []
        zone_counter = {}
        for zone in colormaps.keys():
            zone_counter[zone] = [1, len([x for x in df.columns if zone in x])]
        total_classes = len(df.columns)
        c_others = 0
        for c in df.columns:
            for k in colormaps.keys():
                if k in c:
                    colors.append(colormaps[k](zone_counter[k][0]/zone_counter[k][1]))
                    zone_counter[k][0]+=1
                    break
            else:
                colors.append(cm.bone(c_others/total_classes))
                c_others+=1

    if figsize is None:
        figsize = (4,2.5)
    if linewidth is None:
        linewidth = 0.5
    if pos_legend is None:
        pos_legend = ("upper center", (0.5, -0.25))

    if ax is None:
        fig, ax = plt.subplots()

    if kind=="steps":
        # Double the data, to make a stepwise function
        aux = pd.DataFrame()
        for c in df.columns:
            y = df[c].values
            yy = np.array([y,y]).flatten("F")
            aux[c] = yy[1:]
        x = df.index
        aux["time"] = np.array([x,x]).flatten('F')[:-1]
        df=aux
        p = df.plot(x="time", ax=ax, kind="area", stacked=True, figsize=figsize, 
                    linewidth=linewidth, alpha=0.7, legend=False,
                    color=colors, ylim=(0,None))

        # Create the legend and put it outside
        if legend:
            handles, labels = p.get_legend_handles_labels()
            p.legend(reversed([mpatches.Rectangle((0, 0), 1, 1, fc=handle.get_color(), 
                                              linewidth=handle.get_linewidth(), 
                                              alpha=handle.get_alpha())
                           for handle in handles[:len(df.columns)]]) , reversed(labels[:len(df.columns)]), 
                  loc=pos_legend[0], bbox_to_anchor=pos_legend[1])
    else:
        p = df.plot(ax=ax, kind="bar", stacked=True, figsize=figsize, 
                    linewidth=linewidth, alpha=0.7, legend=False,
                    color=colors, ylim=(0,None))
        for container in ax.containers:
            plt.setp(container, width=1)
        ticks = ax.xaxis.get_ticklocs()
        ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
        ax.xaxis.set_ticks(ticks[::5])
        ax.xaxis.set_ticklabels(ticklabels[::5])
        plt.tight_layout()
        if legend:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(reversed(handles), reversed(labels), loc=pos_legend[0], bbox_to_anchor=pos_legend[1])
        
    ax.set_xlabel("Hour")
    ax.set_ylabel("Number of virtual machines")
    return p
