import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable, Tuple, List, Dict, Union

def lead_comparison_fig(t : Union[np.ndarray, Tuple[np.ndarray]], leads : Union[np.ndarray, Iterable[np.ndarray]], labels : Iterable[str] = None, leads_per_row=4,
                                plot_kwargs : Iterable[Dict] = None, xlabel="Time $t$ [ms]", ylabel="$V(t)$ [mV]", tight_layout=False,
                                legend = True,
                                legend_kwargs : Iterable[Dict] = None,
                                titles : Iterable[str] = None,
                                grid_lines = False):

    if not issubclass(type(leads), (Tuple, List)):
        leads = (leads,)

    if plot_kwargs is None:
        plot_kwargs = ({},) * len(leads)
    elif not issubclass(type(plot_kwargs), (Tuple, List)):
        plot_kwargs = (plot_kwargs,)*len(leads)

    if not issubclass(type(t), (Tuple, List)):
        t = (t,)*len(leads)

    if legend_kwargs is None:
        legend_kwargs = {}

    assert titles is None or len(titles) == leads[0].shape[1], "Titles need to be specified for each of the subplots/leads"
    #assert np.all([np.array(leads[0].shape == l.shape) for l in leads]), "All leads to be compared should have the same dimensionality"
    assert np.all([l.ndim == 2 for l in leads]), "All leads should have 2 dimensions"
    assert labels is None or len(labels) == len(leads), "Please provide no, or the same amount of labels than ecgs to compare"
    assert len(plot_kwargs) == len(leads), "Keyword arguments should be provided for each ecg to be compared"

    if labels is None:
        labels = (None,) * len(leads)

    nr_plots = len(leads)
    nr_leads = leads[0].shape[1]
    t_dims = [l.shape[0] for l in leads]

    assert all([t_dim == t_single.size for t_dim, t_single in zip(t_dims, t)]), f"Incompatible time found should be {t_dims} but is {[t_single.size for t_single in t]}"

    subplot_nrows = int(np.ceil(nr_leads / leads_per_row))
    fig, axes = plt.subplots(nrows=subplot_nrows, ncols=leads_per_row, 
                             sharex=True, sharey="row")
    fig.subplots_adjust(hspace=0, wspace=0)
    axes_flat = axes.flatten()
    if axes.ndim == 1:
        axes = axes[np.newaxis]
    handles = []
    
    for lead_i in range(nr_leads):
        lead_handles = []
        if grid_lines:
            axes_flat[lead_i].grid(which="both", alpha=.5, linestyle="--")
        for plot_i in range(nr_plots):
            lead_handles.append(axes_flat[lead_i].plot(t[plot_i], leads[plot_i][:, lead_i], label=labels[plot_i], **plot_kwargs[plot_i]))

        if titles is not None:
            axes_flat[lead_i].set_title(titles[lead_i], y=0.85, bbox=dict(facecolor='white', alpha=0.5, edgecolor="white"), zorder=10,
                                        ) #x=0.85)
        handles.append(tuple(lead_handles))

    #Remove excess axes (Move up the removed)
    nr_removed = len([ax.remove() for ax_i, ax in enumerate(axes_flat[::-1][:subplot_nrows*leads_per_row-nr_leads])])
    for i in range(min(nr_removed, axes.shape[1])):
        axes[0, -(i+1)].xaxis.set_tick_params(labelbottom=True)
        axes[0, -(i+1)].set_xlabel("Time $t$ [ms]")

    if legend: #nr_leads - 1:
        axes_flat[0].legend(**legend_kwargs)
    
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)

    if tight_layout:
        fig.tight_layout()
        
    return fig, axes, handles
