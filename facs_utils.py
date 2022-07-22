# %%
# ==============================================================================
# // modified by Jackson
# // Notes are added as comments directly above function definition
# ==============================================================================

import os
import os.path
import time
from math import ceil, floor

import FlowCytometryTools
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from FlowCytometryTools import FCMeasurement, PolyGate, ThresholdGate
from ipywidgets import interactive
from lmfit import Minimizer, Model, Parameters
from matplotlib.patches import Polygon

ROUND_INTERVAL = 5.0

def make_titration(specimens, letters, numbers):
    """
    Returns a structure that defines a single titration, with C > 1 concentration stops.
    At least one of the arguments must be a length-C iterable in the same order 
    as the list of concentrations. 
    
    The returned structure is a list of tuples of (specimen, letter, number). For 
    example, the first tuple might be: (1, 'A', 2). This would show that the first
    concentration stop would be found in a file starting with "Specimen_001_A2".
    
    specimens: the specimen number or length-C iterable of specimen numbers
    letters: the letter code or length-C iterable of letter codes
    numbers: the well number or length-C iterable of well numbers
    
    Examples:
    >>> make_titration(1, 'A', range(1, 7))
    [(1, 'A', 1), (1, 'A', 2), (1, 'A', 3), (1, 'A', 4), (1, 'A', 5), (1, 'A', 6)]
    >>> make_titration(2, 'ABCD', range(8, 12))
    [(2, 'A', 8), (2, 'B', 9), (2, 'C', 10), (2, 'D', 11)]
    """
    
    try: _ = iter(specimens)
    except: specimens = [specimens]
    try: _ = iter(letters)
    except: letters = [letters]
    try: _ = iter(numbers)
    except: numbers = [numbers]

    assert len(specimens) > 1 or len(letters) > 1 or len(numbers) > 1, "Must have at least one iterable of length > 1"
    
    # Convert them all to be length-C
    num_concentrations = max(len(specimens), len(letters), len(numbers))
    
    assert len(specimens) in [num_concentrations, 1], "All iterable parameters must have same length"
    if len(specimens) != num_concentrations:
        specimens = specimens * num_concentrations
    assert len(letters) in [num_concentrations, 1], "All iterable parameters must have same length"
    if len(letters) != num_concentrations:
        letters = letters * num_concentrations
    assert len(numbers) in [num_concentrations, 1], "All iterable parameters must have same length"
    if len(numbers) != num_concentrations:
        numbers = numbers * num_concentrations

    # Return the list of concentration stops
    return list(zip((int(x) for x in specimens), letters, (int(x) for x in numbers)))
    
    
def generate_initial_polygon(num_points, xmin, xmax, ymin, ymax):
    """Generates a polygon shape for an initial gate within the given bounds."""
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    radius_x = (xmax - xmin) / 4.0
    radius_y = (ymax - ymin) / 4.0
    counter = np.arange(num_points).reshape(-1, 1)
    points = np.hstack([counter, counter])
    new_points = np.hstack([(center_x + radius_x * np.cos(2.0 * np.pi / num_points * points[:,0])).reshape(-1, 1),
                            (center_y + radius_y * np.sin(2.0 * np.pi / num_points * points[:,1])).reshape(-1, 1)])          
    return new_points


def get_sample_bounds(sample, axes):
    """Returns the xmin, xmax, ymin, and ymax of the given FCMeasurement or list of FCMeasurements, where 
    x corresponds to the first element of axes and y corresponds to the second."""
    try:
        min_x, max_x, min_y, max_y = [], [], [], []
        for samp in sample:
            data = samp.get_data()
            min_x.append(data[axes[0]].min())
            max_x.append(data[axes[0]].max())
            min_y.append(data[axes[1]].min())
            max_y.append(data[axes[1]].max())
        return floor(min(min_x) / ROUND_INTERVAL) * ROUND_INTERVAL, ceil(max(max_x) / ROUND_INTERVAL) * ROUND_INTERVAL, floor(min(min_y) / ROUND_INTERVAL) * ROUND_INTERVAL, ceil(max(max_y) / ROUND_INTERVAL) * ROUND_INTERVAL
    except:
        # Not a list
        return get_sample_bounds([sample], axes)


def vertex_control(sample, axes, num_points, results, key, log=False):
    """
    Creates an interactive UI to draw a gate around the given sample.
    sample: An FCMeasurement object containing the data to plot, or a list of 
            FCMeasurement objects.
    axes: The labels for the axes in a length-2 tuple, such as ("FSC-H", "SSC-H").
    num_points: The number of vertices to use on the gate.
    results: A dictionary in which to place the resulting gate vertices.
    key: The key to use in the results dictionary.
    log: If True, use log sliders and a hyperlog plot instead of linear.
    """
    def plot_me(**kwargs):
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        try:
            for samp in sample:
                samp.plot(axes, ax=ax)
                if log:
                    plt.xscale('log')
                    plt.yscale('log')
        except:
            sample.plot(axes, ax=ax)
            if log:
                plt.xscale('log')
                plt.yscale('log')
        X = []
        Y = []
        for i in range(num_points):
            X.append(kwargs[str(i) + "-x"])
            Y.append(kwargs[str(i) + "-y"])
        X = np.array(X)
        Y = np.array(Y)
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        if not log:
            plt.xlim(min(xmin, X.min()), max(xmax, X.max()))
            plt.ylim(min(ymin, Y.min()), max(ymax, Y.max()))
        results[key] = np.round(np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)]) / ROUND_INTERVAL) * ROUND_INTERVAL
        # Fill
        patch = Polygon(results[key], linewidth=2, edgecolor='b', facecolor='b', alpha=0.2)
        plt.gca().add_patch(patch)
        # Draw points
        plt.scatter(X, Y, color='b', marker='o')        
        # Label points
        for i in range(len(X)):
            ax.annotate(str(i), (X[i], Y[i]), (5.0, 5.0), textcoords='offset points')
            
        # Draw actual data
        #plt.plot(data[:,0], data[:,1], 'g')
        #plt.tight_layout()
        plt.show()
        
    xmin, xmax, ymin, ymax = get_sample_bounds(sample, axes)
    
    kws = {}
    hlayout = widgets.Layout(width="50%")
    vlayout = widgets.Layout(height="90%", width="60px")
    initial_poly = generate_initial_polygon(num_points, xmin, xmax, ymin, ymax)
    if key not in results:
        results[key] = initial_poly
    elif results[key].shape[0] < num_points:
        results[key] = np.vstack([results[key], initial_poly[results[key].shape[0]:]])
    elif results[key].shape[0] > num_points:
        results[key] = results[key][:num_points]
        
    vertices = results[key]
    slider_cls = widgets.FloatLogSlider if log else widgets.FloatSlider
    xmin_slider = np.log10(max(xmin, 1)) if log else xmin
    xmax_slider = np.log10(max(xmax, 1)) if log else xmax
    ymin_slider = np.log10(max(ymin, 1)) if log else ymin
    ymax_slider = np.log10(max(ymax, 1)) if log else ymax
    
    for i in range(num_points):
        xval = vertices[i,0]
        yval = vertices[i,1]
        
        kws[str(i) + "-x"] = slider_cls(min=xmin_slider, max=xmax_slider, step=0.01 if log else 5.0,
                                        value=xval,
                                        continuous_update=False, 
                                        layout=hlayout,
                                        style = {'description_width': '40px'})
        kws[str(i) + "-y"] = slider_cls(min=ymin_slider, max=ymax_slider, step=0.01 if log else 5.0,
                                        value=yval,
                                        continuous_update=False, 
                                        orientation='vertical', 
                                        layout=vlayout)
    w = interactive(plot_me, **kws)
    w.children[-1].layout = widgets.Layout(width='320px', height='320px')
    rows = [widgets.HBox([w.children[-1]] + [kws[str(i) + "-y"] for i in range(num_points)],
                        layout=widgets.Layout(height="320px"))]
    for i in range(num_points):
        rows.append(kws[str(i) + "-x"])
    print("Drag one of the sliders to make the plot visible.")
    display(widgets.VBox(rows))


# modified
def run_lmfit(x, y, init0, sat0, Kd0, graph, report=False, log=False, **kwargs):
    '''
    The Mass-Titer
    x: concentration list
    y: average bin positions
    init0: lower limit
    sat0: upper limit
    Kd0: initial Kd
    graph: whether or not to display the graph
    report: whether or not to print the fit report
    kwargs: additional keyword arguments for plotting
    '''
    # takes input arrays, initial guesses, plotting boolean
    # returns values in a dict, kd/stderr as values

    if np.isnan(y).any():
        return None

    def basic_fit(x, init, sat, Kd):
        return init + (sat - init) * x / (x + Kd)

    gmod = Model(basic_fit)
    # print(gmod.param_names)
    # print(gmod.independent_vars)
    gmod.set_param_hint('Kd', value=Kd0, min=0, max=40000)
    result = gmod.fit(list(y), x=x, init=init0, sat=sat0)
    r2 = 1 - result.redchi / np.var(y, ddof = 2)

    init = result.params['init'].value
    sat = result.params['sat'].value
    kd = result.params['Kd'].value
    # fit_positions = basic_fit(x, init, sat, kd)

    if report:
        print(result.fit_report())
    if graph == True:
        result.plot_fit(numpoints=10000, **kwargs)
        if log:
            plt.xscale('log')
        #result.plot_fit(numpoints=1000)
        return kd, sat, init, result.params['Kd'].stderr, result.chisqr, r2

    else:
        return kd, sat, init, result.params['Kd'].stderr, result.chisqr, r2


def transform_sample(sample, HYPERLOG_B = 100.0):
    """Performs the standard hyperlog transformation on the given sample.
    uses b=100 and transforms the follwing axis: ["FSC-H", "SSC-H",
    "SSC-W", "Alexa Fluor 680-A", "PE-A"]["FSC-H", "SSC-H", "SSC-W",
    "Alexa Fluor 680-A", "PE-A"]"""
    transformable_axes = ["FSC-H", "SSC-H", "SSC-W", "Alexa Fluor 680-A", "PE-A"]
    return sample.transform('hlog', channels=transformable_axes, b=HYPERLOG_B)


# Duplicate function as get_sample() in notebook but the one in notebook uses global variables
# If I had more time, I would put all notebook functions in here and replace all global variables
# with parameters
def path_2_sample(path, id_name, transform=False):
    """Gets a measurement from the given fcs file path, transforming its scatter values using a hyperlog
    transformation if specified."""
    sample = FCMeasurement(ID=id_name, datafile=path)
    if transform:
        return transform_sample(sample)
    return sample


# new function
def multi_plots_files(file_list, axes=["Alexa Fluor 680-A", "PE-A"], log=True, gates = None, transform=False, filename=None):
    """
    imports the fcs files in file_list and plots graphs for up to 16 samples in a 4x4 grid.
    optionally plots gates over the data
    optionally hyperlog transforms the data with HYPERLOG_B=100
    Parameters
    ----------
    file_list : list
        list of filenames
    axes : list
        list of axes to plot
        default = ["Alexa Fluor 680-A", "PE-A"]
    log : bool
        whether or not to plot on a log axis (default = True).
        Do not use when transform = True
    gates : Polygate object
        Gates to display overlaying the FACS data
        default = None (don't draw gates)
    transform : bool
        Whether or not to hyperlog transform data (default = False)
    filename : str
        saves an image of the plot as `filename` (ex: 'plot.jpg')
        default = None (does not save image)
    """
    plt.figure(figsize=(10, 10))
    previous_axes = []
    xmin, xmax, ymin, ymax = 1e9, -1e9, 1e9, -1e9
    for i, path in enumerate(file_list[:16]):
        plt.subplot(4, 4, i + 1)
        # Get the sample, gate it, and plot it
        sample = path_2_sample(path, os.path.splitext(os.path.basename(path))[0], transform=transform)
        sample.plot(axes, ax=plt.gca(), gates = gates)
        plt.title(os.path.splitext(os.path.basename(path))[0])
        if log:
            plt.yscale('log')
            plt.xscale('log')
        # Adjust limits
        previous_axes.append(plt.gca())
        new_xmin, new_xmax = plt.xlim()
        new_ymin, new_ymax = plt.ylim()
        xmin = min(new_xmin, xmin)
        xmax = max(new_xmax, xmax)
        ymin = min(new_ymin, ymin)
        ymax = max(new_ymax, ymax)
        for ax in previous_axes:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
#     plt.show()
    

def multi_plots_sample(sample_list, axes=["Alexa Fluor 680-A", "PE-A"], log=True, gates = None, filename=None):
    """
    graphs FACS plots for up to 16 samples in sample_list
    optionally plots gates over the data
    Parameters
    ----------
    sample_list : list
        list of facs sample objects
    axes : list
        list of axes to plot
        default = ["Alexa Fluor 680-A", "PE-A"]
    log : bool
        whether or not to plot on a log axis (default = True).
        Do not use when transform = True
    gates : Polygate object
        Gates to display overlaying the FACS data
        default = None (no gates drawn)
    filename : str
        saves an image of the plot as `filename` (ex: 'plot.jpg')
        default = None (does not save image)
    """
    plt.figure(figsize=(10, 10))
    xmin, xmax, ymin, ymax = 1, 1, 1, 1
    previous_axes = []
    for i, sample in enumerate(sample_list[:16]):
        plt.subplot(4, 4, i + 1)
        # Get the sample, gate it, and plot it
        sample.plot(axes, ax=plt.gca(), gates = gates)
        plt.title(sample.meta["WELL ID"])
        # plt.xlim(1, 1e5)
        # plt.ylim(1,1e5)
        # print(sample.data['PE-A'].min())
        # print(sample.data['Alexa Fluor 680-A'].min())
        if log:
            plt.yscale('log')
            plt.xscale('log')
        # Adjust limits
        previous_axes.append(plt.gca())
        new_xmin, new_xmax = plt.xlim()
        new_ymin, new_ymax = plt.ylim()
        xmin = min(new_xmin, xmin)
        xmax = max(new_xmax, xmax)
        ymin = min(new_ymin, ymin)
        ymax = max(new_ymax, ymax)
        for ax in previous_axes:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    # plt.show()
