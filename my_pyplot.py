import numpy as np
from matplotlib import rcParams
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import LogLocator
from scipy.optimize import curve_fit

rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = '16'
rcParams['font.sans-serif'] = 'Arial'
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Arial'
rcParams['mathtext.it'] = 'Arial:italic'
rcParams['mathtext.bf'] = 'Arial:bold'
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['agg.path.chunksize'] = 1000

'''
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(1, 2, wspace=0.03)
axes = gs.subplots(sharex=False, sharey=True)

fig, axes = plt.subplots(2, 3, sharex='all', sharey='row', figsize=(30, 12))
fig.subplots_adjust(wspace=0.03, hspace=0.03)

ax.scatter(x, y1, marker='s', s=50, c='C0', edgecolors=mpt.edgecolors[0], linewidths=3)
'''

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'navy', 'purple', 'wheat', 'olive', 'black']
edgecolors = [(0.122, 0.467, 0.706, 0.6), (1.000, 0.498, 0.055, 0.6), (0.173, 0.627, 0.173, 0.6), (0.839, 0.153, 0.157, 0.6), (0.580, 0.404, 0.741, 0.6), (0.549, 0.337, 0.294, 0.6), (0.890, 0.467, 0.761, 0.6), (0.498, 0.498, 0.498, 0.6)]
markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', 'P', '*', 'X', 'H', 'h', 'd', '|', '_']

def init_plot_single(width=8, height=6):
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax

def init_plot_double(width=12, height=6, sharex=False, sharey=False, wspace=0.1):
    fig, axes = plt.subplots(1, 2, sharex=sharex, sharey=sharey, figsize=(width, height))
    fig.subplots_adjust(wspace=wspace)
    return fig, axes

def init_plot_triple(width=18, height=6, sharex=False, sharey=False, wspace=0.1):
    fig, axes = plt.subplots(1, 3, sharex=sharex, sharey=sharey, figsize=(width, height))
    fig.subplots_adjust(wspace=wspace)
    return fig, axes

def forward(x):
    y = 1e3 / x
    return y

def linear_func(x, a, b):
    return a * x + b

def exp_func(x, a, b):
    return np.power(10, a * x + b)

def plot_with_err(ax, x, y, err, c='C0', marker='s', ls='none', label=None, ec=None, alpha=1, zorder=10):
    if ec is None:
        mec = c
        ebc = c
    else:
        mec = ec
        ebc = ec
    ax.errorbar(x, y, yerr=err, ls='none', color=ebc, elinewidth=1, capsize=4, capthick=1, alpha=alpha, zorder=0)
    ax.plot(x, y, marker=marker, markersize=7, color=c, mec=mec, mew=2, linestyle=ls, zorder=zorder, alpha=alpha, label=label)

def fit(x, y):
    params, cov = curve_fit(linear_func, x, y)
    slope, intercept = params
    slope_err = np.sqrt(cov[0, 0])
    return slope, intercept, slope_err

def fit_with_err(x, y, yerr, func='linear', p0=None):
    if func == 'linear':
        fit_func = linear_func
    elif func == 'exp':
        fit_func = exp_func
    params, cov = curve_fit(fit_func, x, y, sigma=yerr, absolute_sigma=True, p0=p0)
    a, b = params
    a_err = np.sqrt(cov[0, 0])
    return a, b, a_err

def fit_and_plot(ax, x, y, c='C0', ls ='--', xrange = None, label = None, alpha = 0.5):
    params, cov = curve_fit(linear_func, x, y)
    slope, intercept = params
    if xrange:
        x_fit = np.linspace(xrange[0], xrange[1], 100)
    else:
        x_fit = np.linspace(min(x), max(x), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color=c, linestyle=ls, alpha = alpha, zorder=0, label=label)
    return slope, intercept

def log_err(x, x_std):
    x_log = np.log10(x)
    x_err_log = np.log10(np.array(x) + 2 * np.array(x_std)) - x_log
    return x_log, x_err_log

def minor(ax):
    ax.minorticks_on()

def logminor(ax):
    ax.xaxis.set_minor_locator(LogLocator(base=10.0,subs=([0.1 * i for i in range(1,10)]),numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0,subs=([0.1 * i for i in range(1,10)]),numticks=12))

def legend(ax, loc='best'):
    ax.legend(fancybox=False, edgecolor='black', loc=loc)

def savefig(file):
    plt.savefig(file, bbox_inches='tight')
def savepdf(file):
    plt.savefig(file + '.pdf', bbox_inches='tight')
def savejpg(file):
    plt.savefig(file + '.jpg', dpi=600, bbox_inches='tight')