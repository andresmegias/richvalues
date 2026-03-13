#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example file to use the library RichValues.
https://github.com/andresmegias/richvalues

It imports a file of abundances of two molecules in several astronomical
sources, computes the ratio between both species for each source, makes
a plot of it and exports the results.

Andrés Megías Toledano
"""

# Libraries.
import time
import numpy as np
import pandas as pd
import richvalues as rv
import matplotlib.pyplot as plt

#%% Operations.

# Importing of the data.
data = pd.read_csv('observed-column-densities.csv', index_col=0, comment='#')
data = rv.rich_dataframe(data, domains=[0,np.inf])

# Calculation of the ratio of the two columns.
t1 = time.time()
use_create_column = False
if use_create_column:
    ratio = data.create_column('{}/{}', ['HC3N','CH3CN'],
                               domain=[0,np.inf], is_vectorizable=True)
else:
    ratio = data['HC3N'] / data['CH3CN']
t2 = time.time()
data['ratio'] = ratio
print('Elapsed time for calculations: {:.3f} s.'.format(t2-t1))

# Exporting of the data.
data.to_csv('observed-ratio.csv')

# Graphical options.
fontsize = 10.
plt.rcParams['font.size'] = fontsize
colors = np.array(['tab:blue']*7 + ['tab:green']*16 + ['tab:red']*4
                  + ['orchid']*2)
names = data.index.values
diffs = np.ones(len(names))
diffs[:6] = [0, 0.5, 1, 0.5, 1, 0.5]
locs = []
for (diff, xi) in zip(diffs, ratio):
    if xi.is_nan:
        diff = 0.
    locs += [float(diff)]
locs = np.cumsum(locs)

# Plot.
plot_approx_uncs = True
plt.close('all')
plt.figure(1, figsize=(7.5,5.0))
rv.errorbar(locs, ratio, fmt=',', color='black')
plt.scatter(locs, ratio.mains, color=colors, zorder=3)
if plot_approx_uncs:
    coldens1 = rv.rich_array(data['HC3N'])
    coldens2 = rv.rich_array(data['CH3CN'])
    ratio_approx = []
    for (coldens1i, coldens2i) in zip(coldens1, coldens2):
        if coldens1i.is_centered and coldens2i.is_centered:
            ratio_i = coldens1i.main / coldens2i.main
            rel_unc = (np.array(coldens1i.rel_unc)**2
                       +np.array(coldens2i.rel_unc)**2)**0.5
            ratio_unc = ratio_i * rel_unc
            ratio_approx += [rv.RichValue(ratio_i, ratio_unc)]
        elif coldens1i.is_centered and coldens2i.is_uplim:
            if max(coldens1i.rel_unc) < 0.20:
                lim = coldens1i.main - 3.*coldens1i.unc[0]
            else:
                # lim = np.nan
                x1 = max(0., coldens1i.main - 3.*coldens1i.unc[0])
                x2 = coldens1i.main - 1.*coldens1i.unc[0]
                sn = min(3., max(coldens1i.signal_noise))
                lim = x1 + (x2 - x1) / (2**(3-1) * 2**((3-sn)**2))
            ratio_lim = lim / coldens2i.main
            ratio_approx += [rv.RichValue(ratio_lim, is_lolim=True)]
        elif coldens1i.is_uplim and coldens2i.is_centered:
            if max(coldens2i.rel_unc) < 0.20:
                lim = coldens2i.main - 3.*coldens2i.unc[0]
            else:
                # lim = np.nan
                x1 = max(0., coldens2i.main - 3.*coldens2i.unc[0])
                x2 = coldens2i.main - 1.*coldens2i.unc[0]
                sn = min(3., max(coldens2i.signal_noise))
                lim = x1 + (x2 - x1) / (2**(3-1) * 2**((3-sn)**2))
            ratio_lim = coldens1i.main / lim
            ratio_approx += [rv.RichValue(ratio_lim, is_uplim=True)]
        else:
            ratio_approx += [rv.RichValue(np.nan)]
    ratio_approx = rv.rich_array(ratio_approx)
    rv.errorbar(locs, ratio_approx, fmt='.', color='darkorange', ecolor='orange',
                ms=15., lw=3., zorder=1.)
plt.xlim([-0.5, locs[-1] + 0.5])
plt.axhline(y=1, linestyle='-', linewidth=0.8, color=(0.4,0.4,0.4))
plt.yscale('log')
cond = ~np.isnan(ratio.mains)
plt.xticks(ticks=locs[cond], labels=names[cond], rotation=90,
           fontsize=0.8*fontsize)
plt.ylabel(r'${\rm HC_3N}$ / ${\rm CH_3CN}$ abundance ratio')
y_top_labels = 1.05
ax = plt.gca()
xrange = np.diff(plt.xlim())[0]
edges = np.array([0, 7, 23, 27, 29])
text_locs = []
for i in range(len(edges)-1):
    i1 = edges[i]
    i2 = edges[i+1]
    text_locs += [0.5 + np.median(locs[i1:i2])]
texts = ['starless cores', 'Class 0/I protostars',
         'Class II \nprotoplanetary discs', 'comets']
fontsizes = fontsize*np.array([0.8, 0.8, 0.7, 0.7])
for (i, text, fs) in zip(text_locs, texts, fontsizes):
    plt.text(i/xrange, y_top_labels, text, ha='center', va='center',
             fontsize=fs, transform=ax.transAxes)
line_locs = [locs[0] - 0.5]
line_locs += list(locs[edges[1:-1]] - 0.5)
line_locs += [locs[-1] + 0.5]
for x in line_locs:
    plt.axvline(x, ymin=-0.5, ymax=1.1, linestyle='--', linewidth=1.,
                color='gray', clip_on=False, zorder=0.5)
plt.tight_layout()
plt.show()

# Restore default font size.
plt.rcParams['font.size'] = plt.rcParamsDefault['font.size']