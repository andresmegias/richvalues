#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Column density comparison

Andrés
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import richvalues as rv

#%%


fontsize = 10
plt.rcParams['font.size'] = fontsize

data = pd.read_csv('observed-column-densities.csv', comment='#', index_col=0)
data = rv.rich_dataframe(data,
                         domain={'HCCCN': [0, np.inf], 'CH3CN': [0, np.inf]})

t1 = time.time()
# ratio = data.create_column(lambda a,b: a/b, ['HCCCN', 'CH3CN'],
#                             unc_function = lambda a,b,da,db:
#                                 a/b * ((da/a)**2 + (db/b)**2)**0.5,
#                             domain=[0,np.inf])
ratio = rv.rich_array(data['HCCCN'] / data['CH3CN'])
t2 = time.time()
data['ratio'] = ratio
ratio.set_lims_factor([2,3])
colors = np.array(['tab:blue']*6 + ['tab:green']*16 + ['tab:red']*4
                  + ['orchid']*2)
names = data.index.values

locs = []
diffs = [0, 0.5, 1, 0.5, 1, 0.5,
         1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
for diff in diffs:
    locs += [diff]
locs = np.cumsum(locs)

plot_propagation_uncs = False

plt.figure(1)
plt.clf()
plt.errorbar(locs, ratio.centers, yerr=ratio.uncs, lolims=ratio.are_lolims,
             uplims=ratio.are_uplims, fmt=',', color='black')
plt.scatter(locs, ratio.centers, color=colors, zorder=3)
if plot_propagation_uncs:
    col_dens_HCCCN = rv.rich_array(data['HCCCN'])
    col_dens_CH3CN = rv.rich_array(data['CH3CN'])
    ratio_centers = col_dens_HCCCN.centers / col_dens_CH3CN.centers
    ratio_uncs = \
        ratio_centers * ((col_dens_HCCCN.uncs/col_dens_HCCCN.centers)**2
                         + (col_dens_CH3CN.uncs/col_dens_CH3CN.centers)**2)**0.5
    for i in range(len(ratio_centers)):
        if col_dens_CH3CN[i].is_uplim:
            ratio_centers[i] = \
                ((col_dens_HCCCN[i].center - col_dens_HCCCN[i].unc[0])
                 / col_dens_CH3CN[i].center)
    ratio_centers[-2] = (col_dens_HCCCN[-2].center
                         / col_dens_CH3CN[-2].center - col_dens_CH3CN[-2].unc[0])
    cond = np.isfinite(ratio.centers)
    locs_ = locs[cond]
    ratio_centers = ratio_centers[cond]
    ratio_uncs = ratio_uncs[:,cond]
    cond = ~ rv.rich_array(ratio[np.isfinite(ratio.centers)]).are_lims
    plt.errorbar(locs_[cond], ratio_centers[cond], yerr=ratio_uncs[:,cond], fmt=',',
                 alpha=0.7, color='chocolate')
    plt.scatter(locs_, ratio_centers, color='chocolate', alpha=0.7, zorder=3)
plt.xlim([-0.5, locs[-1]+0.5])
plt.axhline(y=1, linestyle='-', linewidth=0.8, color=(0.4,0.4,0.4))
plt.yscale('log')
cond = ~ np.isnan(ratio.centers)
plt.xticks(ticks=locs[cond], labels=names[cond], rotation=90,
           fontsize=0.8*fontsize)
plt.ylabel('HC$_3$N / CH$_3$CN abundance ratio')
y_top_labels = 1.05
ax = plt.gca()
xrange = np.diff(plt.xlim())
edges = np.array([0, 6, 22, 26, 28])
text_locs = []
for i in range(len(edges)-1):
    i1 = edges[i]
    i2 = edges[i+1]
    text_locs += [0.5+np.median(locs[i1:i2])]
texts = ['starless cores', 'Class 0/I protostars',
         'Class II proto-\nplanetary discs', 'comets']
fontsizes = fontsize*np.array([0.8, 0.8, 0.7, 0.7])
for i, text, fs in zip(text_locs, texts, fontsizes):
    plt.text(i/xrange, y_top_labels, text, ha='center', va='center',
             fontsize=fs, transform=ax.transAxes)
line_locs = [locs[0] - 0.5]
line_locs += list(locs[edges[1:-1]] - 0.5)
line_locs += [locs[-1] + 0.5]
for x in line_locs:
    plt.axvline(x=x, ymin=-0.5, ymax=1.1, linestyle='--', linewidth=1,
                color='gray', clip_on=False, zorder=0.5)
plt.tight_layout()

plt.show()

print('Elapsed time for calculations: {:.3f} s.'.format(t2-t1))