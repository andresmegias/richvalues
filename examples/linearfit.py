#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example file to use the RichValues library.
https://github.com/andresmegias/richvalues

It makes a linear fit of a set of points whith uncertainties and also ranges
and upper/lower limits.

Andrés Megías Toledano
"""

# Libraries.
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import richvalues as rv

#%% Initial data.

generate_data = False

slope_true, offset_true = 2, 10
dispersion_x, dispersion_y = 0.6, 3
errorbar_x, errorbar_y = 0.9, 3.6

if generate_data:
    
    seed = np.random.randint(int(1e4))
    
    num_points = 30
    np.random.seed(8)
    x = rv.RichArray(np.random.uniform(0., 40., num_points),
                     [dispersion_x]*num_points)
    y = rv.RichArray(slope_true * x.mains + offset_true,
                     [dispersion_y]*num_points)
    x = rv.RichArray(x.sample(),
                np.abs(np.random.normal(errorbar_x, errorbar_x/4, num_points)))
    y = rv.RichArray(y.sample(),
                np.abs(np.random.normal(errorbar_y, errorbar_y/4, num_points)))
    
    inds = np.argsort(x.mains)
    x = x[inds]
    y = y[inds]
    
    x[0].is_uplim = True
    x[0].main += 20*x[0].unc[1]
    y[20].is_uplim = True
    y[20].main += 16*y[20].unc[1]
    y[-1].is_range = True
    x[-1].main += 1*y[-1].unc[1]
    y[-1].unc = [10*y[-1].unc[0], 5*y[1].unc[1]]
    
    data = pd.DataFrame({'x': x, 'y': y})
    data.to_csv('linear-data.csv', index=False)
    
    np.random.seed(seed)
    
data = rv.rich_dataframe(pd.read_csv('linear-data.csv'))
x = rv.rich_array(data['x'].values)
y = rv.rich_array(data['y'].values)

#%% Fit.

t1 = time.time()
result = rv.curve_fit(x, y, lambda x,m,b: m*x+b, guess=[2.,10.],
                      consider_arg_intervs=True)
t2 = time.time()
slope, offset = result['parameters']
dispersion = result['dispersion']
samples = result['parameters samples']
print('Elapsed time for the fit: {:.1f} s.'.format(t2-t1))

#%% Plots.

plot_fit = True
plot_truth = True
color_fit = 'darkblue'
color_samples = 'tab:blue'
color_truth = 'tab:orange'

plt.figure(1, figsize=(7,4))
plt.clf()
rv.errorbar(x, y, color='gray')
xlims = plt.xlim()
ylims = plt.ylim()

if plot_fit:
    
    x_ = np.linspace(0, xlims[1], 4)
    plt.plot(x_, slope.main * x_ + offset.main, color=color_fit, lw=1,
             label='median fit')
    plt.plot([], [], alpha=0, label='slope: {}'.format(slope.latex()))
    plt.plot([], [], alpha=0, label='offset: {}'.format(offset.latex()))
    plt.plot([], [], alpha=0, label='dispersion: {}'
             .format(dispersion.latex()))
    
    if plot_truth:
        plt.plot(x_, slope_true * x_ + offset_true, color=color_truth,
                 label='ground truth', linestyle='--', lw=1)
        plt.plot([], [], alpha=0, label='slope: {}'.format(slope_true))
        plt.plot([], [], alpha=0, label='offset: {}'.format(offset_true))
        plt.plot([], [], alpha=0, label='dispersion: {}'.format(dispersion_y))
        
    num_curves = min(400, samples.shape[0])
    inds = np.arange(samples.shape[0])
    np.random.shuffle(inds)
    for slope_i, offset_i in samples[inds][:num_curves]:
        plt.plot(x_, slope_i * x_ + offset_i, color=color_samples, alpha=0.01,
                 zorder=1)
    plt.legend(fontsize=9)
    plt.title('Linear fit')
    
plt.ylim([0, ylims[1]])
plt.xlim([0, xlims[1]])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.tight_layout()

if plot_fit:
    
    ranges = [samples[:,i].max() - samples[:,i].min() for i in [0,1]]
    q1 = 0.02 if max(ranges) > 100. else 0.
    q2 = 1 - q1
    if q1 > 0:
        x1, x2 = [np.quantile(samples[:,0], qi) for qi in [q1, q2]]
        y1, y2 = [np.quantile(samples[:,1], qi) for qi in [q1, q2]]
        margin = 0.05
        xlims = np.array([min(x1, slope_true), max(x2, slope_true)])
        ylims = np.array([min(y1, offset_true), max(y2, offset_true)])
        xlims += np.array([-1, 1]) * margin * np.diff(xlims)
        ylims += np.array([-1, 1]) * margin * np.diff(ylims)
    else:
        xlims, ylims = None, None
    
    plt.figure(2, figsize=(6.5,3.3))
    plt.clf()
    
    plt.subplot(1,2,1)
    plt.scatter(samples[:,0], samples[:,1], color=color_samples,
                s=10, alpha=0.1)
    if plot_truth:
        plt.scatter(slope_true, offset_true, color=color_truth, s=10,
                    label='ground truth')
        plt.axvline(slope_true, color=color_truth, lw=1.5, ls=':', alpha=0.5)
        plt.axhline(offset_true, color=color_truth, lw=1.5, ls=':', alpha=0.5)
        plt.legend(fontsize=8)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.xlabel('slope')
    plt.ylabel('offset')
    plt.title('Correlation between parameters')
    
    dispersion_sample = result['dispersion sample']
    if q1 > 0:
        lim1, lim2 = [np.quantile(dispersion_sample, qi) for qi in [q1, q2]]
        cond = (dispersion_sample > lim1) & (dispersion_sample < lim2)
        dispersion_sample = dispersion_sample[cond]
    else:
        lim1, lim2 = 0, None
    plt.subplot(1,2,2)
    plt.hist(dispersion_sample, density=True, bins=20,
             histtype='stepfilled', edgecolor='k', alpha=0.9)
    if plot_truth:
        plt.axvline(x=dispersion_y, color=color_truth, lw=1.5, ls='--',
                    label='ground truth')
        plt.legend(fontsize=8)
    plt.xlim(lim1, lim2)
    plt.xlabel('dispersion value')
    plt.ylabel('frequency density')
    plt.title('Obtained dispersion sample')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
