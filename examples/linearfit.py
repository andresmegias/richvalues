#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example file to use the library RichValues.
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

np.random.seed(8)
num_points = 30
slope_true, offset_true = 2., 10.
dispersion_x, dispersion_y = 0.6, 3.
errorbar_x, errorbar_y = 0.9, 3.6
x = rv.RichArray(np.random.uniform(0., 40., num_points),
                 [dispersion_x]*num_points)
y = rv.RichArray(slope_true * x.centers() + offset_true,
                 [dispersion_y]*num_points)
x = rv.RichArray(x.sample(1),
                 np.abs(np.random.normal(errorbar_x, errorbar_x/4, num_points)))
y = rv.RichArray(y.sample(1),
                 np.abs(np.random.normal(errorbar_y, errorbar_y/4, num_points)))
inds = np.argsort(x.centers())
x = x[inds]
y = y[inds]

x[0].is_uplim = True
x[0].center += 20*x[0].unc[1]
y[20].is_uplim = True
y[20].center += 16*y[20].unc[1]
y[-1].is_range = True
x[-1].center += 1*y[-1].unc[1]
y[-1].unc = [10*y[-1].unc[0], 5*y[1].unc[1]]
data = pd.DataFrame({'x': x, 'y': y})
data.to_csv('linear-data.csv', index=False)
data = rv.rich_dataframe(pd.read_csv('linear-data.csv'))
x = rv.rich_array(data['x'].values)
y = rv.rich_array(data['y'].values)

#%% Fit.

t1 = time.time()
result = rv.curve_fit(x, y, lambda x,m,b: m*x+b, guess=[2.,10.])
t2 = time.time()
slope, offset = result['parameters']
samples = result['samples']
print('Elapsed time for the fit: {:.1f} s.'.format(t2-t1))

#%% Plots.

plot_fit = True
plot_truth = True
color_fit = 'darkblue'
color_samples = 'tab:blue'
color_truth = 'tab:orange'

plt.figure(1, figsize=(7,4))
plt.clf()
rv.errorbar(x, y)
xlims = plt.xlim()
ylims = plt.ylim()
if plot_fit:
    x_ = np.linspace(0, xlims[1], 4)
    plt.plot(x_, slope.center * x_ + offset.center, color=color_fit, lw=1,
             label='median fit')
    plt.plot([], [], alpha=0, label='slope: {}'.format(slope.latex()))
    plt.plot([], [], alpha=0, label='offset: {}'.format(offset.latex()))
    if plot_truth:
        plt.plot(x_, slope_true * x_ + offset_true, color=color_truth,
                 label='ground truth', linestyle='--', lw=1)
        plt.plot([], [], alpha=0, label='slope: {:.0f}'.format(slope_true))
        plt.plot([], [], alpha=0, label='offset: {:.0f}'.format(offset_true))
    num_curves = min(400, samples.shape[0])
    inds = np.arange(samples.shape[0])
    np.random.shuffle(inds)
    for slope_i, offset_i in samples[inds][:num_curves]:
        plt.plot(x_, slope_i * x_ + offset_i, color=color_samples, alpha=0.01,
                 zorder=1)
    plt.legend()
    plt.title('Linear fit')
plt.ylim([0, ylims[1]])
plt.xlim([0, xlims[1]])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.tight_layout()

if plot_fit:
    plt.figure(2, figsize=(5,4))
    plt.clf()
    plt.scatter(samples[:,0], samples[:,1], color=color_samples,
                s=10, alpha=0.1)
    if plot_truth:
        plt.scatter(slope_true, offset_true, color=color_truth, s=10,
                    label='ground truth')
        plt.axvline(slope_true, color=color_truth, lw=1.5, ls=':', alpha=0.5)
        plt.axhline(offset_true, color=color_truth, lw=1.5, ls=':', alpha=0.5)
        plt.legend()
    plt.xlabel('slope')
    plt.ylabel('offset')
    plt.title('Correlation between parameters')
    plt.tight_layout()
