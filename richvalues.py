#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rich Values Library
-------------------
Version 1.0

Copyright (C) 2022 - Andrés Megías Toledano

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    (1) Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer. 

    (2) Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.  
    
    (3) The name of the author may not be used to endorse or promote
    products derived from this software without specific prior written
    permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import copy
import math
import itertools
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

defaultparams = {
    'number of significant figures': 1,
    'domain': [-np.inf, np.inf],
    'size of samples': 10000,
    'minimum exponent for scientific notation': 4,
    'allow logarithmic scale': False,
    'limit for extra significant figure': 2.5,
    'minimum relative uncertainty range to use logarithmic scale': 5., 
    'minimum relative distance to the domain edges ' \
    + 'to apply analytic uncertainty propagation': 20.,
    'use 1-sigma combinations to estimate uncertainty propagation': False,
    'fraction of the central value for upper/lower limits': 0.1,
    'number of repetitions to estimate upper/lower limits': 4,
    'decimal exponent to define zero': -90.,
    'decimal exponent to define infinity': 90.,
    'multiplication symbol for scientific notation in LaTeX': '\\cdot'
    }
    

def round_sf(x,
        n=defaultparams['number of significant figures'],
        min_exp=defaultparams['minimum exponent for scientific notation'],
        lim_for_extra_sf=defaultparams['limit for extra significant figure']):
    """
    Round the number to the given number of significant figures.

    Parameters
    ----------
    x : float
        Input number.
    n : int, optional
        Number of significant figures. The default is 1.
    min_exp : int, optional
        If use_exp is True, minimum decimal exponent, in absolute value, to
        display the value in scientific notation. The default is 0.
    lim_for_extra_sf : float, optional
        If the number expressed in scientific notation has a base that is lower
        than this value, an additional significant figure will be used.

    Returns
    -------
    y : str
        Rounded number.
    """
    if np.isnan(x):
        return 'nan'
    use_exp = True
    if abs(np.floor(log10(abs(float(x))))) < min_exp:
        use_exp = False
    if not use_exp and 'e' in str(x):
        x = '{:f}'.format(x)
    x = float(x)
    sign = '-' if x < 0 else ''
    x = abs(x)
    base = '{:e}'.format(x).split('e')[0]
    if round(float(base), n) <= lim_for_extra_sf:
        n += 1
    y = str(float('{:.{}g}'.format(x, n)))
    integers = len(y.split('.')[0])
    if x > 1 and integers >= n:
        y = y.replace('.0','')
    digits = y.replace('.','')
    for i in range(15):
        if digits.startswith('0'):
            digits = digits[1:]
    digits = len(digits)
    if n > digits:
        y = y + '0'*(n-digits)
    if use_exp:
        y = '{:.{}e}'.format(float(y), max(n-1,0))
        y, a = y.split('e')
        if float(y) == 1 and not '.' in y:
            y += '.' + n*'0'
        y = '{}e{}'.format(y, a)
    y = sign + y
    if x == 0:
        y = '0'
    y = y.replace('e+','e').replace('e00','e0')
    return y

def round_sf_unc(x, dx,
        n=defaultparams['number of significant figures'],
        min_exp=defaultparams['minimum exponent for scientific notation'],
        lim_for_extra_sf=defaultparams['limit for extra significant figure']):
    """
    Round a value and its uncertainty depending on their significant figures.

    Parameters
    ----------
    x : float
        Input value.
    dx : float
        Uncertainty of the value.
    n : int, optional
        Number of significant figures. The default is 1.
    min_exp : int, optional
        If use_exp is True, minimum decimal exponent, in absolute value, to
        display the values in scientific notation. The default is 0.
    lim_for_extra_sf : float, optional
        If the number expressed in scientific notation has a base that is lower
        than this value, an additional significant figure will be used.

    Returns
    -------
    y : float
        Rounded value.
    """
    use_exp = True
    if ((float(x) > float(dx)
         and all(abs(np.floor(log10(abs(np.array([x, dx]))))) < min_exp))
         or (float(x) <= float(dx)
             and float('{:e}'.format(float(dx))
                       .split('e')[0]) > lim_for_extra_sf
             and abs(np.floor(log10(abs(float(dx))))) < min_exp)
         or (float(dx) == 0 and abs(np.floor(log10(abs(x)))) < min_exp)
         or np.isinf(min_exp)):
        use_exp = False
    if not use_exp and any(['e' in text for text in [str(x), str(dx)]]):
        x = '{:f}'.format(x)
        dx = '{:f}'.format(dx)
    x, dx = float(x), float(dx)
    sign = '' if x >= 0 else '-'
    if x < 0:
        x = abs(x)
    if x != 0 and dx > 0:
        y, a = '{:e}'.format(x).split('e')
        dy, b = '{:e}'.format(dx).split('e')
        a, b = int(a), int(b)
        y = float(y)
        dy = float(dy) * 10**(b - a)
        if (a >= b) or (b == a+1 and round(dy/10, n) <= lim_for_extra_sf):
            base_dy = float('{:e}'.format(dy).split('e')[0])
            if round(base_dy, n) <= lim_for_extra_sf:
                n += 1
            exp_y = int('{:e}'.format(y).split('e')[1])
            exp_dy = int('{:e}'.format(dy).split('e')[1])
            m = max(0, n + exp_y - exp_dy)
            y = '{:.{}g}'.format(y, m)
            dy = '{:.{}g}'.format(dy, n)
            if 'e' in dy:
                dy, c = dy.split('e')
                c = int(c)
                int_dy = len(dy.split('.')[0]) if '.' in dy else len(dy)
                if c >= 0:
                    dy += '0'*c
                else:
                    dy = '0.' + (abs(c)-int_dy)*'0' + dy.replace('.','')
            num_zeros_y = m - len(y.replace('.',''))
            num_zeros_dy = n - len(dy.replace('.',''))
            y += '.'*(('.' not in y) & (num_zeros_y > 0)) + num_zeros_y*'0'
            dy += '.'*(('.' not in dy) & (num_zeros_dy > 0)) + num_zeros_dy*'0'
        else:
            y = '0e0'
            dy = round_sf(dx, n, min_exp=0, lim_for_extra_sf=lim_for_extra_sf)
        if float(y) == 10:
            _, a = '{:e}'.format(x).split('e')
            y = '{:.{}f}'.format(float(y)/10, m)
            dy = '{:.{}f}'.format(float(dy)/10, m)
            a = int(a) + 1
        if a >= b:
            dec_y = len(y.split('.')[1]) if '.' in y else 0
            dec_dy = len(dy.split('.')[1]) if '.' in dy else 0
            num_zeros_y = dec_dy - dec_y
            num_zeros_dy = dec_y - dec_dy
            y += '.'*(('.' not in y) & (num_zeros_y > 0)) + num_zeros_y*'0'
            dy += '.'*(('.' not in dy) & (num_zeros_dy > 0)) + num_zeros_dy*'0'
        if float(y) != 0:
            y = '{}e{}'.format(y, a)
            dy = '{}e{}'.format(dy, a)
            if use_exp:
                y = y.replace('e+00','')
                dy = dy.replace('e+00','')
        else:
            y = '0e0'
            dy = round_sf(dx, n, min_exp=0, lim_for_extra_sf=lim_for_extra_sf)
    elif x == 0 and np.isinf(dx):
        y = '0e0'
        dy = 'inf'
    elif x == 0 and dx > 0:
        y = '0e0'
        dy = round_sf(dx, n, min_exp=0, lim_for_extra_sf=lim_for_extra_sf)
    elif dx == 0:
        if x != 0:
            y = round_sf(x, n+1, min_exp=0, lim_for_extra_sf=lim_for_extra_sf)
        else:
            y = '0' + use_exp*'e0'
        dy = '0e0'
    else:
        if not np.isinf(x):
            y = round_sf(x, n+1, min_exp=0, lim_for_extra_sf=lim_for_extra_sf)
            dy = y
        else:
            y, dy = 'inf', '0'
    if not use_exp and ('e' in y or 'e' in dy):
        if 'e' in y:
            y, a = y.split('e')
        else:
            a = 0
        if 'e' in dy:
            dy, b = dy.split('e')
        else:
            b = 0
        a, b = int(a), int(b)
        a = [a, b][np.argmax([abs(a), abs(b)])]
        int_y = len(y.split('.')[0]) if '.' in y else len(y)
        int_dy = len(dy.split('.')[0]) if '.' in dy else len(dy)
        if a < 0:
            y = '0.' + (abs(a)-int_y)*'0' + y.replace('.','')
            dy = '0.' + (abs(a)-int_dy)*'0' + dy.replace('.','')
        else:
            dec_y = len(y.split('.')[1]) if '.' in y else 0
            dec_dy = len(dy.split('.')[1]) if '.' in dy else 0
            if float(y) != 0 and float(dy) != 0:
                num_rep_zeros = (int('{:e}'.format(float(y)/float(dy))
                                     .split('e')[1]) + 1)
            else:
                num_rep_zeros = 0
            y = y[:int_y] + y[int_y+1:int_y+1+a] + '.' + y[int_y+1+a:]
            dy = dy[:int_dy] + dy[int_dy+1:int_dy+1+a] + '.' + dy[int_dy+1+a:]
            if y.endswith('.'):
                y = y[:-1]
            if dy.endswith('.'):
                dy = dy[:-1]
            for i in range(num_rep_zeros):
                if dy.startswith('0'):
                    dy = dy[1:]
            if dy.startswith('.'):
                dy = '0' + dy
            new_dec_y = len(y.split('.')[1]) if '.' in y else 0
            new_dec_dy = len(dy.split('.')[1]) if '.' in dy else 0
            y += '0'*(a - (dec_y - new_dec_y))
            dy += '0'*(a - (dec_dy - new_dec_dy))
        if float(y) == 0:
            y = '0' + use_exp*'e0'
    if float(y) != 0:
        y = sign + y
    if not use_exp:
        y = y.replace('e0','')
        dy = dy.replace('e0','')
    else:
        if not np.isinf(x):
            a = int('{:e}'.format(float(y)).split('e')[1])
            if abs(a) < min_exp:
                x = float(sign + str(x))
                y, dy = round_sf_unc(x, dx, n, np.inf, lim_for_extra_sf)
    return y, dy


def round_sf_uncs(x, dx,
        n=defaultparams['number of significant figures'],
        min_exp=defaultparams['minimum exponent for scientific notation'],
        lim_for_extra_sf=defaultparams['limit for extra significant figure']):
    """
    Round a value and its uncertainties depending on their significant figures.

    Parameters
    ----------
    x : float
        Input value.
    dx : list
        Lower and upper uncertainties of the value.
    n : int, optional
        Number of significant figures. The default is 1.
    min_exp : int, optional
        If use_exp is True, minimum decimal exponent, in absolute value, to
        apply scientific notation. The default is 0.
    lim_for_extra_sf : float, optional
        If the number expressed in scientific notation has a base that is lower
        than this value, an additional significant figure will be used.

    Returns
    -------
    y : float
        Rounded value.
    """
    dx1, dx2 = dx
    y1, dy1 = round_sf_unc(x, dx1, min_exp=0,
                           lim_for_extra_sf=lim_for_extra_sf)
    y2, dy2 = round_sf_unc(x, dx2, min_exp=0,
                           lim_for_extra_sf=lim_for_extra_sf)
    num_sf_1 = len(y1.split('e')[0].replace('.',''))
    num_sf_2 = len(y2.split('e')[0].replace('.',''))
    if num_sf_2 > num_sf_1:
        diff = num_sf_2 - num_sf_1
        y1, dy1 = round_sf_unc(x, dx1, n+diff, min_exp, lim_for_extra_sf)
        y2, dy2 = round_sf_unc(x, dx2, n, min_exp, lim_for_extra_sf)
    else:
        diff = num_sf_1 - num_sf_2
        y1, dy1 = round_sf_unc(x, dx1, n, min_exp, lim_for_extra_sf)
        y2, dy2 = round_sf_unc(x, dx2, n+diff, min_exp, lim_for_extra_sf)
    y = y1 if dx2 > dx1 else y2
    dy = [dy1, dy2]
    return y, dy

class RichValue():
    """
    A class to store a value with uncertainties or with upper/lower limits.
    """
    
    def __init__(self, center, unc=0, is_lolim=False, is_uplim=False,
                 num_sf=defaultparams['number of significant figures'],
                 min_exp=defaultparams['minimum exponent for '
                                       + 'scientific notation'],
                 domain=defaultparams['domain'],
                 len_sample=defaultparams['size of samples'],
                 allow_log_scale=defaultparams['allow logarithmic scale']):
        """
        Parameters
        ----------
        center : float
            Central value of the rich value, or value of the upper/lower limit.
        unc : float / list (float), optional
            Lower and upper uncertainties associated with the central value.
            The default is [0,0].
        is_lolim : bool
            If True, it means that the central value is actually a lower limit.
            The default is False.
        is_uplim : bool
            If True, it means that the central value is actually an upper limit.
            The default is False.
        num_sf : int, optional
            Number of significant figures to use for displaying the numbers.
            The default is 1.
        min_exp : int, optional
            If use_exp is True, minimum decimal exponent, in absolute value, to
            apply scientific notation. The default is 4.
        domain : list (float), optional
            The domain of the rich value, that is, the minimum and maximum
            values that it can take.
        len_sample : int, optional
            In case a function has to be applied to this rich value and there
            is no analytic formula for propagating the uncertainties, it will
            be calculated from distributions with this number of samples.
            The default is 10000.
        allow_log_scale : bool, optional
            If True, the value will be displayed as an exponent in decimal
            base. The default is False.
        """
        is_range = False
        unc_or = copy.copy(unc)
        if type(center) in [list, tuple]:
            if center[0] == domain[0]:
                is_uplim = True
                unc = 0
            if center[1] == domain[1]:
                is_lolim = True
                unc = 0
            if is_lolim and is_uplim:
                center = np.nan
                unc = np.inf
            elif is_uplim:
                center = center[1]
            elif is_lolim:
                center = center[0]
            if not (is_lolim or is_uplim):
                is_range = True
                if center[0] >= center[1]:
                    raise Exception('Wrong interval: {}'.format(center))
                unc = (center[1] - center[0]) / 2
                center = (center[0] + center[1]) / 2
        if not hasattr(unc, '__iter__'):
            unc = [unc, unc]
        center = np.nan if any(np.isinf(unc)) else center
        unc = np.nan_to_num(unc, nan=0.) if any(np.isnan(unc)) else unc
        unc = list(unc)
        unc[0] = abs(unc[0])
        if not np.isinf(center):
            unc[0] = min(center - domain[0], unc[0])
            unc[1] = min(domain[1] - center, unc[1])
        if not (is_lolim or is_uplim or is_range) and unc[1] < 0:
            unc_text = ('Superior uncertainty' if hasattr(unc_or, '__iter__')
                        else 'Uncertainty')
            raise Exception('{} cannot be negative.'.format(unc_text))
        if (not np.isnan(center) and unc != [0,0]
                and not domain[0] <= center <= domain[1]):
            raise Exception('Invalid domain {} for center in {}.'
                            .format(domain, center))
        is_lim = is_lolim or is_uplim
        is_range = is_lolim or is_uplim if not is_range else is_range
        len_sample = int(len_sample)
        self.center = center
        self.unc = unc
        self.is_lolim = is_lolim
        self.is_uplim = is_uplim
        self.num_sf = num_sf
        self.min_exp = min_exp
        self.domain = domain
        self.len_sample = len_sample
        self.allow_log_scale = allow_log_scale
        self.is_lim = is_lim
        self.is_range = is_range
        self.lim_for_extra_sf = \
            defaultparams['limit for extra significant figure']
        
    def rel_unc(self):
        """Relative uncertainties of the rich value"""
        m, s = self.center, self.unc
        with np.errstate(all='ignore'):
            runc = list(np.array(s) / m) if m != 0 else [np.inf, np.inf]
        return runc
        
    def ampl(self):
        """Amplitudes of the rich value"""
        m, b = self.center, self.domain
        a = [m - b[0], b[1] - m]
        return a
        
    def rel_ampl(self):
        """Relative amplitudes of the rich value"""
        s, a = self.unc, self.ampl()
        with np.errstate(all='ignore'):
            a_s = [a[0]/s[0] if abs(s[0]) != 0 else np.inf,
                   a[1]/s[1] if abs(s[1]) != 0 else np.inf]
        return a_s
    
    def check_limit(self, sigmas=3.):
        """Autodetect upper/lower limit"""
        a_s = self.rel_ampl()
        if not self.is_lim:
            if a_s[0] <= 1:
                self.is_uplim = True
                sigmas = min(sigmas, (self.domain[1]-self.center)/self.unc[1])
                sigmas = 1. if self.is_range else sigmas - 1e-20
                self.center = self.center + sigmas*self.unc[1]
            if a_s[1] <= 1:
                self.is_lolim = True
                sigmas = min(sigmas, (self.center-self.domain[0])/self.unc[0])
                sigmas = 1. if self.is_range else sigmas - 1e-20
                self.center = self.center - sigmas*self.unc[1]
            if self.is_lolim and self.is_uplim:
                self.center = np.nan
                self.unc = [np.inf, np.inf]
  
    def check_interval(self):
        """Autodetect interval"""
        with np.errstate(all='ignore'):
            a_s = np.array(self.ampl()) / max(self.unc)
        if not self.is_lim:
            if min(a_s) <= 1:
                self.is_range = True
  
    def set_lims_factor(self, c=4.):
        """Set uncertainties of limits with respect to cetral values."""
        if self.is_lolim or self.is_uplim:
            self.unc = [self.center / c, self.center / c]
        
    def _format_as_rich_value(self):
        lim_log_scale = defaultparams['minimum relative uncertainty range '
                                      + 'to use logarithmic scale']
        center = copy.copy(self.center)
        unc = copy.copy(self.unc)
        is_lolim = self.is_lolim
        is_uplim = self.is_uplim
        domain = self.domain
        is_range = self.is_range and not self.is_lim
        allow_log_scale = self.allow_log_scale
        min_exp = self.min_exp
        lim_for_extra_sf = self.lim_for_extra_sf
        x = copy.copy(center)
        dx = copy.copy(unc)
        n = copy.copy(self.num_sf)
        use_exp = True
        if ((float(x) > float(max(dx))
             and abs(np.floor(log10(abs(float(x))))) < min_exp)
             or (float(x) <= float(max(dx))
                 and float('{:e}'.format(max(dx))
                           .split('e')[0]) > lim_for_extra_sf
                 and any(abs(np.floor(log10(abs(np.array(dx)))))
                         < min_exp))
             or (dx == [0,0] and abs(np.floor(log10(abs(x)))) < min_exp)
             or (self.is_lim and abs(np.floor(log10(abs(float(x))))) < min_exp)
             or np.isinf(min_exp)):
            use_exp = False
        x1 = center - unc[0]
        x2 = center + unc[1]
        if not np.isnan(domain[0]):
            x1 = max(domain[0], x1)
        if not np.isnan(domain[1]):
            x2 = min(domain[1], x2)
        with np.errstate(all='ignore'):
            new_unc = [center - x1, x2 - center]
        if ((len(round_sf(new_unc[0])) > len(round_sf(unc[0]))
                 or len(round_sf(new_unc[1])) > len(round_sf(unc[0])))
                and not allow_log_scale):
            n -= 1
        n = max(1, n)
        unc = new_unc
        if (allow_log_scale and not is_range
                and sum(unc) > lim_log_scale * abs(center)):
            log_scale = True
        else:
            log_scale = False
        if log_scale:
            sign = '' if center >= 0 else '-'
            unc = [np.log10(abs(center)) - np.log10(abs(center-unc[0])),
                   np.log10(abs(center+unc[1])) - np.log10(abs(center))]
            center = np.log10(abs(center))
            use_exp = False
            min_exp = 0
        if not is_range and not np.isnan(center):
            x = center
            dx1, dx2 = unc
            if not self.is_lim:
                y, (dy1, dy2) = round_sf_uncs(x, [dx1, dx2], n, min_exp,
                                              lim_for_extra_sf)
                y, dy1, dy2 = str(y), str(dy1), str(dy2)
                if 'e' in y:
                    y, a = y.split('e')
                    a = int(a)
                else:
                    a = 0
                if 'e' in dy1:
                    dy1, _ = dy1.split('e')
                if 'e' in dy2:
                    dy2, _ = dy2.split('e')
                if dy1 == dy2:
                    if float(dy1) != 0:
                        text = '{}+/-{} e{}'.format(y, dy1, a)
                    else:
                        text = '{} e{}'.format(y, a)
                else:
                    text = '{}-{}+{} e{}'.format(y, dy1, dy2, a)
                if not use_exp:
                    text = text.replace(' e0','')
            else:
                y = round_sf(x, n, min_exp, lim_for_extra_sf)
                if 'e' in y:
                    y, a = y.split('e')
                    a = int(a)
                else:
                    a = 0
                if is_lolim:
                    sign = '>'
                elif is_uplim:
                    sign = '<'
                text = '{} {} e{}'.format(sign, y, a)
            if use_exp:
                text = text.replace('e-0', 'e-')
                a = int(text.split('e')[1])
                if abs(a) < min_exp:
                    text = str(RichValue(x, dx, is_lolim, is_uplim, n, np.inf,
                                     domain, self.len_sample, allow_log_scale))
            else:
                text = text.replace(' e0','')
        elif not is_range and np.isnan(center):
            text = 'nan'
        else:
            x1 = RichValue(center - unc[0], min_exp=min_exp, domain=domain)
            x2 = RichValue(center + unc[1], min_exp=min_exp, domain=domain)
            text = '{} -- {}'.format(x1, x2)
        if log_scale:
            text = '{}10^ ({})'.format(sign, text)
        return text
    
    def _format_as_latex_value(self, mult_symbol='\\cdot'):
        lim_log_scale = defaultparams['minimum relative uncertainty range '
                                      + 'to use logarithmic scale']
        center = copy.copy(self.center)
        unc = copy.copy(self.unc)
        domain = self.domain
        is_lolim = self.is_lolim
        is_uplim = self.is_uplim
        allow_log_scale = self.allow_log_scale
        min_exp = self.min_exp
        lim_for_extra_sf = self.lim_for_extra_sf
        is_range = self.is_range and not (self.is_lolim or self.is_uplim)
        use_exp = True
        if (allow_log_scale and not is_range
                and sum(unc) > lim_log_scale *abs(center)):
            log_scale = True
        else:
            log_scale = False
        if log_scale:
            unc = [np.log10(center) - np.log10(center-unc[0]),
                   np.log10(center+unc[1]) - np.log10(center)]
            center = np.log10(center)
            use_exp = False
            min_exp = 0
        x = copy.copy(center)
        dx = copy.copy(unc)
        n = copy.copy(self.num_sf)
        if ((float(x) > float(max(dx))
             and abs(np.floor(log10(abs(float(x))))) < min_exp)
             or (float(x) <= float(max(dx))
                 and float('{:e}'.format(max(dx))
                           .split('e')[0]) > lim_for_extra_sf
                 and any(abs(np.floor(log10(abs(np.array(dx))))) < min_exp))
             or (dx == [0,0] and abs(np.floor(log10(abs(x)))) < min_exp)
             or (self.is_lim and abs(np.floor(log10(abs(float(x))))) < min_exp)
             or np.isinf(min_exp)):
            use_exp = False
        y, dy = round_sf_uncs(x, dx, n, min_exp, lim_for_extra_sf)
        text = ''
        non_numerics = ['nan', 'NaN', 'None', 'inf', '-inf']
        is_numeric = False if str(center) in non_numerics else True
        if is_numeric:
            if not is_range:
                _, unc_r = round_sf_uncs(x, dx, n)
                unc_r = np.array(unc_r, float)
            if not is_range and not use_exp:
                if not (is_lolim or is_uplim):
                    if unc_r[0] == unc_r[1]:
                        if unc_r[0] == unc_r[1] == 0:
                            y = round_sf(x, n+1, np.inf, lim_for_extra_sf)
                            text = '${}$'.format(y)
                        else:
                            y, dy = round_sf_unc(x, dx[0], n, min_exp,
                                                 lim_for_extra_sf)
                            text = '${} \pm {}$'.format(y, dy)
                    else:
                        y, dy = round_sf_uncs(x, dx, n, min_exp,
                                              lim_for_extra_sf)
                        text = '$'+y + '_{-'+dy[0]+'}^{+'+dy[1]+'}$'
                else:
                    if is_lolim:
                        sign = '>'
                    elif is_uplim:
                        sign = '<'
                    y = round_sf(x, n, min_exp, lim_for_extra_sf)
                    text = '${} {}$'.format(sign, y)
            elif not is_range and use_exp:
                if not (is_lolim or is_uplim):
                    if unc_r[0] == unc_r[1]:
                        if unc_r[0] == unc_r[1] == 0:
                            y = round_sf(x, n+1, min_exp, lim_for_extra_sf)
                            y, a = y.split('e')
                            a = str(int(a))
                            text = ('${} {}'.format(y, mult_symbol)
                                    + ' 10^{'+a+'}$')
                        else:
                            y, dy = \
                                round_sf_unc(x, dx[0], n, min_exp,
                                             lim_for_extra_sf)
                            if 'e' in y:
                                y, a = y.split('e')
                                dy, a = dy.split('e')
                            else:
                                a = 0
                            a = str(int(a))
                            text = ('$({} \pm {})'.format(y, dy)
                                     + mult_symbol + '10^{'+a+'}$')
                    else:
                        y, dy = round_sf_uncs(x, [dx[0], dx[1]], n, min_exp,
                                              lim_for_extra_sf)
                        if 'e' in y:
                            y, a = y.split('e')
                            dy1, a = dy[0].split('e')
                            dy2, a = dy[1].split('e')
                        else:
                            dy1, dy2 = dy
                            a = 0
                        a = str(int(a))
                        text = ('$'+y + '_{-'+dy1+'}^{+'+dy2+'} '
                                + mult_symbol + ' 10^{'+a+'}' + '$')
                else:
                    if is_lolim:
                        symbol = '>'
                    elif is_uplim:
                        symbol = '<'
                    y = round_sf(x, n, min_exp=0,
                                 lim_for_extra_sf=lim_for_extra_sf)
                    y, a = y.split('e')
                    a = str(int(a))
                    text = ('${} {} {}'.format(symbol, y, mult_symbol)
                            + ' 10^{'+a+'}$')
                if use_exp:
                    text = text.replace('e-0', 'e-').replace('e+','e')
                    a = int(text.split('10^{')[1].split('}')[0])
                    if abs(a) < min_exp:
                        text = RichValue(x, dx, is_lolim, is_uplim, n, np.inf,
                                         domain, self.len_sample,
                                         allow_log_scale).latex()
            else:
                x1 = RichValue(center - unc[0], min_exp=min_exp, domain=domain)
                x2 = RichValue(center + unc[1], min_exp=min_exp, domain=domain)
                text = '{} -- {}'.format(x1.latex(), x2.latex())
            if log_scale:
                text = '$10^{' + text + '}$'
        else:
            text = (str(center).replace('NaN','nan').replace('nan','...')
                    .replace('inf','$\infty$'))
        return text
        
    def __repr__(self):
        return self._format_as_rich_value()
    def __str__(self):
        return self._format_as_rich_value()
   
    def latex(self, mult_symbol=defaultparams['multiplication symbol for '
                                              +'scientific notation in LaTeX']):
        """Display in LaTeX format"""
        return self._format_as_latex_value(mult_symbol) 
   
    def __neg__(self):
        center = copy.copy(self.center)
        unc = copy.copy(self.unc)
        domain = copy.copy(self.domain)
        if not self.is_range:
            x = -center
        else:
            if self.is_lolim:
                x1, x2 = center, domain[1]
            elif self.is_uplim:
                x1, x2 = domain[0], center
            else:
                x1, x2 = center - unc[0], center + unc[1]
            x = [-x2, -x1]
        dx = unc
        domain = [-domain[0], -domain[1]]
        domain = [min(domain), max(domain)]
        new_rich_value = \
            RichValue(x, dx, False, False, self.num_sf, self.min_exp,  domain,
                      self.len_sample, self.allow_log_scale)
        return new_rich_value
    
    def inv(self):
        """Inverse of the rich value"""
        sigmas = defaultparams['minimum relative distance to the domain edges ' \
                               + 'to apply analytic uncertainty propagation']
        center = copy.copy(self.center)
        unc = copy.copy(self.unc)
        domain = copy.copy(self.domain)
        is_range = self.is_range
        is_lolim = self.is_lolim
        is_uplim = self.is_uplim
        s = np.sign(np.mean(domain))
        domain1 = 1/domain[1] if domain[1] != 0 else s*np.inf
        domain2 = 1/domain[0] if domain[0] != 0 else s*np.inf
        domain = [domain1, domain2]
        if not is_range and max(unc) < 1/sigmas*center:
            x = 1 / center
            dx = np.array(unc) / center * x
            new_rich_value = \
                RichValue(x, dx, is_lolim, is_uplim, self.num_sf, self.min_exp,
                          domain, self.len_sample, self.allow_log_scale)
        elif is_range:
            if is_lolim and not is_uplim:
                a, b = center, domain[1]
            elif is_uplim and not is_lolim:
                a, b = domain[0], center
            else:
                a, b = center - unc[0], center + unc[1]
            if a < 0 and b > 0:
                x = np.nan
                dx = np.inf
                domain = [-np.inf, np.inf]
            else:
                s = np.sign(np.mean([a,b]))
                x1 = 1/b if b != 0 else s*np.inf
                x2 = 1/a if a != 0 else s*np.inf
                x = [x1, x2]
                dx = 0
                if domain[0] < 0 and domain[1] > 0:
                    domain = [-np.inf, np.inf]
                
            new_rich_value = RichValue(x, dx, None, None, self.num_sf,
                                       self.min_exp, domain, self.len_sample,
                                       self.allow_log_scale)
        else:
            new_rich_value = \
                function_with_rich_values(lambda x: 1/x, self,
                                          use_sigma_combs=True)
        return new_rich_value
    
    def __add__(self, other):
        other_ = (other.center if 'RichValue' in str(type(other))
                  and other.unc==[0,0] and other.domain==[-np.inf,np.inf]
                  else other)
        if type(other_) is RichValue:
            new_rich_value = add_two_rich_values(self, other_)
        else:
            if other_ != 0:
                x = self.center + other_
                dx = copy.copy(self.unc)
                new_rich_value = \
                    RichValue(x, dx, self.is_lolim, self.is_uplim,
                              self.num_sf, self.min_exp, self.domain,
                              self.len_sample, self.allow_log_scale)
                new_rich_value.is_range = self.is_range
            else:
                new_rich_value = RichValue(0, domain=self.domain)
        return new_rich_value
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return -(self - other)
    
    def __mul__(self, other):
        other_ = (other.center if 'RichValue' in str(type(other))
                  and other.unc==[0,0] and other.domain==[-np.inf,np.inf]
                  else other)
        if type(other_) is RichValue:
            new_rich_value = multiply_two_rich_values(self, other_)
        else:
            if other_ != 0:
                with np.errstate(all='ignore'):
                    x = self.center * other_
                    dx = np.array(self.unc) * other_
                    domain = np.array(self.domain) * other_
                    domain = [min(domain), max(domain)]
                new_rich_value = \
                    RichValue(x, dx, self.is_lolim, self.is_uplim,
                              self.num_sf, self.min_exp, domain,
                              self.len_sample, self.allow_log_scale)
                new_rich_value.is_range = self.is_range
            else:
                new_rich_value = RichValue(0, domain=self.domain)
        return new_rich_value
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        other_ = (other.center if 'RichValue' in str(type(other))
                  and other.unc==[0,0] and other.domain==[-np.inf,np.inf]
                  else other)
        if type(other_) is RichValue:
            new_rich_value = divide_two_rich_values(self, other_)
        else:
            if other_ != 0:
                with np.errstate(all='ignore'):
                    x = self.center / other_
                    dx = np.array(self.unc) / other_
                    domain = np.array(self.domain) / other_
                    domain = [min(domain), max(domain)]
                new_rich_value = \
                    RichValue(x, dx, self.is_lolim, self.is_uplim,
                              self.num_sf, self.min_exp, domain,
                              self.len_sample, self.allow_log_scale)
                new_rich_value.is_range = self.is_range
            else:
                new_rich_value = RichValue(np.nan, domain=self.domain)
        return new_rich_value

    def __rtruediv__(self, other):
        other_ = (other.center if 'RichValue' in str(type(other))
                  and other.unc==[0,0] and other.domain==[-np.inf,np.inf]
                  else other)
        if type(other_) is RichValue:
            new_rich_value = divide_two_rich_values(other_, self)
        else:
            if other_ != 0:
                with np.errstate(all='ignore'):
                    x = other_ / self.center
                    dx = x * np.array(self.unc) / self.center
                    domain = other_ / np.array(self.domain)
                    domain = [min(domain), max(domain)]
                new_rich_value = \
                    RichValue(x, dx, self.is_lolim, self.is_uplim,
                              self.num_sf, self.min_exp, domain,
                              self.len_sample, self.allow_log_scale)
                new_rich_value.is_range = self.is_range
            else:
                new_rich_value = RichValue(0, domain=self.domain)
        return new_rich_value
    
    def __pow__(self, other):
        sigmas = defaultparams['minimum relative distance to the domain edges ' \
                               + 'to apply analytic uncertainty propagation']
        center = copy.copy(self.center)
        unc = copy.copy(self.unc)
        domain = copy.copy(self.domain)
        if ((domain[0] >= 0
             and (type(other) is RichValue
                  or max(unc) > 1/sigmas * abs(center)))
                    or (domain[0] < 0 and type(other) is not RichValue)
                    and int(other) == other
                    and max(unc) > 1/sigmas * abs(center)):
            other_ = (other if type(other) is RichValue
                      else RichValue(other, num_sf=self.num_sf))
            if center != 0:
                if type(other) is not RichValue and other%2 == 0:
                    domain = [0, np.inf]
                else:
                    domain = self.domain
                new_rich_value = \
                    function_with_rich_values(lambda a,b: a**b, [self, other_],
                                              domain=domain)
            else:
                new_rich_value = RichValue(0, num_sf=self.num_sf)
        elif (type(other) is not RichValue
              and max(unc) <= 1/sigmas * abs(center)):
            x = center ** other
            dx = abs(x * other * np.array(unc) / center)
            if domain != [-np.inf, np.inf]:
                if domain[0] != 0 or (domain[0] == 0 and other>0):
                    x1 = domain[0] ** other
                else:
                    x1 = np.inf
                if domain[1] != 0 or (domain[0] == 0 and other>0):
                    x2 = domain[1] ** other
                else:
                    x2 = np.inf
                domain = [x1, x2]
                domain = [min(domain), max(domain)]
            else:
                domain = [-np.inf, np.inf]
            new_rich_value = RichValue(x, dx, self.is_lolim, self.is_uplim,
                                       self.num_sf, self.min_exp, domain,
                                       self.len_sample, self.allow_log_scale)
        else:
            if (type(other) is RichValue and other.domain[0] < 0
                    and not np.isinf(other.center)):
                print('Warning: Domain of exponent should be positive.')
            new_rich_value = RichValue(np.nan)
        return new_rich_value
    
    def __rpow__(self, other):
        if other > 0:
            domain = [0, np.inf]
        elif other < 0:
            domain = [-np.inf, 0]
        else:
            domain = [-np.inf, np.inf]
        other_ = RichValue(other, num_sf=self.num_sf, domain=domain)
        new_rich_value = other_ ** self
        new_rich_value.allow_log_scale = True
        return new_rich_value
    
    def pdf(self, x):
        """Probability Density Function corresponding to the rich value"""
        center = copy.copy(self.center)
        unc = copy.copy(self.unc)
        domain = copy.copy(self.domain)
        x = np.array(x)
        y = np.zeros(len(x))
        if unc == [0, 0] and not self.is_range:    
            ind = np.argmin(abs(x - center))
            if hasattr(ind, '__iter__'):
                ind = ind[0]
            y[ind] = 1
        else:
            if not self.is_range:
                y = general_pdf(x, center, unc, domain, norm=True)
            elif self.is_lolim and not self.is_uplim:
                y[x > center] = 0.01
            elif self.is_uplim and not self.is_lolim:
                y[x < center] = 0.01
            elif self.is_range:
                x1, x2 = center - unc, center + unc
                y[(x > x1) & (x < x2)] = 1 / (x2 - x1)
        return y
    
    def sample(self, N=None):
        """Sample of the distribution corresponding to the rich value"""
        center = copy.copy(self.center)
        unc = copy.copy(self.unc)
        domain = copy.copy(self.domain)
        if N is None:
            N = self.len_sample
        N = int(N)
        is_range = (self.is_range and not self.is_lim
                    or self.is_uplim and not np.isinf(domain[0])
                    or self.is_lolim and not np.isinf(domain[1]))
        if list(unc) == [0, 0] and not self.is_range:
            x = center * np.ones(N)
        else:
            if not is_range and list(self.unc) != [np.inf, np.inf]:
                if not self.is_lim:
                    x = general_distribution(center, unc, domain, N)
                elif self.is_lolim and not self.is_uplim:
                    x1, x2 = self.center, min(np.inf, domain[1])
                    x = loguniform_distribution(x1, x2, N)
                elif self.is_uplim and not self.is_lolim:
                    x1, x2 = max(-np.inf, domain[0]), center
                    x = loguniform_distribution(x1, x2, N)
            elif not is_range and list(self.unc) == [np.inf, np.inf]:
                x = loguniform_distribution(-np.inf, np.inf, N)
            else:
                if self.is_uplim:
                    x1, x2 = domain[0], center
                elif self.is_lolim:
                    x1, x2 = center, domain[1]
                else:
                    x1, x2 = center - unc[0], center + unc[1]
                N_min = 100
                if N < N_min:
                    x = sample_from_pdf(lambda x: np.ones(len(x)), N, x1, x2)
                else:
                    zero_log = defaultparams['decimal exponent to define zero']
                    x1 += max(10**zero_log, abs(x1)*10**zero_log)
                    x2 -= max(10**zero_log, abs(x2)*10**zero_log)
                    x = np.linspace(x1, x2, N)
                np.random.shuffle(x)
        if len(x) != N:
            np.random.shuffle(x)
            x = x[:N-1]
            x = np.append(x, center)
        return x
    
    def function(self, function, **kwargs):
        """Apply a function to the rich value"""
        new_rich_value = function_with_rich_values(function, self, **kwargs)
        return new_rich_value

class RichArray(np.ndarray):
    """
    A class to store several values with uncertainties or with upper/lower limits.
    """
    
    def __new__(cls, centers, uncs=None,
                are_lolims=None, are_uplims=None, are_ranges=None,
            num_sf=defaultparams['number of significant figures'],
            min_exp=defaultparams['minimum exponent for scientific notation'],
            domain=defaultparams['domain'],
            len_sample=defaultparams['size of samples'],
            allow_log_scale=defaultparams['allow logarithmic scale']):
        """
        Parameters
        ----------
        centers : list / array (float)
            Array of central values.
        uncs : list / array (float), optional
            Array of lower and upper uncertainties associated with the central
            values. The default is None.
        are_lolims : list / array (bool), optional
            Array of logical variables that indicate if each central value is
            actually a lower limit. The default is None.
        are_uplims : list / array (bool), optional
            Array of logical variables that indicate if each central value is
            actually an upper limit. The default is None.
        num_sf : int, optional
            Number of significant figures to use for the uncertainties.
            The default is 1.
        min_exp : int, optional
            If use_exp is True, minimum decimal exponent, in absolute value, to
            apply scientific notation. The default is 4.
        domain : list (float), optional
            The domain of the rich value, that is, the minimum and maximum
            values that it can take. The default is [-np.inf, np.inf].
        len_sample : int, optional
            In case a function has to be applied to this rich value and there
            is no analytic formula for propagating the errors, it will be
            calculated from  distributions with this number of samples.
            The default is 10000
        allow_log_scale : bool, optional
            If True, the values of the rich array will be displayed as an 
            exponent in decimal base. The default is False.
        """
        centers = np.array(centers)
        if uncs is None:
            uncs = np.zeros((2, *centers.shape))
        if are_lolims is None:
            are_lolims = np.zeros(centers.shape, bool)
        if are_uplims is None:
            are_uplims = np.zeros(centers.shape, bool)
        if are_ranges is None:
            are_ranges = np.zeros(centers.shape, bool)
        uncs = np.array(uncs)
        are_lolims = np.array(are_lolims)
        are_uplims = np.array(are_uplims)
        are_lims = are_lolims | are_uplims
        are_ranges = are_ranges | are_lims
        array = np.empty(centers.size, object)
        if uncs.shape != (2, *centers.shape):
            uncs = [[uncs]]*2
            uncs = np.array(uncs).reshape((2, *centers.shape))
        for i in range(centers.size):
            center = centers.flatten()[i]
            unc_flat = uncs.flatten()
            unc = [unc_flat[i], unc_flat[i+len(unc_flat)//2]]
            is_lolim = are_lolims.flatten()[i]
            is_uplim = are_uplims.flatten()[i]
            is_range = are_ranges.flatten()[i]
            if is_range:
                center = [center - unc[0], center + unc[1]]
            array[i] = RichValue(center, unc, is_lolim, is_uplim, num_sf,
                                 min_exp, domain, len_sample, allow_log_scale)
        array = array.reshape(centers.shape)
        array = array.view(cls)
        array.centers = centers
        array.uncs = uncs
        array.are_lolims = are_lolims
        array.are_uplims = are_uplims
        array.num_sf = num_sf
        array.min_exp = min_exp
        array.domain = domain
        array.len_sample = len_sample
        array.allow_log_scale = allow_log_scale
        array.are_lims = are_lims
        array.are_ranges = are_ranges
        return array

    def __copy__(self):
        cls = self.__class__
        new_array = \
            cls.__new__(cls, self.centers, self.uncs, self.are_lolims,
                        self.are_lolims, self.are_ranges, self.num_sf,
                        self.min_exp, self.domain, self.allow_log_scale)
        new_array.__dict__.update(self.__dict__)
        return new_array
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        new_array = \
            cls.__new__(cls, self.centers, self.uncs, self.are_lolims,
                        self.are_lolims, self.are_ranges, self.num_sf,
                        self.min_exp, self.domain, self.allow_log_scale)
        new_array.__dict__.update(self.__dict__)
        memo[id(self)] = new_array
        for k, v in self.__dict__.items():
            setattr(new_array, k, copy.deepcopy(v, memo))
        return new_array
    
    def __getitem__(self, index):
        result = super().__getitem__(index)
        if type(result) is RichArray:
            result = rich_array(result)
        return result
    
    def transpose(self, *axes):
        return rich_array(np.array(self).transpose(*axes))
    
    def reshape(self, shape, order='C'):
        return rich_array(np.array(self).reshape(shape, order))

    def flatten(self, order='C'):
        return rich_array(np.array(self).flatten(order))
    
    def ravel(self, order='C'):
        return rich_array(np.array(self).ravel(order))

    def rel_unc(self):
        new_array = np.empty(0, float)
        for x in self.flat:
            new_array = np.append(new_array, x.rel_unc())
        new_array = new_array.reshape(self.shape)
        return new_array
    
    def ampl(self):
        new_array = np.empty(0, float)
        for x in self.flat:
            new_array = np.append(new_array, x.ampl())
        new_array = new_array.reshape(self.shape)
        return new_array
    
    def rel_ampl(self):
        new_array = np.empty(0, float)
        for x in self.flat:
            new_array = np.append(new_array, x.rel_ampl())
        new_array = new_array.reshape(self.shape)
        return new_array
            
    def latex(self, mult_symbol=defaultparams['multiplication symbol for '
                                              +'scientific notation in LaTeX']):
        """Display the values of the rich array in LaTeX math mode."""
        new_array = np.empty(0, str)
        for x in self.flat:
            new_array = np.append(new_array, x.latex(mult_symbol))
        new_array = new_array.reshape(self.shape)
        return new_array

    def check_limits(self, sigmas=3.):
        """Autodetect upper/lower limits."""
        new_array = np.empty(0, RichArray)
        for x in self.flat:
            x.check_limit(sigmas)
            new_array = np.append(new_array, x)
        new_array = new_array.reshape(self.shape)
        return new_array
    
    def check_intervals(self):
        """Autodetect intervals."""
        new_array = np.empty(0, RichArray)
        for x in self.flat:
            x.check_interval()
            new_array = np.append(new_array, x)
        new_array = new_array.reshape(self.shape)
        return new_array
    
    def set_lims_factor(self, c=4.):
        """Set uncertainties of limits with respect to central values."""
        if not hasattr(c, '__iter__'):
            c = [c, c]
        cl, cu = c
        new_array = np.empty(0, RichValue)
        for x in self.flat:
            if x.is_lolim:
                x.unc = [x.center / cl, x.center / cl]
            elif x.is_uplim:
                x.unc = [x.center / cu, x.center / cu]
            new_array = np.append(new_array, x)
        new_array = rich_array(new_array.reshape(self.shape))
        self.uncs = new_array.uncs

    def sample(self, N=None):
        """Obtain a sample of each entry of the array"""
        N = self.len_sample if N is None else N
        new_array = np.empty(0, float)
        for x in self.flat:
            new_array = np.append(new_array, x.sample(N))
        new_shape = (*self.shape, N) if N != 1 else self.shape
        new_array = new_array.reshape(new_shape)
        return new_array

    def function(self, function, **kwargs):
        """Apply a function to the rich array."""
        new_array = function_with_rich_arrays(function, self, **kwargs)
        return rich_array(new_array)
        
    def __neg__(self):
        return rich_array(-np.array(self))
    
    def __add__(self, other):
        return rich_array(np.array(self) + np.array(other))
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return - (self - other)
    
    def __mul__(self, other):
        return rich_array(np.array(self) * np.array(other))
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return rich_array(np.array(self) / np.array(other))
    
    def __rtruediv__(self, other):
        return rich_array(np.array(other) / np.array(self))
    
    def __pow__(self, other):
        return rich_array(np.array(self) ** np.array(other))
    
    def __rpow__(self, other):
        return rich_array(np.array(other) ** np.array(self))

class RichDataFrame(pd.DataFrame):
    """
    A class to store a dataframe with uncertainties or with upper/lower limits.
    """
      
    def get_params(self):
        """Return the rich value parameters of each column of the dataframe."""
        num_sf, min_exp, domain, len_sample, allow_log_scale = \
            {}, {}, {}, {}, {}
        for col in self:
            x = self[col][0]
            is_rich_value = True if type(x) is RichValue else False
            num_sf[col] = (x.num_sf if is_rich_value else
                           defaultparams['number of significant figures'])
            min_exp[col] = (x.min_exp if is_rich_value
                            else defaultparams['minimum exponent for '
                                               + 'scientific notation'])
            domain[col] = (x.domain if is_rich_value
                           else defaultparams['domain'])
            len_sample[col] = (x.len_sample if is_rich_value else
                               defaultparams['size of samples'])
            allow_log_scale[col] = (x.allow_log_scale if is_rich_value else
                                    defaultparams['allow logarithmic scale'])
        params = {'num_sf': num_sf, 'min_exp': min_exp, 'domain': domain,
                  'len_sample': len_sample, 'allow_log_scale': allow_log_scale}
        return params
    
    def create_column(self, function, columns, **kwargs):
        """
        Create a column applying a function to the given columns of the dataframe.

        Parameters
        ----------
        function : function
            Function to be applied to create the new column.
        columns : list (str)
            List containing the column names of the arguments to be used with
            the given function, in the same order as in the function definition.
        kwargs : optional
            Keyword arguments for the function 'function_with_rich_values'.

        Returns
        -------
        new_df : dataframe
            Resulting dataframe with the new column.
        """
        new_column = np.empty(len(self), RichValue)
        for i,(_,row) in enumerate(self.iterrows()):
            arguments = [row[col] for col in columns]
            new_rich_value = function_with_rich_values(function, arguments,
                                                       **kwargs)
            new_column[i] = new_rich_value
        new_column = rich_array(new_column)
        return new_column

    def create_row(self, function, rows, **kwargs):
        """
        Create a row applying a function to the given rows of the dataframe.
    
        Parameters
        ----------
        function : function
            Function to be applied to create the new column.
        rows: list (str)
            List containing the row names of the arguments to be used with the
            given function, in the same order as in the function definition.
        kwargs : optional
            Keyword arguments for the function 'function_with_rich_values'.
    
        Returns
        -------
        new_row : dataframe
            Dataframe containing the new row.
        """
        new_row = {}
        for i,col in enumerate(self):
            arguments = [self.at[idx,col] for idx in rows]
            new_rich_value = function_with_rich_values(function, arguments,
                                                       **kwargs)
            new_row[col] = new_rich_value
        return new_row

    def latex(self, return_df=False, row_sep='\\tabularnewline',
              mult_symbol=defaultparams['multiplication symbol for '
                                        + 'scientific notation in LaTeX']):
        """Return the content of the dataframe as a table in LaTeX format."""
        row_sep = ' ' + row_sep + ' '
        new_df = copy.copy(self)
        for col in self:
            for i in range(len(self[col])):
                entry = self.at[i,col]
                if 'RichValue' in str(type(entry)):
                    if not np.isnan(entry.center):
                        new_df.at[i,col] = entry.latex(mult_symbol)
                    else:
                        new_df.at[i,col] = '...'
        if return_df:
            output = new_df
        else:
            text = ''
            rows = []
            for i, row in new_df.iterrows():
                cols = []
                for j, column in enumerate(new_df):
                    entry = str(row[column])
                    if entry == 'nan':
                        entry = '...'
                    cols += [entry]
                rows += [' & '.join(cols)]
            text = row_sep.join(rows)
            output = text
        return output

    def set_lims_factor(self, limits_factors={}):
        """Set the uncertainties of limits with respect to central values."""
        if limits_factors == {}:
            limits_factors = 4.
        if type(limits_factors) is not dict:
            limits_factors = {col: limits_factors for col in self}
        for i,row in self.iterrows():
            for col in limits_factors:
                if 'RichValue' in str(type(self.at[i,col])):
                    entry = self.at[i,col]
                    c = limits_factors[col]
                    if not hasattr(c, '__iter__'):
                        c = [c, c]
                    cl, cu = c
                    c = cl if entry.is_lolim else cu
                    self.at[i,col].set_lims_factor(c)

    def __neg__(self):
        return RichDataFrame(-pd.DataFrame(self))
    
    def __add__(self, other):
        other_ = pd.DataFrame(other) if type(other) is RichDataFrame else other
        return RichDataFrame(pd.DataFrame(self) + other_)
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return - (self - other)
    
    def __mul__(self, other):
        other_ = pd.DataFrame(other) if type(other) is RichDataFrame else other
        return RichDataFrame(pd.DataFrame(self) * other_)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        other_ = pd.DataFrame(other) if type(other) is RichDataFrame else other
        return RichDataFrame(pd.DataFrame(self) / other_)
    
    def __rtruediv__(self, other):
        return RichDataFrame(other / self)
    
    def __pow__(self, other):
        other_ = pd.DataFrame(other) if type(other) is RichDataFrame else other
        return RichDataFrame(pd.DataFrame(self) ** other_)
    
    def __rpow__(self, other):
        return RichDataFrame(other ** self)

def rich_value(text, num_sf=None, min_exp=defaultparams['minimum exponent for '
                                                        +'scientific notation'],
               domain=None, len_sample=defaultparams['size of samples'],
               allow_log_scale=None):
    """
    Convert the input text to a rich value.

    Parameters
    ----------
    text : str
        String representing a rich value.
    num_sf : int, optional
        Number of significant figures to use for the uncertainties.
        The default is the maximum of this variable for the elements of the
        resulting rich array.
    min_exp : int, optional
        If use_exp is True, minimum decimal exponent, in absolute value, to
        apply scientific notation. The default is 4.
    domain : list (float), optional
        The domain of the rich value, that is, the minimum and maximum
        values that it can take. The default is the union of the domains of all
        the elements of the resulting rich array.
    len_sample : int, optional
        In case a function has to be applied to this rich value and there
        is no analytic formula for propagating the errors, it will be
        calculated from  distributions with this number of samples.
        The default is the number of elements of the array times the mean of
        this variable for all the elements of the resulting rich array.
    allow_log_scale : bool, optional
        If True, the values of the rich array will be displayed as an exponent
        in decimal base. The default is True if this variable is True for any
        of the elements of the resulting rich array.

    Returns
    -------
    y : rich value
        Resulting rich value.
    """
    
    lim_for_extra_sf = defaultparams['limit for extra significant figure']
    num_sf_or = copy.copy(num_sf)
    domain_or = copy.copy(domain)
    allow_log_scale_or = copy.copy(allow_log_scale)
    
    def parse_as_rich_value(text):
        """
        Obtain the properties of the input text as a rich value.

        Parameters
        ----------
        text : str
            Input text representing a rich value.

        Returns
        -------
        center : float
            Central value.
        unc : list (float)
            Lower and upper uncertainties.
        is_lolim :  bool
            If True, the central value is actually a lower limit.
        is_uplim : bool
            If True, the central value is actually an upper limit.
        num_sf : int
            Number of significant figures.
        domain : list (float)
            Domain of the variable.
        """
        if '[' in text and ']' in text:
            x1, x2 = text.split('[')[1].split(']')[0].split(',')
            x1 = float(x1) if x1 != '-inf' else -np.inf
            x2 = float(x2) if x2 != 'inf' else np.inf
            domain = [x1, x2]
            text = text.split('[')[0][:-1]
        else:
            domain = [-np.inf, np.inf]
        if not '--' in text:
            if text.startswith('+'):
                text = text[1:]
            if 'e' not in text:
                text = '{} e0'.format(text)
            single_value = True
            for symbol, i0 in zip(['<', '>', '+', '-'], [0, 0, 0, 1]):
                if symbol in text[i0:]:
                    single_value = False
            if text in ['nan', 'Nan', 'None']:
                single_value = False
            if single_value:
                x, e = text.split(' ')
                dx = 0
                text = '{}+/-{} {}'.format(x, dx, e)
            if text.startswith('<'):
                x = text.replace('<','').replace(' ','')
                dx1 = float(x)
                dx2 = 0
                is_uplim = True
                is_lolim = False
            elif text.startswith('>'):
                x = text.replace('>','').replace(' ', '')
                dx1 = 0
                dx2 = float(x)
                is_uplim = False
                is_lolim = True
            else:
                is_uplim, is_lolim = False, False
                text = (text.replace('+-', '+/-')
                        .replace(' +/-', '+/-').replace('+/- ', '+/-'))
                if '+/-' in text:
                    x_dx, e = text.split(' ')
                    x, dx = x_dx.split('+/-')
                    text = '{}-{}+{} {}'.format(x, dx, dx, e)
                    dx1, dx2 = dx, dx
                else:
                    if '+' in text:
                        if text.startswith('-'):
                            x = '-' + text.split('-')[1]
                            text = text[1:]
                        else:
                            x = text.split('-')[0]
                        dx1 = text.split('-')[1].split('+')[0]
                        dx2 = text.split('+')[1].split(' ')[0]
                    else:
                        x = text.split(' ')[0]
                        dx1, dx2 = '0', '0'
                if x not in ['nan', 'NaN', 'None']:
                    e = text.split(' ')[1]
                    x = x + e
                    dx1 = dx1 + e
                    dx2 = dx2 + e
                else:
                    x = np.nan
                    dx1, dx2 = 0, 0
            if (not (is_lolim or is_uplim)
                    and not (float(dx1) == float(dx2) == 0)):
                dx1, dx2 = str(dx1), str(dx2)
                dx1_ = dx1.split('e')[0]
                dx2_ = dx2.split('e')[0]
                for i in reversed(range(len(dx1_))):
                    dx1_ = dx1_.replace('0.'+'0'*i, '')
                dx1_ = dx1_.replace('.','')
                for i in reversed(range(len(dx2_))):
                    dx2_ = dx2_.replace('0.'+'0'*i, '')
                dx2_ = dx2_.replace('.','')
                n1 = len(dx1_)
                base1 = float('{:e}'.format(float(dx1)).split('e')[0])
                if base1 < lim_for_extra_sf:
                    n1 -= 1
                n2 = len(dx2_)
                base2 = float('{:e}'.format(float(dx2)).split('e')[0])
                if base2 < lim_for_extra_sf:
                    n2 -= 1
                num_sf = max(1, n1, n2)
            else:
                x = str(x)
                x_ = x.split('e')[0]
                for i in reversed(range(len(x_))):
                    x_ = x_.replace('0.'+'0'*i, '')
                x_ = x_.replace('.','')
                n = len(x_)
                base = float('{:e}'.format(float(x)).split('e')[0])
                if base < lim_for_extra_sf:
                    n -= 1
                num_sf = n
            num_sf = max(1, num_sf)
            center = float(x)
            unc = [float(dx1), float(dx2)]
        else:
            text = text.replace(' --','--').replace('-- ','--')
            x1, x2 = text.split('--')
            x1, _, _, _, n1, domain_1 = parse_as_rich_value(x1)
            x2, _, _, _, n2, domain_2 = parse_as_rich_value(x2)
            center = [x1, x2]
            unc = 0
            is_lolim, is_uplim = False, False
            num_sf = max(n1, n2)
            domain = [min(domain_1[0], domain_2[0]),
                      max(domain_1[1], domain_2[1])]
        return center, unc, is_lolim, is_uplim, num_sf, domain
    
    text = str(text)
    if '10^' in text:
        allow_log_scale = True
        text = (text.replace('( ','(').replace(' )',')')
                    .replace('10^ ','10^').replace('10^(','').replace(')',''))
        center, unc, is_lolim, is_uplim, num_sf, domain = \
            parse_as_rich_value(text)
        unc = [10**center - 10**(center - unc[0]),
               10**(center+unc[1]) - 10**center]
        center = 10**center
    else:
        allow_log_scale = False
        center, unc, is_lolim, is_uplim, num_sf, domain = \
            parse_as_rich_value(text)
    num_sf = num_sf if num_sf_or is None else num_sf_or
    domain = domain if domain_or is None else domain_or
    allow_log_scale = (allow_log_scale if allow_log_scale_or is None
                       else allow_log_scale_or)
    y = RichValue(center, unc, is_lolim, is_uplim, num_sf, min_exp, domain,
                  len_sample, allow_log_scale)   
    return y


def rich_array(array, num_sf=None, min_exp=None, domain=None, len_sample=None,
               allow_log_scale=None, check_limits=True, check_intervals=False):
    """
    Convert the input array to a rich array.

    Parameters
    ----------
    array : array / list (str)
        Input array containing text strings representing rich values.
    num_sf : int, optional
        Number of significant figures to use for the uncertainties.
        The default is the maximum of this variable for the elements of the
        input array.
    min_exp : int, optional
        If use_exp is True, minimum decimal exponent, in absolute value, to
        apply scientific notation. The default is the minimum of this variable
        for all the elements of the input array.
    domain : list (float), optional
        The domain of the rich value, that is, the minimum and maximum
        values that it can take. The default is the union of the domains of all
        the elements of the input array.
    len_sample : int, optional
        In case a function has to be applied to this rich value and there
        is no analytic formula for propagating the errors, it will be
        calculated from  distributions with this number of samples.
        The default is the number of elements of the array times the mean of
        this variable for all the elements of the input array.
    allow_log_scale : bool, optional
        If True, the values of the rich array will be displayed as an exponent
        in decimal base. The default is True if this variable is True for any
        of the elements of the input array.
    check_limits : bool, optional
        If True, if any of the 1-sigma interval bounds equals or exceeds the
        domain, the value will be considered an upper/lower limit, for every
        each of the array. The default is True.
    check_intervals : bool, optional
        If True, if the maximum uncertainty is greater than the distance to the
        closest domain edges, the value will be considered a finite interval,
        for each entry of the array. The default is False.

    Returns
    -------
    rich_array : array
        Resulting rich array.
    """
    array = np.array(array)
    use_default_num_sf = True if num_sf is None else False
    use_default_min_exp = True if min_exp is None else False
    use_default_domain = True if domain is None else False
    use_default_len_sample = True if len_sample is None else False
    use_default_allow_log_scale = True if allow_log_scale is None else False
    centers, uncs, are_lolims, are_uplims, are_ranges = [], [], [], [], []
    min_exps, nums_sf, len_samples, domains, allow_log_scales = \
        [], [], [], [], []
    for element in array.flat:
        x = (element if 'RichValue' in str(type(element))
             else rich_value(element))
        if check_limits:
            x.check_limit()
        if check_intervals:
            x.check_interval()
        centers += [x.center]
        uncs += [x.unc]
        are_lolims += [x.is_lolim]
        are_uplims += [x.is_uplim]
        are_ranges += [x.is_range]
        if use_default_num_sf:
            nums_sf += [x.num_sf]
        if use_default_domain:
            domains += [x.domain]
        if use_default_len_sample:
            len_samples += [x.len_sample]
        if use_default_allow_log_scale:
            allow_log_scales += [x.allow_log_scale]
        if use_default_min_exp:
            min_exps += [x.min_exp]
    centers = np.array(centers).reshape(array.shape)
    uncs = np.array(uncs)
    uncs = np.array([uncs[:,0].reshape(array.shape).tolist(),
                     uncs[:,1].reshape(array.shape).tolist()])
    are_lolims = np.array(are_lolims).reshape(array.shape)
    are_uplims = np.array(are_uplims).reshape(array.shape)
    are_ranges = np.array(are_ranges).reshape(array.shape)
    if use_default_num_sf:
        num_sf = max(nums_sf)
    if use_default_domain:
        domain = [min([dom[0] for dom in domains]),
                  max([dom[1] for dom in domains])]
    if use_default_len_sample:
        len_sample = int(np.mean(len_samples))
    if use_default_allow_log_scale:
        allow_log_scale = any(allow_log_scales)
    if use_default_min_exp:
        min_exp = min(min_exps)
    new_array = RichArray(centers, uncs, are_lolims, are_uplims, are_ranges,
                          num_sf, min_exp, domain, len_sample, allow_log_scale)
    return new_array

def rich_dataframe(df, num_sf={}, min_exp={}, domain={}, len_sample={},
                   allow_log_scale={}, check_limits=True, check_intervals=False):
    """
    Convert the values of the input dataframe of text strings to rich values.

    Parameters
    ----------
    df : dataframe (str)
        Input dataframe which contains text strings formatted as rich values.
    num_sf : dict (int), optional
        Dictionary containing this value for each column of the dataframe.    
        Number of significant figures to use for the uncertainties.
        By default, for each column, it will be the value for the first entry
        if it is a rich value, and 1 if not.
    use_exp : dict (bool), optional
        Dictionary containing this value for each column of the dataframe.
        If True, the values will be displayed in scientific notation
        when printed on screen. By default, for each column, it will be the
        value for the first entry if it is a rich value, and False if not.
    domain : dict (list (float)), optional
        Dictionary containing this value for each column of the dataframe.
        The domain of the rich value, that is, the minimum and maximum
        values that it can take. By default, for each column, it will be the
        value for the first entry if it is a rich value, and [-np.inf, np.inf]
        if not.
    len_sample : dict (int), optional
        Dictionary containing this value for each column of the dataframe.
        In case a function has to be applied to this rich value and there
        is no analytic formula for propagating the errors, it will be
        calculated from normal distributions with this number of samples.
        By default, for each column, it will be the value for the first entry
        if it is a rich value, and 10000 if not.
    allow_log_scale : dict (bool), optional
        Dictionary containing this value for each column of the dataframe.
        If True, the values will be displayed as an exponent in decimal base.
        By default, for each column, it will be the value for the first entry
        if it is a rich value, and False if not.
    min_exp : dict(bool), optional
        Dictionary containing this value for each column of the dataframe.
        If use_exp is True, minimum decimal exponent, in absolute value, to
        apply scientific notation. The default is the minimum of this variable
        for all the elements of the input array. By default, for each column,
        it will be the value for the first entry if it is a rich value, and
        False if not.
    check_limits : bool, optional
        If True, if any of the 1-sigma interval bounds equals or exceeds the
        domain, the value will be considered as an upper/lower limit.
    check_intervals : bool, optional
        If True, if the maximum uncertainty is greater than the distance to the
        closest domain edges, the value will be considered a finite interval,
        for each entry of the array. The default is False.

    Returns
    -------
    new_df : dataframe
        Resulting dataframe of rich values.
    """
    df = pd.DataFrame(df)
    use_default_num_sf = True if num_sf == {} else False
    use_default_min_exp = True if min_exp == {} else False
    use_default_len_sample = True if len_sample == {} else False
    use_default_domain = True if domain == {} else False
    use_default_allow_log_scale = True if allow_log_scale == {} else False
    for col in df:
        is_number = True
        text = str(df[col][0])
        for char in text.replace(' e', ''):
            if char.isalpha():
                is_number = False
                break
        if is_number:
            x = rich_value(text)
        if use_default_num_sf:
            num_sf[col] = (x.num_sf if is_number else
                           defaultparams['number of significant figures'])
        if use_default_min_exp:
            min_exp[col] = (x.min_exp if is_number else
                            defaultparams['minimum exponent for '
                                           + 'scientific notation'])
        if use_default_domain:
            domain[col] = x.domain if is_number else defaultparams['domain']
        if use_default_len_sample:
            len_sample[col] = (x.len_sample if is_number else
                               defaultparams['size of samples'])
        if use_default_allow_log_scale:
            allow_log_scale[col] = (x.allow_log_scale if is_number else
                                    defaultparams['allow logarithmic scale'])
    if type(num_sf) is not dict:
        num_sf = {col: num_sf for col in df}
    if type(min_exp) is not dict:
        min_exp = {col: min_exp for col in df}
    if type(domain) is not dict:
        domain = {col: domain for col in df}
    if type(len_sample) is not dict:
        len_sample = {col: len_sample for col in df}
    if type(allow_log_scale) is not dict:
        allow_log_scale = {col: allow_log_scale for col in df}
    new_df = copy.copy(df)
    for i,row in new_df.iterrows():
        for col in new_df:
            is_rich_value = (True if type(new_df.at[i,col]) is RichValue
                             else False)
            if is_rich_value:
                x = new_df.at[i,col]
                x.num_sf = num_sf[col]
                x.min_exp = min_exp[col]
                x.domain = domain[col]
                x.len_sample = len_sample[col]
                x.allow_log_scale = allow_log_scale[col]
            else:
                is_number = True
                text = str(new_df.at[i,col])
                for char in text.replace(' e', ''):
                    if char.isalpha():
                        is_number = False
                        break
                if is_number:
                    x = rich_value(text, num_sf[col], min_exp[col], domain[col],
                                   len_sample[col], allow_log_scale[col])
            if is_rich_value or is_number:
                if check_limits:
                    x.check_limit()
                if check_intervals:
                    x.check_interval()
                new_df.at[i,col] = x
    new_df = RichDataFrame(new_df)
    return new_df

def general_pdf(x, loc=0, scale=1, bounds=None, norm=False):
    """
    Generic PDF with given median and uncertainties for the given domain.

    If the width of the distributiin is quite lower than the boundaries,
    the PDF (probability density function) will be a modified gaussian for the
    given range. If not, it will be a trapezoidal distribution with a flat with
    a width greater than the uncertainty.
    
    Parameters
    ----------
    x : array (float)
        Independent variable.
    loc : float, optional
        Median of the distribution. The default is 0.
    scale : float, optional
        Uncertainty of the distribution (1-sigma confidence interval).
        The default is 1. It can be a list with lower and upper uncertainties.
        The resulting 1-sigma confidence interval must be lower than the
        boundaries of the independent variable.
    bounds : list (float), optional
        Boundaries of the independent variable. The default is a interval
        centered in the median and with semiwidth equal to 6 times the
        uncertainty.
    norm : bool, optional
        If True, the function will be normalized. The default is False.

    Returns
    -------
    y : array (float)
        Resulting PDF for the input array.
    """
    m, s, b = loc, scale, bounds

    def symmetric_general_pdf(x, m=0., s=1., a=10., norm=False):
        """
        Symmetric general PDF with given median (m) and uncertainty (s) with a
        boundary centered in the median with a given amplitude (a).
        """
        y = np.zeros(len(x))
        if a >= 2*s:
            y = bounded_gaussian(x, m, s, a)
        elif a > s:
            y = mirrored_loggaussian(x, m, s, a)
        else:
            raise Exception('Domain must be greater than uncertainty.')
        if norm:
            dm = min(a, 6*s)
            x1, x2 = m - dm, m + dm
            xl = np.linspace(x1, x2, int(4e3))
            yl = symmetric_general_pdf(xl, m, s, a)
            y /= np.trapz(yl, xl).sum()
        return y
    
    if b is None:
        b = [m - 10*s, m + 10*s]
    if not b[0] < m < b[1]:
        raise Exception('Center ({}) is not inside the boundaries {}.'
                        .format(m, b))
    if not hasattr(s, '__iter__'):
        s = [s, s]
    if np.isinf(b[0]):
        b[0] = m - 10*s[0]
    if np.isinf(b[1]):
        b[1] = m + 10*s[1]
    a = [m - b[0], b[1] - m]
    if not hasattr(a, '__iter__'):
        a = [a, a]
    
    x = np.array(x)
    y = np.zeros(len(x))
    cond = np.less(x, m)
    y[cond] = symmetric_general_pdf(x[cond], m, s[0], a[0], norm=True)
    cond = np.greater_equal(x, m)
    y[cond] = symmetric_general_pdf(x[cond], m, s[1], a[1], norm=True)
    
    return y

def sample_from_pdf(pdf, size, low, high, **kwargs):
    """
    Return a sample of the distribution specified with the input function.

    Parameters
    ----------
    pdf : function
        Probability density function of the distribution.
    size : int
        Size of the sample.
    low : float
        Minimum of the input values for the probability density function.
    high : float
        Maximum of the input values for the probability density function.
    **kwargs : keyword arguments, optional
        Keyword arguments for the probability density function.

    Returns
    -------
    x : array
        Sample of the distribution.
    """
    num_points = max(10, size)
    x = np.random.uniform(low, high, size=num_points)
    x = np.sort(x)
    y = pdf(x, **kwargs)
    y = np.nan_to_num(y, nan=0.)
    y /= y.sum()
    x = np.random.choice(x, p=y, size=size)
    return x

def general_distribution(loc=0, scale=1, bounds=None, size=1):
    """
    General distribution with given median, uncertainty and boundaries.

    Parameters
    ----------
    loc : float, optional
        Median of the distribution. The default is 0.
    scale : float, optional
        Uncertainty of the distribution (1-sigma confidence interval).
        The default is 1. It can be a list with lower and upper uncertainties.
        The resulting 1-sigma confidence interval must be lower than the
        boundaries of the independent variable.
    bounds : list (float), optional
        Boundaries of the independent variable. The default is a interval
        centered in the median and with semiwidth equal to 6 times the
        uncertainty.
    size : int, optional
        Number of samples of the distribution. The default is 1.

    Returns
    -------
    distr : array (float)
        Resulting distribution.
    """
    m, s, b = loc, scale, bounds
    if bounds is None:
        b = [m - 10*s, m + 10*s]
    if not b[0] < m < b[1]:
        raise Exception('Center ({}) is not inside the boundaries {}.'
                        .format(m, b))
    if not hasattr(s, '__iter__'):
        s = [s, s]
    low = max(m-5*s[0], b[0])
    high = min(m+5*s[1], b[1])
    distr = sample_from_pdf(general_pdf, size, low, high,
                            loc=m, scale=s, bounds=b, norm=False)
    return distr

def loguniform_distribution(low=-1, high=1, size=1,
        zero_log=defaultparams['decimal exponent to define zero'],
        inf_log=defaultparams['decimal exponent to define infinity']):
    """
    Create a log-uniform distribution between the given values.

    Parameters
    ----------
    low : float, optional
        Inferior limit. The default is -1.
    high : float, optional
        Superior limit. The default is 1.
    size : int, optional
        Number of samples of the distribution. The default is 1.
    zero_log : float, optional
        Decimal logarithm of the minimum value in absolute alue.
    ing_log : float, optional
        Decimal logarithm of the maximum value in absolute value.

    Returns
    -------
    x : array (float)
        Values of the distribution.
    """
    x1, x2, N = low, high, size
    N_min = 10
    if N < N_min:
        x_ = loguniform_distribution(x1, x2, N_min, zero_log, inf_log)
        p = np.random.uniform(size=N_min)
        p /= p.sum()
        x = np.random.choice(x_, p=p, size=N)
    else:
        if not x1 < x2:
            raise Exception('Inferior limit must be lower than superior limit.')
        if np.isinf(x1):
            log_x1 = inf_log
        elif x1 == 0:
            log_x1 = zero_log
        else:
            log_x1 = log10(abs(x1))
        if np.isinf(x2):
            log_x2 = inf_log
        elif x2 == 0:
            log_x2 = zero_log
        else:
            log_x2 = log10(abs(x2))
        if log_x1 < zero_log:
            log_x1 = zero_log
        if log_x2 > inf_log: 
            log_x2 = inf_log
        if x1 < 0:
            if x2 <= 0:
                exps = np.linspace(log_x2, log_x1, N)
                x = -10**exps
            else:
                exps = np.linspace(zero_log, log_x1, N)
                x = -10**exps
                exps = np.linspace(zero_log, log_x2, N)
                x = np.append(x, 10**exps)
        else:
            exps = np.linspace(log_x1, log_x2, N)
            x = 10**exps
        x1, x2 = x[0], x[-1]
        np.random.shuffle(x)
        if len(x) != N:
            x = x[:N-2]
            x = np.append(x, [x1, x2])
    return x

def center_and_uncs(distr, function=np.median, interval=68.27, fraction=1.):
    """
    Return the central value and uncertainties of the input distribution.

    Parameters
    ----------
    distr : array (float)
        Input distribution.
    function : function, optional.
        Function to calculate the central value of the distribution.
        The default is np.median.
    interval : float, optional
        Size of the interval, in percentile, around the center value which
        defines the uncertainties. The default is 68.27 (1 sigma confidence
        interval).
    fraction : float, optional
        Fraction of the data that is used for the calculation, to exclude
        outliers. The default is 1.

    Returns
    -------
    center : float
        Central value of the distribution.
    uncs : tuple (float)
        Lower and upper uncertainties of the distribution.
    """
    x = np.array(distr)
    x = x[~np.isnan(x)]
    x = np.sort(x)
    margin = (1 - fraction) / 2
    if len(x) > 1:
        x = x[int(margin*len(x)):int((1-margin)*len(x))]
    center = function(x)
    difference = abs(x - center)
    ind = np.argwhere(difference == min(difference))
    if hasattr(ind, '__iter__'):
        ind = int(np.median(ind))
    ind = 100 * float(ind) / len(x)
    perc1 = ind - interval/2
    perc2 = ind + interval/2
    if perc1 < 0:
        perc2 += abs(0 - perc1)
        perc1 = 0
    if perc2 > 100:
        perc1 -= abs(perc2 - 100)
        perc2 = 100
    unc1 = center - np.percentile(x, perc1)
    unc2 = np.percentile(x, perc2) - center
    unc1 *= 1 + margin
    unc2 *= 1 + margin
    uncs = [unc1, unc2]
    return center, uncs

def add_two_rich_values(x, y):
    """
    Sum two rich values to get a new one.
    
    Parameters
    ----------
    x : rich value
        One rich value.
    y : rich value
        Other rich value.
        
    Returns
    -------
    z : rich value
        New rich value
    """
    num_sf = max(x.num_sf, y.num_sf)
    min_exp = min(x.min_exp, y.min_exp)
    domain = [x.domain[0] + y.domain[0], x.domain[1] + y.domain[1]]
    len_sample = max(x.len_sample, y.len_sample)
    allow_log_scale = x.allow_log_scale | y.allow_log_scale
    sigmas = defaultparams['minimum relative distance to the domain edges '
                           + 'to apply analytic uncertainty propagation']
    if (not (x.is_range or y.is_range)
            and min(x.rel_ampl()) > sigmas and min(y.rel_ampl())) > sigmas:
        z = x.center + y.center
        dz = (np.array(x.unc)**2 + np.array(y.unc)**2)**0.5
        z = RichValue(z, dz, False, False, num_sf, min_exp, domain,
                      len_sample, allow_log_scale)
    else:
        z = function_with_rich_values(lambda a,b: a+b, [x, y], domain=domain)
    return z

def multiply_two_rich_values(x, y):
    """
    Multiply two rich values to get a new one.
    
    Parameters
    ----------
    x : rich value
        One rich value.
    y : rich value
        Other rich value.
        
    Returns
    -------
    z : rich value
        New rich value
    """
    num_sf = max(x.num_sf, y.num_sf)
    min_exp = min(x.min_exp, y.min_exp)
    with np.errstate(all='ignore'):
        domain_combs = [x.domain[0] * y.domain[0], x.domain[0] * y.domain[1],
                        x.domain[1] * y.domain[0], x.domain[1] * y.domain[1]]
    domain1, domain2 = min(domain_combs), max(domain_combs)
    domain1 = np.nan_to_num(domain1, nan=0.) if np.isnan(domain1) else domain1
    domain2 = np.nan_to_num(domain2, nan=0.) if np.isnan(domain2) else domain2
    domain = [domain1, domain2]
    len_sample = max(x.len_sample, y.len_sample)
    allow_log_scale = x.allow_log_scale | y.allow_log_scale
    sigmas = defaultparams['minimum relative distance to the domain edges '
                           + 'to apply analytic uncertainty propagation']
    if (not (x.is_range or y.is_range)
         and min(x.rel_ampl()) > sigmas and min(y.rel_ampl()) > sigmas):
        z = x.center * y.center
        dx, dy = np.array(x.unc), np.array(y.unc)
        # dz = z * ((dx/x.center)**2 + (dy/y.center)**2)**0.5
        dz = (dx**2 * dy**2 + dx**2 * y.center**2 + dy**2 * x.center**2)**0.5
        z = RichValue(z, dz, False, False, num_sf, min_exp, domain, len_sample,
                      allow_log_scale)
    else:
        z = function_with_rich_values(lambda a,b: a*b, [x, y], domain=domain)
    return z

def divide_two_rich_values(x, y):
    """
    Divide two rich values to get a new one.
    
    Parameters
    ----------
    x : rich value
        One rich value.
    y : rich value
        Other rich value.
        
    Returns
    -------
    z : rich value
        New rich value
    """
    num_sf = max(x.num_sf, y.num_sf)
    min_exp = min(x.min_exp, y.min_exp)
    with np.errstate(all='ignore'):
        domain_combs = [x.domain[0] * y.domain[0], x.domain[0] * y.domain[1],
                        x.domain[1] * y.domain[0], x.domain[1] * y.domain[1]]
    domain1, domain2 = min(domain_combs), max(domain_combs)
    domain1 = np.nan_to_num(domain1, nan=0.) if np.isnan(domain1) else domain1
    domain2 = np.nan_to_num(domain2, nan=0.) if np.isnan(domain2) else domain2
    domain = [domain1, domain2]
    len_sample = max(x.len_sample, y.len_sample)
    allow_log_scale = x.allow_log_scale | y.allow_log_scale
    sigmas = defaultparams['minimum relative distance to the domain edges '
                           + 'to apply analytic uncertainty propagation']
    if (not (x.is_range or y.is_range)
         and min(x.rel_ampl()) > sigmas and min(y.rel_ampl()) > sigmas):
        z = x.center / y.center
        dx, dy = np.array(x.unc), np.array(y.unc)
        dz = z * ((dx/x.center)**2 + (dy/y.center)**2)**0.5
        z = RichValue(z, dz, False, False, num_sf, min_exp, domain, len_sample,
                      allow_log_scale)
    else:
        z = function_with_rich_values(lambda a,b: a/b, [x, y], domain=domain)
    return z

def distribution_with_rich_values(function, args, len_args_samples=None):
    """
    Same as function_with_rich_values, but just returns the final distribution.
    """
    if not hasattr(args, '__iter__'):
        args = [args]
    if len_args_samples is None:
        len_args_samples = int(len(args)
                               * np.mean([arg.len_sample for arg in args]))
    args_distr = [arg.sample(len_args_samples) for arg in args]
    new_distr = function(*args_distr)
    return new_distr

def function_with_rich_values(
        function, args, len_args_samples=None, unc_function=None,
        sigmas=defaultparams['minimum relative distance to the domain edges '
                             + 'to apply analytic uncertainty propagation'],
        use_sigma_combs=defaultparams['use 1-sigma combinations to '
                                      + 'estimate uncertainty propagation'],
        lims_fraction=defaultparams['fraction of the central value '
                                    + 'for upper/lower limits'],
        num_reps_lims=defaultparams['number of repetitions to estimate'
                                    + ' upper/lower limits'],
        num_sf=None, min_exp=None, domain=None, len_sample=None,
        allow_log_scale=None):
    """
    Apply a function to the input rich values.

    Parameters
    ----------
    function : function
        Function to be applied to the input rich values.
    args : list (rich values)
        List with the input rich values, in the same order as the arguments of
        the given function.
    len_args_samples : int, optional
        Size of the samples of the arguments. The default is the number of
        arguments times the mean of this variable for all the arguments.
    unc_function : function, optional
        Function to estimate the uncertainties, in case that error propagation
        can be used. The arguments should be the central values first and then
        the uncertainties, with the same order as in the input function.
    sigmas : float, optional
        Threshold to apply uncertainty propagation. The value is the distance
        to the bounds of the domain relative to the uncertainty.
        The default is 10.
    use_sigma_combs : bool, optional
        If True, the calculation of the uncertainties will be optimized when
        the relative amplitudes are small and there is no uncertainty function
        provided. The default is False.
    lims_fraction : float, optional
        In case the resulting value is an upper/lower limit, this factor is
        used to calculate the limit. If it is 0, the value will be the maximum/
        lower value of the resulting distributionthe, and if it is 1, the value
        will be result of the function applied to the central value of the
        arguments. For the rest it will be an interpolation.
        The default is 0.1.
    num_reps_lims : int, optional
        Number of repetitions of the sampling done in the cases of having an
        upper/lower limit for better estimating its value. The default is 4.
    num_sf : int, optional
        Number of significant figures to display the resulting value.
        The default is the maximum of this variable for all the arguments.
    min_exp : int, optional
        If use_exp is True, minimum decimal exponent, in absolute value, to
        apply scientific notation to the resulting rich value.
        The default is 4.
    domain : list (float), optional
        Domain of the result. If not specified, it will be estimated
        automatically.
    len_sample : int, optional
        In case a function has to be applied to the resulting rich value and
        there is no analytic formula for propagating the uncertainties, it will
        be calculated from distributions with this number of samples.
        The default is the mean of this value for all the arguments.
    allow_log_scale : bool, optional
        If True, the value will be displayed as an exponent in decimal
        base. The default is False.

    Returns
    -------
    new_rich_value : rich value
        Resulting rich value.
    """

    zero_log = defaultparams['decimal exponent to define zero']
    inf_log = defaultparams['decimal exponent to define infinity']
    
    def magnitude_order_range(interval, zero_log=zero_log):
        x1, x2 = interval
        x1 = 0 if abs(x1) < 10**zero_log else x1
        x2 = 0 if abs(x2) < 10**zero_log else x2
        if x1*x2 > 0:
            d = log10(x2-x1)
        elif x1*x2 < 0:
            d = np.log10(abs(x1)) + 2*abs(zero_log) + np.log10(x2)
        else:
            xlim = x1 if x2 == 0 else x2
            d = abs(zero_log) + np.log10(abs(xlim))
        return d
   
    def remove_zero_infs(interval, zero_log=zero_log, inf_log=inf_log):
        x1, x2 = interval
        if abs(x1) < 10**zero_log:
            x1 = np.sign(x1) * 10**zero_log
        elif x1 < 0 and abs(x1) > 10**inf_log:
            x1 = - 10**inf_log
        if abs(x2) < 10**zero_log:
            x2 = np.sign(x2) * 10**zero_log
        elif x2 > 0 and x2 > 10**inf_log:
            x2 = 10**inf_log
        new_interval = [x1, x2]
        return new_interval
   
    def add_zero_infs(interval, zero_log=zero_log, inf_log=inf_log):
        x1, x2 = interval
        if abs(x1) < 10**zero_log:
            x1 = 0
        elif x1 < 0 and abs(x1) > 10**inf_log:
            x1 = -np.inf
        if abs(x2) < 10**zero_log:
            x2 = 0
        elif x2 > 0 and x2 > 10**inf_log:
            x2 = np.inf
        new_interval = [x1, x2]
        return new_interval
    
    if not hasattr(args, '__iter__'):
        args = [args]
    if len_args_samples is None:
        len_args_samples = int(len(args)
                               * np.mean([arg.len_sample for arg in args]))
    if num_sf is None:
        num_sf = max([arg.num_sf for arg in args])
    if min_exp is None:
        min_exp = min([arg.min_exp for arg in args])
    if len_sample is None:
        len_sample = int(np.mean([arg.len_sample for arg in args]))
    if allow_log_scale is None:
        allow_log_scale = max([arg.allow_log_scale for arg in args])
    unc_propagation = \
        (not any([arg.is_range for arg in args])
         and all([min(arg.rel_ampl()) > sigmas for arg in args]))
    if use_sigma_combs:
        if (unc_function is None
            and (((unc_function is None and len(args) > 5))
                 or all([min(arg.rel_ampl()) > 2*sigmas for arg in args]))):
            unc_propagation = False
    else:
        if unc_function is None:
            unc_propagation = False
    if not any(np.isnan([arg.center for arg in args])):
        if unc_propagation:
            args_center = [arg.center for arg in args]
            center = function(*args_center)
            domain1, domain2 = (domain if domain is not None
                                else [-np.inf, np.inf])
            if unc_function is not None:
                args_unc = [np.array(arg.unc) for arg in args]
                unc = unc_function(*args_center, *args_unc)
            else:
                inds_combs = list(itertools.product(*[[0,1,2]]*len(args)))
                comb_center = tuple([1]*len(args))
                inds_combs.remove(comb_center)
                args_combs = []
                args_all_vals = [[arg.center - arg.unc[0],
                                  arg.center,
                                  arg.center + arg.unc[1]] for arg in args]
                for i, inds in enumerate(inds_combs):
                    args_combs += [[]]
                    for j, arg in enumerate(args):
                        args_combs[i] += [args_all_vals[j][inds[j]]]
                new_comb = [function(*args_comb) for args_comb in args_combs]
                unc = np.array([center - min(new_comb), max(new_comb) - center])
                unc /= len(args)**0.5
        else:
            if domain is not None:
                domain1, domain2 = domain
            else:
                domain_args = \
                    [loguniform_distribution(*arg.domain, len_args_samples//2)
                     for arg in args]
                domain_distr = function(*domain_args)
                domain1, domain2 = min(domain_distr), max(domain_distr)
            domain1, domain2 = remove_zero_infs([domain1, domain2])
            args_distr = [arg.sample(len_args_samples) for arg in args]
            new_distr = function(*args_distr)
            x1, x2 = min(new_distr), max(new_distr)
            center, unc = center_and_uncs(new_distr)
            ord_range_1s = magnitude_order_range([center-unc[0], center+unc[1]])
            ord_range_x = magnitude_order_range([x1, x2])
            new_distr = new_distr[np.isfinite(new_distr)]
            probs_hr, bins_hr = np.histogram(new_distr, bins=4*len_args_samples)
            probs_lr, bins_lr = np.histogram(new_distr, bins=20)
            max_prob_hr = probs_hr.max()
            max_prob_lr = probs_lr.max()
            # plt.clf()
            # plt.plot(bins_hr[:-1], probs_hr,'-')
            # plt.plot(bins_lr[:-1], probs_lr, '--')
            cond_hr1 = (probs_hr[0] > 0.99*max_prob_hr
                       or probs_hr[-1] > 0.99*max_prob_hr)
            cond_hr2 = probs_hr[0] > 0.9*max_prob_hr
            cond_lr = 0.7*max_prob_lr < probs_lr[0] < max_prob_lr
            cond_range = (ord_range_x - ord_range_1s < 0.3 if x1 != x2
                          else False)
            if cond_hr1:
                xx1, xx2, xxc = [x1], [x2], [center]
                for i in range(num_reps_lims):
                    args_distr = [arg.sample(len_args_samples) for arg in args]
                    new_distr = function(*args_distr)
                    x1i, x2i = min(new_distr), max(new_distr)
                    xci = np.median(new_distr)
                    xx1 += [x1i]
                    xx2 += [x2i]
                    xxc += [xci]
                x1 = min(xx1)
                x2 = max(xx2)
                center = np.median(xxc)
                ord_range_b = log10(abs(center)) - log10(abs(x1))
                x1, x2 = add_zero_infs([x1, x2], zero_log-6, inf_log-6)
                if ord_range_b > inf_log-6 and cond_hr2:
                    center = np.nan
                    unc = [np.inf, np.inf]
                else:
                    args_center = [arg.center for arg in args]
                    x0 = function(*args_center)
                    domain_ = add_zero_infs([domain1, domain2],
                                            zero_log+6, inf_log-6)
                    x_ = RichValue([x1,x2], domain=domain_)
                    if x_.is_lolim:
                        x1 = x0 + (1 - lims_fraction) * (x1 - x0)
                    elif x_.is_uplim:
                        x2 = x0 - (1 - lims_fraction) * (x0 - x2)
                    center = [x1, x2]
                    unc = [0, 0]
            elif (cond_range or (not cond_range and cond_hr2)
                  or (not cond_range and not cond_hr2 and cond_lr)):
                xx1, xx2 = [x1], [x2]
                for i in range(num_reps_lims):
                    args_distr = [arg.sample(len_args_samples) for arg in args]
                    new_distr = function(*args_distr)
                    x1i, x2i = min(new_distr), max(new_distr)
                    xx1 += [x1i]
                    xx2 += [x2i]
                x1 = np.median(xx1)
                x2 = np.median(xx2)
                x1, x2 = add_zero_infs([x1, x2], zero_log+6, inf_log-6)
                center = [x1, x2]
                unc = [0, 0]
        domain = add_zero_infs([domain1, domain2], zero_log+6, inf_log-6)
    else:
        center, unc = np.nan, np.nan
        domain = [min([arg.domain[0] for arg in args]),
                  max([arg.domain[1] for arg in args])]
    new_rich_value = RichValue(center, unc, False, False, num_sf, min_exp,
                               domain, len_sample, allow_log_scale)
    return new_rich_value

def function_with_rich_arrays(function, args, **kwargs):
    """
    Apply a function to the input rich arrays, element by element.

    Parameters
    ----------
    function : function
        Function to be applied to the elements of the input rich arrays.
    args : list (rich arrays)
        List with the input rich arrays, in the same order as the arguments of
        the given function.
    * The rest of the arguments are the same as in 'function_with_rich_values'.
    
    Returns
    -------
    new_rich_array : rich array
        Resulting rich array.
    """
    if type(args) not in (tuple, list):
        args = [args]
    shape = args[0].shape
    same_shapes = True
    for arg in args[1:]:
        if arg.shape != shape:
            same_shapes = False
            break
    if not same_shapes:
        raise Exception('Input arrays have different shapes.')
    new_array = np.empty(0, RichValue)
    args_flat = np.array([arg.flatten() for arg in args])
    for i in range(args[0].size):
        args_i = np.array(args_flat)[:,i]
        new_rich_value = \
            function_with_rich_values(function, args_i, **kwargs)
        new_array = np.append(new_array, new_rich_value)
    new_array = rich_array(new_array.reshape(shape))
    return new_array

def evaluate_distr(distr, domain=[-np.inf,np.inf],
        zero_log=defaultparams['decimal exponent to define zero'],
        inf_log = defaultparams['decimal exponent to define infinity']):
    """
    Interpret the given distribution as a rich value.

    Parameters
    ----------
    distr : list/array (float)
        Input distribution of values.
    domain : list (float), optional
        Domain of the variable represented by the distribution.
        The default is [-np.inf,np.inf].

    Returns
    -------
    center : float
        Central value of the distribution. It can be a list representing an
        interval.
    unc : list (float)
        Lower and upper uncertainties of the distribution.
    """
    
    def magnitude_order_range(interval, zero_log=zero_log):
        x1, x2 = interval
        x1 = 0 if abs(x1) < 10**zero_log else x1
        x2 = 0 if abs(x2) < 10**zero_log else x2
        if x1*x2 > 0:
            d = log10(x2-x1)
        elif x1*x2 < 0:
            d = np.log10(abs(x1)) + 2*abs(zero_log) + np.log10(x2)
        else:
            xlim = x1 if x2 == 0 else x2
            d = abs(zero_log) + np.log10(abs(xlim))
        return d
   
    def add_zero_infs(interval, zero_log=zero_log, inf_log=inf_log):
        x1, x2 = interval
        if abs(x1) < 10**zero_log:
            x1 = 0
        elif x1 < 0 and abs(x1) > 10**inf_log:
            x1 = -np.inf
        if abs(x2) < 10**zero_log:
            x2 = 0
        elif x2 > 0 and x2 > 10**inf_log:
            x2 = np.inf
        new_interval = [x1, x2]
        return new_interval
    
    domain1, domain2 = domain
    x1, x2 = min(distr), max(distr)
    center, unc = center_and_uncs(distr)
    ord_range_1s = magnitude_order_range([center-unc[0], center+unc[1]])
    ord_range_x = magnitude_order_range([x1, x2])
    probs_hr, bins_hr = np.histogram(distr, bins=4*len(distr))
    probs_lr, bins_lr = np.histogram(distr, bins=20)
    max_prob_hr = probs_hr.max()
    max_prob_lr = probs_lr.max()
    cond_hr1 = (probs_hr[0] > 0.99*max_prob_hr
               or probs_hr[-1] > 0.99*max_prob_hr)
    cond_hr2 = probs_hr[0] > 0.9*max_prob_hr
    cond_lr = 0.9*max_prob_lr < probs_lr[0] < max_prob_lr
    cond_range = ord_range_x - ord_range_1s < 0.3 if x1 != x2 else False
    if x1 == x2:
        unc = 0
    elif cond_hr1:
        ord_range_b = np.log10(abs(center)) - np.log10(abs(x1))
        x1, x2 = add_zero_infs([x1, x2], zero_log-6, inf_log-6)
        if ord_range_b > inf_log-6 and cond_hr2:
            center = np.nan
            unc = [np.inf, np.inf]
        else:
            center = [x1, x2]
            unc = [0, 0]
    elif (cond_range or (not cond_range and cond_hr2)
          or (not cond_range and not cond_hr2 and cond_lr)):
        x1, x2 = add_zero_infs([x1, x2], zero_log+6, inf_log-6)
        center = [x1, x2]
        unc = [0, 0]
            
    return center, unc

def rich_fmean(array, function=lambda x: x, inverse_function=lambda x: x,
               weights=None, weight_function=lambda x: x, **kwargs):
    """
    Compute the generalized f-mean of the input values.

    Parameters
    ----------
    array : array / list (float)
        Input values.
    function : function, optional
        Function that defines the f-mean.
        The default is nothing (arithmetic mean).
    inverse function : function, optional.
        Inverse of the function that defines the f-mean.
        The default is  nothing (arithmetic mean).
    weights : array / list (float), optional
        Weights to be applied to the input values.
        The default are equal weights.
    weight_function : function, optional
        Function to be applied to the weights before normalization.
        The default is nothing.
    kwargs : optional
        Keyword arguments for the function 'function_with_rich_values'.

    Returns
    -------
    y : array
        Resulting geometric mean.
    """
    if function is not None and inverse_function is None:
        raise Exception('Inverse function not specified.')
    array = rich_array(array)
    if weights is None:
        weights = np.ones(len(array))
    weights = rich_array(weights, domain=[0,np.inf])
    def fmean_function(*args):
        x = args[:len(args)//2]
        w = args[len(args)//2:]
        w = weight_function(w)
        y = inverse_function((1/sum(w)) * sum([wi * function(xi)
                                               for xi,wi in zip(x,w)]))
        return y
    y = function_with_rich_values(fmean_function, [*array, *weights], **kwargs)
    return y

def bounded_gaussian(x, m=0., s=1., a=10., norm=False):
    """
    Bounded gaussian function.

    Parameters
    ----------
    x : array (float)
        Independent variable.
    m : float, optional
        Median of the curve. The default is 0.
    s : float, optional
        Width of the curve (similar to the standard deviation).
        The default is 1.
    a : float, optional
        Amplitude of the curve (distance from the median to the domain edges).
        The default is 10.
    norm : bool, optional
        If True, the curve will be normalized. The default is False.

    Returns
    -------
    y : array (float)
        Resulting array.
    """
    y = np.zeros(len(x))
    s_ = s * np.interp(a/s, width_factor[0,:], width_factor[1,:],
                       right=1.)
    x_ = (4/math.tau * a * np.tan(math.tau/4 * (x-m)/a))
    cond = np.greater(x, m-a) & np.less(x, m+a)
    y[cond] = np.exp(-x_[cond]**2/(2*s_**2))
    if norm:
        dm = min(a, 6*s)
        x1, x2 = m - dm, m + dm
        xl = np.linspace(x1, x2, int(4e3))
        yl = bounded_gaussian(xl, m, s, a)
        y /= np.trapz(yl, xl).sum()
    return y

def mirrored_loggaussian(x, m=0., s=1., a=10., norm=False):
    """
    Mirrored log-gaussian function.

    Parameters
    ----------
    x : array (float)
        Independent variable.
    m : float, optional
        Median of the curve. The default is 0.
    s : float, optional
        Uncertainty (defines the 1 sigma confidence interval).
        The default is 1.
    a : float, optional
        Amplitude of the curve (distance to the domain edges).
        The default is 10.
    norm : bool, optional
        If True, the curve will be normalized. The default is False.

    Returns
    -------
    y : array (float)
        Resulting array.
    """
    y = np.zeros(len(x))
    m_ = np.log(a)
    s_ = np.log(a-s) - m_
    x_ = x - (m-a)
    cond = np.greater(x, m-a) & np.less(x, m)
    y[cond] = (1 / x_[cond]
               * np.exp(-0.5*(np.log(x_[cond]) - m_)**2 / s_**2))
    x_ = x - m
    cond = np.greater(x, m) & np.less(x, m+a)
    y[cond] = (1 / x_[cond][::-1]
               * np.exp(-0.5*(np.log(x_[cond][::-1]) - m_)**2 / s_**2))
    if norm:
        y /= s_ * math.tau**0.5 
    return y

def log10(x):
    """
    Decimal logarithm from NumPy but including x = 0.
    """
    with np.errstate(divide='ignore'):
        y = np.log10(x)
    return y
    
width_factor = np.array(
    [
    [1.5978798107863545, 8.837825688232524],
    [1.6005670566153853, 8.675160051256942],
    [1.6009222672360193, 8.653998717097087],
    [1.6013169776223586, 8.630576767691744],
    [1.60199632090724, 8.590492230761335],
    [1.6031163584202386, 8.525029692077108],
    [1.6033452533800574, 8.51174686358221],
    [1.603804696169071, 8.485182543896775],
    [1.6042361333332888, 8.460355370865946],
    [1.6049220264129038, 8.421119712420127],
    [1.605367819678025, 8.395772329122725],
    [1.607095440504363, 8.298677738395034],
    [1.6079153375159185, 8.253226219829182],
    [1.608902289397169, 8.199045555064492],
    [1.609654325403863, 8.158148646377114],
    [1.6102769494912992, 8.124541634005164],
    [1.610574934237527, 8.108538028417906],
    [1.6107884982286922, 8.097100323775518],
    [1.6107923329078089, 8.096895196226251],
    [1.611388905410217, 8.065087444054269],
    [1.6116844386062283, 8.049407149777469],
    [1.6121319484650452, 8.025759882875857],
    [1.6121820153076918, 8.023121476175227],
    [1.61303035500458, 7.978636116304147],
    [1.6145509177775401, 7.899935228921496],
    [1.6149788077923208, 7.878026472286833],
    [1.6152782524171092, 7.862756156790656],
    [1.6157383167205372, 7.839393832147362],
    [1.6160222508374396, 7.825035115511099],
    [1.6165072086297114, 7.80061543111941],
    [1.6174546490701602, 7.7532880344559105],
    [1.6181725167719725, 7.717761655478075],
    [1.6195214226161387, 7.651777223178618],
    [1.6208095124001003, 7.589699548172522],
    [1.6221595092014633, 7.5256055881210475],
    [1.6231847170725489, 7.477586872214688],
    [1.6232425292103332, 7.474895807613914],
    [1.6240223215152647, 7.438771508474916],
    [1.6251268908138032, 7.388152846502962],
    [1.6252228534768725, 7.383785550219499],
    [1.625955135127176, 7.350618097290874],
    [1.6261396698391688, 7.342304133667039],
    [1.6270820424983001, 7.300123024613865],
    [1.6272868283552588, 7.291017588775223],
    [1.6273756094887035, 7.287076828102535],
    [1.6277318014325712, 7.271307280990803],
    [1.6289107407739107, 7.219577280088809],
    [1.630213130150036, 7.163253990612657],
    [1.6309889205072785, 7.130111289221496],
    [1.6310959667816056, 7.125561871185209],
    [1.6317170332355528, 7.099279900974576],
    [1.6317351845547188, 7.098514678029154],
    [1.631940130802676, 7.089885941043181],
    [1.6336581835321227, 7.0183698047618845],
    [1.6345532242410823, 6.981687187635516],
    [1.6351968803028216, 6.9555484909471526],
    [1.6363384435123385, 6.909682270345135],
    [1.6386567613678857, 6.818451076956774],
    [1.6387251657058026, 6.815797792829185],
    [1.6389457661426516, 6.807256042758334],
    [1.6393900514325614, 6.790122248430148],
    [1.6399920291780292, 6.767053925785203],
    [1.6418223203877813, 6.697944747520734],
    [1.644434595240348, 6.601951729840812],
    [1.6454328778176102, 6.566075227793643],
    [1.6455099321516984, 6.563324380319635],
    [1.6461549924501275, 6.540398259862792],
    [1.6481618357105137, 6.470235144583206],
    [1.6490622950899043, 6.439318642999033],
    [1.6509059631531005, 6.377093810819375],
    [1.6513969234096202, 6.360764834907109],
    [1.6515917443802517, 6.354313116387096],
    [1.6516393505666964, 6.352738986042591],
    [1.6532496314378016, 6.300046381569292],
    [1.6535255621178737, 6.291124209601459],
    [1.6538199120766481, 6.281640712264828],
    [1.6546747544463039, 6.254298552110283],
    [1.6549869702590174, 6.244385960777224],
    [1.6556452360842138, 6.223614879029224],
    [1.6561821670916952, 6.206800652025614],
    [1.6567188602437792, 6.190108319144952],
    [1.6568140995270653, 6.1871580742124905],
    [1.6570391193681488, 6.180201805846152],
    [1.6577614489398185, 6.158006163417444],
    [1.6592568792249855, 6.112700935121183],
    [1.6593271211578806, 6.110594154061858],
    [1.6607755037921916, 6.067573017235441],
    [1.6618317663817193, 6.036700065023896],
    [1.6619735984610902, 6.032586404911286],
    [1.6638234748402063, 5.979616500396014],
    [1.6638716926320496, 5.978252665869169],
    [1.6645663735249787, 5.958697781224797],
    [1.6653372274821185, 5.9372034393942945],
    [1.6655526264007638, 5.931235558770738],
    [1.666537740327789, 5.904153212342533],
    [1.6669611071418964, 5.89262010803019],
    [1.6670601099311708, 5.889932266254534],
    [1.6677749271206461, 5.8706278206318006],
    [1.6688818688180205, 5.8410853430010725],
    [1.671133929581369, 5.782279728340186],
    [1.6715917466720676, 5.770534904125728],
    [1.6728533984240583, 5.738529210276296],
    [1.6733226734981943, 5.72675842432096],
    [1.674019992103029, 5.709400307846973],
    [1.6742052072296527, 5.70481631547264],
    [1.6758993541748985, 5.663397563780642],
    [1.6770000270062724, 5.636975203421973],
    [1.6790629549606477, 5.588464777501108],
    [1.6801847994654628, 5.562627442913153],
    [1.6803988699012722, 5.557739972361382],
    [1.682016640935116, 5.521242040476601],
    [1.6820356453224483, 5.52081784327695],
    [1.6824037826378166, 5.512621312744266],
    [1.6838114310897478, 5.481639603285076],
    [1.6839069733455791, 5.479557225719518],
    [1.6843963844497492, 5.468930736685236],
    [1.6859234307683562, 5.436204344511804],
    [1.6892512335313241, 5.367082706526584],
    [1.6896348263819523, 5.359303592167604],
    [1.6907925260800352, 5.336056138941278],
    [1.6925041229281896, 5.302308747925549],
    [1.6957147376393047, 5.240943105382031],
    [1.69696028424948, 5.217793530578741],
    [1.6977943408474043, 5.2024908625303246],
    [1.7005061301451834, 5.182077449665332],
    [1.7007354855883532, 5.174422874879157],
    [1.7020084643925926, 5.133418022797484],
    [1.703646351208003, 5.087679443413405],
    [1.7050551339301656, 5.058784249175412],
    [1.7059661485891229, 5.046708710287813],
    [1.7101854491652024, 5.0258339656904445],
    [1.7116978949129198, 5.015177012499146],
    [1.712911698253638, 4.9985430012010745],
    [1.7140663836083208, 4.972867855289116],
    [1.714127156602557, 4.97119707991278],
    [1.7142535678509354, 4.967616713564703],
    [1.7159404407583265, 4.911233214425005],
    [1.7161628101292774, 4.903356117974435],
    [1.716754950403503, 4.882922605101845],
    [1.717573760445462, 4.857603532345358],
    [1.7197309465257868, 4.815077119455922],
    [1.7212614818444834, 4.799303750801535],
    [1.721711878606599, 4.795721889974177],
    [1.7225732247123116, 4.789156029684634],
    [1.7232468143351571, 4.783605164259132],
    [1.7239163389714034, 4.777212290555375],
    [1.7242063718162541, 4.774189934833639],
    [1.7249618350438598, 4.765783197992048],
    [1.7249801096431128, 4.765572316334618],
    [1.7257447336385698, 4.756555306660046],
    [1.7263691859389234, 4.749076492980475],
    [1.7294040362408034, 4.71340395253461],
    [1.7297433520378476, 4.709385414333748],
    [1.7320619132880841, 4.680849431575768],
    [1.7325806913110582, 4.674085269887279],
    [1.7360691078562196, 4.622693153388103],
    [1.737110263681717, 4.60480914557801],
    [1.737925898432924, 4.590015213269138],
    [1.7391757526146945, 4.566288056940045],
    [1.7401997278225965, 4.546145144347682],
    [1.7420368582159604, 4.509168014938565],
    [1.74252169457889, 4.4993606520085185],
    [1.7429749593507493, 4.4902144998439315],
    [1.745746406788323, 4.435917081249642],
    [1.7471783883035024, 4.409930691111515],
    [1.7499074904545049, 4.367042175083612],
    [1.7500042735518262, 4.365721493610022],
    [1.753107902574139, 4.330795265629622],
    [1.753391446460782, 4.328226750823082],
    [1.7568018505892355, 4.303043741195577],
    [1.7569241703524425, 4.302289164209875],
    [1.7571366291867683, 4.300995369068442],
    [1.7575153421522733, 4.298737579933105],
    [1.7599611287187156, 4.285050034141488],
    [1.7609755762337322, 4.279440970518453],
    [1.7639507090745727, 4.261034001804752],
    [1.7640034687503234, 4.260662750106765],
    [1.7654006303162677, 4.250006206437728],
    [1.765488469101683, 4.249277384419945],
    [1.76683697312611, 4.237179733450602],
    [1.767687165031461, 4.228730751803446],
    [1.7681434728606031, 4.223954469544617],
    [1.7682164121970272, 4.223176039588101],
    [1.7693346523819407, 4.210754136277527],
    [1.7701753295617781, 4.200855811059894],
    [1.7703920515073734, 4.1982326370676475],
    [1.7745916256163003, 4.142880323376406],
    [1.7746314295528576, 4.14232427553635],
    [1.7764079782368014, 4.11715675444101],
    [1.77793220409275, 4.095235803334708],
    [1.7787659218382155, 4.083222467391044],
    [1.7798786375604476, 4.067260731109907],
    [1.7803717202806681, 4.060238751402009],
    [1.7814527532589888, 4.045012971098418],
    [1.7827569381188253, 4.027069262555717],
    [1.7827745959734247, 4.026830230268215],
    [1.7837284770315704, 4.014100341640242],
    [1.7844130276848746, 4.005209879113749],
    [1.7850007061544684, 3.9977624391451845],
    [1.7857734480402991, 3.9882575522652655],
    [1.786753743887373, 3.9767040216599265],
    [1.7878339204515197, 3.9646010690553597],
    [1.7885191576085546, 3.9572433802582196],
    [1.7896984458561869, 3.9451216214514564],
    [1.7912022892978754, 3.9305699110951076],
    [1.791383037825524, 3.928883768401807],
    [1.791485481138193, 3.9279337674878927],
    [1.7937946345050468, 3.907514557990882],
    [1.7939518502432927, 3.9061872729191953],
    [1.794863936996558, 3.898624408430185],
    [1.7955406550698176, 3.8931527121441887],
    [1.79648266145493, 3.8857093909460496],
    [1.7971293314095915, 3.880701745700142],
    [1.7975105254248969, 3.877783902709247],
    [1.7982621971252237, 3.872094576037564],
    [1.7984825141561758, 3.870441357943902],
    [1.7989188913495016, 3.867183491737487],
    [1.8028225817468697, 3.8385323893941985],
    [1.8048279089984296, 3.8236817729160535],
    [1.8062232978909512, 3.8130750696799245],
    [1.8078729818246466, 3.8000853725328185],
    [1.8091611827199587, 3.7894988905319216],
    [1.8094171735678548, 3.787341032701515],
    [1.8102300675521203, 3.7803688148807364],
    [1.8113544922976863, 3.7704447637668586],
    [1.8121169388272254, 3.763547746522183],
    [1.812632979492065, 3.7588097262983644],
    [1.8127647545180063, 3.7575913113086967],
    [1.8145018510812156, 3.7412396311423324],
    [1.8145864861178178, 3.740430483478973],
    [1.8154478430978058, 3.732139292349304],
    [1.816191906819516, 3.7249035456131248],
    [1.817159749746342, 3.715408279182858],
    [1.8175720619343443, 3.711339868025428],
    [1.8180421335742676, 3.7066878994295944],
    [1.818066758743379, 3.706443849734426],
    [1.8202257840218234, 3.6849654807696677],
    [1.8229379499822105, 3.6580409031763086],
    [1.8233099883296595, 3.6543788819203917],
    [1.8241989604380826, 3.6456787784185924],
    [1.8249737126309222, 3.6381657128252654],
    [1.825173013996963, 3.6362450224242093],
    [1.825332952894828, 3.634707513519961],
    [1.8258610429566884, 3.629656634129183],
    [1.8284749735009438, 3.6052675844352784],
    [1.8322670188683647, 3.5716120604267703],
    [1.8416377894917293, 3.4961199287834224],
    [1.8453268553479014, 3.468945168834727],
    [1.8457687906740228, 3.4657724047150653],
    [1.8466883447274005, 3.4592238525450076],
    [1.8468704889543033, 3.4579350697253064],
    [1.8471980116425266, 3.455624454827607],
    [1.8473187836659544, 3.4547746192495743],
    [1.847432783344772, 3.4539735126528583],
    [1.8478141694751007, 3.4513009262478924],
    [1.8492609534999518, 3.441264866524845],
    [1.8498004410933049, 3.437562601211354],
    [1.851999774326964, 3.4226806269831602],
    [1.852493010568871, 3.419387243274398],
    [1.853969873228188, 3.40961602453935],
    [1.8541646325424632, 3.4083371268342866],
    [1.8566573880410238, 3.3921516904007762],
    [1.8577501332081963, 3.385155481809745],
    [1.8586935744916209, 3.379158846726944],
    [1.861527516624817, 3.361360163099843],
    [1.864444396893078, 3.3433135422553684],
    [1.8651605637040574, 3.338915787066081],
    [1.8652986495189459, 3.3380691913278047],
    [1.8654673636204624, 3.337035405250389],
    [1.865647329579526, 3.3359333886780975],
    [1.869619254091328, 3.3117982386156375],
    [1.870295873394111, 3.3077222257080696],
    [1.8739521950216025, 3.2858728280342633],
    [1.878927890489699, 3.2566123758875194],
    [1.879806869034349, 3.2514994133745185],
    [1.8839929774474804, 3.2273769777057812],
    [1.8855767593663197, 3.218347818524106],
    [1.886926234367478, 3.2106962372079457],
    [1.887428961247073, 3.207855540823155],
    [1.8876030198690268, 3.2068732441840195],
    [1.8882708679880746, 3.203110138941812],
    [1.8901056015617193, 3.1928198914266397],
    [1.8902531017097874, 3.1919956630836963],
    [1.8904095163698147, 3.191122113100951],
    [1.8935283021558613, 3.17380955604425],
    [1.8980237205201422, 3.149204565887466],
    [1.8983336755595015, 3.1475231117990403],
    [1.9055406023432822, 3.1089646949080616],
    [1.906875232260517, 3.1019359802053788],
    [1.9075021978746647, 3.0986460254961776],
    [1.9095364859749742, 3.08802336409461],
    [1.90984260046168, 3.086431759929103],
    [1.9114585301077076, 3.07805953430847],
    [1.9170034847469273, 3.0497051121286267],
    [1.9178962918249072, 3.0451932381253988],
    [1.9253016505540554, 3.0083303377708517],
    [1.9259217525996597, 3.0052883453777053],
    [1.9295441733066865, 2.987653935809647],
    [1.92992499392752, 2.985813431199494],
    [1.9343454279833758, 2.964633276816209],
    [1.9350579902531568, 2.9612504861107127],
    [1.9371498357443897, 2.9513695064050873],
    [1.93877824071358, 2.9437286427275207],
    [1.938951774555493, 2.942916997394256],
    [1.9433719625326327, 2.9224112497379013],
    [1.9441791843917633, 2.9187010906969872],
    [1.9485443519685919, 2.8988201967202967],
    [1.9513453266795022, 2.88622319690788],
    [1.9528129996391552, 2.8796716907267847],
    [1.9532430045667373, 2.8777585521945745],
    [1.9535897641603082, 2.876217868597675],
    [1.9541422913935722, 2.8737667839053676],
    [1.9562916352974569, 2.8642767116670673],
    [1.9686308567752595, 2.8111396978958108],
    [1.9733963861892863, 2.7912124369974642],
    [1.9738660223176105, 2.789266068071266],
    [1.9755989151585371, 2.782110987430161],
    [1.9773983241800477, 2.7747255178808974],
    [1.9815059483951332, 2.758033013551053],
    [1.9817297317893916, 2.757130198484852],
    [1.9859874635925958, 2.7400809843653087],
    [1.9866083343599934, 2.737614950951749],
    [1.9891196235117985, 2.7276919328286984],
    [1.9934112094656935, 2.710923144867124],
    [1.995804343444916, 2.7016740477504473],
    [1.9970299911035077, 2.696964906324475],
    [2.0147899013917936, 2.6266368116527814],
    [2.01659588157142, 2.620580880842677],
    [2.0168887985130626, 2.619603391337742],
    [2.026354489210936, 2.5885101566827324],
    [2.0269252633171813, 2.5866538904342944],
    [2.031682443503364, 2.5711887740472275],
    [2.0362480440516415, 2.5562710858524493],
    [2.036268827853039, 2.5562028266094923],
    [2.0365191907133524, 2.555380255762286],
    [2.0388715774175665, 2.547624356514787],
    [2.040147349539069, 2.5434003675479127],
    [2.0407589033369016, 2.5413719010986924],
    [2.0422364407413154, 2.536462891400017],
    [2.04795831340439, 2.517393738796674],
    [2.0540118820858737, 2.4972650815947097],
    [2.0551255820509247, 2.4935831276213687],
    [2.061266865216962, 2.473490557226331],
    [2.073398188669348, 2.435519879244982],
    [2.0763718276923857, 2.4267098430840295],
    [2.076373659768253, 2.4267044883838973],
    [2.076824456362622, 2.4253897720258553],
    [2.0808559917394507, 2.4138970878640698],
    [2.0830526814334362, 2.4078498880003245],
    [2.086511389744611, 2.3986653197731154],
    [2.0885040155128154, 2.3935739217192844],
    [2.0930607813673476, 2.3824932080369448],
    [2.096778284222603, 2.3739763001679006],
    [2.1009373693622138, 2.364923030790019],
    [2.101776349707054, 2.3631502996960965],
    [2.1066278791179713, 2.353197042322681],
    [2.109515344930321, 2.3474758409905516],
    [2.110649281312125, 2.345262309345945],
    [2.1142003346673484, 2.338426398992253],
    [2.1168557465722824, 2.3333855023387007],
    [2.1189550796504353, 2.3294265413922415],
    [2.1215470848519624, 2.3245526265587246],
    [2.12452872138757, 2.3189398210762002],
    [2.1261867693289043, 2.3158045743017173],
    [2.1272331827953086, 2.313817492287777],
    [2.129611080623757, 2.309269554989593],
    [2.139334050455762, 2.2898947188940135],
    [2.1407003124602726, 2.2870309970842464],
    [2.14360798827792, 2.2807946295733315],
    [2.144688678643154, 2.2784306158179115],
    [2.14602246857099, 2.275481213175677],
    [2.1510911620644775, 2.2639920634523825],
    [2.151204339763818, 2.263731064380164],
    [2.1514439568988943, 2.2631779140022217],
    [2.1528119508515395, 2.2600057291064095],
    [2.1542442824497616, 2.256660406816593],
    [2.1542541881012975, 2.256637192545458],
    [2.159328816466513, 2.244629204364797],
    [2.1608390083000937, 2.241021722573691],
    [2.1637811613379196, 2.233968957594618],
    [2.163999450342606, 2.2334449052911447],
    [2.169334448488197, 2.2206438637819033],
    [2.171814902367789, 2.214721628775572],
    [2.1786464699889225, 2.1986572196551077],
    [2.1794372401877538, 2.1968307623112855],
    [2.1842595834744527, 2.185865007542517],
    [2.189712155086588, 2.173814246996469],
    [2.1918645972619477, 2.1691558956944452],
    [2.1931796090082263, 2.1663368523982096],
    [2.1981635295389292, 2.155834185639141],
    [2.198674731360897, 2.154772895327526],
    [2.200753817882936, 2.150486597135703],
    [2.2019417578280303, 2.1480589386857734],
    [2.204124496935695, 2.1436383666713654],
    [2.2050923052312337, 2.1416947271316897],
    [2.209207241660681, 2.1335410798899543],
    [2.210082150381251, 2.131830125154795],
    [2.211614083413831, 2.128853093862009],
    [2.2187572530325097, 2.1152788697686717],
    [2.21929182903794, 2.114282813425623],
    [2.2230011563446084, 2.1074447238188925],
    [2.2242677935636275, 2.1051385705161687],
    [2.225672920845475, 2.102597122472196],
    [2.233491658258982, 2.0887682364552043],
    [2.236464715416416, 2.0836433633893887],
    [2.236492504426762, 2.083595795506723],
    [2.239709539360334, 2.0781299090733834],
    [2.2401723588953666, 2.0773501362788833],
    [2.2437314066216216, 2.0714075353709154],
    [2.2468644612117, 2.0662528536693294],
    [2.2476042881764435, 2.065045812865778],
    [2.2481857138782804, 2.0640998825490295],
    [2.2506957931221554, 2.0600427988569803],
    [2.2536325486614985, 2.0553495172663165],
    [2.2631453399381605, 2.0405170993233592],
    [2.275182054842996, 2.022468430300262],
    [2.2766975003372556, 2.020246169519325],
    [2.2787912579073843, 2.0171926809180443],
    [2.282982844239861, 2.0111371497097523],
    [2.284854160329998, 2.008457908761423],
    [2.2854109352836987, 2.0076635905244524],
    [2.286635024263361, 2.0059217914014567],
    [2.287634050529592, 2.0045048317897542],
    [2.28893314555472, 2.0026683870585447],
    [2.296733041506501, 1.9917837805595178],
    [2.2975778253882266, 1.9906190886129265],
    [2.2978694518610983, 1.990217655481686],
    [2.303100209227688, 1.983071147472692],
    [2.303124699425532, 1.983037923741974],
    [2.306786612316116, 1.978094237698824],
    [2.307041609572402, 1.9777517476758115],
    [2.3107434535595783, 1.9728050298069901],
    [2.3222480227114657, 1.9577193075963935],
    [2.322926081779786, 1.9568430747952121],
    [2.323213151275088, 1.956472518873096],
    [2.326590083773821, 1.9521317152728594],
    [2.3312957419455813, 1.946137260256832],
    [2.337769258230546, 1.9379886006066716],
    [2.3470628908795312, 1.9264724057324143],
    [2.3491345221159814, 1.9239324362214119],
    [2.353643819866852, 1.9184353795393339],
    [2.355893059916062, 1.9157089616394334],
    [2.358610565979031, 1.9124280395839504],
    [2.366268616679693, 1.9032541324943972],
    [2.371465934823483, 1.8970830114838466],
    [2.371535092532948, 1.8970011708515033],
    [2.374093253897338, 1.893978661564622],
    [2.3757127116246997, 1.892069923196237],
    [2.3824652961128217, 1.8841469735585545],
    [2.3905415865148156, 1.8747364553039205],
    [2.3936267849214063, 1.8711574341399335],
    [2.398511953495511, 1.86550839881092],
    [2.4014107059375727, 1.8621676690594846],
    [2.4032899685080222, 1.8600066521987197],
    [2.403515803703247, 1.8597472189556026],
    [2.404476475258764, 1.8586442633881168],
    [2.4048854990878956, 1.858174975759992],
    [2.408425853405624, 1.8541211231605774],
    [2.40876716866033, 1.853731091586932],
    [2.4115384246401237, 1.850569616270991],
    [2.4141188449080286, 1.8476346356882585],
    [2.422931899728742, 1.837679266156338],
    [2.4287134337604246, 1.8312108575734043],
    [2.430603324750826, 1.8291079745981433],
    [2.4329319772694036, 1.8265250254129046],
    [2.4341473533535054, 1.825180575104937],
    [2.4346235378353933, 1.8246545124887512],
    [2.4385332416338548, 1.820350361409279],
    [2.443533802920916, 1.8148858649202955],
    [2.4435736788026112, 1.8148424784840396],
    [2.451692870459835, 1.8060736933921957],
    [2.4556654820251387, 1.8018324619739325],
    [2.4558509617560604, 1.8016352622487688],
    [2.456267890264062, 1.801192260528503],
    [2.459550525818149, 1.7977176230001755],
    [2.465586193602617, 1.791392264699939],
    [2.467216013963671, 1.7896987439311072],
    [2.470476245628029, 1.7863302042739642],
    [2.4718988714279013, 1.784868445581762],
    [2.475521055405054, 1.7811694248056185],
    [2.4813134144326527, 1.7753242057623142],
    [2.4873059112923808, 1.769371315977648],
    [2.4929620969653334, 1.763844317195994],
    [2.4944965594708495, 1.7623607411485929],
    [2.4953527638168205, 1.761535885111871],
    [2.525250288737084, 1.7340027654548882],
    [2.5341113477645565, 1.7262904568636943],
    [2.5509420453968485, 1.7121592591963253],
    [2.555093960211657, 1.7087725233287163],
    [2.5570921354080323, 1.7071560566342117],
    [2.574302478307895, 1.693580374320437],
    [2.575770629716961, 1.6924499702781197],
    [2.578557625568321, 1.6903156026657662],
    [2.5817648783013767, 1.687877724914819],
    [2.5851509788725977, 1.6853247903342063],
    [2.5876164056242548, 1.683479238188873],
    [2.588908234046803, 1.6825165992790554],
    [2.5900551526347404, 1.6816644464017811],
    [2.5942311270664296, 1.6785813305937125],
    [2.5970081301216648, 1.6765478010425756],
    [2.6061067931519157, 1.6699753362525593],
    [2.6101792917716304, 1.6670766982552574],
    [2.611175265329593, 1.6663717277862735],
    [2.6131275289842772, 1.6649942701971896],
    [2.615067852034159, 1.6636309328730905],
    [2.6151596457685162, 1.6635665746789967],
    [2.6238708180365853, 1.657514660464361],
    [2.6268624908268103, 1.6554608603997725],
    [2.628748668561619, 1.654172213117872],
    [2.631803759992296, 1.6520949385363213],
    [2.632861186509545, 1.6513787782483333],
    [2.6334039470736026, 1.651011740679701],
    [2.6525098262902835, 1.6383169879582073],
    [2.6545448514574863, 1.6369888646467852],
    [2.665497362017048, 1.6299115378935127],
    [2.668922755057819, 1.6277210005944334],
    [2.671116467866824, 1.6263234573344054],
    [2.686177191469921, 1.6168369502747746],
    [2.703540857258465, 1.606131154207687],
    [2.7043007184856034, 1.605668243230808],
    [2.7070705076722272, 1.6039848146536964],
    [2.7111942789701695, 1.6014898844059338],
    [2.722368621226841, 1.5947975289911334],
    [2.7240822589015328, 1.5937799885177464],
    [2.7258169461711037, 1.5927523099308096],
    [2.7648630742135913, 1.5702389333372067],
    [2.770608311213728, 1.567024570549229],
    [2.7784879046096584, 1.5626562411384552],
    [2.7995679685018, 1.5511951156308723],
    [2.800730796572014, 1.5505723266140148],
    [2.81179036919115, 1.544697682291728],
    [2.812522805171327, 1.54431171825308],
    [2.8125767630004437, 1.54428329978792],
    [2.8161152646247265, 1.5424241667219623],
    [2.8165663800514884, 1.542187789315407],
    [2.82678720138083, 1.5368707529590635],
    [2.828011152486273, 1.5362389557238236],
    [2.8284819534078234, 1.5359962096971855],
    [2.8321852906168434, 1.5340921533652985],
    [2.835539557729093, 1.5323758028389032],
    [2.8409999346709496, 1.5295984247999015],
    [2.8829287652228683, 1.5089444570212338],
    [2.8949248188064853, 1.503248727382181],
    [2.9029955790624324, 1.499468719973519],
    [2.9042509698625256, 1.49888446971906],
    [2.917599142950983, 1.492733657742723],
    [2.921493632482736, 1.490960034249616],
    [2.9255552687748496, 1.4891202707654836],
    [2.939069677148417, 1.4830714008604817],
    [2.942576966689836, 1.4815196574436513],
    [2.948817436786528, 1.4787768828330785],
    [2.9535501449612283, 1.476712265561692],
    [2.961759364408669, 1.473162487039452],
    [2.98094549140685, 1.4650199755116984],
    [2.9873094013854926, 1.4623661795324994],
    [3.0073216499210544, 1.4541710872730969],
    [3.012870842434508, 1.4519385550055113],
    [3.0174867930917055, 1.4500945282871185],
    [3.01816451121154, 1.4498247801239277],
    [3.0643960224626423, 1.4320125581861314],
    [3.0917916921743207, 1.4219898318335513],
    [3.114613398186244, 1.4139315447309824],
    [3.129286345396797, 1.4088865926714644],
    [3.1313494266398174, 1.408185646091238],
    [3.1327613495374025, 1.4077071190830752],
    [3.144458737517312, 1.4037794162092976],
    [3.146588876204356, 1.4030711811434653],
    [3.1470339844556268, 1.4029234614707529],
    [3.1670422910123572, 1.3963791329487962],
    [3.171390235858927, 1.3949815494567253],
    [3.1824865665753372, 1.3914538880467948],
    [3.195017662737316, 1.3875367791336453],
    [3.2011219902575214, 1.3856538913863254],
    [3.2018483968312186, 1.3854309226702686],
    [3.22529131815439, 1.3783581366138273],
    [3.229739729941503, 1.3770425986701436],
    [3.245081187598224, 1.372569270537314],
    [3.250075260035253, 1.3711341026359136],
    [3.2512339385571143, 1.3708025883258546],
    [3.2515432639107393, 1.3707141786655817],
    [3.274363026123402, 1.3642983801955735],
    [3.2789377265546586, 1.3630371082528314],
    [3.28646737811021, 1.36097892562017],
    [3.289032941702588, 1.3602826602854552],
    [3.314556520640156, 1.3534920334737721],
    [3.3390349192957274, 1.347205332934382],
    [3.3397316123603358, 1.3470295482566466],
    [3.342383956351381, 1.3463618945433122],
    [3.3431291776075307, 1.3461747514369948],
    [3.3510424652314814, 1.3441995112311327],
    [3.3911938803997845, 1.3345049311287396],
    [3.393391781548897, 1.333989565535932],
    [3.410987627369356, 1.3299188401184865],
    [3.4162161714519734, 1.3287278306570431],
    [3.427873368237913, 1.326102404459864],
    [3.438688375947175, 1.3237029576412664],
    [3.4700095199319807, 1.3169437560873207],
    [3.492258280809284, 1.3123058727055088],
    [3.507574028069325, 1.3091880289246072],
    [3.516631613702746, 1.307371809241728],
    [3.519903306247057, 1.3067207077838037],
    [3.5377848393041016, 1.3032071947890531],
    [3.5546264597684316, 1.2999654448042077],
    [3.583990067442874, 1.2944608667285173],
    [3.590088366815328, 1.2933399987575533],
    [3.607090842802388, 1.2902533776888392],
    [3.6115612529165935, 1.2894509417093751],
    [3.6129000820775548, 1.289211343258891],
    [3.62021163431012, 1.287908633367001],
    [3.6279524792639712, 1.2865398758065458],
    [3.647176912133992, 1.2831851700066979],
    [3.6713289905501894, 1.2790552008658505],
    [3.6713539651847866, 1.279050976304763],
    [3.677607087419256, 1.2779961066666523],
    [3.6833751579000524, 1.27702804813036],
    [3.718044085550217, 1.2713035719591752],
    [3.735755914216523, 1.2684361977953529],
    [3.7375229626628155, 1.2681521591977718],
    [3.7504938802738916, 1.266078403776144],
    [3.7542840065077705, 1.2654761640009502],
    [3.754914474108436, 1.2653761471262053],
    [3.759255440871844, 1.2646887565419422],
    [3.7756350471106397, 1.2621147507529886],
    [3.7852964071009216, 1.2606110285695595],
    [3.8091955550267587, 1.2569372472350848],
    [3.8207173649821424, 1.2551893219859211],
    [3.826228694467358, 1.2543585241908166],
    [3.83217899748962, 1.2534653917451277],
    [3.8477672626847443, 1.2511444228352295],
    [3.8516963560589645, 1.2505636896977077],
    [3.883449324891642, 1.2459331534161477],
    [3.8984452165056203, 1.2437847396216257],
    [3.9032367080382264, 1.2431034305798883],
    [3.909512771052536, 1.2422147862854362],
    [3.910263656691838, 1.24210875130262],
    [3.9143948835713758, 1.2415264543276974],
    [3.9170498825161877, 1.241153201549771],
    [3.9219032466860018, 1.2404728490364596],
    [3.925465471614229, 1.2399750975564883],
    [3.9279052291613437, 1.2396349725022853],
    [3.9541035797328354, 1.2360225945290781],
    [3.954210545260445, 1.23600799445986],
    [3.959909757707361, 1.2352318323751612],
    [3.979248519442387, 1.232623505660397],
    [3.9983871305349616, 1.2300804588608698],
    [4.000580053271048, 1.2297914879680039],
    [4.033068337981131, 1.2255679485911715],
    [4.039138463520054, 1.2247906929156644],
    [4.048316686335823, 1.2236224942188743],
    [4.050649382423249, 1.2233269356964092],
    [4.055220814546073, 1.2227493005731565],
    [4.080428675202298, 1.2196013773097143],
    [4.0840172137003385, 1.2191583453833474],
    [4.085368247277059, 1.218991877650534],
    [4.088579638692452, 1.2185969035920035],
    [4.101983023600129, 1.216959273559987],
    [4.130105428705562, 1.2135797997018627],
    [4.139927810733705, 1.2124173056574643],
    [4.142152800693725, 1.2121552493281056],
    [4.152299875317724, 1.2109660842281287],
    [4.153179071589197, 1.2108635064429463],
    [4.168006330460352, 1.2091445179606166],
    [4.173234718507944, 1.2085432739515922],
    [4.215117721375297, 1.2038180642516356],
    [4.219327253306617, 1.2033520199755092],
    [4.231135586720534, 1.2020532461295703],
    [4.240126563913385, 1.201072753441046],
    [4.255436423492335, 1.1994197659185588],
    [4.260062372532041, 1.1989243970331076],
    [4.281374128586687, 1.196666530300281],
    [4.303326037681684, 1.19438216490277],
    [4.305437045736599, 1.194164677248451],
    [4.327450478353826, 1.191919384912185],
    [4.331248386515712, 1.1915361675624825],
    [4.348696246926157, 1.1897912051431299],
    [4.352960257642474, 1.1893686263031014],
    [4.357863971973818, 1.1888845163200947],
    [4.36522287547692, 1.1881617536431552],
    [4.370395917704531, 1.1876563476719781],
    [4.409193969478481, 1.1839352233769462],
    [4.429560202893744, 1.1820302201164248],
    [4.445784468894729, 1.1805360352807546],
    [4.445947946381007, 1.1805210843429952],
    [4.474369766049276, 1.1779531395675158],
    [4.507891246699752, 1.1750034572523473],
    [4.5179666561125815, 1.1741333048869187],
    [4.5357187696539425, 1.1726183462137572],
    [4.540603951470409, 1.1722054836171711],
    [4.543469227322805, 1.1719641352185002],
    [4.5588867742382, 1.170675643453022],
    [4.571452446375093, 1.1696380664445767],
    [4.644198968248831, 1.163847316236843],
    [4.65165407242235, 1.163274178011771],
    [4.666004199037043, 1.162181307640189],
    [4.672565440625473, 1.1616861205434421],
    [4.677532018379905, 1.161313149543579],
    [4.682661540671189, 1.1609296203334487],
    [4.697062818720539, 1.1598619019772525],
    [4.701248212896009, 1.1595540806174904],
    [4.701443345595917, 1.1595397564437486],
    [4.740240502947523, 1.1567392274660517],
    [4.762092638760625, 1.155202687074267],
    [4.7712277532919085, 1.1545688848566915],
    [4.773717481440245, 1.1543970091021412],
    [4.841671667420171, 1.1498451639298493],
    [4.846586158528337, 1.1495261403610983],
    [4.890083832718824, 1.146760000285841],
    [4.897148320719359, 1.1463203186795232],
    [4.897229509634285, 1.1463152808882622],
    [4.909979001930778, 1.1455284464318418],
    [4.917899795385453, 1.1450438614042675],
    [4.920073049528223, 1.144911468500755],
    [4.994820242258619, 1.140501267094594],
    [5.000282203230057, 1.1401895873970713],
    [5.003154669851046, 1.1400262336404883],
    [5.039120488282004, 1.1380129855894963],
    [5.044833165690714, 1.1376985785756173],
    [5.05027933348187, 1.1374001832425658],
    [5.083281948911297, 1.1356195067409298],
    [5.110269859773504, 1.1341975002202143],
    [5.1205629247169835, 1.1336629938751484],
    [5.1375368955609595, 1.1327907779634607],
    [5.199490846030355, 1.1297005958770534],
    [5.2121018811807724, 1.1290887214012013],
    [5.217507486622997, 1.128828150145237],
    [5.25260672897309, 1.1271603865683806],
    [5.274363049263102, 1.126146918966774],
    [5.304527686708617, 1.124766196210115],
    [5.319714191780301, 1.1240813756216865],
    [5.392979579240146, 1.120867368288074],
    [5.396782815292655, 1.120704337849223],
    [5.401609071610972, 1.1204979638620969],
    [5.403279507511452, 1.1204266666051292],
    [5.413984194568187, 1.119971357493538],
    [5.42629354030653, 1.1194511237629179],
    [5.461955699297492, 1.1179630782803853],
    [5.46812097176206, 1.117708582030432],
    [5.479484573337184, 1.117241534526747],
    [5.495211483686139, 1.1165993644169332],
    [5.50331930696767, 1.1162701475138121],
    [5.51864606903457, 1.1156511069108563],
    [5.524906762307743, 1.11539944479877],
    [5.539509961838693, 1.1148150494485136],
    [5.553709163723795, 1.1142501932694655],
    [5.555391464337308, 1.1141834826049966],
    [5.556027391776221, 1.114158276774745],
    [5.56591786426554, 1.1137670554953074],
    [5.580690881358869, 1.1131854196073292],
    [5.586342572071827, 1.1129637334925329],
    [5.597183141657323, 1.1125397798176395],
    [5.6127369246926095, 1.1119343990908876],
    [5.619313477273299, 1.1116794527922693],
    [5.642102111636256, 1.110800725867955],
    [5.666325803571217, 1.1098746231923264],
    [5.730771239516835, 1.107450401111327],
    [5.79840294992689, 1.1049675115909685],
    [5.798471358239619, 1.1049650315967383],
    [5.813634002204211, 1.1044168991190073],
    [5.827161382118696, 1.1039304910025434],
    [5.830240414241688, 1.1038201202261044],
    [5.8810037772643815, 1.1020186947891752],
    [5.9207074602272005, 1.1006335292177172],
    [5.957202262913631, 1.0993785466150587],
    [5.9622145242657, 1.0992075403224073],
    [5.97314418778903, 1.0988357767980201],
    [5.998349520390534, 1.0979843268810463],
    [6.007591475139511, 1.097674179434639],
    [6.091234980428194, 1.0949167876722563],
    [6.108062932364408, 1.0943727282144766],
    [6.162061425024644, 1.0926507945446333],
    [6.16363028736787, 1.09260130673396],
    [6.165761083686124, 1.0925341420934993],
    [6.221721474676941, 1.0907901920495622],
    [6.2262524575190525, 1.0906506634616087],
    [6.240328073524648, 1.0902188046241996],
    [6.245076373130527, 1.0900736620737674],
    [6.256884154368185, 1.0897139120424364],
    [6.260214219859699, 1.0896127584393454],
    [6.275367598257674, 1.0891541471270323],
    [6.323401504560067, 1.0877185795018973],
    [6.334103409686413, 1.0874024734072192],
    [6.344821115973035, 1.0870872570556327],
    [6.372122498146949, 1.0862904082588305],
    [6.385172368336001, 1.0859126041402887],
    [6.456866887757771, 1.0838722200988289],
    [6.4650615205673025, 1.0836427699317812],
    [6.504532788000536, 1.0825482734547336],
    [6.511639900316019, 1.0823530741725014],
    [6.5479478360737415, 1.0813647168834803],
    [6.605553521704332, 1.0798267002865611],
    [6.702887185260508, 1.077310382960316],
    [6.704990532017912, 1.0772571318531718],
    [6.711220122701372, 1.0770996918235514],
    [6.713738465234832, 1.077036163149408],
    [6.74195205710559, 1.076329033399912],
    [6.7630418700844945, 1.0758059350090805],
    [6.786142585444478, 1.075238305840631],
    [6.811818622977689, 1.0746139095665408],
    [6.853553500166567, 1.0736134898012426],
    [6.876927842544344, 1.073060958712547],
    [6.901333288092695, 1.072489954167967],
    [6.910160077436663, 1.0722849128238752],
    [6.923755850390236, 1.0719706167292489],
    [6.95001942195167, 1.0713686905436342],
    [6.950127833514506, 1.0713662200744563],
    [6.961930027064116, 1.0710979675530476],
    [6.979587643948638, 1.0706991885046673],
    [6.982220822881269, 1.0706399830903284],
    [6.992252948000491, 1.0704150384670845],
    [7.000728700346008, 1.0702257571548814],
    [7.024119404379601, 1.0697070155322426],
    [7.050191727609591, 1.0691350303074132],
    [7.068006231422805, 1.068747956371934],
    [7.0707931778520345, 1.0686876753777559],
    [7.132799836296648, 1.0673654750527393],
    [7.1507242047930415, 1.066989974267893],
    [7.15337968223222, 1.0669345977880837],
    [7.188209740084751, 1.0662142836665114],
    [7.210036647997927, 1.0657685498356109],
    [7.2249738851202085, 1.0654660069752533],
    [7.305781811847608, 1.0638639781670698],
    [7.313878556695871, 1.0637066463848304],
    [7.357385542242381, 1.0628710272443131],
    [7.374575837678372, 1.0625453716362931],
    [7.394270636160625, 1.0621753803584277],
    [7.417138446926203, 1.061749916705883],
    [7.453404387926054, 1.0610842024065965],
    [7.463006692174168, 1.0609097761132535],
    [7.483714998177512, 1.0605362064230266],
    [7.487576425173901, 1.0604669387431442],
    [7.531149308889672, 1.059693755912297],
    [7.559283599786365, 1.0592026866034416],
    [7.561210684790941, 1.059169282496745],
    [7.568004931231078, 1.0590517472959524],
    [7.7235992205520185, 1.0564586924405652],
    [7.745716391962058, 1.0561050601902997],
    [7.748343047342976, 1.0560633037916556],
    [7.760221932368906, 1.0558750994978126],
    [7.793372479762307, 1.0553553518088126],
    [7.8190092678220005, 1.0549588813977044],
    [7.822545996569073, 1.0549045576069593],
    [7.931833654623338, 1.053269401456443],
    [7.934210165804243, 1.0532347643023188],
    [8.03768590109493, 1.0517633754459155],
    [8.069209709639736, 1.0513290913879392],
    [8.077190151278407, 1.0512201614368268],
    [8.087129262240017, 1.0510850633509738],
    [8.094559443643806, 1.0509844771846897],
    [8.11932463791795, 1.0506517281111563],
    [8.148125662182856, 1.050269560670426],
    [8.151032464866697, 1.0502312741454487],
    [8.158418181897462, 1.05013422736492],
    [8.44478838575887, 1.0466163424807247],
    [8.454830347176452, 1.0465011794375727],
    [8.465089613722048, 1.0463840645383438],
    [8.510162882576005, 1.0458759137682199],
    [8.523626204513132, 1.0457261189728906],
    [8.551047876537403, 1.045423801117623],
    [8.589038413767476, 1.0450110225995337],
    [8.599402820059893, 1.044899612339494],
    [8.7987695390839, 1.042851453636723],
    [8.800885311841538, 1.0428306380486236],
    [8.845298111464741, 1.0423978890735817],
    [8.869425567922194, 1.0421660940725481],
    [8.875161007343154, 1.0421113280421537],
    [8.88895199447631, 1.0419801624241922],
    [8.89282841787938, 1.0419434254250097],
    [8.925686409688444, 1.0416343155121615],
    [8.985063665693005, 1.0410858269875363],
    [9.003877889665887, 1.0409146613025018],
    [9.040435141684428, 1.0405855702513007],
    [9.05022133508916, 1.0404982418615183],
    [9.067670405919067, 1.040343321351492],
    [9.072638702440107, 1.0402993935787934],
    [9.212579534297532, 1.039093684481883],
    [9.228615914362756, 1.038959204726214],
    [9.279319971457452, 1.0385386287964433],
    [9.338357493533028, 1.0380573576178194],
    [9.349647806563635, 1.0379663023943138],
    [9.357512227637141, 1.037903056975888],
    [9.441992705876254, 1.0372325856489815],
    [9.453954614523871, 1.0371389079458297],
    [9.462646687910532, 1.0370710257691311],
    [9.515369024206759, 1.0366626658555058],
    [9.549359628473342, 1.0364024567711214],
    [9.557897266668324, 1.0363374739680293],
    [9.612468718781864, 1.035925656232407],
    [9.621791764276615, 1.0358559113837382],
    [9.657838165641808, 1.035587918574688],
    [9.738974707007051, 1.0349943201176082],
    [9.871576061906891, 1.0340524959614965],
    [9.8806216303734, 1.0339895113409157],
    [9.882252112982423, 1.0339781752521546],
    [9.887129929091135, 1.0339442927755618],
    [9.913541602330914, 1.0337616367994693],
    [9.924919778683263, 1.033683366655627],
    [9.943086143150342, 1.0335589211475884],
    [9.988402718808329, 1.0332512650952481],
    [10.08947761439092, 1.0325792116189734],
    [10.157075929079477, 1.032140511916081],
    [10.172968636549992, 1.03203861054739],
    [10.209990889741245, 1.0318030463728063],
    [10.226726745552696, 1.031697390557585],
    [10.257655077911746, 1.0315034896708541],
    [10.264372279465137, 1.0314616086028916],
    [10.319891039389242, 1.0311185999327395],
    [10.409894413581906, 1.0305743471714222],
    [10.461753969479295, 1.0302673038313206],
    [10.462800229015832, 1.0302611581333547],
    [10.51224675994815, 1.029972900174928],
    [10.605571063095375, 1.0294404259088252],
    [10.727963154259825, 1.0287646511191124],
    [10.826942419615987, 1.0282364760564648],
    [10.833515119401886, 1.0282019741619115],
    [10.928088940803464, 1.027713302927346],
    [10.933343616181103, 1.0276865746646249],
    [11.070070974194703, 1.027006502827491],
    [11.072349772414642, 1.0269954170027313],
    [11.115469071791715, 1.0267871688771526],
    [11.178348486435267, 1.0264886133888986],
    [11.21455425709549, 1.0263194363990564],
    [11.27836008832675, 1.0260260924376465],
    [11.28608512507951, 1.0259909891750854],
    [11.332676935246242, 1.0257811428279053],
    [11.41102349656757, 1.025435439501886],
    [11.411533682610878, 1.0254332174751113],
    [11.441862144979815, 1.025301799387653],
    [11.464836028032929, 1.0252031263754455],
    [11.591095121223114, 1.0246741398328745],
    [11.679014376027487, 1.0243188246026533],
    [11.684847575618903, 1.0242956223767863],
    [11.727978975592016, 1.0241254824883563],
    [11.75562982277073, 1.0240177163447073],
    [11.81766684704964, 1.023779610377705],
    [11.86217045311649, 1.0236118973237618],
    [11.900138436044664, 1.0234708341930707],
    [11.924020551031704, 1.0233830487598994],
    [11.945931169654113, 1.0233031467876839],
    [12.022718877515196, 1.0230278803807582],
    [12.02415689086174, 1.0230227954095483],
    [12.028744306791513, 1.023006590876278],
    [12.066420579147335, 1.0228744829656602],
    [12.074560792144982, 1.022846168479726],
    [12.115953594202033, 1.0227034351254891],
    [12.22540047912155, 1.0223359124892728],
    [12.263554845543975, 1.0222111006521604],
    [12.269922440008347, 1.0221904349752728],
    [12.275292304188342, 1.0221730437092023],
    [12.299680072537477, 1.0220944764257804],
    [12.337106586015866, 1.0219752239273199],
    [12.413085185923164, 1.02173798310003],
    [12.547722965742917, 1.021333150879529],
    [12.661772797079186, 1.021005281156731],
    [12.670114978275963, 1.020981826496293],
    [12.68335022844755, 1.0209447599559374],
    [12.790936942229173, 1.0206499751981948],
    [12.803898897206569, 1.0206152320486863],
    [13.153661678860356, 1.019737094308477],
    [13.164580225121446, 1.019711441899386],
    [13.169050922786498, 1.019700967638009],
    [13.221999909937736, 1.0195782004223004],
    [13.227007516283187, 1.019566711499473],
    [13.280232273706675, 1.0194458784726446],
    [13.292601350993532, 1.019418129609821],
    [13.31577322025603, 1.0193664785293304],
    [13.386076572261592, 1.0192123863280147],
    [13.421829124860489, 1.019135505763145],
    [13.467244736546302, 1.0190392562962065],
    [13.480596738728199, 1.0190112553494994],
    [13.584854624065551, 1.0187971261894402],
    [13.8664870870107, 1.0182562762470135],
    [13.91137746857564, 1.0181747833068844],
    [13.965698340633473, 1.0180777892363622],
    [14.107918733573506, 1.0178318622707483],
    [14.111574987503488, 1.0178256869180977],
    [14.125381789573552, 1.017802431767624],
    [14.18439604523496, 1.01770416042105],
    [14.232566858501443, 1.0176252658576153],
    [14.24143854448786, 1.0176108618410904],
    [14.244356278001588, 1.0176061330826556],
    [14.254501126700278, 1.0175897237839377],
    [14.279897871734827, 1.0175488633847487],
    [14.281523119960168, 1.0175462591067443],
    [14.428374955855329, 1.017315974626543],
    [14.514284517567235, 1.0171856062327438],
    [14.520647101422508, 1.0171760708247828],
    [14.587020511855101, 1.0170775418260385],
    [14.658797061308057, 1.0169728483178166],
    [14.734807204505108, 1.0168639479613162],
    [14.85386804342278, 1.016697091317355],
    [14.891434996392205, 1.016645310055399],
    [14.892105285241628, 1.0166443896966957],
    [14.940421908859426, 1.0165783635116312],
    [14.94690710649081, 1.0165695476758403],
    [15.006623495358909, 1.0164888612248923],
    [15.083854148298707, 1.0163857381156887],
    [15.085092679027152, 1.016384094885446],
    [15.129567585377895, 1.0163252938006933],
    [15.27636462009583, 1.0161337580555656],
    [15.280935706833972, 1.01612784904902],
    [15.313613480348618, 1.0160856913161165],
    [15.338451416341396, 1.0160537420700813],
    [15.440288638664455, 1.015923484351489],
    [15.481605162465298, 1.015870919333139],
    [15.551053773631361, 1.0157828385246244],
    [15.633269120571851, 1.015678877463653],
    [15.69297276323188, 1.01560349396105],
    [15.69763320575605, 1.0155976110451472],
    [15.714825196967476, 1.0155759096019756],
    [15.765476113912092, 1.015511960426374],
    [15.816051944305626, 1.0154480534446484],
    [15.879755571427486, 1.0153674163675876],
    [15.9150871220302, 1.0153225955956042],
    [16.007225378515063, 1.0152053061078319],
    [16.017929027617175, 1.01519164283372],
    [16.040510975539313, 1.0151627915111232],
    [16.094681954651858, 1.0150934429943026],
    [16.149662394890104, 1.0150228629703342],
    [16.21506244062625, 1.0149386581071518],
    [16.505744082806206, 1.0145613481462075],
    [16.6321652334442, 1.014395833454197],
    [16.779632215279207, 1.0142017994960864],
    [16.780435525105116, 1.0142007398440394],
    [16.820625955695046, 1.014147688999851],
    [16.888700948485084, 1.0140576770395555],
    [16.965395262276807, 1.0139560472840654],
    [17.078931217129277, 1.0138051974045814],
    [17.094625044839763, 1.013784310378582],
    [17.153436736738616, 1.0137059654097382],
    [17.15635787760069, 1.0137020711706268],
    [17.202583960402425, 1.013640410990129],
    [17.231263556182178, 1.0136021234230899],
    [17.301867136739947, 1.0135077667401118],
    [17.32257797307244, 1.0134800623553566],
    [17.525329812203914, 1.0132082980247714],
    [17.540560654706862, 1.0131878469484121],
    [17.783722530340057, 1.0128608159712473],
    [17.913704317584585, 1.0126856944578846],
    [18.074150505211808, 1.012469364876635],
    [18.132788752197474, 1.0123902814136454],
    [18.146953974855894, 1.0123711767912131],
    [18.311478516309478, 1.0121493066100549],
    [18.43639568695539, 1.0119809365543744],
    [18.53240626926786, 1.0118516228914716],
    [18.913059819541996, 1.0113403184374279],
    [18.94425811719412, 1.0112985446704574],
    [19.04383042964497, 1.0111653893966812],
    [19.162168764939786, 1.011007508786092],
    [19.31693780590616, 1.0108017139821932],
    [19.353843730431354, 1.0107527675741106],
    [19.391511425236985, 1.0107028647155676],
    [19.56794849889147, 1.0104698918872275],
    [19.59460925041877, 1.010434806103875],
    [19.599555442669885, 1.010428300446706],
    [19.603032172538406, 1.0104237282279158],
    [19.647901133668977, 1.01036477205017],
    [19.827387441802433, 1.0101299218582431],
    [19.854416607095523, 1.0100946997578042],
    [19.92585513145714, 1.010001799499897],
    [19.988063588723257, 1.0099211366050875],
    [20.13510690208075, 1.009731385261181],
    [20.17878893067654, 1.0096752743697852],
    [20.219244640918298, 1.0096234175058754],
    [20.435937597883015, 1.0093475414007875],
    [20.473617544204675, 1.0092999094705046],
    [20.603893054299206, 1.0091360444920963],
    [21.05987666998213, 1.0085733778447112],
    [21.206935634370605, 1.0083958675547253],
    [21.209448607706314, 1.0083928521891259],
    [21.229801893169657, 1.008368452549827],
    [21.25457102477545, 1.0083388138758276],
    [21.40596597969495, 1.0081589883049202],
    [21.41347807655288, 1.008150126335594],
    [21.417663700451158, 1.0081451911210209],
    [21.46603081897459, 1.008088294612595],
    [21.549239464879278, 1.0079909910827962],
    [21.72040678040884, 1.0077932005647632],
    [21.825181135464714, 1.0076737591066647],
    [21.87626074375294, 1.0076159914213347],
    [21.961720772988414, 1.0075200344291437],
    [22.014077567175487, 1.0074616816583382],
    [22.048558220237886, 1.0074234339487473],
    [22.073718172084757, 1.007395616491665],
    [22.150386547141252, 1.0073113259933832],
    [22.26587374414386, 1.0071857155448503],
    [22.36637913931569, 1.0070777356629161],
    [22.420578698844903, 1.0070200236597724],
    [22.504740860063933, 1.006931130188663],
    [22.51177293725796, 1.0069237427035989],
    [22.517477298725183, 1.0069177545676948],
    [22.822130036341942, 1.0066038737511653],
    [22.855694383947462, 1.006570008663179],
    [22.868176274497138, 1.0065574514016264],
    [22.938862893282856, 1.0064867117210832],
    [22.942672045245143, 1.0064829177782875],
    [23.127671313250044, 1.0063008912025697],
    [23.127681901324557, 1.0063008809103244],
    [23.178123034954417, 1.006252013040504],
    [23.211734337530103, 1.0062196323397423],
    [23.430714392059702, 1.0060122565820666],
    [23.445026889709553, 1.005998920038548],
    [23.476646580901534, 1.005969551487591],
    [23.529733121267977, 1.005920539110611],
    [23.717067485667464, 1.0057505470230959],
    [23.761495733719403, 1.0057109127793125],
    [23.785480456151102, 1.0056896251129726],
    [23.827451531733402, 1.005652557860913],
    [23.884538693123176, 1.0056025177484544],
    [24.05607478100783, 1.0054547834449123],
    [24.143407091363432, 1.005381091142289],
    [24.2456051196967, 1.0052961666510796],
    [24.26172185809651, 1.0052829035305244],
    [24.334054357676266, 1.0052238142070782],
    [24.46018074983272, 1.005122491834989],
    [24.569873521293268, 1.005036149056112],
    [24.676672667971644, 1.0049536816682714],
    [24.817000691245177, 1.0048477341348914],
    [24.878424719265052, 1.0048022244218022],
    [25.036555214341, 1.0046875019512165],
    [25.346280558718743, 1.0044730586295005],
    [25.36374675258362, 1.004461373299509],
    [25.376913377441735, 1.0044525934133628],
    [25.534152828260904, 1.0043496685156836],
    [25.726935466954192, 1.0042283574509088],
    [25.924707434084045, 1.0041095385579775],
    [25.926192788485064, 1.0041086678825084],
    [26.127735053245722, 1.0039935481198394],
    [26.146180524186484, 1.003983312542174],
    [26.227409312923882, 1.0039388398578613],
    [26.35746564866689, 1.0038696841806127],
    [26.429819760121504, 1.0038323080137832],
    [26.51344801998228, 1.0037900892408858],
    [26.752267476376247, 1.0036753503108535],
    [26.856995809283738, 1.0036277734287407],
    [26.951524324834462, 1.0035862731469265],
    [26.963411051272896, 1.0035811517850972],
    [27.021069073915026, 1.003556618822111],
    [27.09001803253284, 1.003527955319763],
    [27.166494737202896, 1.003497023113395],
    [27.18879860378726, 1.003488172878757],
    [27.34750957016227, 1.0034274327289794],
    [27.447078853811632, 1.0033913368099519],
    [27.549670330153425, 1.0033557746285857],
    [27.62970064320572, 1.0033291857222368],
    [27.641960744979595, 1.0033252019635577],
    [27.86915306577209, 1.0032557000967082],
    [28.175914105621533, 1.0031747506165138],
    [28.49143905919613, 1.0031060263431648],
    [28.491674332660217, 1.0031059803051798],
    [28.566435057898367, 1.0030917302813676],
    [28.579135322834485, 1.0030893839804835],
    [28.696828021015893, 1.0030686498851484],
    [28.800574754830023, 1.0030518489159976],
    [28.99941864170119, 1.0030233533725923],
    [29.2580673690842, 1.0029931532684127],
    [29.394406195485743, 1.0029801556982945],
    [29.457296651109388, 1.002974803461547],
    [29.7810103697244, 1.002953263674665],
    [29.864372424694846, 1.002949245002635],
    [29.988811685911465, 1.002944323301554],
    [30.157260834460256, 1.0029395982693927],
    [30.16579850280058, 1.002939415227073],
    [30.180673246675646, 1.0029391089052917],
    [30.236790503115106, 1.0029380951966362],
    [30.36926939164922, 1.0029365599511515],
    [30.41161547354536, 1.0029363125474657],
    [30.41604092954182, 1.0029362932596182],
    [30.509476765700292, 1.0029361680120108],
    [30.97291009202411, 1.0029425851739968],
    [30.9932845910036, 1.0029430999905533],
    [31.070169197655435, 1.0029451925696966],
    [31.364981068646365, 1.0029551399861407],
    [31.403352381910416, 1.0029566258004872],
    [31.516295835672476, 1.0029612062243924],
    [31.66436123432976, 1.0029676056755534],
    [31.910764994913652, 1.0029789346255356],
    [32.20079502527375, 1.0029926801244924],
    [32.23888199920608, 1.002994472574646],
    [32.426539859026484, 1.0030031225483367],
    [32.52436944852101, 1.0030074548155252],
    [32.52652308413554, 1.0030075484276315],
    [32.69463640250984, 1.0030145732185938],
    [32.69802186732925, 1.0030147083112522],
    [32.78045622093644, 1.0030179086888693],
    [32.87937625246974, 1.003021502201927],
    [32.94194695266665, 1.003023620513555],
    [32.983507541013516, 1.0030249556846371],
    [33.102233973908085, 1.0030284269924812],
    [33.10677586736025, 1.0030285491138315],
    [33.28763247357163, 1.003032707901978],
    [33.34944112961419, 1.0030337907034725],
    [33.508521715116075, 1.003035694888228],
    [33.6269296003501, 1.0030362160477728],
    [33.772738957779914, 1.0030357034268902],
    [33.886026321256615, 1.003034355226412],
    [34.28936217045154, 1.0030219348833171],
    [34.42316883019513, 1.0030148889873174],
    [34.817819884275295, 1.0029843843385442],
    [34.823464925797815, 1.0029838349844369],
    [34.88135308013068, 1.0029780068751006],
    [34.93171789449454, 1.002972643700546],
    [34.966639879957654, 1.0029687624807235],
    [34.98381716124664, 1.0029668039934623],
    [35.12915918716906, 1.0029488989143647],
    [35.19308589407814, 1.0029402489897554],
    [35.21952823825193, 1.0029365291336447],
    [35.30217791660944, 1.0029243566513992],
    [35.3195459316411, 1.002921692234156],
    [35.34806081728322, 1.0029172365079404],
    [35.434452641130385, 1.002903111948009],
    [35.55405186178376, 1.0028819703655005],
    [35.65932715359921, 1.002861788249631],
    [35.671904155178765, 1.002859276468051],
    [35.81889652687997, 1.0028283408382939],
    [36.02179996079061, 1.002781034977575],
    [36.281129582936494, 1.0027133044737195],
    [36.29533458348593, 1.0027093716080333],
    [36.91007315062819, 1.002519362022241],
    [37.15542945979633, 1.0024339837771246],
    [37.4656170547298, 1.0023196444921652],
    [38.09148641826565, 1.0020719479138744],
    [38.09363089147063, 1.0020710688527825],
    [38.34717339554801, 1.0019660695058066],
    [38.780885705465174, 1.0017829650556314],
    [38.7929191671683, 1.0017778477925323],
    [38.82367905380545, 1.0017647623120038],
    [38.827633324664276, 1.0017630796848462],
    [38.856257795761, 1.001750896697909],
    [38.91389338175691, 1.001726355073423],
    [38.923543686502214, 1.0017222448888587],
    [39.038301744573424, 1.0016733635098358],
    [39.75991974976459, 1.0013693178704848],
    [39.93334114376286, 1.0012981849459837],
    [39.94884129433083, 1.0012918820924155],
    [40.03748077645377, 1.0012560291907917],
    [40.24208398647277, 1.0011746386116989],
    [40.5180729571707, 1.0010683940554979],
    [40.81840248118606, 1.0009583343787136],
    [40.968888906891316, 1.0009057037725348],
    [41.095474699088356, 1.0008628729711755],
    [41.20836367804738, 1.000825859799745],
    [41.364033068809974, 1.000776763787159],
    [41.37481667574844, 1.0007734496913159],
    [41.52593917694728, 1.0007282438421934],
    [41.64159791266181, 1.000695266899482],
    [41.83182032436881, 1.0006441990580042],
    [41.88058045760582, 1.00063173490835],
    [41.91264992785347, 1.0006236742283647],
    [42.085369934549625, 1.000582100398957],
    [42.28441524704338, 1.000537939546774],
    [42.628272817538964, 1.0004706680947837],
    [42.70918982769089, 1.000456427238506],
    [42.79414478358315, 1.0004421036907813],
    [43.567301560379775, 1.0003392557362076],
    [43.6367550320595, 1.0003322639035626],
    [43.9120352705645, 1.000307845104614],
    [43.935974433963146, 1.000305961448941],
    [44.10698323135863, 1.0002935702284412],
    [44.12383395664876, 1.0002924481087772],
    [44.13249005526041, 1.0002918784368673],
    [44.329255425165464, 1.0002801355898525],
    [44.36589750073797, 1.000278197281391],
    [44.52622737330225, 1.0002705924960813],
    [44.58538644824575, 1.0002681358212508],
    [45.355953361491444, 1.0002512485560995],
    [45.4533288708887, 1.0002508533472785],
    [45.87898051638183, 1.0002527548698412],
    [45.92432417110684, 1.000253265479933],
    [46.00289388542206, 1.0002542730016872],
    [46.00617444715796, 1.0002543183420887],
    [46.061546442082715, 1.0002551217474906],
    [46.30344846648201, 1.0002594040820179],
    [46.33671802947605, 1.000260081398276],
    [46.73482750313948, 1.0002694618041756],
    [46.83742131179224, 1.0002721700904913],
    [47.135206766767624, 1.000280391887752],
    [47.14588982759803, 1.0002806923435796],
    [47.290833183904084, 1.0002847726620339],
    [47.590666409123415, 1.0002931246724114],
    [47.76614941248089, 1.0002979446528728],
    [47.81577626291668, 1.0002992986490815],
    [47.895097912760136, 1.0003014545083782],
    [47.92017907258865, 1.000302134054864],
    [48.251083420337906, 1.0003110041449155],
    [48.613021744244236, 1.0003205040043832],
    [48.840684615361496, 1.0003263719911957],
    [48.97563917898181, 1.0003298114231307],
    [49.07358111927753, 1.0003322894313158],
    [49.083050278275415, 1.0003325282016216],
    [49.20737846757686, 1.0003356500328884],
    [49.25098790182458, 1.0003367392546338],
    [49.31399789036999, 1.0003383077327777],
    [49.82928511341699, 1.0003509001689974],
    [49.867546049889555, 1.000351818591501],
    [49.949791584371305, 1.0003537851002895],
    [50.05084002350168, 1.0003561867709196],
    [51.245621019172226, 1.0003833899139853],
    [51.28631141893583, 1.0003842779639778],
    [51.48767777073504, 1.000388635840713],
    [51.564413245377715, 1.000390280415001],
    [51.65233382918606, 1.0003921538130067],
    [51.89960999305819, 1.000397360553626],
    [52.28053305176791, 1.0004052028200436],
    [52.38519486287389, 1.0004073198195416],
    [52.40094849843612, 1.0004076370659525],
    [52.54907199890495, 1.0004106020546388],
    [52.732537738743886, 1.0004142296935266],
    [52.8387721333109, 1.00041630766033],
    [53.090663213126476, 1.0004211687548399],
    [53.10027536195103, 1.0004213524213317],
    [53.305519593910766, 1.0004252421365472],
    [53.36778801443332, 1.000426410147767],
    [53.46364830397677, 1.0004281973038103],
    [53.712050542625796, 1.0004327667037773],
    [53.78625821427279, 1.0004341145618973],
    [53.81389805207832, 1.000434614575613],
    [54.29089461890311, 1.0004430718250943],
    [55.00122568404018, 1.000455069816092],
    [55.08710168588248, 1.0004564724471765],
    [55.09248062619561, 1.000456559960995],
    [55.273982877630715, 1.0004594893977217],
    [55.663127211544314, 1.0004656166606343],
    [55.851283555926, 1.000468504629337],
    [56.0099376577326, 1.000470902183311],
    [56.98396372337716, 1.0004848756238856],
    [57.045263989620075, 1.0004857125435604],
    [57.195903108805055, 1.000487748013811],
    [57.23077258901091, 1.0004882148967633],
    [57.38194572800303, 1.0004902204569657],
    [57.45382079313958, 1.0004911634382985],
    [57.62392992486379, 1.0004933681979193],
    [58.60499156776304, 1.0005053495672056],
    [58.64315581592475, 1.000505790622351],
    [58.68516136665831, 1.0005062739227042],
    [58.95741783339378, 1.0005093519984918],
    [59.26107267960701, 1.0005126744382744],
    [59.33322738003109, 1.0005134468613301],
    [59.43322738003109, 1.000514506585805],
    [59.60140699118094, 1.0005162606533122],
    [59.77006250415963, 1.000517984332775],
    [59.93919526563791, 1.0005196774939389],
    [60.10880662609714, 1.0005213400072581],
    [60.27889793984018, 1.0005229717439161],
    [60.44947056500217, 1.0005245725758447],
    [60.620525863561504, 1.0005261423757428],
    [60.79206520135049, 1.0005276810171002],
    [60.9640899480664, 1.000529188374214],
    [61.13660147728234, 1.000530664322212],
    [61.30960116645835, 1.000532108737072],
    [61.483090396952186, 1.0005335214956437],
    [61.65707055403052, 1.0005349024756711],
    [61.8315430268799, 1.0005362515558127],
    [62.006509208617985, 1.0005375686156646],
    [62.18197049630448, 1.000538853535782],
    [62.35792829095238, 1.0005401061977028],
    [62.534383997539265, 1.00054132648397],
    [62.7113390250183, 1.0005425142781543],
    [62.88879478632964, 1.0005436694648795],
    [63.06675269841164, 1.0005447919298431],
    [63.2452141822123, 1.0005458815598434],
    [63.4241806627004, 1.0005469382428018],
    [63.60365356887702, 1.0005479618677893],
    [63.78363433378691, 1.0005489523250497],
    [63.96412439453, 1.0005499095060257],
    [64.14512519227276, 1.000550833303384],
    [64.32663817225976, 1.0005517236110428],
    [64.5086647838252, 1.000552580324195],
    [64.69120648040459, 1.0005534033393382],
    [64.87426471954615, 1.000554192554299],
    [65.05784096292261, 1.0005549478682603],
    [65.24193667634279, 1.0005556691817914],
    [65.42655332976345, 1.000556356396873],
    [65.6116923973008, 1.0005570094169254],
    [65.7973553572424, 1.0005576281468402],
    [65.98354369205909, 1.0005582124930048],
    [66.17025888841653, 1.0005587623633356],
    [66.35750243718732, 1.0005592776673045],
    [66.54527583346275, 1.0005597583159709],
    [66.73358057656495, 1.0005602042220099],
    [66.92241817005855, 1.000560615299746],
    [67.11179012176295, 1.0005609914651814],
    [67.30169794376418, 1.000561332636028],
    [67.4921431524272, 1.0005616387317406],
    [67.68312726840772, 1.0005619096735463],
    [67.87465181666452, 1.0005621453844802],
    [68.06671832647167, 1.000562345789416],
    [68.2593283314306, 1.000562510815099],
    [68.45248336948237, 1.000562640390183],
    [68.64618498292, 1.000562734445261],
    [68.84043471840091, 1.0005627929129],
    [69.03523412695897, 1.0005628157276787],
    [69.23058476401711, 1.000562802826219],
    [69.42648818939963, 1.0005627541472246],
    [69.62294596734483, 1.0005626696315157],
    [69.81995966651724, 1.0005625492220662],
    [70.01753086002032, 1.0005623928640404],
    [70.21566112540894, 1.0005622005048296],
    [70.41435204470216, 1.0005619720940921],
    [70.61360520439553, 1.0005617075837892],
    [70.81342219547406, 1.0005614069282251],
    [71.01380461342472, 1.0005610700840861],
    [71.21475405824941, 1.0005606970104794],
    [71.41627213447745, 1.000560287668974],
    [71.61836045117857, 1.0005598420236415],
    [71.82102062197582, 1.0005593600410958],
    [72.02425426505823, 1.0005588416905355],
    [72.22806300319391, 1.0005582869437863],
    [72.43244846374292, 1.0005576957753426],
    [72.63741227867038, 1.0005570681624105],
    [72.84295608455939, 1.0005564040849513],
    [73.04908152262406, 1.0005557035257266],
    [73.25579023872271, 1.00055496647034],
    [73.4630838833711, 1.000554192907285],
    [73.67096411175535, 1.0005533828279882],
    [73.87943258374534, 1.000552536226857],
    [74.08849096390792, 1.0005516531013245],
    [74.29814092152026, 1.000550733451897],
    [74.50838413058308, 1.0005497772822027],
    [74.71922226983403, 1.000548784599038],
    [74.93065702276112, 1.000547755412417],
    [75.14269007761631, 1.0005466897356199],
    [75.35532312742865, 1.000545587585245],
    [75.56855787001807, 1.0005444489812554],
    [75.78239600800894, 1.0005432739470324],
    [75.99683924884349, 1.0005420625094266],
    [76.21188930479556, 1.0005408146988086],
    [76.42754789298422, 1.0005395305491231],
    [76.64381673538763, 1.000538210097941],
    [76.86069755885656, 1.0005368533865133],
    [77.07819209512833, 1.0005354604598256],
    [77.29630208084053, 1.0005340313666529],
    [77.51502925754514, 1.0005325661596156],
    [77.73437537172205, 1.0005310648952346],
    [77.95434217479325, 1.0005295276339896],
    [78.17493142313691, 1.000527954440375],
    [78.3961448781011, 1.0005263453829583],
    [78.61798430601807, 1.000524700534439],
    [78.8404514782183, 1.00052301997171],
    [79.0635481710447, 1.0005213037759122],
    [79.28727616586669, 1.0005195520325005],
    [79.51163724909445, 1.0005177648313037],
    [79.73663321219321, 1.0005159422665835],
    [79.96226585169765, 1.0005140844371014],
    [80.18853696922605, 1.0005121914461799],
    [80.41544837149478, 1.0005102634017664],
    [80.6430018703327, 1.0005083004164974],
    [80.87119928269577, 1.000506302607767],
    [81.10004243068128, 1.0005042700977884],
    [81.32953314154263, 1.0005022030136657],
    [81.55967324770378, 1.000500101487457],
    [81.79046458677409, 1.0004979656562463],
    [82.02190900156268, 1.0004957956622103],
    [82.25400834009335, 1.0004935916526883],
    [82.48676445561941, 1.0004913537802553],
    [82.72017920663822, 1.0004890822027899],
    [82.95425445690626, 1.000486777083549],
    [83.18899207545384, 1.0004844385912388],
    [83.42439393660024, 1.0004820669000902],
    [83.66046191996847, 1.0004796621899328],
    [83.89719791050032, 1.0004772246462685],
    [84.13460379847146, 1.0004747544603503],
    [84.37268147950661, 1.0004722518292568],
    [84.6114328545945, 1.0004697169559722],
    [84.85085983010306, 1.0004671500494617],
    [85.09096431779493, 1.0004645513247556],
    [85.33174823484232, 1.000461921003024],
    [85.57321350384257, 1.0004592593116626],
    [85.81536205283344, 1.000456566484372],
    [86.05819581530845, 1.0004538427612415],
    [86.30171673023251, 1.0004510883888316],
    [86.54592674205716, 1.000448303620263],
    [86.79082780073617, 1.0004454887152947],
    [87.03642186174126, 1.000442643940417],
    [87.2827108860775, 1.0004397695689364],
    [87.52969684029907, 1.0004368658810625],
    [87.77738169652493, 1.000433933164],
    [88.02576743245474, 1.000430971712037],
    [88.27485603138436, 1.0004279818266355],
    [88.52464948222183, 1.0004249638165268],
    [88.77514977950328, 1.0004219179978011],
    [89.0263589234089, 1.0004188446940017],
    [89.27827891977874, 1.0004157442362236],
    [89.5309117801288, 1.0004126169632044],
    [89.78425952166727, 1.000409463221425],
    [90.03832416731032, 1.000406283365208],
    [90.29310774569849, 1.0004030777568127],
    [90.54861229121273, 1.0003998467665403],
    [90.8048398439909, 1.0003965907728305],
    [91.06179244994374, 1.0003933101623677],
    [91.31947216077137, 1.0003900053301815],
    [91.57788103397964, 1.0003866766797525],
    [91.8370211328967, 1.000383324623118],
    [92.0968945266892, 1.0003799495809769],
    [92.35750329037899, 1.0003765519827983],
    [92.61884950485977, 1.0003731322669311],
    [92.88093525691345, 1.0003696908807134],
    [93.14376263922703, 1.0003662282805825],
    [93.40733375040922, 1.0003627449321906],
    [93.67165069500716, 1.0003592413105125],
    [93.93671558352344, 1.0003557178999654],
    [94.20253053243262, 1.0003521751945221],
    [94.4690976641983, 1.0003486136978295],
    [94.73641910729025, 1.0003450339233237],
    [95.00449699620101, 1.0003414363943535],
    [95.27333347146326, 1.0003378216442997],
    [95.54293067966671, 1.0003341902166936],
    [95.81329077347547, 1.0003305426653457],
    [96.08441591164497, 1.000326879554466],
    [96.35630825903935, 1.0003232014587922],
    [96.62896998664867, 1.0003195089637138],
    [96.90240327160645, 1.000315802665404],
    [97.17661029720674, 1.0003120831709467],
    [97.45159325292171, 1.0003083510984674],
    [97.72735433441929, 1.0003046070772688],
    [98.00389574358039, 1.0003008517479592],
    [98.28121968851666, 1.0002970857625928],
    [98.55932838358814, 1.0002933097848017],
    [98.83822404942082, 1.000289524489937],
    [99.11790891292459, 1.0002857305652064],
    [99.39838520731075, 1.0002819287098141],
    [99.67965517210996, 1.0002781196351063],
    [99.96172105319029, 1.0002743040647095]
    ]
).transpose()