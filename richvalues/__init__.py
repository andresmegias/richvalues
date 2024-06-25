#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rich Values Library
-------------------
Version 4.1

Copyright (C) 2024 - Andrés Megías Toledano

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

__version__ = '4.1.3'
__author__ = 'Andrés Megías Toledano'

import copy
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import linregress

variable_count = 0
variable_dict = {}

defaultparams = {
    'domain': [-np.inf, np.inf],
    'size of samples': int(8e3),
    'number of significant figures': 1,
    'minimum exponent for scientific notation': 4,
    'maximum number of decimals': 5,
    'limit for extra significant figure': 2.5,
    'use extra significant figure for exact values': True,
    'use extra significant figure for finite intervals': True, 
    'omit ones in scientific notation in LaTeX': False,
    'multiplication symbol for scientific notation in LaTeX': '\\cdot',
    'sigmas to define upper/lower limits from read values': 2.,
    'sigmas to use approximate uncertainty propagation': 20.,
    'use 1-sigma combinations to approximate uncertainty propagation': False,
    'fraction of the central value for upper/lower limits': 0.2,
    'number of repetitions to estimate upper/lower limits': 4,
    'decimal exponent to define zero': -90.,
    'decimal exponent to define infinity': 90.,
    'sigmas for intervals': 3.,
    'sigmas for overlap': 1.,
    'show domain': False,
    'assume integers': False,
    'show asterisk for rich values with custom PDF': True,
    'save PDF in rich values': False
    }

original_defaultparams = copy.copy(defaultparams)

def set_default_params(new_params_dict):
    """Set the values of the specified default parameters."""
    for param in new_params_dict:
        if param in original_defaultparams:
            defaultparams[param] = new_params_dict[param]
        else:
            print("Warning: Parameter '{}' does not exist. ".format(param)
                  + "Check the variable 'original_defaultparams' to see the"
                  + " parameter names ('richvalues.original_defaultparams').")

def restore_default_params():
    """Restore original values of the default parameters"""
    for var in original_defaultparams:
        defaultparams[var] = copy.copy(original_defaultparams[var])

def set_default_value(var, param):
    """Set the current variable as the general parameter value"""
    if var is None:
        var = copy.copy(defaultparams[param])
    return var

def round_sf(x, n=None, min_exp=None, extra_sf_lim=None):
    """
    Round the number to the given number of significant figures.

    Parameters
    ----------
    x : float
        Input number.
    n : int, optional
        Number of significant figures. The default is 1.
    min_exp : int, optional
        Minimum decimal exponent, in absolute value, to display the value in
        scientific notation. The default is 4.
    extra_sf_lim : float, optional
        If the number expressed in scientific notation has a base that is lower
        than this value, an additional significant figure will be used.
        The default is 2.5.

    Returns
    -------
    y : str
        Rounded number.
    """
    n = set_default_value(n, 'number of significant figures')
    min_exp = set_default_value(min_exp,
                                'minimum exponent for scientific notation')
    extra_sf_lim = set_default_value(extra_sf_lim,
                                     'limit for extra significant figure')
    x = float(x)
    n = max(0, n)
    n_ = copy.copy(n)
    if np.isnan(x):
        y = 'nan'
        return y
    elif np.isinf(x):
        y = str(x)
        return y
    use_exp = True
    if x == 0 or abs(np.floor(_log10(abs(x)))) < min_exp:
        use_exp = False
    sign = '-' if x < 0 else ''
    x = abs(x)
    base = '{:e}'.format(x).split('e')[0]
    m = n+1 if round(float(base), n) <= extra_sf_lim else n
    y = str(float('{:.{}g}'.format(x, m)))
    base = '{:e}'.format(float(y)).split('e')[0]
    if round(float(base), n) <= extra_sf_lim:
        n += 1
    integers = len(y.split('.')[0])
    if x >= 1 and integers >= n:
        y = y.replace('.0','')
    digits = str(y).replace('.','')
    for i in range(len(digits)-1):
        if digits.startswith('0'):
            digits = digits[1:]
    digits = len(digits)
    if n > digits and 'e' not in y:
        y = y + '0'*(n-digits)
    if use_exp:
        y = '{:.{}e}'.format(float(y), max(n-1,0))
        if 'e' in y:
            y, a = y.split('e')
            y = round_sf(y, n_, np.inf, extra_sf_lim)
            y = '{}e{}'.format(y, a)
    else:
        if 'e' in y:
            base, exp = y.split('e')
            y = '0.' + '0'*int(abs(float(exp))-1) + base.replace('.', '')
    y = sign + y
    if x == 0:
        y = '0'
    y = y.replace('e+','e').replace('e00','e0')
    return y

def round_sf_unc(x, dx, n=None, min_exp=None, max_dec=None, extra_sf_lim=None):
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
        Minimum decimal exponent, in absolute value, to display the values in
        scientific notation. The default is 4.
    max_dec : int, optional
        Maximum number of decimals, to use the notation with parenthesis.
        The default is 5.
    extra_sf_lim : float, optional
        If the number expressed in scientific notation has a base that is lower
        than this value, an additional significant figure will be used.
        The default is 2.5.

    Returns
    -------
    y : float
        Rounded value.
    dy : float
        Rounded uncertainty.
    """
    n = set_default_value(n, 'number of significant figures')
    min_exp = set_default_value(min_exp, 'minimum exponent for scientific notation')
    max_dec = set_default_value(max_dec, 'maximum number of decimals')
    extra_sf_lim = set_default_value(extra_sf_lim, 'limit for extra significant figure')
    use_exp = True
    if ((float(x) > float(dx)
          and all(abs(np.floor(_log10(abs(np.array([x, dx]))))) < min_exp))
          or (float(x) <= float(dx)
              and float('{:e}'.format(float(dx))
                        .split('e')[0]) > extra_sf_lim
              and abs(np.floor(_log10(abs(float(dx))))) < min_exp)
          or (float(dx) == 0 and abs(np.floor(_log10(abs(x)))) < min_exp)
          or np.isinf(min_exp)):
        use_exp = False
    x, dx = float(x), float(dx)
    sign = '' if x >= 0 else '-'
    if x < 0:
        x = abs(x)
    if np.isinf(x):
        y = 'inf'
        dy = '0'
    elif np.isnan(x):
        y = 'nan'
        dy = 'nan'
    elif dx > 0:
        dy = round_sf(dx, n, min_exp, extra_sf_lim)
        if not use_exp:
            m = len(dy.split('.')[1]) if '.' in dy else 0
            y = '{:.{}f}'.format(x, m) if x != 0. else '0'
            if m == 0:
                num_digits_y = len(y)
                num_digits_dy = len(dy)
                m = n + int(num_digits_y - num_digits_dy)
                num_digits_x = len(str(x).split('.')[0])
                if num_digits_y > num_digits_x:
                    m -= 1
                base_dy = '{:e}'.format(float(dy)).split('e')[0]
                if float(base_dy) <= extra_sf_lim:
                    m += 1
                y = round_sf(x, m, min_exp, extra_sf_lim=1-1e-8)
        else:
            base_y, exp_y = '{:e}'.format(x).split('e')
            base_dy, exp_dy = '{:e}'.format(dx).split('e')
            exp_y, exp_dy = int(exp_y), int(exp_dy)
            d = exp_dy - exp_y
            if x != 0. and d < 0:
                o = 1 if float(base_dy) <= extra_sf_lim else 0
                m = max(n+d+o, n)
                min_exp_ = np.inf
                base_dy = round_sf(float(base_dy)*10**d, m,
                                   min_exp_, extra_sf_lim)
                m = len(base_dy.split('.')[1]) if '.' in base_dy else 0
                base_y = '{:.{}f}'.format(float(base_y), m)
                base_dy = '{:.{}f}'.format(float(base_dy), m)
                y = '{}e{}'.format(base_y, exp_y)
                dy = '{}e{}'.format(base_dy, exp_y)
            elif d == 0:
                if 'e' in dy:
                    base_dy, exp_dy = dy.split('e')
                else:
                    base_dy, exp_dy = dy, '0'
                m = len(base_dy.split('.')[1]) if '.' in dy else 0
                base_y = ('{:.{}f}'.format(float(base_y)*10**(-d), m) if x != 0
                          else '0')
                y = '{}e{}'.format(base_y, exp_dy)
            else:
                f = 10**(-int(exp_y))
                base_y, dy = round_sf_unc(x*f, dx*f, n, np.inf, max_dec, extra_sf_lim)
                y = '{}e{}'.format(base_y, exp_y)
    elif dx == 0:
        y = round_sf(x, n, min_exp, extra_sf_lim)
        dy = '0e0'
    else:
        y = 'nan'
        dy = 'nan'
    if float(y) != 0 and y != 'nan':
        y = sign + y
    if not use_exp:
        y = y.replace('e0','')
        dy = dy.replace('e0','')
    else:
        if not np.isinf(x):
            if not 'e' in y:
                y = '{:e}'.format(float(y))
            exp = int(y.split('e')[1]) if 'e' in y else 0
            if abs(exp) < min_exp:
                x = float(sign + str(x))
                min_exp = np.inf
                y, dy = round_sf_unc(x, dx, n, min_exp, max_dec, extra_sf_lim)
        y = y.replace('e+', 'e').replace('e00', 'e0')
        dy = dy.replace('e+','e').replace('e00', 'e0')
    d = len(y.split('e')[0].split('.')[-1])
    if d > max_dec and ')' not in dy:
        if 'e' in y:
            y, exp = y.split('e')
            dy, _ = dy.split('e')
        else:
            exp = None
        d = len(y.split('.')[-1])
        dy_ = round_sf(float(dy)*10**d, n, min_exp=np.inf, extra_sf_lim=0.)
        dy = '(' + dy_.split('e')[0] + ')'
        if exp is not None:
            y = '{}e{}'.format(y, exp)
    return y, dy

def round_sf_uncs(x, dx, n=None, min_exp=None, max_dec=None, extra_sf_lim=None):
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
        Minimum decimal exponent, in absolute value, to apply scientific
        notation. The default is 4.
    max_dec : int, optional
        Maximum number of decimals, to apply notation with parenthesis.
        The default is 5.
    extra_sf_lim : float, optional
        If the number expressed in scientific notation has a base that is lower
        than this value, an additional significant figure will be used.
        The default is 2.5.

    Returns
    -------
    y : float
        Rounded value.
    dy : list (float)
        Rounded uncertainties.
    """
    n = set_default_value(n, 'number of significant figures')
    min_exp = set_default_value(min_exp, 'minimum exponent for scientific notation')
    max_dec = set_default_value(max_dec, 'maximum number of decimals')
    extra_sf_lim = set_default_value(extra_sf_lim, 'limit for extra significant figure')
    dx1, dx2 = dx
    y1, dy1 = round_sf_unc(x, dx1, n, min_exp, max_dec, extra_sf_lim)
    y2, dy2 = round_sf_unc(x, dx2, n, min_exp, max_dec, extra_sf_lim)
    num_dec_1 = len(y1.split('e')[0].split('.')[1]) if '.' in y1 else 0
    num_dec_2 = len(y2.split('e')[0].split('.')[1]) if '.' in y2 else 0
    if num_dec_2 > num_dec_1:
        diff = num_dec_2 - num_dec_1
        y1, dy1 = round_sf_unc(x, dx1, n+diff, min_exp, max_dec, extra_sf_lim)
        y2, dy2 = round_sf_unc(x, dx2, n, min_exp, max_dec, extra_sf_lim)
    else:
        diff = num_dec_1 - num_dec_2
        off1, off2 = 0, 0
        if num_dec_1 == 0 == num_dec_2:
            base_dy1 = '{:e}'.format(dx1).split('e')[0]
            base_dy2 = '{:e}'.format(dx2).split('e')[0]
            b1 = float(base_dy1) if np.isfinite(dx1) else 10.
            b2 = float(base_dy2) if np.isfinite(dx2) else 10.
            if dx2 > dx1 and b1 <= extra_sf_lim and b2 > extra_sf_lim:
                off2 = 1
            if dx1 > dx2 and b2 <= extra_sf_lim and b1 > extra_sf_lim:
                off1 = 1
        y1, dy1 = round_sf_unc(x, dx1, n+off1, min_exp, max_dec, extra_sf_lim)
        y2, dy2 = round_sf_unc(x, dx2, n+diff+off2, min_exp, max_dec, extra_sf_lim)
    y = y1 if dx2 > dx1 else y2
    if dy1 != dy2 and (')' in dy1 or ')' in dy2):
        dy1 = dy1.replace('(', '(-')
        dy2 = dy2.replace('(', '(+')
    if ')' in dy1 and ')' not in dy2 or ')' in dy2 and '.' in dy2:
        _, dy2 = round_sf_unc(x, dx[1], n, min_exp, max_dec-1, extra_sf_lim)
        dy2 = '(' + dy2[1:-1] + '0' + ')'
    elif ')' in dy2 and ')' not in dy1 or ')' in dy1 and '.' in dy1:
        _, dy1 = round_sf_unc(x, dx[0], n, min_exp, max_dec-1, extra_sf_lim)
        dy1 = '(' + dy1[1:-1] + '0' + ')'
    if ')' in dy1 or ')' in dy2:
        if not dy1.startswith('(-'):
            dy1 = '(-' + dy1[1:]
        if not dy2.startswith('(+'):
            dy2 = '(+' + dy2[1:]
    dy = [dy1, dy2]
    return y, dy

class RichValue():
    """
    A class to store a value with uncertainties or with upper/lower limits.
    """
    
    def __init__(self, main=None, unc=0., is_lolim=False, is_uplim=False,
                 is_range=False, domain=None, is_int=None, **kwargs):
        """
        Parameters
        ----------
        main : float
            Central value of the rich value, or value of the upper/lower limit.
        unc : float / list (float), optional
            Lower and upper uncertainties associated with the central value.
            The default is [0,0].
        is_lolim : bool, optional
            If True, it means that the main value is actually a lower limit.
            The default is False.
        is_uplim : bool, optional
            If True, it means that the main value is actually an upper limit.
            The default is False.
        is_range : bool, optional
            If True, it means that the rich value is actually a constant range
            of values defined by the main value and the uncertainties.
        domain : list (float), optional
            The domain of the rich value, that is, the minimum and maximum
            values that it can take.
        is_int : bool, optional
            If True, the variable corresponding to the rich value will be an
            integer, so when creating samples it will have integer values.
        """
        
        input_domain = copy.copy(domain)
        domain = set_default_value(domain, 'domain')
        is_int = set_default_value(is_int, 'assume integers')
        acronyms = ['main_value', 'uncertainty', 'is_integer',
                    'is_lower_limit', 'is_upper_limit', 'is_finite_range']
        for kwarg in kwargs:
            if kwarg not in acronyms:
                raise TypeError("RichValue() got an unexpected keyword argument"
                                + " '{}'".format(kwarg))
        if 'main_value' in kwargs:
            main = kwargs['main_value']
        if main is None:
            raise TypeError("RichValue() missing required argument 'main'"
                            + " (pos 1)")
        if 'uncertainty' in kwargs:
            unc = kwargs['uncertainty']
        if 'is_lower_limit' in kwargs:
            is_lolim = kwargs['is_lower_limit']
        if 'is_upper_limit' in kwargs:
            is_uplim = kwargs['is_upper_limit']
        if 'is_finite_range' in kwargs:
            is_range = kwargs['is_finite_range']
        if 'is_integer' in kwargs:
            is_int = kwargs['is_integer']
        
        input_main = copy.copy(main)
        if not domain[0] <= domain[1]:
            raise Exception('Invalid domain {}.'.format(domain))
        
        if type(main) in [list, tuple]:
            is_lolim, is_uplim, is_range = False, False, False
            is_int = all(['int' in str(type(xi)) for xi in main])
            main = [float(main[0]), float(main[1])]
            if input_main[0] <= domain[0] and input_main[1] < domain[1]:
                is_uplim = True
                main = main[1]
            elif input_main[1] >= domain[1] and input_main[0] > domain[0]:
                is_lolim = True
                main = main[0]
            else:
                is_range = True
            if is_lolim and is_uplim:
                is_range = True
                main = domain
            if main == domain:
                main = np.nan
                is_range = False
        if type(main) is not list:
            if is_uplim and np.isfinite(domain[0]) and domain[0] != 0:
                main = [domain[0], main]
                is_range = True
                is_uplim = False
            elif is_lolim and np.isfinite(domain[1]) and domain[1] != 0:
                main = [main, domain[1]]
                is_range = True
                is_lolim = False
        if is_range and type(main) is list:
            unc = [(main[1] - main[0]) / 2] * 2
            main = (main[0] + main[1]) / 2
            
        if not np.isnan(main) and not domain[0] <= main <= domain[1]:
            raise Exception('Invalid main value {} for domain {}.'
                            .format(main, domain))
        
        if type(unc) is np.ndarray:
            unc = unc.tolist()
        if not hasattr(unc, '__iter__'):
            unc = [unc]*2
        if any(np.isinf(unc)):
            main = np.nan
        if any(np.isnan(unc)):
            unc = [0.]*2
        if is_lolim or is_uplim or not np.isfinite(main):
            unc = [np.nan]*2
        unc = list(unc)
        unc = [float(unc[0]), float(unc[1])]
        unc = [abs(unc[0]), abs(unc[1])]
        if input_domain is None and unc[0] == 0. == unc[1]:
            domain = [main]*2
        
        is_int = 'int' in str(type(main)) and unc[0] == 0. == unc[1]
            
        with np.errstate(divide='ignore', invalid='ignore'):
            ampl = [main - domain[0], domain[1] - main]
            rel_ampl = list(np.array(ampl) / np.array(unc))
        if not (is_lolim or is_uplim or is_range):
            is_range_domain = False
            if min(rel_ampl) <= 1.:
                sigmas = defaultparams['sigmas to define upper/lower limits'
                                       ' from read values']
                x1 = max(main - unc[0], domain[0])
                x2 = min(main + unc[1], domain[1])
                if x1 == domain[0] and x2 != domain[1]:
                    main += sigmas * unc[1]
                    if main < domain[1]:
                        is_uplim = True
                        is_range = False
                    else:
                        is_range = True
                        is_range_domain = True
                elif x2 == domain[1] and x1 != domain[0]:
                    main -= sigmas * unc[0]
                    if main > domain[0]:
                        is_lolim = True
                    else:
                        is_range = True
                        is_range_domain = True
                else:
                    is_range = True
            if min(rel_ampl) <= 1. and is_range_domain:
                main = (domain[0] + domain[1]) / 2
                unc = [(domain[1] - domain[0]) / 2] * 2
        
        if (is_lolim or is_uplim) and np.isinf(np.diff(domain)):
            is_range = False
        
        if is_int and is_range:
            x1, x2 = main - unc[0], main + unc[1]
            x1 = np.ceil(x1)
            x2 = np.floor(x2)
            if x1 == x2:
                main = x1
                unc = [0, 0]
                is_range = False
            else:
                main = (x1 + x2) / 2
                unc = [(x2 - x1) / 2] * 2
        
        global variable_count
        if domain[0] != domain[1]:
            variable_count += 1
            variable = 'x{}'.format(variable_count)
            variables = [variable]
            expression = variable
        else:
            variables = []
            expression = str(main)
                
        self.main = main
        self._unc = unc
        self.is_lolim = is_lolim
        self.is_uplim = is_uplim
        self.is_range = is_range
        self.is_int = is_int
        self.domain = domain
        self.num_sf = defaultparams['number of significant figures']
        self.min_exp = defaultparams['minimum exponent for scientific notation']
        self.max_dec = defaultparams['maximum number of decimals']
        self.extra_sf_lim = defaultparams['limit for extra significant figure']
        self.pdf_info = 'default'
        self.variables = variables
        self.expression = expression
        
        global variable_dict
        variable_dict[expression] = self
 
    @property
    def unc(self): return self._unc
    @unc.setter
    def unc(self, x):
        if not hasattr(x, '__iter__'):
            x = [x, x]
        x = list(x)
        self._unc = x
         
    @property
    def is_lim(self):
        """Upper/lower limit."""
        islim = self.is_lolim or self.is_uplim
        return islim
    @property
    def is_interv(self):
        """Upper/lower limit or constant range of values."""
        isinterv = self.is_range or self.is_lim
        return isinterv
    @property
    def is_centr(self):
        """Centered value."""
        iscentr = not self.is_interv
        return iscentr
    @property
    def is_exact(self):
        """Exact value."""
        isexact = self.unc[0] == 0 == self.unc[1]
        return isexact
    @property
    def is_const(self):
        """Constant value."""
        isconst = self.is_exact and self.domain[0] == self.domain[1]
        return isconst
    @property    
    def center(self):
        """Central value."""
        cent = self.main if self.is_centr else np.nan
        return cent  
    @property
    def unc_eb(self):
        """Uncertainties with shape (2,1)."""
        unceb = [[self.unc[0]], [self.unc[1]]]
        return unceb
    @property
    def rel_unc(self):
        """Relative uncertainties."""
        if self.is_centr:
            m, s = self.main, self.unc
            with np.errstate(divide='ignore', invalid='ignore'):
                runc = list(np.array(s) / abs(m))
        else:
            runc = [np.nan]*2
        return runc
    @property
    def signal_noise(self):
        """Signal-to-noise ratios (S/N)."""
        if self.is_centr:
            m, s = self.main, self.unc
            with np.errstate(divide='ignore', invalid='ignore'):
                s_n = list(np.nan_to_num(abs(m) / np.array(s),
                           nan=0, posinf=np.inf))
        else:
            s_n = [np.nan]*2
        return s_n
    @property    
    def ampl(self):
        """Amplitudes."""
        m, b = self.main, self.domain
        a = [m - b[0], b[1] - m]
        return a
    @property
    def rel_ampl(self):
        """Relative amplitudes."""
        if self.is_centr:
            s, a = self.unc, self.ampl
            with np.errstate(divide='ignore', invalid='ignore'):
                a_s = list(np.array(a) / np.array(s))
        else:
            a_s = [np.nan]*2
        return a_s
    @property
    def norm_unc(self):
        """Normalized uncertainties."""
        if self.is_centr:
            s, a = self.unc, self.ampl
            s_a = list(np.array(s) / np.array(a))
        else:
            s_a = [np.nan]*2
        return s_a
    @property
    def prop_score(self):
        """Minimum of the signals-to-noise and the relative amplitudes."""
        if self.is_exact:
            ps = np.inf
        elif self.is_centr:
            s_n = self.signal_noise
            a_s = self.rel_ampl
            ps = np.min([s_n, a_s])
        else:
            ps = 0.
        return ps
    @property
    def is_nan(self):
        """Not a number value."""
        isnan = np.isnan(np.diff(self.interval(sigmas=3.)))[0]
        return isnan
    @property
    def is_inf(self):
        """Infinite value."""
        isinf = np.isinf(np.diff(self.interval(sigmas=3.)))[0]
        return isinf
    @property
    def is_finite(self):
        """Finite value."""
        isfinite = np.isfinite(np.diff(self.interval(sigmas=3.)))[0]
        return isfinite
    @property
    def real(self): return self
    @property
    def imag(self): return 0
    
    def interval(self, sigmas=None):
        """Interval of possible values of the rich value."""
        sigmas = set_default_value(sigmas, 'sigmas for intervals')
        if not self.is_interv:
            if np.isfinite(self.main):
                ampl1 = sigmas * self.unc[0] if self.unc[0] != 0 else 0
                ampl2 = sigmas * self.unc[1] if self.unc[1] != 0 else 0
                interv = [max(self.domain[0], self.main - ampl1),
                          min(self.domain[1], self.main + ampl2)]
            elif np.isinf(self.main):
                interv = [self.main, self.main]
            else:
                interv = [np.nan, np.nan]
        else:
            if self.is_uplim and not self.is_lolim:
                interv = [self.domain[0], self.main]
            elif self.is_lolim and not self.is_uplim:
                interv = [self.main, self.domain[1]]
            else:
                interv = [self.main - self.unc[0], self.main + self.unc[1]]
        if self.is_int and self.is_interv:
            x1, x2 = interv
            x1, x2 = np.ceil(x1), np.floor(x2)
            x1 = int(x1) if np.isfinite(x1) else x1
            x2 = int(x2) if np.isfinite(x2) else x2
            interv = [x1, x2]
        return interv
    
    def sign(self, sigmas=np.inf):
        """Sign of the rich value."""
        interv = self.interval(sigmas)
        signs_interv = np.sign(interv)
        if all(signs_interv == 0):
            s = 0
        elif all(signs_interv >= 0):
            s = 1
        elif all(signs_interv <= 0):
            s = -1
        else:
            s = np.nan
        return s
  
    def set_lim_unc(self, factor=4.):
        """Set uncertainties of limits with respect to cetral values."""
        if self.is_lim:
            self.unc = [self.main / factor, self.main / factor]
        
    def _format_as_rich_value(self):
        main = self.main
        unc = self.unc
        is_lolim = self.is_lolim
        is_uplim = self.is_uplim
        is_range = self.is_range
        domain = copy.copy(self.domain)
        is_int = self.is_int
        min_exp = abs(self.min_exp)
        max_dec = abs(self.max_dec)
        extra_sf_lim = self.extra_sf_lim
        show_domain = defaultparams['show domain']
        show_asterisk = defaultparams['show asterisk for rich values with custom PDF']
        use_extra_sf_in_exacts = \
            defaultparams['use extra significant figure for exact values']
        use_extra_sf_in_ranges = \
            defaultparams['use extra significant figure for finite intervals']
        x = main
        dx = unc
        n = self.num_sf
        if self.is_exact and np.diff(domain) > 0.:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = np.array(domain) / main
                if all(ratios < 7.):
                    n = max(n, 2)
        range_bound = (self.range_bound if hasattr(self, 'range_bound')
                       else False)
        use_exp = True
        if ((self.is_centr and 'e' not in
             str(round_sf_uncs(x, dx, n, min_exp, max_dec, extra_sf_lim)))
              or self.is_lim and abs(np.floor(_log10(abs(float(x))) < min_exp))
              or np.isinf(min_exp)):
            use_exp = False
        if not is_range and not np.isnan(main):
            dx1, dx2 = dx
            if not self.is_lim:
                if dx1 == dx2 == 0:
                    if (not range_bound and use_extra_sf_in_exacts
                        or range_bound and use_extra_sf_in_ranges):
                        n += 1
                y, (dy1, dy2) = round_sf_uncs(x, [dx1, dx2], n, min_exp,
                                              max_dec, extra_sf_lim)
                if 'e' in y:
                    y, a = y.split('e')
                    a = int(a)
                else:
                    a = 0
                if 'e' in dy1:
                    dy1, _ = dy1.split('e')
                if 'e' in dy2:
                    dy2, _ = dy2.split('e')
                if ')' not in dy1:
                    if dy1 == dy2:
                        if float(dy1) != 0:
                            text = '{}+/-{} e{}'.format(y, dy1, a)
                        else:
                            y = int(round(float(y))) if is_int else y
                            text = '{} e{}'.format(y, a)
                    else:
                        text = '{}-{}+{} e{}'.format(y, dy1, dy2, a)
                else:
                    dy1 = dy1.replace('(-', '(')
                    dy2 = dy2.replace('(+', '(')
                    if dy1 == dy2:
                        text = '{}{} e{}'.format(y, dy1, a)
                    else:
                        text = '{}{}{} e{}'.format(y, dy1, dy2, a)
                if not use_exp:
                    text = text.replace(' e0','')
            else:
                if is_int:
                    n += 1
                y = round_sf(x, n, min_exp, extra_sf_lim)
                if 'e' in y:
                    y, a = y.split('e')
                    a = int(a)
                else:
                    a = 0
                if is_lolim:
                    sign = '>'
                    y = int(np.floor(float(y))) if is_int else y
                elif is_uplim:
                    sign = '<'
                    y = int(np.ceil(float(y))) if is_int else y
                text = '{} {} e{}'.format(sign, y, a)
            if show_asterisk and self.pdf_info != 'default':
                text = '*' + text
            if use_exp:
                text = text.replace('e-0', 'e-').replace(' *','')
                a = int(text.split('e')[1])
                if abs(a) < min_exp:
                    z = RichValue(x, dx, is_lolim, is_uplim,
                                  is_range, domain, is_int)
                    z.num_sf = n
                    z.min_exp = np.inf
                    z.extra_sf_lim = extra_sf_lim
                    text = str(z)
            else:
                text = text.replace(' e0','')
            if show_domain and domain[0] != domain[1]:
                d1, d2 = round_sf(domain[0], n=2), round_sf(domain[1], n=2)
                text += ' [{},{}]'.format(d1, d2)
        elif not is_range and np.isnan(main):
            text = 'nan'
        else:
            x1, x2 = main - unc[0], main + unc[1]
            if is_int:
                x1 = np.ceil(x1)
                x2 = np.floor(x2)
            x1 = RichValue(x1, domain=domain, is_int=is_int)
            x2 = RichValue(x2, domain=domain, is_int=is_int)
            x1.min_exp = min_exp
            x2.min_exp = min_exp
            x1.range_bound = True
            x2.range_bound = True
            while str(x1) == str(x2):
                x1.num_sf = x1.num_sf + 1
                x2.num_sf = x2.num_sf + 1
            text = '{} -- {}'.format(x1, x2)
        if show_domain and domain[0] != domain[1]:
            domain = ' [' + text.split('[')[1].split(']')[0] + ']'
            text = text.replace(domain, '') + domain
        return text
        
    def __repr__(self):
        return self._format_as_rich_value()
    
    def __str__(self):
        return self._format_as_rich_value()
   
    def latex(self, show_dollars=True, mult_symbol=None,
              use_extra_sf_in_exacts=None, omit_ones_in_sci_notation=None):
        """Display rich value in LaTeX format."""
        mult_symbol = set_default_value(mult_symbol,
                      'multiplication symbol for scientific notation in LaTeX')
        use_extra_sf_in_exacts = set_default_value(use_extra_sf_in_exacts,
                                'use extra significant figure for exact values')
        omit_ones_in_sci_notation = set_default_value(omit_ones_in_sci_notation,
                                    'omit ones in scientific notation in LaTeX')
        kwargs = (show_dollars, mult_symbol, use_extra_sf_in_exacts,
                  omit_ones_in_sci_notation)
        main = self.main
        unc = self.unc
        domain = self.domain
        is_lolim = self.is_lolim
        is_uplim = self.is_uplim
        is_range = self.is_range
        is_int = self.is_int
        min_exp = abs(self.min_exp)
        max_dec = abs(self.max_dec)
        extra_sf_lim = self.extra_sf_lim
        show_domain = defaultparams['show domain']
        use_exp = True
        x = main
        dx = unc
        n = self.num_sf
        if ((float(x) > float(max(dx))
             and abs(np.floor(_log10(abs(float(x))))) < min_exp)
             or (float(x) <= float(max(dx))
                 and float('{:e}'.format(max(dx))
                           .split('e')[0]) > extra_sf_lim
                 and any(abs(np.floor(_log10(abs(np.array(dx))))) < min_exp))
             or self.is_exact and abs(np.floor(_log10(abs(x)))) < min_exp
             or self.is_lim and abs(np.floor(_log10(abs(float(x))))) < min_exp
             or np.isinf(min_exp) or main == 0 and unc[0] == 0 == unc[1]):
            use_exp = False
        text = ''
        non_numerics = ['nan', 'NaN', 'None', 'inf', '-inf']
        is_numeric = False if str(main) in non_numerics else True
        range_bound = (self.range_bound if hasattr(self, 'range_bound')
                       else False)
        if is_numeric:
            if not is_range:
                _, unc_r = round_sf_uncs(x, dx, n, min_exp, np.inf, extra_sf_lim)
                unc_r = np.array(unc_r, float)
            if not is_range and not use_exp:
                if not self.is_lim:
                    if unc_r[0] == unc_r[1]:
                        if unc_r[0] == 0:
                            if not range_bound and use_extra_sf_in_exacts:
                                n += 1
                            y = (int(round(x)) if is_int else
                                 round_sf(x, n, np.inf, extra_sf_lim))
                            text = '${}$'.format(y)
                        else:
                            y, dy = round_sf_unc(x, dx[0], n, min_exp,
                                                 max_dec, extra_sf_lim)
                            text = '${} \pm {}$'.format(y, dy)
                    else:
                        y, dy = round_sf_uncs(x, dx, n, min_exp, max_dec, extra_sf_lim)
                        text = '$'+y + '_{-'+dy[0]+'}^{+'+dy[1]+'}$'
                else:
                    if is_lolim:
                        symbol = '>'
                        y = int(np.floor(x)) if is_int else x               
                    elif is_uplim:
                        symbol = '<'
                        y = int(np.ceil(x)) if is_int else x
                    y = round_sf(y, n, min_exp, extra_sf_lim)
                    text = '${} {}$'.format(symbol, y)
            elif not is_range and use_exp:
                if not self.is_lim:
                    if unc_r[0] == unc_r[1]:
                        if unc_r[0] == 0:
                            if not range_bound and use_extra_sf_in_exacts:
                                n += 1
                            min_exp = 0
                            y = str(round(x)) if is_int else round_sf(x, n,
                                                        min_exp, extra_sf_lim)
                            if 'e' in y:
                                y, a = y.split('e')
                            else:
                                a = '0'
                            a = str(int(a))
                            text = ('${} {}'.format(y, mult_symbol)
                                    + ' 10^{'+a+'}$')
                        else:
                            y, dy = round_sf_unc(x, dx[0], n, min_exp,
                                                 max_dec, extra_sf_lim)
                            if 'e' in y:
                                y, a = y.split('e')
                                dy, a = dy.split('e')
                            else:
                                a = 0
                            a = str(int(a))
                            text = ('$({} \pm {}) '.format(y, dy)
                                     + mult_symbol + ' 10^{'+a+'}$')
                    else:
                        y, dy = round_sf_uncs(x, [dx[0], dx[1]], n, min_exp,
                                              max_dec, extra_sf_lim)
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
                        y = int(np.floor(x)) if is_int else x  
                    elif is_uplim:
                        symbol = '<'
                        y = int(np.ceil(x)) if is_int else x  
                    y = round_sf(y, n, min_exp, extra_sf_lim)
                    if 'e' in y:
                        y, a = y.split('e')
                    else:
                        a = '0'
                    text = ('${} {} {}'.format(symbol, y, mult_symbol)
                            + ' 10^{'+a+'}$')
                a = int(text.split('10^{')[1].split('}')[0])
                if abs(a) < min_exp:
                    y = RichValue(x, dx, is_lolim, is_uplim,
                                  is_range, domain, is_int)
                    y.num_sf = n
                    y.min_exp = np.inf
                    y.extra_sf_lim = extra_sf_lim
                    text = y.latex(*kwargs)
                if (not use_extra_sf_in_exacts and omit_ones_in_sci_notation
                     and dx[0] == dx[1] and dx[0] == 0 or np.isnan(dx[0])):
                    if '.' not in text:
                        text = text.replace('1 {} '.format(mult_symbol), '')
            else:
                x1 = RichValue(main - unc[0], domain=domain)
                x2 = RichValue(main + unc[1], domain=domain)
                x1.min_exp = min_exp
                x2.min_exp = min_exp
                x1.range_bound = True
                x2.range_bound = True
                while str(x1) == str(x2):
                    x1.num_sf = x1.num_sf + 1
                    x2.num_sf = x2.num_sf + 1
                text = '{} -- {}'.format(x1.latex(*kwargs), x2.latex(*kwargs))
        else:
            text = (str(main).replace('NaN','nan').replace('nan','...')
                    .replace('inf','$\infty$'))
        if show_domain and domain[0] != domain[1]:
            d1, d2 = domain[0], domain[1]
            if np.isinf(d1):
                d1 = '\infty'
            elif round(d1) != d1:
                d1 = round_sf(d1, n=2)
            if np.isinf(d2):
                d2 = '\infty'
            elif round(d2) != d2:
                d2 = round_sf(d2, n=2)
            domain = ' $[{},\\,{}]$'.format(d1, d2)
            text = text.replace(domain,'') + domain
        if not show_dollars:
            text = text.replace('$','')
        return text
   
    def __neg__(self):
        if not self.is_interv:
            x = -self.main
        else:
            x1, x2 = self.interval()
            x = [-x2, -x1]
        dx = [self.unc[1], self.unc[0]]
        domain = [-self.domain[0], -self.domain[1]]
        domain = [min(domain), max(domain)]
        rvalue = RichValue(x, dx, domain=domain, is_int=self.is_int)
        rvalue.num_sf = self.num_sf
        rvalue.min_exp = self.min_exp
        rvalue.extra_sf_lim = self.extra_sf_lim
        rvalue.variables = self.variables
        expression = self.expression
        rvalue.expression = ('-{}'.format(expression) if
               self.expression.count('x') == 1 else '-({})'.format(expression))
        if self.pdf_info != 'default':
            rvalue.pdf_info = edit_pdf_info(self.pdf_info, lambda x: -x)
        return rvalue
    
    def __abs__(self):
        sigmas = defaultparams['sigmas to use approximate '
                               + 'uncertainty propagation']
        domain = copy.copy(self.domain)
        is_int = copy.copy(self.is_int)
        x1, x2 = self.interval(sigmas=np.inf)
        if all(np.array([x1, x2]) >= 0):
            if (self.is_centr and self.prop_score > sigmas) or self.is_interv:
                rvalue = self
            else:
                rvalue = self.function(lambda x: abs(x), domain=domain)
        elif all(np.array([x1, x2]) <= 0):
            rvalue = -self
        else:
            domain[0] = 0
            if self.is_centr:
                if self.prop_score > sigmas:
                    x = abs(self.main)
                    dx = self.unc
                    rvalue = RichValue(x, dx, domain=domain, is_int=is_int)
                else:
                    rvalue = self.function(lambda x: abs(x), domain=domain)
            else:
                x2 = max(np.abs([x1, x2]))
                x1 = 0
                rvalue = RichValue([x1, x2], domain=domain, is_int=is_int)
        rvalue.domain[0] = max(0, rvalue.domain[0])
        rvalue.num_sf = self.num_sf
        rvalue.min_exp = self.min_exp
        rvalue.extra_sf_lim = self.extra_sf_lim
        rvalue.variables = self.variables
        rvalue.expression = 'abs({})'.format(self.expression)
        if self.pdf_info != 'default':
            rvalue.pdf_info = edit_pdf_info(self.pdf_info, lambda x: abs(x))
        return rvalue
    
    def __add__(self, other):
        if type(other) in (np.ndarray, RichArray, ComplexRichValue):
            return other + self
        elif 'complex' in str(type(other)):
            return ComplexRichValue(self, 0) + other
        elif type(other) is RichValue:
            other_vars = other.variables
            other_expression = other.expression
        else:
            other_vars = []
            other_expression = str(other)
        is_other_numeric = type(other) is not RichValue
        expression = ('{}+{}'.format(self.expression, other_expression)
                      .replace('+-','-'))
        variables = list(dict.fromkeys(self.variables + other_vars).keys())
        common_vars = set(self.variables) & set(other_vars)
        if len(common_vars) == 0:
            if is_other_numeric:
                x = self.main + other
                dx = self.unc
                domain = [self.domain[0] + other, self.domain[1] + other]
                is_int = self.is_int and (round(other) == other)
                rvalue = RichValue(x, dx, self.is_lolim, self.is_uplim,
                                   self.is_range, domain, is_int)
            else:
                rvalue = add_rich_values(self, other)
        else:
            vars_str = ','.join(variables)
            function = eval('lambda {}: {}'.format(vars_str, expression))
            args = [variable_dict[var] for var in variables]
            rvalue = function_with_rich_values(function, args)
        rvalue.variables = list(variables)
        rvalue.expression = expression
        if self.pdf_info != 'default':
            is_other_rich_value = type(other) is RichValue
            is_other_numeric |= is_other_rich_value and other.is_exact
            if is_other_numeric:
                c = other.main if is_other_rich_value else other
                rvalue.pdf_info = edit_pdf_info(self.pdf_info, lambda x: x+c)
        return rvalue
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return -(self - other)
    
    def __mul__(self, other):
        if type(other) in (np.ndarray, RichArray, ComplexRichValue):
            return other * self
        elif 'complex' in str(type(other)):
            return ComplexRichValue(self, 0) * other
        elif type(other) is RichValue:
            other_vars = other.variables
            other_expression = other.expression
        else:
            other_vars = []
            other_expression = str(other)
        is_other_numeric = type(other) is not RichValue
        expression = '({})*({})'.format(self.expression, other_expression)
        variables = list(dict.fromkeys(self.variables + other_vars).keys())
        common_vars = set(self.variables) & set(other_vars)
        if len(common_vars) == 0:
            if is_other_numeric:
                x = self.main * other
                dx = [self.unc[0] * other, self.unc[1] * other]
                is_int = self.is_int and round(other) == other
                domain = propagate_domain(self.domain, [other]*2,
                                          lambda a,b: a*b)
                rvalue = RichValue(x, dx, self.is_lolim, self.is_uplim,
                                   self.is_range, domain, is_int)
            else:
                rvalue = multiply_rich_values(self, other)
        else:
            vars_str = ','.join(variables)
            function = eval('lambda {}: {}'.format(vars_str, expression))
            args = [variable_dict[var] for var in variables]
            rvalue = function_with_rich_values(function, args)
        rvalue.variables = list(variables)
        rvalue.expression = expression
        if self.pdf_info != 'default':
            is_other_rich_value = type(other) is RichValue
            is_other_numeric |= is_other_rich_value and other.is_exact
            if is_other_numeric:
                c = other.main if is_other_rich_value else other
                rvalue.pdf_info = edit_pdf_info(self.pdf_info, lambda x: x*c)
        return rvalue
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        if type(other) is ComplexRichValue or 'complex' in str(type(other)):
            return ComplexRichValue(self, 0) / other
        elif type(other) in (np.ndarray, RichArray):
            return rich_array(self) / other
        elif type(other) is RichValue:
            other_vars = other.variables
            other_expression = other.expression
        else:
            other_vars = []
            other_expression = str(other)
        is_other_numeric = type(other) is not RichValue
        expression = '({})/({})'.format(self.expression, other_expression)
        variables = list(dict.fromkeys(self.variables + other_vars).keys())
        common_vars = set(self.variables) & set(other_vars)
        if len(common_vars) == 0:
            if is_other_numeric:
                if other != 0:
                    x = self.main / other
                    dx = [self.unc[0] / other, self.unc[1] / other]
                    is_int = self.is_int and round(other) == other
                else:
                    x = np.nan
                    dx = np.nan
                domain = propagate_domain(self.domain, [other]*2,
                                          lambda a,b: a/b)
                rvalue = RichValue(x, dx, self.is_lolim, self.is_uplim,
                                   self.is_range, domain, is_int)
            else:
                rvalue = divide_rich_values(self, other)
        else:
            vars_str = ','.join(variables)
            function = eval('lambda {}: {}'.format(vars_str, expression))
            args = [variable_dict[var] for var in variables]
            rvalue = function_with_rich_values(function, args)
        rvalue.variables = list(variables)
        rvalue.expression = expression
        if self.pdf_info != 'default':
            is_other_rich_value = type(other) is RichValue
            is_other_numeric |= is_other_rich_value and other.is_exact
            if is_other_numeric:
                c = other.main if is_other_rich_value else other
                rvalue.pdf_info = edit_pdf_info(self.pdf_info, lambda x: x/c)
        return rvalue

    def __rtruediv__(self, other):
        type_other = str(type(other))
        if 'int' in type_other or 'float' in type_other:
            other_ = RichValue(other)
        elif 'complex' in type_other:
            other_ = ComplexRichValue(other.real, other.imag)
        else:
            other_ = other
        return other_ / self
    
    def __floordiv__(self, other):
        type_other = str(type(other))
        other_ = (RichValue(other, is_int=(type(other) is int))
                  if 'int' in type_other or 'float' in type_other else other)
        rvalue = function_with_rich_values('{}//{}', [self, other_])
        return rvalue

    def __rfloordiv__(self, other):
        type_other = str(type(other))
        other_ = (RichValue(other, is_int=(type(other) is int))
                  if 'int' in type_other or 'float' in type_other else other)
        other_ = RichValue(other) if type(other) is not RichValue else other
        return other_ // self

    def __mod__(self, other):
        type_other = str(type(other))
        other_ = (RichValue(other, is_int=(type(other) is int))
                  if 'int' in type_other or 'float' in type_other else other)
        domain = other_.interval(6.)
        if domain[0] > 0 and other_.sign() != -1:
            domain[0] = 0
        if domain[1] < 0 and other_.sign() != 1:
            domain[1] = 0    
        rvalue = function_with_rich_values('{}%{}', [self, other_],
                               domain=domain, is_domain_cyclic=True)
        return rvalue

    def __rmod__(self, other):
        other_ = RichValue(other) if type(other) is not RichValue else other
        return other_ // self

    def __pow__(self, other):
        if type(other) is ComplexRichValue or 'complex' in str(type(other)):
            return ComplexRichValue(self, 0) ** other
        elif type(other) in (np.ndarray, RichArray):
            return rich_array(self) ** other
        elif type(other) is RichValue:
            other_vars = other.variables
            other_expression = other.expression
        else:
            other_vars = []
            other_expression = str(other)
        expression = '({})**({})'.format(self.expression, other_expression)
        variables = list(dict.fromkeys(self.variables + other_vars).keys())
        common_vars = set(self.variables) & set(other_vars)
        if len(common_vars) == 0:
            sigmas = defaultparams['sigmas to use approximate '
                                   + 'uncertainty propagation']
            domain = copy.copy(self.domain)
            if ((domain[0] >= 0 and (type(other) is RichValue
                                     or self.prop_score < sigmas))
                or (domain[0] < 0 and type(other) is not RichValue)
                    and int(other) == other and self.prop_score < sigmas):
                other_ = (other if type(other) is RichValue
                          else RichValue(other))
                if self.main != 0:
                    if type(other) is not RichValue and other%2 == 0:
                        domain = [0, np.inf]
                    rvalue = function_with_rich_values(lambda a,b: a**b,
                                                [self, other_], domain=domain)
                else:
                    rvalue = RichValue(0, domain=domain, is_int=self.is_int)
            elif (type(other) is not RichValue and self.prop_score > sigmas):
                x = self.main ** other
                dx = (np.abs(x * other * np.array(self.unc) / self.main)
                      if (x*other) != 0. else 0.)
                is_int = self.is_int and type(other) is int
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
                rvalue = RichValue(x, dx, self.is_lolim, self.is_uplim,
                                     domain=domain, is_int=is_int)
            else:
                if (type(other) is RichValue and other.domain[0] < 0
                        and not np.isinf(other.main)):
                    print('Warning: Domain of exponent should be positive.')
                rvalue = RichValue(np.nan)
            rvalue.num_sf = self.num_sf
            rvalue.min_exp = self.min_exp
            rvalue.extra_sf_lim = self.extra_sf_lim
        else:
            vars_str = ','.join(variables)
            function = eval('lambda {}: {}'.format(vars_str, expression))
            args = [variable_dict[var] for var in variables]
            rvalue = function_with_rich_values(function, args)
        rvalue.variables = list(variables)
        rvalue.expression = expression
        return rvalue
    
    def __rpow__(self, other):
        type_other = str(type(other))
        if 'int' in type_other or 'float' in type_other:
            other_ = RichValue(other)
        elif 'complex' in type_other:
            other_ = ComplexRichValue(other.real, other.imag)
        else:
            other_ = other
        other_.num_sf = self.num_sf
        rvalue = other_ ** self
        return rvalue
    
    def __gt__(self, other):
        return greater(self, other)
    
    def __lt__(self, other):
        return less(self, other)

    def __eq__(self, other):
        return equiv(self, other)
    
    def __ge__(self, other):
        return greater_equiv(self, other)
    
    def __le__(self, other):
        return less_equiv(self, other)
    
    def pdf(self, x):
        """Probability density function corresponding to the rich value."""
        pdf_info = self.pdf_info
        if pdf_info == 'default':
            main = self.main
            unc = self.unc
            domain = copy.copy(self.domain)
            x = np.array(x)
            y = np.zeros(len(x))
            if self.is_exact:    
                ind = np.argmin(abs(x - main))
                if hasattr(ind, '__iter__'):
                    ind = ind[0]
                y[ind] = 1.
            else:
                if not self.is_interv:
                    y = general_pdf(x, main, unc, domain)
                elif self.is_lolim and not self.is_uplim:
                    y[x > main] = 1e-3
                elif self.is_uplim and not self.is_lolim:
                    y[x < main] = 1e-3
                elif self.is_range:
                    x1, x2 = main - unc[0], main + unc[1]
                    y[(x > x1) & (x < x2)] = 1 / (x2 - x1)
        else:
            x_, y_ = pdf_info['values'], pdf_info['probs']
            mask = np.argsort(x_)
            pdf_ = lambda x: np.interp(x, x_[mask], y_[mask],
                                       left=0., right=0.)
            y = pdf_(x)
        return y
    
    def sample(self, len_sample=1):
        """Sample of the distribution corresponding to the rich value"""
        domain = copy.copy(self.domain)
        is_int = self.is_int
        pdf_info = self.pdf_info
        N = int(len_sample)
        if pdf_info == 'default':
            main = self.main
            unc = list(self.unc)
            is_finite_interv = (self.is_range
                                or self.is_uplim and np.isfinite(domain[0])
                                or self.is_lolim and np.isfinite(domain[1]))
            if self.is_exact:
                distr = main * np.ones(N)
            elif self.is_nan:
                distr = np.nan * np.ones(N)
            else:
                if not is_finite_interv and not all(np.isinf(unc)):
                    if not self.is_lim:
                        distr = general_distribution(main, unc, domain, N)
                    else:
                        x1, x2 = self.interval()    
                        distr = loguniform_distribution(x1, x2, N)
                elif not is_finite_interv and all(np.isinf(unc)):
                    distr = loguniform_distribution(-np.inf, np.inf, N)
                else:
                    x1, x2 = self.interval()
                    N_min = 100
                    if N < N_min:
                        distr = sample_from_pdf(lambda x: np.ones(len(x)),
                                                N, x1, x2)
                    else:
                        zero_log = defaultparams['decimal exponent'
                                                 ' to define zero']
                        x1 += max(10**zero_log, abs(x1)*10**zero_log)
                        x2 -= max(10**zero_log, abs(x2)*10**zero_log)
                        distr = np.linspace(x1, x2, N)
                        np.random.shuffle(distr)
        else:
            x_, y_ = pdf_info['values'], pdf_info['probs']
            mask = np.argsort(x_)
            pdf = lambda x: np.interp(x, x_[mask], y_[mask], left=0., right=0.)
            distr = sample_from_pdf(pdf, N, low=domain[0], high=domain[1])
        if N == 1:
            distr = distr[0]
        if is_int:
            distr = np.round(distr).astype(int)
        return distr
    
    def function(self, function, **kwargs):
        """Apply a function to the rich value"""
        return function_with_rich_values(function, self, **kwargs)
    
    # Instance variable acronyms.
    
    @property
    def main_value(self): return self.main
    @main_value.setter
    def main_value(self, x): self.main = x
    
    @property
    def uncertainty(self): return self.unc
    @uncertainty.setter
    def uncertainty(self, x): self.unc = x
    
    @property
    def is_lower_limit(self): return self.is_lolim
    @is_lower_limit.setter
    def is_lower_limit(self, x): self.is_lolim = x
    
    @property
    def is_upper_limit(self): return self.is_uplim
    @is_upper_limit.setter
    def is_upper_limit(self, x): self.is_uplim = x
    
    @property
    def is_finite_range(self): return self.is_range
    @is_finite_range.setter
    def is_finite_range(self, x): self.is_range = x
    
    @property
    def is_integer(self): return self.is_int
    @is_integer.setter
    def is_integer(self, x): self.is_int = x
    
    @property
    def number_of_significant_figures(self): return self.num_sf
    @number_of_significant_figures.setter
    def number_of_significant_figures(self, x): self.num_sf = x
    
    @property
    def minimum_exponent_for_scientific_notation(self): return self.min_exp
    @minimum_exponent_for_scientific_notation.setter
    def minimum_exponent_for_scientific_notation(self, x): self.min_exp = x
    
    @property
    def maximum_number_of_decimals(self): return self.max_dec
    @maximum_number_of_decimals.setter
    def maximum_number_of_decimals(self, x): self.max_dec = x
    
    @property
    def limit_for_extra_significant_figure(self): return self.extra_sf_lim
    @limit_for_extra_significant_figure.setter
    def limit_for_extra_significant_figure(self, x): self.extra_sf = x
    
    # Attribute acronyms.
    is_limit = is_lim
    is_interval = is_interv
    is_centered = is_centr
    is_constant = is_const
    relative_uncertainty = rel_unc
    signal_to_noise = signal_noise
    amplitude = ampl
    relative_amplitude = rel_ampl
    normalized_uncertainty = norm_unc
    propagation_score = prop_score
    is_not_a_number = is_nan
    is_infinite = is_inf
    # Method acronyms.
    probability_density_function = pdf
    set_limit_uncertainty = set_lim_unc

class RichArray(np.ndarray):
    """
    A class to store values with uncertainties or upper/lower limits.
    """
    
    def __new__(cls, mains=None, uncs=None, are_lolims=None, are_uplims=None,
                are_ranges=None, domains=None, are_ints=None,
                variables=None, expressions=None, **kwargs):
        """
        Parameters
        ----------
        mains : list / array (float)
            Array of main values.
        uncs : list / array (float), optional
            Array of lower and upper uncertainties associated with the central
            values. The default is all 0.
        are_lolims : list / array (bool), optional
            Array of logical variables that indicate if each mian value is
            actually a lower limit. The default is all False.
        are_uplims : list / array (bool), optional
            Array of logical variables that indicate if each main value is
            actually an upper limit. The default is all False.
        are_ranges : list / array (bool), optional
            Array of logical variables that indicate if each rich value is
            actually a constant range of values defined by the main value and
            the uncertainties. The default is all False.
        domains : list / array (float), optional
            Array of domains for each entry of the rich value.
            The default is [-np.inf, np.inf].
        are_ints : list / array (bool), optional
            Array of logical variables that indicate if each rich value
            corresponds to an integer variable. The default is False.
        """
        
        acronyms = ['main_values', 'uncertainties', 'are_integers',
                   'are_lower_limits', 'are_upper_limits', 'are_finite_ranges']
        for kwarg in kwargs:
            if kwarg not in acronyms:
                raise TypeError("RichArray() got an unexpected keyword argument"
                                + " '{}'".format(kwarg))
        if 'main_values' in kwargs:
            mains = kwargs['main_values']
        if mains is None:
            raise TypeError("RichArray() missing required argument 'mains'"
                            + " (pos 1)")
        if 'uncertainties' in kwargs:
            uncs = kwargs['uncertainties']
        if 'are_lower_limits' in kwargs:
            are_lolims = kwargs['are_lower_limits']
        if 'are_upper_limits' in kwargs:
            are_uplims = kwargs['are_upper_limits']
        if 'are_finite_ranges' in kwargs:
            are_ranges = kwargs['are_finite_ranges']
        if 'are_integers' in kwargs:
            are_ints = kwargs['are_integers']
        
        mains = np.array(mains)
        shape = mains.shape
        if uncs is None:
            uncs = 0.
        if are_lolims is None:
            are_lolims = False
        if are_uplims is None:
            are_uplims = False
        if are_ranges is None:
            are_ranges = False
        if domains is None:
            domains = (np.array([defaultparams['domain'] for x in mains.flat])
                       .reshape((*shape, 2)))
        if are_ints is None:
            are_ints = defaultparams['assume integers']
        uncs = np.array(uncs)
        are_lolims = np.array(are_lolims)
        are_uplims = np.array(are_uplims)
        are_ranges = np.array(are_ranges)
        domains = np.array(domains)
        are_ints = np.array(are_ints)

        if uncs.size == 1:
            uncs = uncs * np.ones((*shape, 2))
        elif len(uncs) == 2 and type(uncs[0]) is not np.ndarray:
            uncs = np.array([uncs[0] * np.ones(shape),
                             uncs[1] * np.ones(shape)])
        elif uncs.shape == (*shape, 2):
            uncs = uncs.transpose()
        elif uncs.shape == shape:
            uncs = np.array([[uncs]]*2).reshape((2, *shape))
        if are_lolims.size == 1:
            are_lolims = are_lolims * np.ones(shape, bool)
        if are_uplims.size == 1:
            are_uplims = are_uplims * np.ones(shape, bool)
        if are_ranges.size == 1:
            are_ranges = are_ranges * np.ones(shape, bool)
        if len(domains) == 2 and type(domains[0]) is not np.ndarray:
            domains = np.array([domains[0] * np.ones(shape),
                                domains[1] * np.ones(shape)])
        elif domains.shape == (*shape, 2):
            domains = domains.transpose()
        elif domains.flatten().shape == (2,):
            domains = (np.array([domains for x in mains.flat])
                       .reshape((*shape, 2)).transpose())
        if are_ints.size == 1:
            are_ints = are_ints * np.ones(shape, bool)
        
        mains_flat = mains.flatten()
        uncs_flat = uncs.flatten()
        are_lolims_flat = are_lolims.flatten()
        are_uplims_flat = are_uplims.flatten()
        are_ranges_flat = are_ranges.flatten()
        domains_flat = domains.flatten()
        are_ints_flat = are_ints.flatten()
        offset = len(uncs_flat) // 2
        array = np.empty(mains.size, object)
        for i in range(mains.size):
            main = mains_flat[i]
            unc = [uncs_flat[i], uncs_flat[i+offset]]
            is_lolim = are_lolims_flat[i]
            is_uplim = are_uplims_flat[i]
            is_range = are_ranges_flat[i]
            domain = [domains_flat[i], domains_flat[i+offset]]
            is_int = are_ints_flat[i]
            is_real = 'complex' not in str(type(main))
            if is_real:
                array[i] = RichValue(main, unc, is_lolim, is_uplim, is_range,
                                     domain, is_int)
            else:
                real = RichValue(main.real, np.array(unc).real,
                   bool(complex(is_lolim).real), bool(complex(is_uplim).real),
                   bool(complex(is_range).real), np.array(domain).real, is_int)
                imag = RichValue(main.imag, np.array(unc).imag,
                   bool(complex(is_lolim).imag), bool(complex(is_uplim).imag),
                   bool(complex(is_range).imag), np.array(domain).imag, is_int)
                array[i] = ComplexRichValue(real, imag, domain, is_int)
            if variables is not None and expressions is not None:
                array[i].variables = variables[i]
                array[i].expression = expressions[i]
        array = array.reshape(mains.shape)
        array = array.view(cls)
        return array

    @property
    def mains(self):
        return np.array([x.main for x in self.flat]).reshape(self.shape)
    @property
    def uncs(self):
        return np.array([x.unc for x in self.flat]).reshape([*self.shape,2]) 
    @property
    def are_lolims(self):
        return np.array([x.is_lolim for x in self.flat]).reshape(self.shape)
    @property
    def are_uplims(self):
        return np.array([x.is_uplim for x in self.flat]).reshape(self.shape) 
    @property
    def are_ranges(self):
        return np.array([x.is_range for x in self.flat]).reshape(self.shape)
    @property 
    def domains(self):
        return np.array([x.domain for x in self.flat]).reshape([*self.shape,2])
    @property
    def are_ints(self):
        return np.array([x.is_int for x in self.flat]).reshape(self.shape)
    @property
    def nums_sf(self):
        return np.array([x.num_sf for x in self.flat]).reshape(self.shape)
    @property
    def min_exps(self):
        return np.array([x.min_exp for x in self.flat]).reshape(self.shape)
    @property
    def max_decs(self):
        return np.array([x.max_dec for x in self.flat]).reshape(self.shape)
    @property
    def extra_sf_lims(self):
        return np.array([x.extra_sf_lim for x in self.flat]).reshape(self.shape)
    @property
    def are_lims(self):
        return np.array([x.is_lim for x in self.flat]).reshape(self.shape)
    @property
    def are_intervs(self):
        return np.array([x.is_interv for x in self.flat]).reshape(self.shape)
    @property
    def are_centrs(self):
        return np.array([x.is_centr for x in self.flat]).reshape(self.shape)
    @property
    def are_exacts(self):
        return np.array([x.is_exact for x in self.flat]).reshape(self.shape)
    @property
    def are_consts(self):
        return np.array([x.is_const for x in self.flat]).reshape(self.shape)
    @property
    def are_nans(self):
        return np.array([x.is_nan for x in self.flat]).reshape(self.shape)
    @property
    def are_infs(self):
        return np.array([x.is_inf for x in self.flat]).reshape(self.shape)
    @property
    def centers(self):
        return np.array([x.center for x in self.flat]).reshape(self.shape) 
    @property
    def rel_uncs(self):
        return (np.array([x.rel_unc for x in self.flat])
                .reshape((*self.shape,2)))
    @property
    def signals_noises(self):
        return (np.array([x.signal_noise for x in self.flat])
                .reshape((*self.shape,2)))
    @property
    def ampls(self):
        return (np.array([x.ampl for x in self.flat])
                .reshape((*self.shape,2)))
    @property
    def rel_ampls(self):
        return (np.array([x.rel_ampl for x in self.flat])
                .reshape((*self.shape,2)))
    @property
    def norm_uncs(self):
        return (np.array([x.norm_unc for x in self.flat])
                .reshape((*self.shape,2)))
    @property
    def prop_scores(self):
        return np.array([x.prop_score for x in self.flat]).reshape(self.shape)
    @property
    def uncs_eb(self):
        return self.uncs.transpose()

    @property
    def variables(self):
        allvars = list(np.concatenate(tuple(x.variables for x in self.flat)))
        allvars = list(set(allvars))
        return allvars
    
    @property
    def expression(self):
        self_flat = self.flat
        expr_list = [x.expression for x in self_flat]
        if all(['[{}]'.format(i) in expr for (i,expr) in enumerate(expr_list)]):
            expr = expr_list[0].replace('[0]','')
        else:
            expr_list = np.array(expr_list).reshape(self.shape).tolist()
            expr = 'np.array({})'.format(expr_list)
        expr = expr.replace("'","")
        if expr[0] == '(' and expr[-1] == ')':
            expr = expr[1:-1]
        return expr
    
    @property
    def datatype(self):
        """Type of numbers of the entries of the rich array."""
        types = np.array([type(x) for x in self.flat]).reshape(self.shape)
        if any(types == ComplexRichValue):
            arr_type = complex
        elif any(types == RichValue):
            if not any(self.are_ints):
                arr_type = float
            else:
                arr_type = int
        else:
            arr_type = types[0]
        return arr_type
    
    def intervals(self, sigmas=None):
        sigmas = set_default_value(sigmas, 'sigmas for intervals')
        return (np.array([x.interval(sigmas) for x in self.flat])
                .reshape((*self.shape,2)))

    def signs(self, sigmas=np.inf):
        return (np.array([x.sign(sigmas) for x in self.flat])
                .reshape(self.shape))
    
    def set_params(self, params):
        """Set the rich value parameters of each entry of the rich array."""
        abbreviations = {'is integer': 'is_int',
                         'number of significant figures': 'num_sf',
                         'minimum exponent for scientific notation': 'min_exp',
                         'maximum number of decimals': 'max_dec',
                         'limit for extra significant figure': 'extra_sf_lim'}
        attributes = ['domain'] + list(abbreviations.values()) 
        for entry in abbreviations:
            name = abbreviations[entry]
            if entry in params:
                params[name] = params[entry]
                del params[entry]
        for name in params:
            if name not in attributes:
                print("Warning: Parameter '{}' does not exist.".format(name))
        for x in self.flat:
            if 'domain' in params:
                x.domain = params['domain']
            if 'is_int' in params:
                x.is_int = params['is_int']
            if 'num_sf' in params:
                x.num_sf = params['num_sf']
            if 'min_exp' in params:
                x.min_exp = params['min_exp']
            if 'max_dec' in params:
                x.max_dec = params['max_dec']
            if 'extra_sf_lim' in params:
                x.extra_sf_lim = params['extra_sf_lim']
    
    def set_lims_uncs(self, factor=4.):
        """Set uncertainties of limits with respect to central values."""
        c = factor
        if not hasattr(c, '__iter__'):
            c = [c, c]
        cl, cu = c
        if cl == 0:
            cl = 1
        if cu == 0:
            cu = 1
        for x in self.flat:
            if x.is_lolim:
                x.unc = [x.main / cl] * 2
            elif x.is_uplim:
                x.unc = [x.main / cu] * 2

    def latex(self, **kwargs):
        """Display the values of the rich array in LaTeX math mode."""
        array = (np.array([x.latex(**kwargs) for x in self.flat])
                 .reshape(self.shape))
        return array

    def sample(self, len_sample=1):
        """Obtain a sample of each entry of the array"""
        array = np.empty(0, float)
        for x in self.flat:
            array = np.append(array, x.sample(len_sample))
        shape = (*self.shape, len_sample) if len_sample != 1 else self.shape
        array = array.reshape(shape).transpose()
        return array

    def function(self, function, **kwargs):
        """Apply a function to the rich array."""
        return function_with_rich_arrays(function, self, **kwargs)
    
    def mean(self):
        return np.array(self).mean()
    
    def std(self):
        std_function = lambda u: (np.sum((u-u.mean())**2)/(len(self)-1))**0.5
        return self.function(std_function)

    # Attribute acronyms.
    main_values = mains
    uncertainties = uncs
    are_lower_limits = are_lolims
    are_upper_limits = are_uplims
    are_finite_ranges = are_ranges
    are_integers = are_ints
    numbers_of_significant_figures = nums_sf
    minimum_exponents_for_scientific_notation = min_exps
    maximum_number_of_decimals = max_decs
    limits_for_extra_significant_figure = extra_sf_lims
    are_limits = are_lims
    are_intervals = are_intervs
    are_centereds = are_centrs
    are_constants = are_consts
    are_not_a_number = are_nans
    are_infinites = are_infs
    relative_uncertainties = rel_uncs
    signals_to_noises = signals_noises
    amplitudes = ampls
    relative_amplitudes = rel_ampls
    normalized_uncertainties = norm_uncs
    propagation_scores = prop_scores
    # Method acronyms.
    set_limits_uncertainties = set_lims_uncs
    set_parameters = set_params

class RichDataFrame(pd.DataFrame):
    """
    A class to store a dataframe with uncertainties or with upper/lower limits.
    """
    
    @property
    def _constructor(self):
        return RichDataFrame
    @property
    def _constructor_sliced(self):
        return RichSeries
    @property
    def values(self):
        return pd.DataFrame(self).values.view(RichArray)

    def _property(self, name):
        """Apply the input RichArray attribute/method with 1 element."""
        code = [
            'try:',
            '    data = self.values.{}'.format(name),
            'except:',
            '    array = self.values',
            '    shape = array.shape',
            '    types = (np.array([type(x) for x in array.flat])'
                       + '.reshape(shape))',
            '    data = np.zeros(shape, object)',
            '    cond = types == RichValue',
            '    data[cond] = array[cond].{}'.format(name),
            '    cond = ~cond',
            '    data[cond] = array[cond]',
            'df = pd.DataFrame(data, self.index, self.columns)']
        code = '\n'.join(code)
        output = {}
        exec(code, {**{'self': self}, **globals()}, output)
        return output['df']

    def _property2(self, name):
        """Apply the input RichArray attribute/method with 2 elements."""
        code = [
            'array = self.values',
            'shape = array.shape',
            'types = (np.array([type(x) for x in array.flat])'
                   + '.reshape(shape))',
            'data = np.zeros(shape, object)',
            'cond = types == RichValue',
            'new_elements = array[cond].{}'.format(name),
            'new_elements = [[x[0], x[1]] for x in new_elements]',
            'new_subarray = np.frompyfunc(list, 0, 1)'
                 + '(np.empty(cond.sum(), dtype=object))',
            'new_subarray[:] = new_elements',
            'data[cond] = new_subarray',
            'cond = ~cond',
            'data[cond] = array[cond]',
            'df = pd.DataFrame(data, self.index, self.columns)']
        code = '\n'.join(code)
        output = {}
        exec(code, {**{'self': self}, **globals()}, output)
        return output['df']

    @property    
    def mains(self): return self._property('mains')
    @property
    def uncs(self): return self._property2('uncs')
    @property
    def are_lolims(self): return self._property('are_lolims')
    @property
    def are_uplims(self): return self._property('are_uplims')
    @property
    def are_ranges(self): return self._property('are_ranges')
    @property
    def domains(self): return self._property2('domains')
    @property
    def are_ints(self): return self._property('are_ints')
    @property
    def nums_sf(self): return self._property('nums_sf')
    @property
    def min_exps(self): return self._property('min_exps')
    @property
    def max_decs(self): return self._property('max_decs')
    @property
    def extra_sf_lims(self): return self._property('extra_sf_lims')
    @property
    def are_lims(self): return self._property('are_lims')
    @property
    def are_intervs(self): return self._property('are_intervs')
    @property
    def are_centrs(self): return self._property('are_centrs')
    @property
    def are_exacts(self): return self._property('are_exacts')
    @property
    def are_consts(self): return self._property('are_consts')
    @property
    def are_nans(self): return self._property('are_nans')
    @property
    def are_infs(self): return self._property('are_infs')
    @property
    def centers(self): return self._property('centers')
    @property
    def rel_uncs(self): return self._property2('rel_uncs')
    @property
    def signals_noises(self): return self._property2('signal_noises')
    @property
    def ampls(self): return self._property2('ampls')
    @property
    def rel_ampls(self): return self._property2('rel_ampls')
    @property
    def norm_uncs(self): return self._property2('norm_uncs')
    @property
    def prop_scores(self): return self._property('prop_scores')

    def intervals(self, sigmas=None):
        sigmas = set_default_value(sigmas, 'sigmas for intervals')
        sigmas = str(sigmas).replace('inf', 'np.inf')
        return self._property2('intervals({})'.format(sigmas))

    def signs(self, sigmas=np.inf):
        sigmas = str(sigmas).replace('inf', 'np.inf')
        return self._property('signs({})'.format(sigmas))
    
    def flatten_property(self, name):
        """Separate the list elements from the given attribute/method output."""
        df = eval('self.{}'.format(name))
        df1, df2 = df.copy(), df.copy()
        columns = df.columns
        are_lists = False
        for (i,row) in df.iterrows():
            for (entry,col) in zip(row,columns):
                if (type(entry) is list and len(entry) == 2
                        and all([type(entry[i]) != str for i in [0,1]])):
                    are_lists = True
                    df1.at[i,col] = entry[0]
                    df2.at[i,col] = entry[1]
                else:
                    df1.at[i,col] = entry
                    df2.at[i,col] = entry
        output = [df1, df2] if are_lists else df1
        return output
    
    def set_params(self, params):
        """Set the rich value parameters of each column of the dataframe."""
        abbreviations = {'is integer': 'is_int',
                         'number of significant figures': 'num_sf',
                         'minimum exponent for scientific notation': 'min_exp',
                         'maximum number of decimals': 'max_dec',
                         'limit for extra significant figure': 'extra_sf_lim'}
        attributes = ['domain'] + list(abbreviations.values())
        for entry in abbreviations:
            name = abbreviations[entry]
            if entry in params:
                params[name] = params[entry]
                del params[entry]
        for name in params:
            if name not in attributes:
                print("Warning: Parameter '{}' does not exist.".format(name))
        for name in attributes:
            if name in params and type(params[name]) is not dict:
                default_param = params[name]
                params[name] = {}
                for col in self:
                    params[name][col] = default_param
        set_domain = 'domain' in params
        set_is_int = 'is_int' in params
        set_num_sf = 'num_sf' in params
        set_min_exp = 'min_exp' in params
        set_max_dec = 'max_dec' in params
        set_extra_sf_lim = 'extra_sf_lim' in params
        row_inds = self.index
        for col in self:
            idx = self.index[0]
            is_rich_value = type(self[col][idx]) in (RichValue, ComplexRichValue)
            if is_rich_value:
                if set_domain and col in params['domain']:
                    for i in row_inds:
                        self[col][i].domain = params['domain'][col]
                if set_is_int and col in params['is_int']:
                    for i in row_inds:
                        self[col][i].is_int = params['is_int'][col]
                if set_num_sf and col in params['num_sf']:
                    for i in row_inds:
                        self[col][i].num_sf = params['num_sf'][col]
                if set_min_exp and col in params['min_exp']:
                    for i in row_inds:
                        self[col][i].min_exp = params['min_exp'][col]
                if set_max_dec and col in params['max_dec']:
                    for i in row_inds:
                        self[col][i].max_dec = params['max_dec'][col]
                if set_extra_sf_lim and col in params['extra_sf_lim']:
                    for i in row_inds:
                        self[col][i].extra_sf_lim = params['extra_sf_lim'][col]
 
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
        new_column : rich array
            Resulting column as a rich array.
        """
        new_column = np.empty(len(self), object)
        for (i,(_,row)) in enumerate(self.iterrows()):
            arguments = [row[col] for col in columns]
            rvalue = function_with_rich_values(function, arguments, **kwargs)
            new_column[i] = rvalue
        new_column = new_column.view(RichArray)
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
            New row as a dataframe.
        """
        new_row = {}
        for (i,col) in enumerate(self):
            arguments = [self.at[idx,col] for idx in rows]
            rvalue = function_with_rich_values(function, arguments, **kwargs)
            new_row[col] = rvalue
        new_row = pd.DataFrame(new_row, idex=[0])
        return new_row

    def latex(self, return_df=False, export_frame=True, export_index=True,
              row_sep='\\tabularnewline', **kwargs):
        """Return the content of the dataframe as a table in LaTeX format."""
        row_sep = ' ' + row_sep + ' \n'
        df = copy.copy(self)
        for col in self:
            for i in self.index:
                entry = self.at[i,col]
                if 'RichValue' in str(type(entry)):
                    if not np.isnan(entry.main):
                        df.at[i,col] = entry.latex(**kwargs)
                    else:
                        df.at[i,col] = '...'
        if return_df:
            output = df
        else:
            columns = list(df.columns)
            num_columns = len(columns)
            text = ''
            if export_frame:
                text += '\\renewcommand*{\\arraystretch}{1.4}' + ' \n'
                text += ('\\begin{tabular}{' + 'l'*export_index
                         + num_columns*'c' + '}' + ' \n')
                text += '\\hline \n'
            columns = ['{\\bf ' + column + '}' for column in columns]
            index_name = df.index.name if df.index.name is not None else ' '
            if export_index:
                columns = ['{\\bf ' + index_name + '}'] + columns
            if export_frame:
                text += ' & '.join(columns) + row_sep + '\\hline \n'
            rows = []
            for (ind,row) in df.iterrows():
                cols = [str(ind)] if export_index else []
                for (j,column) in enumerate(df):
                    entry = str(row[column])
                    if entry == 'nan':
                        entry = '...'
                    cols += [entry]
                rows += [' & '.join(cols)]
            text += row_sep.join(rows)
            if export_frame:
                text += row_sep
                text += '\\hline \n'
                text += '\\end{tabular}' + ' \n'
                text += '\\renewcommand*{\\arraystretch}{1.0}' + ' \n'
            output = text
        return output

    def set_lims_uncs(self, factors={}):
        """Set the uncertainties of limits with respect to central values."""
        if factors == {}:
            factors = 4.
        if type(factors) is not dict:
            factors = {col: factors for col in self}
        for (i,row) in self.iterrows():
            for col in factors:
                if 'RichValue' in str(type(self.at[i,col])):
                    entry = self.at[i,col]
                    if entry.is_lim:
                        c = factors[col]
                        if not hasattr(c, '__iter__'):
                            c = [c, c]
                        cl, cu = c
                        c = cl if entry.is_lolim else cu
                        self.at[i,col].set_lim_unc(c)
    
    # Attribute acronyms.
    main_values = mains
    uncertainties = uncs
    are_lower_limits = are_lolims
    are_upper_limits = are_uplims
    are_finite_ranges = are_ranges
    are_integers = are_ints
    are_limits = are_lims
    are_intervals = are_intervs
    are_centereds = are_centrs
    are_constants = are_consts
    are_not_a_number = are_nans
    are_infinites = are_infs
    relative_uncertainties = rel_uncs
    signals_to_noises = signals_noises
    amplitudes = ampls
    relative_amplitudes = rel_ampls
    normalized_uncertainties = norm_uncs
    propagation_scores = prop_scores
    # Method acronyms.
    set_parameters = set_params
    set_limits_factors = set_lims_uncs

class RichSeries(pd.Series):
    """A class to store a series with the RichArray methods."""
    
    @property
    def _constructor(self):
        return RichSeries
    @property
    def _constructor_expanddim(self):
        return RichDataFrame
    @property
    def values(self):
        return pd.Series(self).values.view(RichArray)
    
    @property
    def mains(self):
        return pd.Series(self.values.view(RichArray).mains, self.index)
    @property
    def uncs(self):
        return [pd.Series(self.values.view(RichArray).uncs.T[i].T,
                          self.index) for i in (0,1)]
    @property
    def are_lolims(self):
        return pd.Series(self.values.view(RichArray).are_lolims, self.index)
    @property
    def are_uplims(self):
        return pd.Series(self.values.view(RichArray).are_uplims, self.index)
    @property
    def are_ranges(self):
        return pd.Series(self.values.view(RichArray).are_ranges, self.index)
    @property
    def domains(self):
        return [pd.Series(self.values.view(RichArray).domains.T[i].T,
                          self.index) for i in (0,1)]
    @property
    def are_ints(self):
        return pd.Series(self.values.view(RichArray).are_ints, self.index)
    @property
    def nums_sf(self):
        return pd.Series(self.values.view(RichArray).num_sf, self.index)
    @property
    def min_exps(self):
        return pd.Series(self.values.view(RichArray).min_exps, self.index)
    @property
    def max_decs(self):
        return pd.Series(self.values.view(RichArray).max_decs, self.index)
    @property
    def extra_sf_lims(self):
        return pd.Series(self.values.view(RichArray).extra_sf_lim, self.index)
    @property
    def are_lims(self):
        return pd.Series(self.values.view(RichArray).are_lims, self.index)
    @property
    def are_intervs(self):
        return pd.Series(self.values.view(RichArray).are_intervs, self.index)
    @property
    def are_centrs(self):
        return pd.Series(self.values.view(RichArray).are_centrs, self.index)
    @property
    def are_exacts(self):
        return pd.Series(self.values.view(RichArray).are_exacts, self.index)
    @property
    def are_consts(self):
        return pd.Series(self.values.view(RichArray).are_consts, self.index)
    @property
    def are_nans(self):
        return pd.Series(self.values.view(RichArray).are_nans, self.index)
    @property
    def are_infs(self):
        return pd.Series(self.values.view(RichArray).are_infs, self.index)
    @property
    def centers(self):
        return pd.Series(self.values.view(RichArray).centers, self.index)
    @property
    def rel_uncs(self):
        return [pd.Series(self.values.view(RichArray).rel_uncs.T[i].T,
                          self.index) for i in (0,1)]
    @property
    def signals_noises(self):
        return [pd.Series(self.values.view(RichArray).signals_noises.T[i].T,
                          self.index) for i in (0,1)]
    @property
    def ampls(self):
        return [pd.Series(self.values.view(RichArray).ampls.T[i].T,
                          self.index) for i in (0,1)]
    @property
    def rel_ampls(self):
        return [pd.Series(self.values.view(RichArray).rel_ampls.T[i].T,
                          self.index) for i in (0,1)]
    @property
    def norm_uncs(self):
        return [pd.Series(self.values.view(RichArray).norm_uncs.T[i].T,
                          self.index) for i in (0,1)]
    @property
    def prop_scores(self):
        return pd.Series(self.values.view(RichArray).prop_scores, self.index)
    
    def intervals(self, sigmas=None):
        sigmas = set_default_value(sigmas, 'sigmas for intervals')
        return [pd.Series(self.values.view(RichArray).intervals(sigmas).T[i].T,
                          self.index) for i in (0,1)]
            
    def signs(self, sigmas=None):
        sigmas = set_default_value(sigmas, 'sigmas for intervals')
        return pd.Series(self.values.view(RichArray).signs(sigmas), self.index)
    
    def set_params(self, params):
        data = self.values.view(RichArray)
        data.set_params(params)
        self.update(pd.Series(data, self.index))
    
    def set_lims_uncs(self, factor=4.):
        data = self.values.view(RichArray)
        data.set_lims_uncs(factor)
        self.update(pd.Series(data, self.index))

    def latex(self, **kwargs):
        return pd.Series(self.values.view(RichArray).latex(**kwargs), self.index)
    
    def function(self, function, **kwargs):
        data = self.values.view(RichArray).function(function, **kwargs)
        return pd.Series(data, self.index)

    # Attribute acronyms.
    main_values = mains
    uncertainties = uncs
    are_lower_limits = are_lolims
    are_upper_limits = are_uplims
    are_finite_ranges = are_ranges
    are_integers = are_ints
    numbers_of_scientific_figures = nums_sf
    minimum_exponents_for_scientific_notation = min_exps
    maximum_numbers_of_decimals = max_decs
    are_limits = are_lims
    are_intervals = are_intervs
    are_centereds = are_centrs
    are_constants = are_consts
    are_not_a_number = are_nans
    are_infinites = are_infs
    relative_uncertainties = rel_uncs
    signals_to_noises = signals_noises
    amplitudes = ampls
    relative_amplitudes = rel_ampls
    normalized_uncertainties = norm_uncs
    propagation_scores = prop_scores
    # Method acronyms.
    set_limits_uncertainties = set_lims_uncs
    set_parameters = set_params
 
class ComplexRichValue():
    """
    A class to store complex value with uncertainties or upper/lower limits.
    """
    
    def __init__(self, real=0, imag=0, domain=None, is_int=None, **kwargs):
        """
        Parameters
        ----------
        real : rich value
            Real part of the complex rich value. 
        imag : rich value
            Imaginary part of the complex rich value
        """
        
        acronyms = ['real_part', 'imaginary_part', 'imaginary', 'is_integer']
        for kwarg in kwargs:
            if kwarg not in acronyms:
                raise TypeError("RichValue() got an unexpected keyword argument"
                                + " '{}'".format(kwarg))
        if 'real_part' in kwargs:
            real = kwargs['real_part']
        if 'imaginary_part' in kwargs:
            imag = kwargs['imaginary_part']
        if 'imaginary' in kwargs:
            imag = kwargs['imaginary']
        if 'is_integer' in kwargs:
            is_int = kwargs['is_integer']
        
        if type(real) is not RichValue:
            real = rich_value(real, domain, is_int)
        if type(imag) is not RichValue:
            imag = rich_value(imag, domain, is_int)
        
        if is_int is None:
            is_int = real.is_int and imag.is_int
        real.is_int = is_int
        imag.is_int = is_int
        
        num_sf = min(real.num_sf, imag.num_sf)
        min_exp = round(np.mean([real.min_exp, imag.min_exp]))
        max_dec = min(real.max_dec, imag.max_dec)
        extra_sf_lim = min(real.extra_sf_lim, imag.extra_sf_lim)
        real.num_sf = num_sf
        imag.num_sf = num_sf
        real.min_exp = min_exp
        imag.min_exp = min_exp
        real.max_dec = max_dec
        real.max_dec = max_dec
        real.extra_sf_lim = extra_sf_lim
        imag.extra_sf_lim = extra_sf_lim
        
        self._real = real
        self._imag = imag
 
    @property
    def real(self): return self._real
    @real.setter
    def real(self, x):
        rvalue = x if type(x) is RichValue else rich_value(x)
        self._real = rvalue

    @property
    def imag(self): return self._imag
    @imag.setter
    def imag(self, x):
        rvalue = x if type(x) is RichValue else rich_value(x)
        self._imag = rvalue

    @property
    def is_lolim(self): return self.real.is_lolim + 1j*self.imag.is_lolim
    @property
    def is_uplim(self): return self.real.is_uplim + 1j*self.imag.is_uplim
    @property
    def is_range(self): return self.real.is_range + 1j*self.imag.is_range
    
    @property
    def is_int(self): return self.real.is_int and self.imag.is_int
    @is_int.setter
    def is_int(self, x):
        self.real.is_int = x
        self.imag.is_int = x
    
    @property
    def num_sf(self): return min(self.real.num_sf, self.imag.is_int)
    @num_sf.setter
    def num_sf(self, x):
        self.real.num_sf = x
        self.imag.num_sf = x
        
    @property
    def min_exp(self): return round(np.mean([self.real.min_exp, self.imag.min_exp]))  
    @min_exp.setter
    def min_exp(self, x):
        self.real.min_exp = x
        self.imag.min_exp = x
        
    @property
    def max_dec(self): return min(self.real.max_dec, self.imag.max_dec)  
    @max_dec.setter
    def max_dec(self, x):
        self.real.max_dec = x
        self.imag.max_dec = x
    
    @property
    def extra_sf_lim(self): return max(self.real.extra_sf_lim, self.imag.extra_sf_lim)
    @extra_sf_lim.setter
    def extra_sf_lim(self, x):
        self.real.extra_sf_lim = x
        self.imag.extra_sf_lim = x

    @property
    def domain(self):
        domain_real = self.real.domain
        domain_imag = self.imag.domain
        domain = [complex(domain_real[0], domain_imag[0]),
                  complex(domain_real[1], domain_imag[1])]
        return domain
    @domain.setter
    def domain(self, x):
        self.real.domain = x
        self.imag.domain = x

    @property
    def common_domain(self):
        if self.real.is_const and self.imag.is_const:
            domain = None
        else:
            domain_real = np.array(self.domain).real
            domain_imag = np.array(self.domain).imag
            domain = [min(domain_real[0], domain_imag[0]),
                      max(domain_real[0], domain_imag[1])]
        return domain

    @property
    def main(self):
        """Main value."""
        x = self.real.main + 1j*self.imag.main
        return x 

    @property
    def unc(self):
        """Uncertainty."""
        dx = list(np.array(self.real.unc) + 1j*np.array(self.imag.unc))
        return dx
        
    @property
    def mod(self):
        """Module."""
        modu = abs(self)
        return modu

    @property
    def ang(self):
        """Angle."""
        if self.real.is_exact and self.imag.is_exact:
            ang = RichValue(np.angle(self.real.main + 1j*self.imag.main),
                            domain=self.common_domain, is_int=self.is_int)
        else:
            ang = self.function('np.angle({})', domain=[-np.pi,np.pi],
                                is_domain_cyclic=True)
        return ang

    @property
    def is_lim(self): return self.real.is_lim or self.imag.is_lim
    @property
    def is_interv(self): return self.real.is_interv or self.imag.is_interv
    @property
    def is_centr(self): return self.real.is_centr or self.imag.is_centr
    @property
    def is_exact(self): return self.real.is_exact and self.imag.is_exact
    @property
    def is_const(self): return self.real.is_const and self.imag.is_const
    @property
    def is_finite(self): return self.real.is_finite and self.imag.is_finite
    @property
    def is_inf(self): return self.real.is_inf and self.imag.is_inf
    @property
    def is_nan(self): return self.real.is_nan and self.imag.is_nan
    @property
    def prop_score(self): return min(self.real.prop_score, self.imag.prop_score)

    @property
    def variables(self):
        variables = np.concatenate((self.real.variables,
                                    self.imag.variables)).tolist()
        return variables
    
    @property
    def expression(self):
        expression = 'complex({},{})'.format(self.real.expression,
                                             self.imag.expression)
        return expression

    def _format_as_complex_rich_value(self):
        show_domain = defaultparams['show domain']
        real = self.real
        imag = self.imag
        if imag.is_exact and imag.main == 0:
            text = '{}'.format(real)
        elif real.is_exact and real.main == 0:
            text = '{} j'.format(imag)
        else:
            text = '{} + ({}) j'.format(real, imag)
            if imag.is_exact:
                text = text.replace('(','').replace(')','')
        text = text.replace('+ -', '- ').replace('+ (-', '- (')
        if show_domain:
            domain = ' [' + text.split('[')[1].split(']')[0] + ']'
            text = text.replace(domain, '') + domain
        return text

    def __repr__(self):
        return self._format_as_complex_rich_value()
    
    def __str__(self):
        return self._format_as_complex_rich_value()
    
    def latex(self, **kwargs):
        show_domain = defaultparams['show domain']
        real = self.real
        imag = self.imag
        if imag.is_exact and imag.main == 0:
            text = '{}'.format(real.latex(**kwargs))
        elif real.is_exact and real.main == 0:
            text = '{}\\,i'.format(imag.latex(**kwargs))
        else:
            text = '{} + ({})\\,i'.format(real.latex(**kwargs),
                                        imag.latex(**kwargs))
            if imag.is_exact:
                text = text.replace('(','').replace(')','')
        text = (text.replace('$ + ', ' + ').replace(' + $', ' + ')
                .replace('+ ($', '+ (').replace('$)\\,', ')\\,$')
                .replace('+ -', '- ').replace('+ (-', '- ('))
        if show_domain:
            domain = ' [' + text.split('[')[1].split(']')[0] + ']'
            text = text.replace(domain, '') + domain
        return text
    
    def __neg__(self):
        rvalue = copy.copy(self)
        rvalue.real = -self.real
        rvalue.imag = -self.imag
        return rvalue
    
    def __abs__(self):
        if self.real.is_exact and self.imag.is_exact:
            rvalue = RichValue(abs(self.real.main + 1j*self.imag.main),
                                 domain=self.common_domain, is_int=self.is_int)
        else:
            rvalue = self.function('abs({})')
        return rvalue
    
    def __add__(self, other):
        if type(other) in (np.ndarray, RichArray):
            return other + self
        else:
            rvalue = copy.copy(self)
            rvalue.real = self.real + other.real
            rvalue.imag = self.imag + other.imag
        return rvalue
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return -(self - other)

    def __mul__(self, other):
        if type(other) in (np.ndarray, RichArray):
            return other * self
        rvalue = copy.copy(self)
        is_other_numeric = (type(other) not in (RichValue, ComplexRichValue)
                            or 'complex' in str(type(other)))
        if is_other_numeric:
            rvalue.real *= other
            rvalue.imag *= other
        else:
            a, b = self.real, self.imag
            c, d = other.real, other.imag
            rvalue.real = a*c - b*d
            rvalue.imag = a*d + b*c
        return rvalue

    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        if type(other) in (np.ndarray, RichArray):
            return rich_array(self) / other
        rvalue = copy.copy(self)
        is_other_real = (type(other) not in (RichValue, ComplexRichValue)
                         and 'complex' not in str(type(other)))
        if is_other_real:
            rvalue.real /= other
            rvalue.imag /= other
        else:
            a, b = self.real, self.imag
            c, d = other.real, other.imag
            norm = c**2 + d**2
            rvalue.real = (a*c + b*d) / norm
            rvalue.imag = (b*c - a*d) / norm
        return rvalue

    def __rtruediv__(self, other):
        other_ = (ComplexRichValue(other) if type(other) is not ComplexRichValue
                  else other)
        return other_ / self

    def __pow__(self, other):
        rvalue = copy.copy(self)
        is_other_numeric = (type(other) not in (RichValue, ComplexRichValue)
                            or 'complex' in str(type(other)))
        if is_other_numeric and self.is_exact:
            self_ = self.real.main + 1j*self.imag.main
            self_domain = self.common_domain
            domain = (self_domain if self.is_const
                      else propagate_domain(self_domain, [other, other]))
            value = self_**other
            rvalue = ComplexRichValue(value.real, value.imag,
                                        domain=domain)
            variables = self.variables
            expression = '({})**({})'.format(self.expression, str(other))
            rvalue.real.variables = variables
            rvalue.real.variables = variables
            rvalue.real.expression = 'np.real({})'.format(expression)
            rvalue.imag.expression = 'np.imag({})'.format(expression)
        else:
            rvalue = function_with_rich_values('{}**{}', [self, other])
        return rvalue

    def __rpow__(self, other):
        is_other_real = (type(other) not in (RichValue, ComplexRichValue)
                         and 'complex' not in str(type(other)))
        if is_other_real:
            is_int = type(other) is int
            other_ = RichValue(other, is_int=is_int)
            other_.variables = []
            other_.expression = str(other)
            other_ = ComplexRichValue(other_, 0)
        other_.num_sf = self.num_sf
        rvalue = other_ ** self
        return rvalue
    
    def sample(self, len_sample=1):
        """Sample of the distribution corresponding to the rich value"""
        distr_real = self.real.sample(len_sample)
        distr_imag = self.imag.sample(len_sample)
        distr = distr_real + 1j*distr_imag
        return distr

    def function(self, function, **kwargs):
        """Apply a function to the rich value"""
        return function_with_rich_values(function, self, **kwargs)
    
    # Instance variable acronyms.  

    @property
    def real_part(self): return self.real
    @real_part.setter
    def real_part(self, x): self.real = x

    @property
    def imaginary_part(self): return self.imag
    @imaginary_part.setter
    def imaginary_part(self, x): self.imag = x

    @property
    def imaginary(self): return self.imag
    @imaginary.setter
    def imaginary(self, x): self.imag = x
    
    # Attribute abbreviations.
    module = mod
    angle = ang
    is_limit = is_lim
    is_interval = is_interv
    is_centered = is_centr
    propagation_score = prop_score
    is_not_a_number = is_nan
    is_infinite = is_inf

 
def propagate_domain(x_domain, y_domain, function):
    """Estimate the domain of the function applied to two domains."""
    x_domain, y_domain = np.array(x_domain), np.array(y_domain)
    with np.errstate(divide='ignore', invalid='ignore'):
        domain_combs = [function(x_domain[i], y_domain[j])
                        for (i,j) in [(0,0), (0,1), (1,0), (1,1)]]
    if any(np.isfinite(domain_combs)):
        domain1, domain2 = np.nanmin(domain_combs), np.nanmax(domain_combs)
    else:
        domain1, domain2 = [-np.inf, np.inf]
    domain = [domain1, domain2]
    return domain

def add_rich_values(x, y):
    """Sum two rich values to get a new one."""
    num_sf = min(x.num_sf, y.num_sf)
    min_exp = round(np.mean([x.min_exp, y.min_exp]))
    extra_sf_lim = max(x.extra_sf_lim, y.extra_sf_lim)
    is_int = x.is_int and y.is_int
    domain = [x.domain[0] + y.domain[0], x.domain[1] + y.domain[1]]
    sigmas = defaultparams['sigmas to use approximate uncertainty propagation']
    if (x.is_exact or y.is_exact) and (x.is_interv or y.is_interv):
        z = list(np.array(x.interval()) + np.array(y.interval()))
        z = RichValue(z, domain=domain, is_int=is_int)
    elif (not (x.is_interv or y.is_interv)
            and min(x.rel_ampl) > sigmas and min(y.rel_ampl) > sigmas):
        z = x.main + y.main
        dz = (np.array(x.unc)**2 + np.array(y.unc)**2)**0.5
        z = RichValue(z, dz, domain=domain, is_int=is_int)
    else:
        z = function_with_rich_values(lambda a,b: a+b, [x, y], domain=domain,
                                      is_vectorizable=True)
    z.num_sf = num_sf
    z.min_exp = min_exp
    z.extra_sf_lim = extra_sf_lim
    return z

def multiply_rich_values(x, y):
    """Multiply two rich values to get a new one."""
    num_sf = min(x.num_sf, y.num_sf)
    min_exp = round(np.mean([x.min_exp, y.min_exp]))
    extra_sf_lim = max(x.extra_sf_lim, y.extra_sf_lim)
    is_int = x.is_int and y.is_int
    domain = propagate_domain(x.domain, y.domain, lambda a,b: a*b)
    sigmas = defaultparams['sigmas to use approximate uncertainty propagation']
    if (x.is_exact or y.is_exact) and (x.is_interv or y.is_interv):
        z = list(np.array(x.interval()) * np.array(y.interval()))
        z = RichValue(z, domain=domain, is_int=is_int)
    elif (not (x.is_interv or y.is_interv)
         and x.prop_score > sigmas and y.prop_score > sigmas):
        z = x.main * y.main
        dx, dy = np.array(x.unc), np.array(y.unc)
        dz = abs(z) * ((dx/x.main)**2 + (dy/y.main)**2)**0.5 if z != 0. else 0.
        z = RichValue(z, dz, domain=domain, is_int=is_int)
    else:
        z = function_with_rich_values(lambda a,b: a*b, [x, y], domain=domain,
                                      is_vectorizable=True)
    z.num_sf = num_sf
    z.min_exp = min_exp
    z.extra_sf_lim = extra_sf_lim
    return z

def divide_rich_values(x, y):
    """Divide two rich values to get a new one."""
    num_sf = min(x.num_sf, y.num_sf)
    min_exp = round(np.mean([x.min_exp, y.min_exp]))
    extra_sf_lim = max(x.extra_sf_lim, y.extra_sf_lim)
    is_int = x.is_int and y.is_int
    domain = propagate_domain(x.domain, y.domain, lambda a,b: a/b)
    sigmas = defaultparams['sigmas to use approximate uncertainty propagation']
    if (x.is_exact or y.is_exact or (not (x.is_interv or y.is_interv)
            and x.prop_score > sigmas and y.prop_score > sigmas)):
        if y.main != 0:
            z = x.main / y.main
            dx, dy = np.array(x.unc), np.array(y.unc)
            dz = (abs(z) * ((dx/x.main)**2 + (dy/y.main)**2)**0.5 if z != 0.
                  else 0.)
            is_uplim = True if x.is_uplim or y.is_lolim else False
            is_lolim = True if x.is_lolim or y.is_uplim else False
            is_range = x.is_range or y.is_range
            if not domain[0] <= z <= domain[1]:
                domain = [min(x.domain[0], y.domain[0]),
                          max(x.domain[1], y.domain[1])]
            z = RichValue(z, dz, is_lolim, is_uplim, is_range, domain, is_int)
        else:
            zero = y
            zero_signs = np.sign(zero.domain)
            if all(zero_signs == 0):
                zero_sign = np.nan
            elif all(zero_signs >= 0):
                zero_sign = 1
            elif all(zero_signs <= 0):
                zero_sign = -1
            else:
                zero_sign = np.nan
            sign = x.sign() * zero_sign
            value = sign * np.inf
            if np.isinf(value):
                domain = [0, np.inf] if value > 0 else [-np.inf, 0]
            else:
                domain = (sign * abs(x)).domain
            z = RichValue(value, domain=domain)
    else:
        z = function_with_rich_values(lambda a,b: a/b, [x, y], domain=domain,
                                      is_vectorizable=True, sigmas=sigmas)
    z.num_sf = num_sf
    z.min_exp = min_exp
    z.extra_sf_lim = extra_sf_lim
    return z

def greater(x, y, sigmas=None):
    """Determine if a rich value/array (x) is greater than another one (y)."""
    sigmas = set_default_value(sigmas, 'sigmas for intervals')
    are_single_values = all([type(var) is str
                             or not hasattr(var, '__iter__') for var in (x,y)])
    if are_single_values:
        x = x if type(x) is RichValue else rich_value(x)
        y = y if type(y) is RichValue else rich_value(y)
        x1, x2 = x.interval(sigmas=sigmas)
        y1, y2 = y.interval(sigmas=sigmas)
        output = True if x2 > y2 else False
    else:
        x = x if type(x) is RichArray else rich_array(x)
        y = y if type(y) is RichValue else rich_array(y)
        if x.size == 0 and x.shape != y.shape:
            x = x.flatten()[0] * np.ones(y.shape)
        if y.size == 0 and x.shape != y.shape:
            y = y.flatten()[0] * np.ones(x.shape)
        if x.shape != y.shape:
            raise Exception('Input arrays must have the same shape.')
        output = np.empty(0, bool)
        for (xi, yi) in zip(x.flat, y.flat):
            output = np.append(output, greater(xi,yi))
        output = output.reshape(x.shape)
    return output

def less(x, y, sigmas=None):
    """Determine if a rich value/array (x) is less than another one (y)."""
    sigmas = set_default_value(sigmas, 'sigmas for intervals')
    are_single_values = all([type(var) is str
                             or not hasattr(var, '__iter__') for var in (x,y)])
    if are_single_values:
        x = x if type(x) is RichValue else rich_value(x)
        y = y if type(y) is RichValue else rich_value(y)
        x1, x2 = x.interval(sigmas=sigmas)
        y1, y2 = y.interval(sigmas=sigmas)
        output = True if x2 < y2 else False
    else:
        x = x if type(x) is RichArray else rich_array(x)
        y = y if type(y) is RichValue else rich_array(y)
        if x.size == 1 and x.shape != y.shape:
            x = x.flatten()[0] * np.ones(y.shape)
        if y.size == 1 and x.shape != y.shape:
            y = y.flatten()[0] * np.ones(x.shape)
        if x.shape != y.shape:
            raise Exception('Input arrays must have the same shape.')
        output = np.empty(0, bool)
        for (xi, yi) in zip(x.flat, y.flat):
            output = np.append(output, less(xi,yi))
        output = output.reshape(x.shape)
    return output

def equiv(x, y, sigmas=None):
    """Check if a rich value/array (x) is equivalent than another one (y)."""
    sigmas = set_default_value(sigmas, 'sigmas for overlap')
    are_single_values = all([type(var) is str
                             or not hasattr(var, '__iter__') for var in (x,y)])
    if are_single_values:
        x = x if type(x) is RichValue else rich_value(x)
        y = y if type(y) is RichValue else rich_value(y)
        minor, major = (x, y) if x < y else (y, x)
        x1, x2 = minor.interval(sigmas=sigmas)
        y1, y2 = major.interval(sigmas=sigmas)
        output = True if x2 >= y1 else False
    else:
        x = x if type(x) is RichArray else rich_array(x)
        y = y if type(y) is RichValue else rich_array(y)
        if x.size == 1 and x.shape != y.shape:
            x = x.flatten()[0] * np.ones(y.shape)
        if y.size == 1 and x.shape != y.shape:
            y = y.flatten()[0] * np.ones(x.shape)
        if x.shape != y.shape:
            raise Exception('Input arrays must have the same shape.')
        output = np.empty(0, bool)
        for (xi, yi) in zip(x.flat, y.flat):
            output = np.append(output, equiv(xi,yi))
        output = output.reshape(x.shape)
    return output

def greater_equiv(x, y, sigmas_interval=None, sigmas_overlap=None):
    """Check if a rich value/array is greater or equivalent than another one."""
    sigmas_interval = set_default_value(sigmas_interval, 'sigmas for intervals')
    sigmas_overlap = set_default_value(sigmas_overlap, 'sigmas for overlap')
    are_single_values = all([type(var) is str
                             or not hasattr(var, '__iter__') for var in (x,y)])
    if are_single_values:
        x = x if type(x) is RichValue else rich_value(x)
        y = y if type(y) is RichValue else rich_value(y)
        output = greater(x, y, sigmas_interval) or equiv(x, y, sigmas_overlap)
    else:
        x = x if type(x) is RichArray else rich_array(x)
        y = y if type(y) is RichValue else rich_array(y)
        if x.size == 1 and x.shape != y.shape:
            x = x.flatten()[0] * np.ones(y.shape)
        if y.size == 1 and x.shape != y.shape:
            y = y.flatten()[0] * np.ones(x.shape)
        if x.shape != y.shape:
            raise Exception('Input arrays must have the same shape.')
        output = np.empty(0, bool)
        for (xi, yi) in zip(x.flat, y.flat):
            output = np.append(output, greater_equiv(xi,yi))
        output = output.reshape(x.shape)
    return output

def less_equiv(x, y, sigmas_interval=None, sigmas_overlap=None):
    """Check if a rich value/array is less or equivalent than another one."""
    sigmas_interval = set_default_value(sigmas_interval, 'sigmas for intervals')
    sigmas_overlap = set_default_value(sigmas_overlap, 'sigmas for overlap')
    are_single_values = all([type(var) is str
                             or not hasattr(var, '__iter__') for var in (x,y)])
    if are_single_values:
        x = x if type(x) is RichValue else rich_value(x)
        y = y if type(y) is RichValue else rich_value(y)
        output = less(x, y, sigmas_interval) or equiv(x, y, sigmas_overlap)
    else:
        x = x if type(x) is RichArray else rich_array(x)
        y = y if type(y) is RichValue else rich_array(y)
        if x.size == 1 and x.shape != y.shape:
            x = x.flatten()[0] * np.ones(y.shape)
        if y.size == 1 and x.shape != y.shape:
            y = y.flatten()[0] * np.ones(x.shape)
        if x.shape != y.shape:
            raise Exception('Input arrays must have the same shape.')
        output = np.empty(0, bool)
        for (xi, yi) in zip(x.flat, y.flat):
            output = np.append(output, less_equiv(xi,yi))
        output = output.reshape(x.shape)
    return output

def rich_value(text=None, domain=None, is_int=None, pdf=None,
               use_default_extra_sf_lim=False):
    """
    Convert the input text to a rich value.

    Parameters
    ----------
    text : str
        String representing a rich value.
    domain : list (float), optional
        The domain of the rich value, that is, the minimum and maximum
        values that it can take. By default, it is the domain written in the
        input text, but if it is not written it will be [-np.inf, np.inf].
    is_int : bool, optional
        If True, the variable corresponding to the rich value will be an
        integer, so when creating samples it will have integer values.
    pdf : function / dict, optional
        Probability density function (PDF) that defines the rich value. It can
        be specified instead of the text representing the rich value.
    use_default_extra_sf_lim : bool, optional
        If True, the default limit for extra significant figure will be used
        instead of inferring it from the input text. This will reduce the
        computation time a little bit.

    Returns
    -------
    rvalue : rich value
        Resulting rich value.
    """

    if ('function' in str(type(text)) or type(text) in
                                        (dict, tuple, list, np.ndarray)):
        pdf = text
        text = None
    else:
        pdf = None
    if text is None and pdf is None:
        raise Exception('You should specify either a text or a PDF to'
                        ' represent the rich value.')
    if text is not None and pdf is not None:
        raise Exception('You have to give a text or a PDF to represent'
                        ' represent the rich value, not both.')

    input_domain = copy.copy(domain)
    default_num_sf = defaultparams['number of significant figures']
    default_extra_sf_lim = defaultparams['limit for extra significant figure']
    abbreviations = {'inf': 'np.inf', 'tau': 'math.tau', 'pi': 'np.pi',
                     'nan': 'np.nan', 'NaN': 'np.nan', 'none': 'np.nan'}
    
    def parse_as_rich_value(text):
        """Obtain the properties of the input text as a rich value."""
        def parse_value(text):
            """Parse input text as a numeric value."""
            text = str(text)
            if any([char.isalpha() for char in text.replace(' e','')]):
                for short_name in abbreviations:
                    full_name = abbreviations[short_name]
                    if full_name not in text:
                        text = text.replace(short_name, full_name)
            text = text.replace(' e', 'e')
            return text
        def read_domain(text):
            """Read the domain in the input text."""
            if '[' in text and ']' in text:
                x1, x2 = text.split('[')[1].split(']')[0].split(',')
                x1 = eval(parse_value(x1))
                x2 = eval(parse_value(x2))
                domain = [x1, x2]
            else:
                domain = None
            return domain
        domain = read_domain(text)
        if domain is not None:
            text = text.split('[')[0][:-1]
        if not '--' in text:    
            if text.startswith('+'):
                text = text[1:]
            if 'e' in text:
                text = text.replace('e+', 'e')
                if not ' e' in text:
                    text = text.replace('e', ' e')
                min_exp = abs(int(text.split('e')[1]))
            else:
                min_exp = np.inf
                text = '{} e0'.format(text)
            min_exp = min(min_exp, defaultparams['minimum exponent for '
                                                 + 'scientific notation'])
            single_value = True
            for (symbol, i0) in zip(['<', '>', '+', '-'], [0, 0, 0, 1]):
                if symbol in text[i0:]:
                    single_value = False
            if text in ['None', 'none', 'NaN', 'nan', 'inf', '-inf']:
                single_value = False
            if single_value:
                x, e = text.split(' ')
                dx = 0.
                text = '{}+/-{} {}'.format(x, dx, e)
            if text.startswith('<'):
                x = text.replace('<','').replace(' ','')
                dx1, dx2 = [np.nan]*2
                is_uplim = True
                is_lolim = False
            elif text.startswith('>'):
                x = text.replace('>','').replace(' ', '')
                dx1, dx2 = [np.nan]*2
                is_uplim = False
                is_lolim = True
            else:
                is_uplim, is_lolim = False, False
                text = (text.replace('+-', '+/-').replace(' -', '-')
                        .replace(' +/-', '+/-').replace('+/- ', '+/-'))
                x_dx, e = text.split(' ')
                if ')' in text:
                    if text.count(')') == 1:
                        x, dx = x_dx.split('(')
                        dx = dx.split(')')[0]
                        dx1 = dx2 = dx
                    else:
                        x, dx1, dx2 = x_dx.split('(')
                        dx1 = dx1[:-1]
                        dx2 = dx2[:-1]
                        if dx1.startswith('+'):
                            dx1, dx2 = dx2, dx1
                        dx1 = dx1[1:]
                        dx2 = dx2[1:]
                    d = len(x.split('.')[1]) if '.' in x else 0
                    dx1 = '{:f}'.format(float(dx1)*10**(-d))
                    dx2 = '{:f}'.format(float(dx2)*10**(-d))
                elif '+/-' in text:
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
                        if '+' not in x:
                            dx1 = text.split('-')[1].split('+')[0]
                            dx2 = text.split('+')[1].split(' ')[0]
                        else:
                            x = x.split('+')[0]
                            dx2 = text.split('+')[1].split('-')[0]
                            dx1 = text.split('-')[1].split(' ')[0]
                    else:
                        x = text.split(' ')[0]
                        dx1, dx2 = '0', '0'
                x = '{} {}'.format(x, e)
                dx1 = '{} {}'.format(dx1, e)
                dx2 = '{} {}'.format(dx2, e)
            x = parse_value(x)
            dx1 = parse_value(dx1)
            dx2 = parse_value(dx2)
            if not use_default_extra_sf_lim:
                if (not (is_lolim or is_uplim)
                        and not (eval(dx1) == eval(dx2) == 0)):
                    dx1_ = dx1.split('e')[0]
                    dx2_ = dx2.split('e')[0]
                    for i in reversed(range(len(dx1_))):
                        dx1_ = dx1_.replace('0.'+'0'*i, '')
                    dx1_ = dx1_.replace('.','')
                    for i in reversed(range(len(dx2_))):
                        dx2_ = dx2_.replace('0.'+'0'*i, '')
                    dx2_ = dx2_.replace('.','')
                    n1 = len(dx1_)
                    n2 = len(dx2_)
                    num_sf = max(1, n1, n2)
                    val = np.array([dx1, dx2])[np.argmax([n1, n2])]
                else:
                    x_ = x.split('e')[0]
                    for i in reversed(range(len(x_))):
                        x_ = x_.replace('0.'+'0'*i, '')
                    x_ = x_.replace('.','')
                    n = len(x_)
                    num_sf = n
                    val = x
                num_sf = max(1, num_sf)
                if eval(dx1) == eval(dx2) == 0:
                    extra_sf_lim = 1 - 1e-8
                else:
                    extra_sf_lim = default_extra_sf_lim 
                    base = float('{:e}'.format(eval(val)).split('e')[0])
                    if base <= default_extra_sf_lim:
                        if num_sf < default_num_sf + 1:
                            extra_sf_lim = base - 1e-8        
            else:
                extra_sf_lim = default_extra_sf_lim
            x = x.replace('e0','')
            main = eval(x)
            unc = [eval(dx1), eval(dx2)]
            is_range = False
            if domain is None and np.isfinite(main) and unc[0] == 0. == unc[1]:
                domain = [main]*2          
        else:
            text = text.replace(' --','--').replace('-- ','--')
            text1, text2 = text.split('--')
            x1, _, _, _, _, _, me1, el1 = parse_as_rich_value(text1)
            x2, _, _, _, _, _, me2, el2 = parse_as_rich_value(text2)
            main = [x1, x2]
            unc = 0
            is_lolim, is_uplim, is_range = False, False, True
            min_exp = round(np.mean([me1, me2]))
            extra_sf_lim = max(el1, el2)
        return (main, unc, is_lolim, is_uplim, is_range, domain,
                min_exp, extra_sf_lim)
    
    if pdf is None:
        text = str(text)
        is_complex = 'j' in text
        if not is_complex:
            (main, unc, is_lolim, is_uplim, is_range, domain,
                         min_exp, extra_sf_lim) = parse_as_rich_value(text)
            if input_domain is not None:
                domain = input_domain
            rvalue = RichValue(main, unc, is_lolim, is_uplim,
                               is_range, domain, is_int)
            rvalue.min_exp = min_exp
            rvalue.extra_sf_lim = extra_sf_lim
        else:
            if '+/-' in text or '-' in text[1:] and '+' in text[1:]:
                if ' + ' in text or ' - ' in text:
                    separator = ' + ' if ' + ' in text else ' - '
                    sign = '' if separator == ' + ' else '-'
                    text_real, text_imag = text.split(separator)
                    text_imag = (sign + text_imag)
                else:
                    text_real = '0'
                    text_imag = text.replace('- (','-(')
                text_imag = (text_imag.replace(' j','').replace('j','')
                             .replace('-(','(-'))
                if text_imag[0] == '(':
                    text_imag = text_imag[1:]
                if text_imag[-1] == ')':
                    text_imag = text_imag[:-1]
                args = (domain, is_int, pdf, use_default_extra_sf_lim)
                real = rich_value(text_real, *args)
                imag = rich_value(text_imag, *args)
            else:
                text = text.replace(' + ', '+').replace(' - ', '-')
                val = complex(text) 
                real = val.real
                imag = val.imag
            rvalue = ComplexRichValue(real, imag, domain, is_int)
    else:
        if 'function' in str(type(pdf)):
            pdf_ = pdf
        else:
            if type(pdf) is dict:
                x_, y_ = pdf['values'], pdf['probs']
            elif type(pdf) in (tuple, list, np.ndarray):
                x_, y_ = pdf[0], pdf[1]
            pdf_ = lambda x: np.interp(x, x_, y_, left=0., right=0.)
        domain = set_default_value(input_domain, 'domain')
        distr = sample_from_pdf(pdf_, size=4e4, low=domain[0], high=domain[1])
        if is_int:
            distr = np.round(distr).astype(int)
        rvalue = evaluate_distr(distr, domain)
        x1, x2 = rvalue.interval(4.)
        x = np.linspace(x1, x2, int(1e4))
        y = pdf_(x)
        norm = np.trapz(y, x)
        x = np.linspace(x1, x2, 400)
        y = pdf_(x)
        pdf_info = {'values': x, 'probs': y / norm}
        rvalue.pdf_info = pdf_info
        
    return rvalue

def rich_array(array, domain=None, is_int=None,
               use_default_extra_sf_lim=False):
    """
    Convert the input array to a rich array.

    Parameters
    ----------
    array : array / list (str)
        Input array containing text strings representing rich values.
    domain : list (float), optional
        The domain of all the entries of the rich array. If not specified,
        there are two possibilities: if the entry of the input array is already
        a rich value, its original domain will be preserved; if not, the
        default domain will be used, that is, [-np.inf, np.inf].
    is_int : bool, optional
        If True, the variable corresponding to the rich array will be an
        integer, so when creating samples it will have integer values.
    use_default_extra_sf_lim : bool, optional
        If True, the default limit for extra significant figure will be used
        instead of inferring it from the input text. This will reduce the
        computation time a little bit.

    Returns
    -------
    rarray : rich array
        Resulting rich array.
    """
    array = np.array(array)
    shape = array.shape
    mains, uncs, are_lolims, are_uplims, are_ranges, domains, are_ints = \
        [], [], [], [], [], [], []
    min_exps, extra_sf_lims, variables, expressions = [], [], [], []
    for entry in array.flat:
        if type(entry) in (RichValue, ComplexRichValue):
            if domain is not None:
                entry.domain = domain
            if is_int is not None:
                entry.is_int = is_int
        else:
            entry = rich_value(entry, domain, is_int, use_default_extra_sf_lim)
        mains += [entry.main]
        uncs += [entry.unc]
        are_lolims += [entry.is_lolim]
        are_uplims += [entry.is_uplim]
        are_ranges += [entry.is_range]
        domains += [entry.domain]
        are_ints += [entry.is_int]
        min_exps += [entry.min_exp]
        extra_sf_lims += [entry.extra_sf_lim]
        variables += [entry.variables]
        expressions += [entry.expression]
    mains = np.array(mains).reshape(shape)
    uncs = np.array(uncs)
    uncs = (np.array([uncs[:,0].reshape(shape).tolist(),
                      uncs[:,1].reshape(shape).tolist()])
            .transpose().reshape((*shape, 2)))
    are_lolims = np.array(are_lolims).reshape(shape)
    are_uplims = np.array(are_uplims).reshape(shape)
    are_ranges = np.array(are_ranges).reshape(shape)
    domains = np.array(domains)
    domains = (np.array([domains[:,0].reshape(shape).tolist(),
                         domains[:,1].reshape(shape).tolist()])
               .transpose().reshape((*shape, 2)))
    are_ints = np.array(are_ints).reshape(shape)
    rarray = RichArray(mains, uncs, are_lolims, are_uplims, are_ranges,
                       domains, are_ints, variables, expressions)
    min_exp = round(np.mean(min_exps))
    extra_sf_lim = max(extra_sf_lims)
    rarray.set_params({'min_exp': min_exp, 'extra_sf_lim': extra_sf_lim})
    return rarray

def rich_dataframe(df, domains=None, are_ints=None,
                   use_default_extra_sf_lim=False, **kwargs):
    """
    Convert the values of the input dataframe of text strings to rich values.

    Parameters
    ----------
    df : dataframe (str)
        Input dataframe which contains text strings formatted as rich values.
    domains : dict (list (float)), optional
        Dictionary containing the domain for each column of the dataframe.
        Instead, a common domain for all the columns can be directly specified.
        If not specified, there are two possibilities: if the entry of the
        input dataframe is already a rich value, its original domain will be
        preserved; if not, the default domain will be used, that is,
        [-np.inf, np.inf].
    domains : dict (list (float)), optional
        Dictionary containing the information of whether the rich values are
        correspond to integer variables for each column of the dataframe.
        Instead, a common value for all the columns can be directly specified.
        If not specified, there are two possibilities: if the entry of the
        input dataframe is already a rich value, its original domain will be
        preserved; if not, the default domain will be used, that is,
        [-np.inf, np.inf].
    use_default_extra_sf_lim : bool, optional
        If True, the default limit for extra significant figure will be used
        instead of inferring it from the input text. This will reduce the
        computation time a little bit.
    **kwargs : keyword arguments, optional
        Keyword arguments for the DataFrame class.

    Returns
    -------
    rdf : dataframe
        Resulting dataframe of rich values.
    """
    df = pd.DataFrame(df, **kwargs)
    if type(domains) is not dict:
        domains = {col: domains for col in df}
    if type(are_ints) is not dict:
        are_ints = {col: are_ints for col in df}
    df = copy.copy(df)
    for (i,row) in df.iterrows():
        for col in df:
            entry = df.at[i,col]
            is_rich_value = type(entry) in (RichValue, ComplexRichValue)
            domain = domains[col] if col in domains else None
            is_int = are_ints[col] if col in are_ints else None
            if is_rich_value:
                if domain is not None:
                    entry.domain = domain
                if is_int is not None:
                    entry.is_int = is_int
            else:
                is_number = True
                text = str(entry)
                for char in text.replace('e','').replace('j',''):
                    if char.isalpha():
                        is_number = False
                        break
                if is_number:
                    if domain is None:
                        domain = defaultparams['domain']
                    if is_int is None:
                        is_int = defaultparams['assume integers']
                    try:
                        entry = rich_value(text, domain, is_int, use_default_extra_sf_lim)
                    except:
                        entry = text
            if is_rich_value or is_number:
                df.at[i,col] = entry
    rdf = RichDataFrame(df)
    return rdf

def bounded_gaussian(x, m=0., s=1., a=np.inf):
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
        The default is np.inf (a usual gaussian function).

    Returns
    -------
    y : array (float)
        Resulting array.
    """
    sqrt_tau = 2.50662827
    if np.isfinite(a):
        x = np.array(x)
        y = np.zeros(x.size)
        cond = (x-m > -a) & (x-m < a)
        x_ = x[cond]
        x_m = a * np.arctanh((x_-m)/a)
        s_ = a * np.arctanh(s/a)
        y_ = np.exp(-0.5*(x_m/s_)**2) / (1 - ((x_-m)/a)**2)
        y_ /= s_ * sqrt_tau
        y[cond] = y_
        if x.shape == ():
            y = y[0]
    else:
        y = np.exp(-0.5 * ((x-m) / s)**2)
        y /= s * sqrt_tau
    return y

def general_pdf(x, loc=0, scale=1, bounds=[-np.inf,np.inf]):
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
        Boundaries of the independent variable.
        The default is [-np.inf, np.inf].

    Returns
    -------
    y : array (float)
        Resulting PDF for the input array.
    """
    
    def symmetric_general_pdf(x, m, s, a):
        """
        Symmetric general PDF with given median (m) and uncertainty (s) with a
        boundary mained in the median with a given amplitude (a).
        """
        if a > s:
            y = bounded_gaussian(x, m, s, a)
        else:
            raise Exception('Amplitude must be greater than uncertainty.')
        return y
    
    m, s, b = loc, scale, bounds
    x = np.array(x)
    
    if not b[0] < m < b[1]:
        raise Exception('Center ({}) is not inside the boundaries {}.'
                        .format(m, b))
    a = [m - b[0], b[1] - m]
    if not hasattr(s, '__iter__'):
        s = [s]*2
    s1, s2 = s
    a1, a2 = a
    
    if s1 == s2 and a1 == a2:
        y = symmetric_general_pdf(x, m, s1, a1)
    else:
        def correction(x, m, s, c, sign=1):
            """Correction to smooth the final asymmetric PDF."""
            n = 3/2
            g = np.zeros(x.size)
            cond = np.abs(x-m) < s
            x_ = (x[cond]-m)/s*(3/4)*math.tau
            g[cond] = - c * np.abs(np.cos(x_))**n * np.sign(x_)
            cond = (s/3 < np.abs(x-m)) & (np.abs(x-m) < s)
            x_ = (x[cond]-m)/s*(3/4)*math.tau
            g[cond] = c/2 * np.abs(np.cos(x_))**n * np.sign(x_)
            g *= sign
            return g
        y = np.zeros(x.size)
        cond1 = x < m
        cond2 = x > m
        y1 = symmetric_general_pdf(x[cond1], m, s1, a1)
        y2 = symmetric_general_pdf(x[cond2], m, s2, a2)
        h1 = symmetric_general_pdf(m, m, s1, a1)
        h2 = symmetric_general_pdf(m, m, s2, a2)
        h1_, h2_ = h1.copy(), h2.copy()
        h1, h2 = min(h1, h2), max(h1, h2)
        cond1_ = x[cond1] > m - s1
        cond2_ = x[cond2] < m + s2
        frac = 0.
        y_min = 0.
        lim = 1/6 * h1
        i = 0
        while y_min < lim:
            if i > 0:
                frac += 1/8
            c1 = frac * (h2 - h1)
            c2 = h2 - h1 - c1
            if h1_ < h2_:
                c1, c2 = -c2, c1
            x_ = np.linspace(m - s1, m + s2, int(1e3))
            cond1_ = x_ < m
            cond2_ = x_ > m
            y1_ = symmetric_general_pdf(x_[cond1_], m, s1, a1)
            y2_ = symmetric_general_pdf(x_[cond2_], m, s2, a2)
            y1_ -= correction(x_[cond1_], m, s1, c1, sign=1)
            y2_ += correction(x_[cond2_], m, s2, c2, sign=-1)
            y_min = np.min(np.append(y1_, y2_))
            i += 1
        y[cond1] = y1 - correction(x[cond1], m, s1, c1, sign=1)
        y[cond2] = y2 + correction(x[cond2], m, s2, c2, sign=-1)
        y[x==m] = h1 + (1-frac) * (h2 - h1)
        if x.shape == ():
            y = y[0]
    
    return y

def sample_from_pdf(pdf, size, low=None, high=None, **kwargs):
    """
    Return a sample of the distribution specified with the input function.

    Parameters
    ----------
    pdf : function
        Probability density function of the distribution.
    size : int
        Size of the sample.
    low, high : float, optional
        Minimum and maximum of the input values for the probability density
        function. If not specified, they will be estimated automatically.
    **kwargs : keyword arguments, optional
        Keyword arguments for the probability density function.

    Returns
    -------
    distr : array
        Sample of the distribution.
    """
    size = int(size)
    min_num_points = 12
    num_points = max(min_num_points, size)
    edit_low = low is None or np.isinf(low)
    edit_high = high is None or np.isinf(high)
    if edit_low or edit_high:
        x1 = -1e30 if edit_low else low
        x2 = 1e30 if edit_high else high
        x = symlogspace(x1, x2, int(4e4))
        y = pdf(x)
        x_ = x[y > 1e-8 * max(y[np.isfinite(y)])]
        x1, x2 = min(x_), max(x_)
        y = pdf(x)
        x_ = x[y > 1e-3 * max(y[np.isfinite(y)])]
        x1, x2 = min(x_), max(x_)
        if edit_low:
            low = x1
        if edit_high:
            high = x2
    x = np.random.uniform(low, high, num_points)
    y = pdf(x, **kwargs)
    y /= y.sum()
    distr = np.random.choice(x, p=y, size=size)
    return distr

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
        mained in the median and with semiwidth equal to 6 times the
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
        b = [m - 5*s, m + 5*s]
    if not b[0] < m < b[1]:
        raise Exception('main ({}) is not inside the boundaries {}.'
                        .format(m, b))
    if not hasattr(s, '__iter__'):
        s = [s, s]
    low = max(m-5*s[0], b[0])
    high = min(m+5*s[1], b[1])
    distr = sample_from_pdf(general_pdf, size, low, high,
                            loc=m, scale=s, bounds=b)
    return distr

def loguniform_distribution(low=-1, high=1, size=1,
                            zero_log=None, inf_log=None):
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
        Decimal logarithm of the minimum value in absolute value.
    inf_log : float, optional
        Decimal logarithm of the maximum value in absolute value.

    Returns
    -------
    distr : array (float)
        Resulting distribution of numeric values.
    """
    zero_log = set_default_value(zero_log, 'decimal exponent to define zero')
    inf_log = set_default_value(inf_log, 'decimal exponent to define infinity')
    x1, x2, N = low, high, size
    N_min = 10
    if N < N_min:
        distr_ = loguniform_distribution(x1, x2, N_min, zero_log, inf_log)
        p = np.random.uniform(size=N_min)
        p /= p.sum()
        distr = np.random.choice(distr_, p=p, size=N)
    else:
        if not x1 <= x2:
            raise Exception('Inferior limit must be lower than superior limit.')
        if np.isinf(x1):
            log_x1 = inf_log
        elif x1 == 0:
            log_x1 = zero_log
        else:
            log_x1 = _log10(abs(x1))
        if np.isinf(x2):
            log_x2 = inf_log
        elif x2 == 0:
            log_x2 = zero_log
        else:
            log_x2 = _log10(abs(x2))
        if log_x1 < zero_log:
            log_x1 = zero_log
        if log_x2 > inf_log: 
            log_x2 = inf_log
        if x1 < 0:
            if x2 <= 0:
                exps = np.linspace(log_x2, log_x1, N)
                distr = -10**exps
            else:
                exps = np.linspace(zero_log, log_x1, N)
                distr = -10**exps
                exps = np.linspace(zero_log, log_x2, N)
                distr = np.append(distr, 10**exps)
        else:
            exps = np.linspace(log_x1, log_x2, N)
            distr = 10**exps
        x1, x2 = distr[0], distr[-1]
        np.random.shuffle(distr)
        if len(distr) != N:
            distr = distr[:N-2]
            distr = np.append(distr, [x1, x2])
    return distr

def distr_with_rich_values(function, args, len_samples=None,
                           is_vectorizable=False, **kwargs):
    """
    Same as function_with_rich_values, but just returns the final distribution.
    """
    if type(args) not in (tuple, list):
        args = [args]
    args = [rich_value(arg) if type(arg) not in (RichValue, ComplexRichValue)
            else arg for arg in args]
    if type(function) is str:
        variables = list(np.concatenate(tuple(arg.variables for arg in args)))
        variables = list(dict.fromkeys(variables).keys())
        vars_str = ','.join(variables)
        expressions = [arg.expression for arg in args]
        function = function.replace('{}','({})')
        expression = function.format(*expressions).replace(' ', '')
        function = eval('lambda {}: {}'.format(vars_str, expression))
        args = [variable_dict[var] for var in variables]
    if len_samples is None:
        len_samples = int(len(args)**0.5 * defaultparams['size of samples'])
    args_distr = np.array([arg.sample(len_samples) for arg in args])
    distr = (function(*args_distr) if is_vectorizable else
            np.array([function(*args_distr[:,i]) for i in range(len_samples)]))
    return distr

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
        Size of the interval, in percentile, around the main value which
        defines the uncertainties. The default is 68.27 (1 sigma confidence
        interval).
    fraction : float, optional
        Fraction of the data that is used for the calculation, to exclude
        outliers. The default is 1.

    Returns
    -------
    main : float
        Central value of the distribution.
    uncs : tuple (float)
        Lower and upper uncertainties of the distribution.
    """
    distr = np.array(distr)
    distr = np.sort(distr[np.isfinite(distr)].flatten())
    size = len(distr)
    if fraction != 1 and size > 1:
        margin = (1 - fraction) / 2
        distr = distr[round(margin*size):round((1-margin)*size)]
    main = function(distr)
    ind = np.argmin(np.abs(distr - main))
    if hasattr(ind, '__iter__'):
        ind = int(np.median(ind))
    ind = 100 * ind / len(distr)
    perc1 = ind - interval/2
    perc2 = ind + interval/2
    if perc1 < 0:
        perc2 += abs(0 - perc1)
        perc1 = 0
    if perc2 > 100:
        perc1 -= abs(perc2 - 100)
        perc2 = 100
    unc1 = main - np.percentile(distr, perc1)
    unc2 = np.percentile(distr, perc2) - main
    if fraction != 1:
        unc1 *= 1 + margin
        unc2 *= 1 + margin
    uncs = [unc1, unc2]
    return main, uncs

# Pair of functions used when evaluating distributions.
def add_zero_infs(interval, zero_log, inf_log):
    """Add 0 and infinity to the input interval with the given threshold."""
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
def remove_zero_infs(interval, zero_log, inf_log):
    """Replace 0 and infinity for the given values in the input interval."""
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

def evaluate_distr(distr, domain=None, function=None, args=None,
            len_samples=None, is_vectorizable=False, consider_intervs=None,
            is_domain_cyclic=False, lims_fraction=None, num_reps_lims=None,
            save_pdf=None, **kwargs):
    """
    Interpret the given distribution as a rich value.

    Parameters
    ----------
    distr : list/array (float)
        Input distribution of values.
    domain : list (float), optional
        Domain of the variable represented by the distribution.
        The default is [-np.inf,np.inf].
    consider_intervs : bool, optional
        If True, the resulting distribution could be interpreted as an upper/
        lower limit or a constant range of values. The default is True.
    * The rest of the arguments are only used if the distribution was the
      result of a known function, and are the same as in the function
      'function_with_rich_values'.

    Returns
    -------
    rvalue : rich value
        Rich value representing the input distribution.
    """

    len_samples = set_default_value(len_samples, 'size of samples')
    lims_fraction = set_default_value(lims_fraction,
                        'fraction of the central value for upper/lower limits')
    num_reps_lims = set_default_value(num_reps_lims,
                        'number of repetitions to estimate upper/lower limits')
    save_pdf = set_default_value(save_pdf, 'save PDF in rich values')
    zero_log = defaultparams['decimal exponent to define zero']
    inf_log = defaultparams['decimal exponent to define infinity']

    if 'arguments' in kwargs:
        args = kwargs['arguments']
    if 'samples_length' in kwargs:
        len_samples = kwargs['samples_length']
    if 'samples_size' in kwargs:
        len_samples = kwargs['samples_size']
    
    def magnitude_order_range(interval, zero_log=zero_log):
        """Return the range in order of magnitude of the input interval."""
        x1, x2 = interval
        if abs(x1) < 10**zero_log:
            x1 = 0
        if abs(x2) < 10**zero_log:
            x2 = 0
        if x1*x2 > 0:
            d = _log10(x2-x1)
        elif x1*x2 < 0:
            d = _log10(abs(x1)) + 2*abs(zero_log) + _log10(x2)
        else:
            xlim = x1 if x2 == 0 else x2
            d = abs(zero_log) + _log10(abs(xlim))
        return d
   
    input_function = copy.copy(function)
    if type(function) is str:
        variables = list(np.concatenate(tuple(arg.variables for arg in args)))
        variables = list(dict.fromkeys(variables).keys())
        vars_str = ','.join(variables)
        expressions = [arg.expression for arg in args]
        function = function.replace('{}','({})')
        expression = function.format(*expressions).replace(' ', '')
        function = eval('lambda {}: {}'.format(vars_str, expression))
        args = [variable_dict[var] for var in variables]
    
    def repeat_functions(functions, initial_values, num_reps_lims=1):
        """Repeat the given functions to the input arguments."""
        num_vars = len(initial_values)
        all_vars = [[initial_values[j]] for j in range(num_vars)]
        for i in range(num_reps_lims):
            args_distr = np.array([arg.sample(len_samples)
                                   for arg in args])
            if is_vectorizable:
                distr = function(*args_distr)
            else:
                distr = np.array([function(*args_distr[:,j])
                                  for j in range(len_samples)])
            for j in range(num_vars):
                all_vars[j] += [functions[j](distr)]
        return tuple(all_vars)
    
    if args is not None:
        if type(args) is np.ndarray:
            args = [rich_value(arg) for arg in args]
        if type(args) not in (tuple, list):
            args = [args]
        if type(args[0]) is not RichArray:
            args = [rich_value(arg) if type(arg) not in
                    (RichValue, ComplexRichValue) else arg for arg in args]
    if consider_intervs is None:
        consider_intervs = True
        if args is not None and all([arg.is_centr for arg in args]):
            consider_intervs = False
        
    distr = np.array(distr)
    
    is_complex = 'complex' in str(distr.dtype)
    if is_complex:
        if type(input_function) is str:
            function_real = 'np.real({})'.format(input_function)
            function_imag = 'np.imag({})'.format(input_function)
        else:
            function_real = lambda x: np.real(function(x))
            function_imag = lambda x: np.imag(function(x))
        fargs =  (args, len_samples, is_vectorizable, consider_intervs,
                 is_domain_cyclic, lims_fraction, num_reps_lims, save_pdf)
        real = evaluate_distr(distr.real, domain, function_real, *fargs)
        imag = evaluate_distr(distr.imag, domain, function_imag, *fargs)
        rvalue = ComplexRichValue(real, imag)
        if type(input_function) is str:
            rvalue.variables = variables
            rvalue.expression = expression
        return rvalue
    
    original_size = distr.size
    distr = distr[np.isfinite(distr)].flatten()
    size = distr.size
    if size == 0:
        return RichValue(np.nan)
    else:
        finite_fraction = size / original_size
        if finite_fraction < 0.7:
            decimals = (0 if finite_fraction >= 0.01
                        else int(abs(np.floor(np.log10(100*finite_fraction)))))
            print('Warning: Valid values are only {:.{}f} % of'
                  ' the distribution.'.format(100*finite_fraction, decimals))

    distr_unique = np.unique(distr)
    is_exact = len(distr_unique) == 1
    if is_exact:
        value = distr_unique[0]
        main, unc = value, 0.
        domain = [value, value]
        consider_intervs = False
    elif domain is None:
        domain = [-np.inf, np.inf]
        
    is_int = np.array_equal(np.round(distr_unique), distr_unique)

    if is_int and not is_exact:
        q = 1e-3
        x1 = np.quantile(distr, q)
        x2 = np.quantile(distr, 1-q)
        is_range_small = (x2 - x1) < int(1e3)
        if is_range_small:
            bins = np.arange(x1, x2+2) - 0.5
            probs, _ = np.histogram(distr, bins=bins, density=True)
            values = np.arange(x1, x2+1)
            pdf = lambda x: np.interp(x, values, probs, left=0., right=0.)
            distr = sample_from_pdf(pdf, size=size, low=x1, high=x2)
            
    if not is_exact:
        main, unc = center_and_uncs(distr)
    
    if consider_intervs:
        x1, x2 = np.min(distr), np.max(distr)
        if is_int:
            x1 -= 1
            x2 += 1
        ord_range_1s = magnitude_order_range([main-unc[0], main+unc[1]])
        ord_range_x = magnitude_order_range([x1, x2])
        probs_hr, bins_hr = np.histogram(distr, bins=60, density=True)
        probs_lr, bins_lr = np.histogram(distr, bins=20, density=True)
        probs_hr /= probs_hr.max()
        probs_lr /= probs_lr.max()
        # plt.plot(np.mean([bins_hr[:-1], bins_hr[1:]], axis=0), probs_hr,'.')
        # plt.plot(np.mean([bins_lr[:-1], bins_lr[1:]], axis=0), probs_lr, '*')
        hr1f, hr2f, lrf, rf = 0.9, 0.8, 0.5, 0.4
        cond_hr1 = (probs_hr[0] > hr1f or probs_hr[-1] > hr1f)
        cond_hr2 = probs_hr[0] > hr2f
        cond_lr = lrf < probs_lr[0] < 1.
        cond_range = ord_range_x - ord_range_1s < rf if x1 != x2 else False
        cond_limit = cond_hr1 or cond_hr2
        cond_range = (cond_range or (not cond_range and cond_hr2)
                      or (not cond_range and not cond_hr2 and cond_lr))
        if cond_limit:
            if num_reps_lims > 0 and args is not None:
                xx1, xx2, xxc = repeat_functions([np.min, np.max, np.median],
                 [x1, x2, main], num_reps_lims)
                x1, x2 = np.min(xx1), np.max(xx2)
                main = np.median(xxc)
            ord_range_b1 = magnitude_order_range([x1, main])
            ord_range_b2 = magnitude_order_range([main, x2])
            x1, x2 = add_zero_infs([x1, x2], zero_log-6, inf_log-6)
            if (ord_range_b1 > inf_log-6 and ord_range_b2 > inf_log-6
                    and cond_hr2):
                main = np.nan
                unc = [np.inf, np.inf]
            else:
                if args is not None:
                    args_main = [arg.main for arg in args]
                    x0 = function(*args_main)
                    domain_ = add_zero_infs([domain[0], domain[1]],
                                            zero_log+6, inf_log-6)
                    x_ = RichValue([x1,x2], domain=domain_)
                    if x_.is_lolim:
                        x1 = x0 + (1 - lims_fraction) * (x1 - x0)
                    elif x_.is_uplim:
                        x2 = x0 - (1 - lims_fraction) * (x0 - x2)
                main = [x1, x2]
                unc = [0, 0]
        elif cond_range:
            if num_reps_lims > 0 and args is not None:
                xx1, xx2 = repeat_functions([np.min, np.max], [x1, x2],
                                            num_reps_lims)
                x1, x2 = np.median(xx1), np.median(xx2)
            x1, x2 = add_zero_infs([x1, x2], zero_log+6, inf_log-6)
            main = [x1, x2]
            unc = [0, 0]
    
    is_range = type(main) is list and np.isfinite(x2-x1)
    if is_range:
        dist = [main[0] - domain[0], domain[1] - main[1]]
        rel_dist = np.array(dist) / (main[1] - main[0])
        threshold = 0.01
        if rel_dist[0] < threshold:
            main[0] = domain[0]
        if rel_dist[1] < threshold:
            main[1] = domain[1]
    
    rvalue = RichValue(main, unc, domain=domain, is_int=is_int)
    
    if is_domain_cyclic:
        if domain is None:
            domain = np.min(distr), np.max(distr)
        period = domain[1] - domain[0]
        domain = [domain[0], domain[1] + period]
        distr_extended = np.concatenate((distr, distr + period))
        num_divisions = 8
        x1, x2 = distr_extended.min(), distr_extended.max() - period
        distrs, rvals, widths = [], [], []
        for j in range(num_divisions):
            x1j, x2j = np.array([x1, x2]) + j * period / num_divisions
            mask = (distr_extended >= x1j) & (distr_extended <= x2j)
            distr_j = distr_extended[mask]
            rval_j = evaluate_distr(distr_j, domain=domain)
            width_j = np.std(distr_j)
            widths += [width_j]
            rvals += [rval_j]
            distrs += [distr_j]
            # plt.hist(distr_j, label=str(j+1), bins=60, alpha=0.6)
        are_centrs = np.array([rvalue.is_centr for rvalue in rvals], bool)
        mask = np.isfinite(widths) & are_centrs
        idx = np.argmin(np.array(widths)[mask])
        rvalue = np.array(rvals)[mask][idx]
    
    if type(input_function) is str:
        rvalue.variables = variables
        rvalue.expression = expression
    
    if save_pdf:
        x1, x2 = rvalue.interval(3.)
        if is_int and is_range_small:
            bins = np.arange(x1, x2+2) - 0.5
        else:
            num_bins = max(80, size//200)
            bins = np.linspace(x1, x2, num_bins)
        probs, _ = np.histogram(distr, bins=bins, density=True)
        values = np.mean([bins[0:-1], bins[1:]], axis=0)
        pdf_info = {'values': values, 'probs': probs}
        rvalue.pdf_info = pdf_info
        
    return rvalue

def function_with_rich_values(function, args, unc_function=None,
        is_vectorizable=False, len_samples=None, optimize_len_samples=False,
        consider_intervs=None, domain=None, is_domain_cyclic=False,
        sigmas=None, use_sigma_combs=None, force_approx_propagation=False,
        lims_fraction=None, num_reps_lims=None, save_pdf=None, **kwargs):
    """
    Apply a function to the input rich values.

    Parameters
    ----------
    function : str / function
        Text string of the source code of the function to be applied to the
        input rich values. It should have empty ({}) brackets to indicate the
        position of the arguments in the same order as in 'args'. It can also
        be a Python function directly, but then correlation between variables
        will not be taken into account, and the mathematical expression of the
        resulting rich value will not be stored.
    args : list (rich values)
        List with the input rich values, in the same order as the arguments of
        the given function.
    unc_function : function, optional
        Function to estimate the uncertainties, in case that error propagation
        can be used. It should be the text string of the source code of the
        function to be applied to the input rich values. It will only be used
        if the arguments of the principal function ('args') are all independent
        variables. The arguments of 'unc_function' should be the central values
        first and then the uncertainties, with the same order as in the input
        function.
    is_vectorizable : bool, optional
        If True, the calculations of the function will be optimized making use
        of vectorization. It only works with functions that return only one
        output. The default is False.
    len_samples : int, optional
        Size of the samples of the arguments. The default is the number of
        arguments times the default size of samples (12000).
    optimize_len_samples : bool, optional
        If True, the samples size will be reduced up to the half if the minimum
        propagation score of the arguments is high enough. The default is
        False.
    consider_intervs : bool, optional
        If True, the resulting distribution could be interpreted as an upper/
        lower limit or a constant range of values. The default is None (it is
        False if all of the arguments are centered values).
    domain : list (float), optional
        Domain of the result. If not specified, it will be estimated.
    is_domain_cyclic : bool, optional
        If True, the domain of the result will be considered as cyclic.
        The default is False.
    sigmas : float, optional
        Threshold to apply uncertainty propagation. The value is the distance
        to the bounds of the domain relative to the uncertainty.
        The default is 10.
    use_sigma_combs : bool, optional
        If True, the calculation of the uncertainties will be optimized when
        the relative amplitudes are small and there is no uncertainty function
        provided. The default is False.
    force_approx_propagation : bool, optiona
        If True, approximate uncertainty propagation will be performed even if
        the relative uncertainty of the arguments is high their propagation
        score of the arguments is low.
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
    save_pdf : bool, optional
        If True and a distribution is created to perform the calculations,
        the obtained probability density function (PDF) will be stored into
        the rich value.

    Returns
    -------
    rvalue : rich value
        Resulting rich value.
    """
    
    sigmas = set_default_value(sigmas,
                        'sigmas to use approximate uncertainty propagation')
    use_sigma_combs = set_default_value(use_sigma_combs,
             'use 1-sigma combinations to approximate uncertainty propagation')
    zero_log = defaultparams['decimal exponent to define zero']
    inf_log = defaultparams['decimal exponent to define infinity']
    
    if 'arguments' in kwargs:
        args = kwargs['arguments']
    if 'samples_length' in kwargs:
        len_samples = kwargs['samples_length']
    if 'samples_size' in kwargs:
        len_samples = kwargs['samples_size']
    if 'uncertainty_function' in kwargs:
        unc_function = kwargs['uncertainty_function']
        
    def create_domain_distr(args, function):
        """Create a distribution of domains with the given function."""
        args_distr = np.array([loguniform_distribution(*arg.domain,
                                              len_samples//3) for arg in args])
        distr = (function(*args_distr) if is_vectorizable
                 else np.array([function(*args_distr[:,i])
                                for i in range(len_samples//3)]))
        if output_size == 1 and len(distr.shape) == 1:
            distr = np.array([distr]).transpose()
        return distr
        
    def evaluate_domain_distr(distr):
        """Obtain the domain from the given sample of possible domains."""
        domain1 = np.min(distr)
        domain2 = np.max(distr)
        if not np.isfinite(domain1):
            domain1 = -np.inf
        if not np.isfinite(domain2):
            domain2 = np.inf
        domain1, domain2 = remove_zero_infs([domain1, domain2],
                                            zero_log, inf_log)
        domain = [float(round_sf(domain1, num_sf+3)),
                  float(round_sf(domain2, num_sf+3))]
        domain = add_zero_infs(domain, zero_log+6, inf_log-6)
        return domain
    
    if type(args) not in (tuple, list):
        args = [args]
    args = [rich_value(arg) if type(arg) not in (RichValue, ComplexRichValue)
            else arg for arg in args]
    args_copy = copy.copy(args)
    
    input_function = copy.copy(function)
    if type(function) is str:
        variables = (list(np.concatenate(tuple(arg.variables for arg in args)))
                     if len(args) > 0 else [])
        variables = list(dict.fromkeys(variables).keys())
        vars_str = ','.join(variables)
        expressions = [arg.expression for arg in args]
        function = function.replace('{}','({})')
        expression = function.format(*expressions).replace(' ', '')
        function = eval('lambda {}: {}'.format(vars_str, expression))
        args = [variable_dict[var] for var in variables]
        if len(args) > 1:
            common_vars = set(args[0].variables)
            for i in range(len(args)-1):
                common_vars = common_vars & set(args[i+1].variables)
        else:
            common_vars = []
        if len(common_vars) > 0:
            unc_function = None
            use_sigma_combs = False

    if len(args) == 0:
        len_samples = set_default_value(len_samples, 'size of samples')
        main = function()
        output_size = np.array(main).size
        output_type = RichArray if type(main) is np.ndarray else type(main)
        output = []
        for k in range(output_size):
            distr = [(function() if output_size == 1 else function()[k])
                     for i in range(len_samples)]
            rvalue = evaluate_distr(distr, domain)
            output += [rvalue]
        if output_size == 1 and output_type is not list:
            output = output[0]
        elif output_type is tuple and output_size > 1:
            output = tuple(output)
        elif output_type is RichArray:
            output = np.array(output).view(RichArray)
        return output
            
    if len_samples is None:
        len_samples = int(len(args)**0.5 * defaultparams['size of samples'])
    num_sf = int(np.median([arg.num_sf for arg in args]))
    min_exp = round(np.mean([arg.min_exp for arg in args]))
    extra_sf_lim = max([arg.extra_sf_lim for arg in args])
    
    if consider_intervs is None:
        consider_intervs = (False if all([arg.is_centr for arg in args])
                            else True)
    use_analytic_propagation = (not any([arg.is_interv for arg in args])
                        and all([arg.prop_score > sigmas for arg in args]))
    if use_sigma_combs:
        if (unc_function is None and (((unc_function is None and len(args) > 5))
                 or any([arg.prop_score < sigmas for arg in args]))):
            use_analytic_propagation = False
    elif unc_function is None:
            use_analytic_propagation = False
    
    args_main = np.array([arg.main for arg in args])
    with np.errstate(divide='ignore', invalid='ignore'):
        main = function(*args_main)
    output_size = np.array(main).size
    output_type = RichArray if type(main) is np.ndarray else type(main)
    if is_vectorizable and output_size > 1:
        print("Warning: The argument 'is_vectorizable' only works with"
              "functions that return only one output.")

    if type(input_function) is str:
        if output_size == 1:
            expressions = [expression]
        else:
            if expression.startswith('['):
                bracket_count = 0
                text = expression[1:-1]
                text_list = list(text)
                for (i,char) in enumerate(text):
                    if char == '(':
                        bracket_count += 1
                    elif char == ')':
                        bracket_count -= 1
                    if bracket_count == 0 and char == ',':
                        text_list[i] = ';'
                text = ''.join(text_list)
                expressions = text.split(';')
                for (k,expr) in enumerate(expressions):
                    if expr[0] == '(' and expr[-1] == ')':
                        expressions[k] = expr[1:-1]
            else:
                expressions = ['({})[{}]'.format(expression,k)
                               for k in range(output_size)]

    if ((domain is None)
            or (domain is not None and not hasattr(domain[0], '__iter__'))):
        domain = [domain] * output_size
    if not hasattr(is_domain_cyclic, '__iter__'):
        is_domain_cyclic = [is_domain_cyclic] * output_size
    domains = domain
    are_domains_periodic = is_domain_cyclic

    size = min(100, len_samples//100)
    distr = distr_with_rich_values(function, args, size, is_vectorizable)
    if output_size == 1 and len(distr.shape) == 1:
        distr = np.array([distr]).transpose()
    are_real = ['complex' not in str(distr[:,k].dtype)
                for k in range(output_size)]
    
    if force_approx_propagation:
        use_analytic_propagation = True
        if any(are_domains_periodic) and unc_function is None:
            use_analytic_propagation = False
            print('Warning: Approximate propagation cannot be performed, '
                   'using distributions instead.')
    
    if any([element is None for element in np.array(domain).flat]):
        are_args_real = any([type(arg) is RichValue for arg in args])
        if are_args_real:
            domain_distr = create_domain_distr(args, function)
        else:
            args_real = [arg.real for arg in args]
            args_imag = [arg.imag for arg in args]
            domain_distr_real = create_domain_distr(args_real, function)
            domain_distr_imag = create_domain_distr(args_imag, function)
            domain_distr = domain_distr_real + 1j*domain_distr_imag
            
    for k in range(output_size):
        if domains[k] is not None:
            domain_k = domains[k]
        else:
            if are_real[k]:
                domain_k = evaluate_domain_distr(domain_distr[:,k])
            else:
                domain_real = evaluate_domain_distr(domain_distr[:,k].real)
                domain_imag = evaluate_domain_distr(domain_distr[:,k].imag)
                domain_k = [min(domain_real[0], domain_imag[0]),
                            max(domain_real[1], domain_imag[1])]
            if are_domains_periodic[k]:
                period = (domain_k[1] - domain_k[0]) / 2
                domain_k = [domain_k[0] - period, domain_k[1] + period]
        domains[k] = domain_k
    
    if use_analytic_propagation:
        
        mains = [main] if output_size == 1 else main
        if unc_function is not None:
            uncs = []
            args_main = np.array([arg.main for arg in args_copy])
            for i in (0,1):
                args_unc = [arg.unc[i] for arg in args_copy]
                uncs += [unc_function(*args_main, *args_unc)]
            uncs = [uncs] if output_size == 1 else uncs
            for k in range(output_size):
                uncs[k][1] = abs(uncs[k][1])
        else:
            inds_combs = list(itertools.product(*[[0,1,2]]*len(args)))
            comb_main = tuple([1]*len(args))
            inds_combs.remove(comb_main)
            args_combs = []
            args_all_vals = [[arg.main - arg.unc[0], arg.main,
                              arg.main + arg.unc[1]] for arg in args]
            for (i,inds) in enumerate(inds_combs):
                args_combs += [[]]
                for (j,arg) in enumerate(args):
                    args_combs[i] += [args_all_vals[j][inds[j]]]
            combinations = [function(*args_comb) for args_comb in args_combs]
            uncs = [[mains[k] - np.min(combinations),
                     np.max(combinations) - mains[k]]
                    for k in range(output_size)]
        output = []
        for k in range(output_size):
            is_real = 'complex' not in str(type(mains[k]))
            main_k = mains[k]
            unc_k = np.array(uncs[k])
            domain_k = domains[k]
            if is_real:
                rval_k = RichValue(main_k, unc_k, domain=domain_k)
            else:
                real_k = RichValue(main_k.real, unc_k.real, domain=domain_k)
                imag_k = RichValue(main_k.imag, unc_k.imag, domain=domain_k)
                rval_k = ComplexRichValue(real_k, imag_k)
            rval_k.num_sf = num_sf
            rval_k.min_exp = min_exp
            rval_k.extra_sf_lim = extra_sf_lim
            output += [rval_k]
            
    else:
        
        if optimize_len_samples:
            prop_score = min([arg.prop_score for arg in args])
            lim1, lim2 = 4., 20.
            if prop_score > lim1:
                factor = 1. - 0.5 * (min(prop_score,20.)-lim1) / (lim2-lim1)
                len_samples = int(factor * len_samples)
        distr = distr_with_rich_values(function, args, len_samples,
                                       is_vectorizable)
        if output_size == 1 and len(distr.shape) == 1:
            distr = np.array([distr]).transpose()
        output = []
        for k in range(output_size):
            def function_k(*argsk):
                y = function(*argsk)
                if output_size > 1:
                    y = y[k]
                return y
            rval_k = evaluate_distr(distr[:,k], domains[k], function_k, args,
                          len_samples, is_vectorizable, consider_intervs,
                          are_domains_periodic[k], lims_fraction,
                          num_reps_lims, save_pdf)
            rval_k.num_sf = num_sf
            rval_k.min_exp = min_exp
            rval_k.extra_sf_lim = extra_sf_lim
            if type(input_function) is str:
                if type(rval_k) is RichValue:
                    rval_k.variables = variables
                    rval_k.expression = expressions[k]
                else:
                    rval_k.real.variables = variables
                    rval_k.imag.variables = variables
                    rval_k.real.expression = 'np.real({})'.format(expression)
                    rval_k.imag.expression = 'np.imag({})'.format(expression)
            output += [rval_k]
        
    if output_size == 1 and output_type is not list:
        output = output[0]
    elif output_type is tuple and output_size > 1:
        output = tuple(output)
    elif output_type is RichArray:
        output = np.array(output).view(RichArray)
            
    return output

def function_with_rich_arrays(function, args, elementwise=False, **kwargs):
    """
    Apply a function to the input rich arrays.
    (abbreviation: array_function)

    Parameters
    ----------
    function : str / function
        Text string of the source code of the function to be applied to the
        input rich values. It should have empty ({}) brackets to indicate the
        position of the arguments in the same order as in 'args'. It can also
        be a Python function directly, but then correlation between variables
        will not be taken into account, and the mathematical expression of the
        resulting rich value will not be stored.
    args : list (rich arrays)
        List with the input rich arrays, in the same order as the arguments of
        the given function.
    elementwise : bool, optional
        If True, the function will be aplied to the input arrays element by
        element.
    * The rest of the arguments are the same as in 'function_with_rich_values'.
    
    Returns
    -------
    output : rich array / rich value
        Result of the function.
    """
    if 'domain' in kwargs:
        domain = kwargs['domain']
        del kwargs['domain']
    else:
        domain = None
    distr = distr_with_rich_arrays(function, args, elementwise, **kwargs)
    if len(distr.shape) == 1:
        output = evaluate_distr(distr, domain, function, args, **kwargs)
    else:
        output_size = distr.shape[1]
        output = []
        for k in range(output_size):
            function_k = lambda *args: function(args)[k]
            rval_k = evaluate_distr(distr[:,k], domain, function_k, **kwargs)
            output += [rval_k]
        if type(args) not in (tuple, list):
            args = [args]
        for (i,arg) in enumerate(args):
            if type(arg) is not RichArray:
                args[i] = RichArray(arg)
        args_mains = [arg.mains for arg in args]
        if type(function) is not str:
            with np.errstate(divide='ignore', invalid='ignore'):
                main = function(*args_mains)
        else:
            main = distr[0,:]
        output_type = RichArray if type(main) is np.ndarray else type(main)
        if output_type is tuple and output_size > 1:
            output = tuple(output)
        elif output_type is RichArray:
            output = np.array(output).view(RichArray)
    return output

def distr_with_rich_arrays(function, args, elementwise=False, **kwargs):
    """
    Same as function_with_rich_arrays, but just returns the final distribution.
    """
    if type(args) not in (tuple, list):
        args = [args]
    args = [rich_array(arg) if type(arg) != RichArray else arg
            for arg in args]
    if elementwise and (len(args) > 0
            and type(args[0]) is str or not hasattr(args[0], '__iter__')):
        args = [rich_array(args)]
        shape = args[0].shape
    else:
        args = [rich_array(arg) if type(arg) != RichArray else arg
                for arg in args]
        shape = args[0].shape
    if 'len_samples' not in kwargs:
        kwargs['len_samples'] = int(len(args)**0.5
                                    * defaultparams['size of samples'])
    if 'consider_intervs' in kwargs:
        del kwargs['consider_intervs']
    if elementwise:
        same_shapes = True
        for arg in args[1:]:
            if arg.shape != shape:
                same_shapes = False
                break
        if not same_shapes:
            raise Exception('Input arrays have different shapes.')
        distr = []
        args_flat = np.array([arg.flatten() for arg in args])
        for i in range(args[0].size):
            args_i = np.array(args_flat)[:,i].tolist()
            distr_i = distr_with_rich_values(function, args_i, **kwargs)
            distr += [distr_i]
        distr = np.array(distr).T
        output = distr
    else:
        if type(function) is str:
            variables = list(np.concatenate(tuple(arg.variables for arg in args)))
            variables = list(set(variables))
            expressions = [arg.expression for arg in args]
            function = function.replace('{}','({})')
            expression = function.format(*expressions).replace(' ', '') + ' '
            var_sizes = np.array([len(var) for var in variables])
            inds = np.argsort(var_sizes)
            variables_sorted = np.array(variables)[inds][::-1]
            alt_args, var, possible_var = [], '', False
            for char in expression:
                if char == 'x':
                    possible_var = True
                elif possible_var and not char.isdigit():
                    possible_var = False
                    if len(var) > 1:
                        alt_args += [variable_dict[var]]
                    var = ''
                if possible_var:
                    var += char
            for var in variables_sorted:
                if var in expression:
                    expression = expression.replace(var, '{}')
            alt_function = expression[:-1]
        else:
            num_args = len(args)
            arg_sizes = [arg.size for arg in args]
            arg_shapes = [arg.shape for arg in args]
            inds = [0, *np.cumsum(arg_sizes)]
            def alt_function(*argsf):
                rec_args = []
                for i in range(num_args):
                    arg_i = argsf[inds[i]:inds[i+1]]
                    arg_i = np.array(arg_i).reshape(arg_shapes[i])
                    rec_args += [arg_i]
                y = function(*rec_args)
                return y
            alt_args = []
            for arg in args:
                alt_args += list(arg.flat)
        output = distr_with_rich_values(alt_function, alt_args, **kwargs)
    return output

def fmean(array, function='None', inverse_function='None',
          weights=None, weight_function=None, **kwargs):
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
    if function != 'None' and inverse_function in ('None', None):
        raise Exception('Inverse function not specified.')
    if type(array) is not RichArray:
        array = rich_array(array)
    if function == 'None' and weights is None:
        expression = 'np.mean({})'.format(array.expression)
    elif function == 'None' and weights is not None and weight_function is None:
        expression = 'np.average({}, weights={})'.format(array.expression,
                                                rich_array(weights).expression)
    else:
        expression = None
    if function == 'None':
        function = lambda x: x
        inverse_function = lambda x: x
    if weight_function is None:
        weight_function = lambda x: x
    if weights is None:
        weights = np.ones(len(array))
    weights = rich_array(weights, domain=[0,np.inf])
    def fmean_function(x,w):
        x_f = function(x)
        w_f = weight_function(w)
        w_f /= sum(w_f)
        y = inverse_function(np.sum(x_f * w_f))
        return y
    y = function_with_rich_arrays(fmean_function, [array, weights], **kwargs)
    if expression is not None:
        y.expression = expression
    return y

def mean(array, weights=None, weight_function=lambda x: x, **kwargs):
    """Arithmetic mean of the input values."""
    mean = fmean(array, weights=weights, weight_function=weight_function,
                 **kwargs)
    return mean

def errorbar(x, y, lims_factor=None, **kwargs):
    """
    Plot the input rich arrays (y versus x) with Matplotlib.

    Parameters
    ----------
    x : rich array
        Variable to be plotted on the horizontal axis.
    y : rich array
        Variable to be plotted on the vertical axis.
    lims_factor : list / float, optional
        List containing the factors that define the sizes of the arrows for
        displaying the upper/lower limits. By default it will be calculated
        automatically.
    kwargs : arguments, optional
        Matplotlib's 'errorbar' keyword arguments.

    Returns
    -------
    plot : matplotlib.container.ErrorbarContainer
        Matplotib's 'errorbar' output.
    """
    def set_kwarg(keyword, default):
        """Set a certain keyword argument with a default value."""
        if keyword in kwargs:
            kwarg = kwargs[keyword]
            del kwargs[keyword]
        else:
            kwarg = default
        return kwarg
    def lim_factor(x):
        xc = np.sort(x.mains)
        xc = xc[np.isfinite(xc)]
        if len(xc) > 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                try:
                    r = abs(linregress(xc, np.arange(len(xc))).rvalue)
                except:
                    r = 0
        else:
            r = 0
        factor = 2. + 12.*r**8
        return factor
    xa, ya = rich_array(x), rich_array(y)
    xc = rich_array([x]) if len(xa.shape) == 0 else xa
    yc = rich_array([y]) if len(ya.shape) == 0 else ya
    if type(lims_factor) in (float, int):
        lims_factor_x, lims_factor_y = [lims_factor]*2
    elif type(lims_factor) in (list, tuple):
        lims_factor_x, lims_factor_y = lims_factor
    else:
        lims_factor_x, lims_factor_y = None, None
    if lims_factor_x is None:
        lims_factor_x = lim_factor(xc)
    if lims_factor_y is None:
        lims_factor_y = lim_factor(yc)
    xc.set_lims_uncs(lims_factor_x)
    yc.set_lims_uncs(lims_factor_y)
    plt.plot()
    ax = plt.gca()
    color = ax.plot([])[0].get_color()
    color = set_kwarg('color', color)
    ecolor = set_kwarg('ecolor', 'black')
    fmt = set_kwarg('fmt', '.')
    cond = ~ (xc.are_ranges | yc.are_ranges)
    plot = plt.errorbar(xc.mains[cond], yc.mains[cond],
                xerr=xc.uncs_eb[:,cond], yerr=yc.uncs_eb[:,cond],
                uplims=yc.are_uplims[cond], lolims=yc.are_lolims[cond],
                xlolims=xc.are_lolims[cond], xuplims=xc.are_uplims[cond],
                color=color, ecolor=ecolor, fmt=fmt, **kwargs)
    cond = xc.are_ranges
    for (xi, yi) in zip(xc, yc):
        if xi.is_range & ~xi.is_lim:
            plt.errorbar(xi.main, yi.main, xerr=xi.unc_eb,
                         uplims=yi.is_uplim, lolims=yi.is_uplim,
                         fmt=fmt, color='None', ecolor=ecolor, **kwargs)
            for xij in yi.interval():
                plt.errorbar(xij, yi.main, xerr=xi.unc_eb, fmt=fmt,
                             color='None', ecolor=ecolor, **kwargs)
    cond = yc.are_ranges
    for (xi, yi) in zip(xc[cond], yc[cond]):
        if yi.is_range:
            plt.errorbar(xi.main, yi.main, yerr=yi.unc_eb,
                         xuplims=xi.is_uplim, xlolims=xi.is_uplim,
                         fmt=fmt, color='None', ecolor=ecolor, **kwargs)
            for yij in yi.interval():
                plt.errorbar(xi.main, yij, xerr=xi.unc_eb, fmt=fmt,
                             color='None', ecolor=ecolor, **kwargs)
    return plot

def curve_fit(x, y, function, guess, num_samples=3000,
              loss=lambda a,b: (a-b)**2, lim_loss_factor=4.,
              consider_arg_intervs=False, consider_param_intervs=True,
              use_easy_sampling=False, **kwargs):
    """
    Perform a fit of y over x with respect to the given function.

    Parameters
    ----------
    x : rich array
        Independent variable.
    y : rich array
        Dependent variable.
    function : function
        Function to be optimized, that is, the function of y with respect to x.
        It has to contain as arguments the independent variable (x) and the
        parameters to be optimized.
    guess : list (float)
        List of initial values of the arguments of the function.
    num_samples : int, optional
        Number of different samples of the input data used for calculating the
        parameter distributions. The default is 3000.
    loss : function, optional
        Function that defines the error between two numbers: a sample of a rich
        value (first argument) and a numeric prediction of it (second
        argument). The default is the squared error.
    lim_loss_factor : float, optional
        Factor to enlarge the loss if the rich value is not a centered value
        and the prediction falls outside the interval of possible values of the
        rich value. The default is 4.
    consider_arg_intervs : bool, optional
        If True, upper/lower limits and constant ranges of values in the input
        data will be taken into account during the fit. This option increases
        considerably the computation time. The default is False.
    consider_param_intervs : bool, optional
        If True, upper/lower limits and constant ranges of values will be taken
        into account for calculating the fit parameters. The default is True.
    use_easy_sampling : bool, optional.
        If True, upper/lower limits and constant ranges of values will be
        sampled as usual, with uniform distributions for finite intervals of
        values and lognormal distributions for infinite intervals. If False,
        intervals will not be sampled for the fitting itself, but will be taken
        into account for calculating the loss function for the fit.
        The default is False.
    **kwargs : arguments
        Keyword arguments of SciPy's function 'minimize'.

    Returns
    -------
    result : dict
        Dictionary containing the following entries:
        - parameters : list (rich value)
            List containing the optimized values for the parameters.
        - dispersion : rich value
            Estimated real dispersion between the model and the fitted data.
        - loss : rich value
            Final mean loss between the original points and the modeled ones.
        - parameters samples : array (float)
            Array containing the samples of the fitted parameters used to
            compute the rich values. Its shape is (num_samples, num_params),
            with num_params being the number of parameters to be fitted.
        - dispersion sample : array (float)
            Sample of the calculated real dispersion of the points with respect
            to the model.
        - loss sample : array (float)
            Array containing the loss between the original data and each group
            of fitted parameters in the 'parameters samples' entry.
        - number of fails : int
            Number of times that the fit failed, for the iterations among the
            different samples (the number of iterations is num_samples).
    """
    if len(x) != len(y):
        raise Exception('Input arrays have not the same size.')
    num_points = len(y)
    xa, ya = rich_array(x), rich_array(y)
    x = rich_array([x]) if len(xa.shape) == 0 else xa
    y = rich_array([y]) if len(ya.shape) == 0 else ya
    if not hasattr(guess, '__iter__'):
        guess = [guess]
    num_params = len(guess)
    condx = x.are_centrs
    condy = y[condx].are_centrs
    if use_easy_sampling and consider_arg_intervs or sum(condx) == 0:
        condx = np.ones(num_points, bool)
        condy = np.ones(num_points, bool)
    num_intervs_x = (~condx).sum()
    num_intervs_y = (~condy).sum()
    num_lim_samples = 8
    xlims_sample = np.append(x[~condx].sample(num_lim_samples),
                             x[~condx].intervals().T, axis=0).T
    ylims_sample = np.append(y[~condx].sample(num_lim_samples),
                             y[~condx].intervals().T, axis=0).T
    def loss_function(params, xs, ys):
        y_condx = np.array(function(xs[condx], *params))
        error = sum(loss(ys[condx][condy], y_condx[condy]))
        if consider_arg_intervs:
            y_condxy = y_condx[~condy]
            if num_intervs_y > 0:
                ylims = np.empty(num_intervs_y)
                for j, (yj, y_j) in enumerate(zip(y[condx][~condy], y_condxy)):
                    y1, y2 = yj.interval()
                    yl = (y_j if y1 <= y_j <= y2
                          else [y1, y2][np.argmin([abs(y1-y_j), abs(y2-y_j)])])
                    ylims[j] = yl
                    factor = 1. if y1 <= y_j <= y2 else lim_loss_factor
                error += sum(factor*loss(ylims, y_condxy))
            if num_intervs_x > 0:
                for (xi, yi) in zip(xlims_sample, ylims_sample):
                    yi_ = [function(xij, *params) for xij in xi]
                    y1, y2 = min(yi_), max(yi_)
                    error_i = []
                    for yij in yi:
                        yl = (yij if y1 <= yij <= y2 else
                              [y1, y2][np.argmin([abs(y1-yij),abs(y2-yij)])])
                        factor = 1. if y1 <= yij <= y2 else lim_loss_factor
                        error_ij = factor*loss(yl, yij)
                        error_i += [error_ij]
                        if error_ij == 0:
                            break
                    error += min(error_i)
        error /= num_points
        return error
    losses, dispersions = [], []
    samples = [[] for i in range(num_params)]
    print('Fitting...')
    num_fails = 0
    x_sample = x.sample(num_samples)
    y_sample = y.sample(num_samples)
    cond = x.are_centrs & y.are_centrs
    num_disp_points = cond.sum()
    for (i, (xs, ys)) in enumerate(zip(x_sample, y_sample)):
        result = minimize(loss_function, guess, args=(xs,ys), **kwargs)
        if result.success:
            params_i = result.x
            for j in range(num_params):
                samples[j] += [params_i[j]]
            if num_disp_points > 0:
                ys_cond = function(xs[cond], *params_i)
                dispersions += [(np.sum((ys[cond] - ys_cond)**2)
                                / (num_disp_points - 1))**0.5]
            losses += [result.fun]
        else:
            num_fails += 1
        if ((i+1) % (num_samples//4)) == 0:
            print('  {} %'.format(100*(i+1)//num_samples))
    if num_fails > 0.9*num_samples:
        raise Exception('The fit failed more than 90 % of the time.')
    params_fit = [evaluate_distr(samples[i],
           consider_intervs=consider_param_intervs) for i in range(num_params)]
    if num_disp_points > 0:
        mean_unc = y[cond].uncs.mean()
        dispersions = np.array(dispersions)
        dispersions1 = np.maximum(0, dispersions - mean_unc)
        dispersions2 = np.maximum(0, dispersions**2 - mean_unc**2)**0.5
        disp_coef = np.median(dispersions1) / mean_unc
        lim1, lim2 = 0.2, 1.2
        if disp_coef <= lim1:
            frac1 = 1.
        elif disp_coef < lim2: 
            frac1 = 1. - disp_coef / (lim2 - lim1)
        else:
            frac1 = 0.
        dispersions = frac1 * dispersions1 + (1-frac1) * dispersions2
    dispersion = evaluate_distr(dispersions, domain=[0,np.inf],
                                consider_intervs=False)
    losses = np.array(losses)
    loss = evaluate_distr(losses, consider_intervs=False)
    samples = np.array(samples).transpose()
    result = {'parameters': params_fit, 'dispersion': dispersion, 'loss': loss,
              'parameters samples': samples, 'dispersion sample': dispersions,
              'loss sample': losses, 'number of fails': num_fails}
    return result   

def point_fit(y, function, guess, num_samples=3000,
              loss=lambda a,b: (a-b)**2, lim_loss_factor=4.,
              consider_arg_intervs=True, consider_param_intervs=True,
              use_easy_sampling=False, **kwargs):
    """
    Perform a fit of the input points (y) with respect to the given function.

    The parameters and the outputs are the same as in the 'curve_fit' function.
    """
    ya = rich_array(y)
    y = rich_array([y]) if len(ya.shape) == 0 else ya
    num_points = len(y)
    if not hasattr(guess, '__iter__'):
        guess = [guess]
    example_pred = np.array(function(*guess))
    function_copy = copy.copy(function)
    if len(example_pred.shape) == 0 or len(example_pred) != num_points:
        function = lambda *params: [function_copy(*params)]*num_points
    num_params = len(guess)
    cond = y.are_centrs
    if use_easy_sampling and consider_arg_intervs or sum(cond) == 0:
        cond = np.ones(num_points, bool)
    num_intervs = (~cond).sum()
    def loss_function(params, ys):
        y_ = np.array(function(*params))
        error = sum(loss(ys[cond], y_[cond]))
        y_cond = y_[~cond]
        if num_intervs > 0:
            ylims = np.empty(num_intervs)
            for (j, (yj, y_j)) in enumerate(zip(y[~cond], y_cond)):
                y1, y2 = yj.interval()
                yl = (y_j if y1 <= y_j <= y2
                      else [y1, y2][np.argmin([abs(y1-y_j), abs(y2-y_j)])])
                ylims[j] = yl
                factor = 1. if y1 <= y_j <= y2 else lim_loss_factor
            error += sum(factor*loss(ylims, y_cond))
        error /= len(ys)
        return error
    losses, dispersions = [], []
    samples = [[] for i in range(num_params)]
    print('Fitting...')
    num_fails = 0
    y_sample = y.sample(num_samples)
    cond = y.are_centrs
    num_disp_points = cond.sum()
    for (i,ys) in enumerate(y_sample):
        result = minimize(loss_function, guess, args=ys, **kwargs)
        if result.success:
            params_i = result.x
            for j in range(num_params):
                samples[j] += [params_i[j]]
            if num_disp_points > 0:
                ys_cond = function(*params_i)
                dispersions += [(np.sum((ys[cond] - ys_cond)**2)
                                / (num_disp_points - 1))**0.5]
            losses += [result.fun]
        else:
            num_fails += 1
        if ((i+1) % (num_samples//4)) == 0:
            print('  {} %'.format(100*(i+1)//num_samples))
    if num_fails > 0.9*num_samples:
        raise Exception('The fit failed more than 90 % of the time.')
    params_fit = [evaluate_distr(samples[i],
           consider_intervs=consider_param_intervs) for i in range(num_params)]
    if num_disp_points > 0:
        mean_unc = y[cond].uncs.mean()
        dispersions = np.array(dispersions)
        dispersions1 = np.maximum(0, dispersions - mean_unc)
        dispersions2 = np.maximum(0, dispersions**2 - mean_unc**2)**0.5
        disp_coef = np.median(dispersions1) / mean_unc
        lim1, lim2 = 0.2, 1.2
        if disp_coef <= lim1:
            frac1 = 1.
        elif disp_coef < lim2: 
            frac1 = 1. - disp_coef / (lim2 - lim1)
        else:
            frac1 = 0.
        dispersions = frac1 * dispersions1 + (1-frac1) * dispersions2
    dispersion = evaluate_distr(dispersions, domain=[0,np.inf],
                                consider_intervs=False)
    losses = np.array(losses)
    loss = evaluate_distr(losses, consider_intervs=False)
    samples = np.array(samples).transpose()
    result = {'parameters': params_fit, 'dispersion': dispersion, 'loss': loss,
              'parameters samples': samples, 'dispersion sample': dispersions,
              'loss sample': losses, 'number of fails': num_fails}
    return result    

def edit_pdf_info(pdf_info, function):
    """Modify input PDF for a rich value with the given function."""
    pdf_info = copy.copy(pdf_info)
    pdf_info['values'] = function(pdf_info['values'])
    return pdf_info

def symlogspace(x1, x2, size, zero_lim=None):
    """
    Create an array of values in a symmetrical logarithmic scale.

    Parameters
    ----------
    x1, x2 : float
        Inferior and superior limits.
    size: int
        Size of the desired array.
    zero_lim : float, optional
        Limit value to define zero. The default is 1e-90.

    Returns
    -------
    x: array
        Resulting array.

    """
    if zero_lim is None:
        zero_log = defaultparams['decimal exponent to define zero']
        zero_lim = 10**zero_log
    else:
        zero_log = np.log10(abs(zero_lim))
    if x1 > x2:
        raise Exception('Inferior limit must be lower than superior limit.')
    sign1 = np.sign(x1) if x1 != 0. else 1
    if abs(x1) < zero_lim:
        x1 = sign1 * zero_lim
    sign2 = np.sign(x2) if x2 != 0. else 0
    if abs(x2) < zero_lim:
        x2 = sign2 * zero_lim
    if np.sign(x1) * np.sign(x2) > 0:
        if any(np.sign([x1, x2])) > 0:
            x = np.logspace(np.log10(x1), np.log10(x2), size)
        else:
            x = -np.logspace(np.log10(abs(x2)), _log10(abs(x1)), size)[::-1]
    else:
        _x = -np.logspace(zero_log, _log10(abs(x1)), size)[::-1]
        x_ = np.logspace(zero_log, _log10(x2), size)
        x = np.concatenate((_x, x_))
    return x

def _log10(x):
    """Decimal logarithm from NumPy but including x = 0."""
    with np.errstate(divide='ignore'):
        y = np.log10(x)
    return y

# Abbreviations from NumPy.
inf = np.inf
nan = np.nan

# Functions for masking arrays.
def isnan(x):
    x = rich_array(x) if type(x) is not RichArray else x
    return np.array([xi.is_nan for xi in x.flat]).reshape(x.shape)
def isinf(x):
    x = rich_array(x) if type(x) is not RichArray else x
    return np.array([xi.is_inf for xi in x.flat]).reshape(x.shape)
def isfinite(x):
    x = rich_array(x) if type(x) is not RichArray else x
    return np.array([xi.is_finite for xi in x.flat]).reshape(x.shape)

# Functions for concatenating arrays.
def append(arr, values, **kwargs):
    return np.append(arr, values, **kwargs).view(RichArray)
def concatenate(arrs, **kwargs):
    return np.concatenate(arrs, **kwargs).view(RichArray)

# Mathematical functions.
get_operation_function = lambda x: (function if type(x) in
                        (str, RichValue, ComplexRichValue) else array_function)
def sqrt(x):
    function_ = get_operation_function(x)
    return function_('np.sqrt({})', x, domain=[0,np.inf], elementwise=True,
                     unc_function= lambda x,dx: dx/x**0.5 / 2)
def exp(x):
    function_ = get_operation_function(x)
    return function_('np.exp({})', x, domain=[0,np.inf], elementwise=True,
                     unc_function= lambda x,dx: dx*np.exp(x))
def log(x):
    function_ = get_operation_function(x)
    return function_('np.log({})', x, domain=[-np.inf,np.inf],
                     unc_function= lambda x,dx: dx/x, elementwise=True)
def log10(x):
    function_ = get_operation_function(x)
    return function_('np.log10({})', x, domain=[-np.inf,np.inf],
                unc_function= lambda x,dx: dx/x / np.log(10), elementwise=True)
def sin(x):
    function_ = get_operation_function(x)
    return function_('np.sin({})', x, domain=[-1,1], elementwise=True,
                     unc_function= lambda x,dx: dx * np.abs(np.cos(x)))
def cos(x):
    function_ = get_operation_function(x)
    return function_('np.cos({})', x, domain=[-1,1], elementwise=True,
                     unc_function= lambda x,dx: dx * np.abs(np.sin(x)))
def tan(x):
    function_ = get_operation_function(x)
    return function_('np.tan({})', x, domain=[-np.inf, np.inf],
      elementwise=True, unc_function= lambda x,dx: dx * np.abs(1/np.cos(x)**2))
def arcsin(x):
    function_ = get_operation_function(x)
    return function_('np.arcsin({})', x, domain=[-np.pi,np.pi],
                     is_domain_cyclic=True, elementwise=True,
                     unc_function= lambda x,dx: dx / (1 - x**2)**0.5)
def arccos(x):
    function_ = get_operation_function(x)
    return function_('np.arccos({})', x, domain=[-np.pi,np.pi],
                     is_domain_cyclic=True, elementwise=True,
                     unc_function= lambda x,dx: dx / (1 - x**2)**0.5)
def arctan(x):
    function_ = get_operation_function(x)
    return function_('np.arctan({})', x, domain=[-np.pi,np.pi],
                     is_domain_cyclic = True, elementwise=True,
                     unc_function= lambda x,dx: dx / (1 + x**2))
def sinh(x):
    function_ = get_operation_function(x)
    return function_('np.sinh({})', x, domain=[-np.inf,np.inf],
          unc_function= lambda x,dx: dx * np.abs(np.cosh(x)), elementwise=True)
def cosh(x):
    function_ = get_operation_function(x)
    return function_('np.cos({})', x, domain=[1,np.inf],
          unc_function= lambda x,dx: dx * np.abs(np.sinh(x)), elementwise=True)
def tanh(x):
    function_ = get_operation_function(x)
    return function_('np.tan({})', x, domain=[-1,1],
       unc_function= lambda x,dx: dx*np.abs(1/np.cosh(x)**2), elementwise=True)
def arcsinh(x):
    function_ = get_operation_function(x)
    return function_('np.arcsinh({})', x, domain=[-np.inf, np.inf],
             unc_function= lambda x,dx: dx / (x**2 + 1)**0.5, elementwise=True)
def arccosh(x):
    function_ = get_operation_function(x)
    return function_('np.arccosh({})', x, domain=[0, np.inf],
             unc_function= lambda x,dx: dx / (x**2 - 1)**0.5, elementwise=True)
def arctanh(x):
    function_ = get_operation_function(x)
    return function_('np.arctanh({})', x, domain=[-np.inf, np.inf],
                    unc_function= lambda x,dx: dx / (1-x**2), elementwise=True)

# Function acronyms.
rval = rich_value
rarray = rich_array
rich_df = rdataframe = rich_dataframe
function = function_with_rich_values
array_function = function_with_rich_arrays
distribution = distr_with_rich_values
array_distribution = distr_with_rich_arrays
evaluate_distribution = evaluate_distr
center_and_uncertainties = center_and_uncs
is_not_a_number = is_nan = isnan
is_infinite = is_inf = isinf
is_finite = is_finite = isfinite
