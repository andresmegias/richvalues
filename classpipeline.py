#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated GILDAS-CLASS Pipeline
-------------------------------
Main script
Version 1.3

Copyright (C) 2025 - Andrés Megías Toledano

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

# Libraries.
import os
import sys
import glob
import copy
import time
import shutil
import platform
import argparse
import subprocess
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Functions.

def full_path(text):
    """
    Obtain the full path described in a text string.
    
    Corrects the format of the path, allowing the use of operating system
    abbreviations (for example, './', '../' and '~' for Unix).
    
    Parameters
    ----------
    text : str
        Text of the path, which can include operating system abbreviations.
        
    Returns
    -------
    path : str
        Text of the full path, so that Python can use it.
    """
    path = text
    if path.startswith('~'):
        path = os.path.expanduser('~') + path[1:]  
    path = str(os.path.realpath(path))
    return path

def parse_bad_scans(input_bad_scans):
    """
    Return a list of all the specified scans to be ignored.

    Parameters
    ----------
    input_bad_scans : list
        Scans to be ignored, including intervals of bad scans.

    Returns
    -------
    bad_scans : list
        All scans to be ignored.
    """
    first = 0
    last = int(1e6)
    bad_scans = []
    for scan in input_bad_scans:
        scan = str(scan)
        if scan.startswith(':'):
            i2 = int(scan[1:])
            for i in range(first, i2+1):
                bad_scans += [i]
        elif scan.endswith(':'):
            i1 = int(scan[:-1])
            for i in range(i1, last):
                bad_scans += [i]
        elif ':' in scan:
            i1, i2 = scan.split(':')
            i1, i2 = int(i1), int(i2)
            for i in range(i1, i2+1):
                bad_scans += [i]
        else:
            bad_scans += [scan]
    bad_scans = sorted(bad_scans)
    return bad_scans

def remove_extra_spaces(input_text):
    """
    Remove extra spaces from a text string.

    Parameters
    ----------
    input_text : str
        Input text string.

    Returns
    -------
    text : str
        Resulting text.
    """
    text = input_text
    for i in range(12):
        if '  ' in text:
            text = text.replace('  ', ' ')
    if text.startswith(' '):
        text = text[1:]
    return text

def save_yaml_dict(dictionary, file_path, default_flow_style=False,
                   replace=False):
    """
    Save the input YAML dictionary into a file.

    Parameters
    ----------
    dictionary : dict
        Dictionary that wants to be saved.
    file_path : str
        Path of the output file.
    default_flow_style : bool, optional
        The flow style of the output YAML file. The default is False.
    replace : bool, optional
        If True, replace the output file in case it existed. If False, load the
        existing output file and merge it with the input dictionary.
        The default is False.

    Returns
    -------
    None.
    """
    file_path = os.path.realpath(file_path)
    if not replace and os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            old_dict = yaml.safe_load(file)
        new_dict = {**old_dict, **dictionary}
    else:
        new_dict = dictionary
    with open(file_path, 'w') as file:
        yaml.dump(new_dict, file, default_flow_style=default_flow_style)

def ticks_format(value, index):
    """
    Format the input value.
    
    Francesco Montesano
    
    Get the value and returns the value formatted.
    To have all the number of the same size they are all returned as LaTeX
    strings
    """
    exp = np.floor(np.log10(value))
    base = value/10**exp
    if exp == 0 or exp == 1:   
        return '${0:d}$'.format(int(value))
    if exp == -1:
        return '${0:.1f}$'.format(value)
    else:
        return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))

def default_elements(lists):
    """
    Return the common elements for the given lists of elements.

    Parameters
    ----------
    lists : list (list)
        Input lists of elements to compare.

    Returns
    -------
    default_els : list
        List of the common elements.
    """
    default_els = set(lists[0])
    for i in range(len(lists)-1):
        default_els = default_els.intersection(lists[i+1])
    default_els = list(default_els)
    return default_els

#%%

# Folder sep.
sep = '/'
operating_system = platform.system()
if operating_system == 'Windows':
    sep = '\\'

# Arguments.
parser = argparse.ArgumentParser()
parser.add_argument('config', default='./config.yaml')
parser.add_argument('--rms_check', action='store_true')
parser.add_argument('--check_rms_plots', action='store_true')
parser.add_argument('--selection', action='store_true')
parser.add_argument('--line_search', action='store_true')
parser.add_argument('--reduction', action='store_true')
parser.add_argument('--merging', action='store_true')
parser.add_argument('--averaging', action='store_true')
parser.add_argument('--spectra_tables', action='store_true')
parser.add_argument('--use_julia', action='store_true')
parser.add_argument('--using_windows_py', action='store_true')
args = parser.parse_args()
if not (sys.argv[0].startswith('/Users/') or sys.argv[0].startswith('C:')):
    local_run = True
else:
    local_run = False
    

#%% Preparation.

# Default options.
default_options = {
    'data folder': 'input',
    'output folder': 'output',
    'exporting folder': 'exported',
    'plots folder': 'plots',
    'input files': [],
    'bad scans': [],
    'default telescopes': ['*'],
    'observatory': '',
    'scale to main beam': False,
    'telescope main beam efficiencies': {},
    'frequency units': 'MHz',
    'weighting mode': 'time',
    'line frequencies (MHz)': {},
    'radial velocities (km/s)': {},
    'new source names': {},
    'fold spectra': False,
    'average all input files': True,
    'use only daily bad scans': False,
    'check Doppler corrections': True,
    'only rms noise plots': False,
    'extra note': '',
    'reduction': {
        'check windows': True,
        'save plots': False,
        'rolling sigma clip': False,
        'reference width': 14,
        'smoothing factor': 20,
        'intensity threshold (rms)': 8.,
        'relative frequency margin for rms noise': 0.1
        },
    'ghost lines' : {
        'clean lines': False,
        'absolute intensity threshold (rms)': 10.,
        'relative intensity threshold': 0.3,
        'smoothing factor': 40
        },
    'merging': 'auto',
    'only rms plots': False
    }

default_rms_opts = {
    'sources-lines-telescopes': [],
    'frequency range (GHz)': [],
    'scans per group': 1
    }

# Initial check.
if not (args.selection or args.line_search or args.reduction or args.merging
        or args.averaging or args.spectra_tables or args.rms_check
        or args.check_rms_plots):
    raise Exception('No processing mode is selected.')

# Starting.
time1 = time.time()
print()
print('Automated GILDAS-CLASS Pipeline')
print("-------------------------------")
print('\nStarting the processing of the data.')

# Folders.
original_folder = os.getcwd()
if local_run:
    codes_folder = sep.join([original_folder]
                                  + sys.argv[0].split(sep)[:-1])
    codes_folder += sep

# Reading of the configuration file.
config_path = full_path(args.config)
if os.path.isfile(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
else:
    raise FileNotFoundError('Configuration file not found.')
options = {**default_options, **config}
config_folder = full_path(sep.join(args.config.split(sep)[:-1])) + sep
if not config_folder.endswith(sep):
    config_folder += sep
os.chdir(config_folder)

# Reading of the variables.
data_folder = options['data folder']
input_files = options['input files']
exporting_folder = options['exporting folder']
output_folder = options['output folder']
plots_folder = options['plots folder']
default_telescopes = options['default telescopes']
observatory = options['observatory']
bad_scans = options['bad scans']
telescope_effs = options['telescope main beam efficiencies']
modify_beam_eff = options['scale to main beam']
if modify_beam_eff and 'beam efficiency' in telescope_effs:
    telescope_effs = {'default': telescope_effs}
frequency_units = options['frequency units']
line_frequencies = options['line frequencies (MHz)']
radial_velocities = options['radial velocities (km/s)']
new_source_names = options['new source names']
weight_mode = options['weighting mode']
fold_spectra = options['fold spectra']
average_all_input_files = options['average all input files']
only_daily_bad_scans = options['use only daily bad scans']
check_doppler_corrections = options['check Doppler corrections']
only_rms_plots = options['only rms noise plots']
extra_note = options['extra note']
if 'reduction' in options:
    options['reduction'] = {**default_options['reduction'],
                            **options['reduction']}
else:
    options['reduction'] = default_options['reduction']
if 'averaging' in options and 'ghost lines' in options['averaging']:
    options['averaging']['ghost lines'] = {**default_options['ghost lines'],
                                           **options['averaging']['ghost lines']}
elif 'averaging' in options:
    options['averaging']['ghost lines'] = default_options['ghost lines']
check_windows = options['reduction']['check windows']
save_plots = options['reduction']['save plots']
line_width = str(options['reduction']['reference width'])
smooth_factor = str(options['reduction']['smoothing factor'])
intensity_threshold = str(options['reduction']['intensity threshold (rms)'])
rolling_sigma_clip = str(options['reduction']['rolling sigma clip'])
rms_margin = str(options['reduction']['relative frequency margin for rms noise'])
if 'rms noise check' in options:
    for rms_opt_list in options['rms noise check']:
        options['rms noise check'][rms_opt_list] = \
            {**default_rms_opts, **options['rms noise check'][rms_opt_list]}
    rms_option_list = options['rms noise check']
    
if frequency_units == 'MHz':
      to_GHz = 0.001
elif frequency_units == 'GHz':
      to_GHz = 1.0

# Checks and setting of the common options.
if not data_folder.endswith(sep):
    data_folder += sep
if not exporting_folder.endswith(sep):
    exporting_folder += sep
if not output_folder.endswith(sep):
    output_folder += sep
if not plots_folder.endswith(sep):
    plots_folder += sep
if len(input_files) == 0:
    raise Exception('There are no input files specified.')
if not os.path.isdir(full_path(data_folder)):
    raise Exception(f'Folder {data_folder} does not exist.')
if not only_daily_bad_scans:
    bad_files = []
    for file in input_files:
        if file.startswith('--'):
            bad_files += [file]
    for file in bad_files:
        del input_files[file]
input_files = {file.replace('--',''): input_files[file] for file in input_files}
for file in input_files:
    if not os.path.isfile(full_path(data_folder + file)):
        raise Exception(f'The file {file} is missing.')
    if not 'sources-lines-telescopes' in input_files[file]:
        raise Exception(f'There are no sources specified in file {file}.')
    else:
        sources = input_files[file]['sources-lines-telescopes']
        for (source, lines) in zip(sources, sources.values()):
            for line, telescopes in zip(lines, lines.values()):
                if telescopes == 'default':
                    (input_files[file]['sources-lines-telescopes'][source]
                     [line]) = default_telescopes
    if not average_all_input_files:
        if not 'note' in input_files[file]:
            raise Exception(f'There is no note for file {file}.')
        elif len(input_files[file]['note']) == 0:
            raise Exception(f'The note of the file {file} is empty.')
if options['frequency units'] not in ['MHz', 'GHz']:
    raise Exception('Wrong units for frequency, should be MHz or GHz.')
if not os.path.isdir(full_path(exporting_folder)):
    os.makedirs(full_path(exporting_folder))
if not os.path.isdir(full_path(output_folder)):
    os.makedirs(full_path(output_folder))
if not os.path.isdir(full_path(output_folder + 'all') + sep):
    os.makedirs(full_path(output_folder + 'all') + sep)
if not os.path.isdir(full_path(plots_folder)):
    os.makedirs(full_path(plots_folder))


#%% Selection mode.

# Initialization of variables.
output_files = []
output_spectra = {}
output_telescopes = {}
script = []
# Loop for files.
for file in input_files:
    script += ['file in ' + data_folder + file]
    script += ['set weight ' + weight_mode]
    ext = '.' + file.split('.')[-1]
    if not average_all_input_files:
        note = input_files[file]['note']
    # Loop for ignoring the bad scans.
    bad_scans_i = []
    if type(bad_scans) == list and not only_daily_bad_scans:
        bad_scans_i += parse_bad_scans(bad_scans)
    if 'bad scans' in input_files[file]:
        bad_scans_i += parse_bad_scans(input_files[file]['bad scans'])
    for scan in set(bad_scans_i):
        script += ['ignore /scan {}'.format(scan)]
    # Loop for sources.
    sources = input_files[file]['sources-lines-telescopes']
    for (source, lines) in zip(sources, sources.values()):
        script += ['set source ' + source]
        # Loop for lines.
        for (line, telescopes) in zip(lines, lines.values()):
            observations = []
            script += ['set line ' + line]
            # Name of the output files.
            output_file = '-'.join([source, line.replace('*','')])
            if len(extra_note) > 0:
                output_file += '-' + extra_note
            if not average_all_input_files:
                output_file = '-'.join([output_file, note])
            output_file = (output_file + ext).replace('-'+ext, ext)
            # Definition of the output file.
            script += ['file out ' + output_folder + output_file]
            if output_file not in output_files:
                script[-1] += ' m /overwrite'
            output_files += [output_file]
            # Loop for telescopes.
            for telescope in telescopes:
                script += ['set telescope ' + telescope]
                # Bad scans.
                bad_scans_i = []
                if (type(bad_scans) == dict and not only_daily_bad_scans
                    and source in bad_scans and line in bad_scans[source]
                        and telescope in bad_scans[source][line]):
                    bad_scans_i += \
                        parse_bad_scans(bad_scans[source][line][telescope])
                for scan in set(bad_scans_i):
                    script += ['ignore /scan {}'.format(scan)]
                # Average.
                script += ['find /all', 'list', 'stitch']
                if fold_spectra:
                    script += ['fold']
                if line in line_frequencies and telescope in line_frequencies[line]:
                    script += ['modify frequency {}'
                               .format(line_frequencies[line][telescope])]
                if '*' in line:
                    script += ['modify line {}'.format(line.replace('*',''))]
                if source in radial_velocities:
                    script += ['modify velocity {}'.format(radial_velocities[source])]
                if source in new_source_names:
                    script += ['modify source {}'
                               .format(new_source_names[source].upper())]
                script += ['write'] 
                output_obs = '-'.join([source, line.replace('*','')])
                if not average_all_input_files:
                    output_obs = '-'.join([output_obs, note])
                output_obs = ('-'.join([output_obs, telescope.replace('*','')]))
                if not average_all_input_files:
                    script += ['greg ' + exporting_folder + output_obs + '.dat'
                             + ' /formatted']
                    output_obs_fits = output_obs + '.fits'
                    if args.selection and os.path.isfile(exporting_folder
                                                         + output_obs_fits):
                        os.remove(exporting_folder + output_obs_fits)
                    script += ['fits write {} /mode spectrum'
                             .format(exporting_folder + output_obs_fits)]
                observations += [output_obs]
            if output_file not in list(output_spectra.keys()):
                output_spectra[output_file] = copy.copy(observations)
                output_telescopes[output_file] = copy.copy(telescopes)
            else:
                output_spectra[output_file] += observations
                output_telescopes[output_file] += telescopes
        
# Removing repeated elements from output files list and combining all spectra.
all_spectra = []
output_files = list(np.unique(output_files))
for file in output_files:
    output_spectra[file] = list(np.unique(output_spectra[file]))
    output_telescopes[file] = list(np.unique(output_telescopes[file]))
    all_spectra += output_spectra[file]

if args.selection:

    # Combining observations coming from different files.
    if average_all_input_files and len(input_files) >= 1:
        for file in output_files:
            script += ['file in ' + output_folder + file]
            ext = '.' + file.split('.')[-1]
            script += ['file out ' + output_folder
                     + file.replace(ext, '-temp'+ext) + ' m /overwrite']
            script += ['set source *', 'set line *']
            for (spectrum, telescope) in zip(output_spectra[file],
                                             output_telescopes[file]):
                script += ['set telescope ' + telescope]
                script += ['find /all', 'list', 'stitch', 'write']
                script += ['greg ' + exporting_folder + spectrum + '.dat'
                         + ' /formatted']
                spectrum_fits = spectrum + '.fits'
                if os.path.isfile(exporting_folder + spectrum_fits):
                    os.remove(exporting_folder + spectrum_fits)
                script += ['fits write {} /mode spectrum'
                          .format(exporting_folder + spectrum_fits)]
                
    # Creating CLASS files for each spectrum.
    script += ['set source *', 'set line *', 'set telescope *']
    all_subfolder = 'all' + sep
    for file in output_files:
        script += ['file in {}/{}'.format(output_folder, file)]
        script += ['find /all']
        for spectrum in output_spectra[file]:
            script += ['file out {}{}{} m /overwrite'
                       .format(output_folder, all_subfolder, spectrum + ext)]
            telescope = spectrum.split('-')[-1]
            script += ['set telescope ' + '*'+telescope+'*']
            script += ['find /all', 'get first']
            script += ['write']
            if not average_all_input_files:
                i = 3 if spectrum.split('-')[1][0].isdigit() else 2
                note = '-'.join(spectrum.split('-')[i:-1])
                spectrum = spectrum.replace('-'+note+'-','-') + '-'+note
                script += ['file out {}{}{} m /overwrite'
                           .format(output_folder, all_subfolder, spectrum + ext)]
                script += ['write']
    
    # End of the script.
    script += ['exit']
    script = [line + '\n' for line in script]
 
    # Writing of the first class file.
    with open('selection.class', 'w') as file:
        file.writelines(script)

# Running of the first class file.
    
    print('\nStarting selection.\n')

    subprocess.run(['class', '@selection.class'])
    
    if average_all_input_files:
        for file in output_files:
            ext = '.' + file.split('.')[-1]
            original_name = output_folder + file.replace(ext,'-temp'+ext)
            new_name = output_folder + file
            if os.path.isfile(new_name):
                os.remove(new_name)
            os.rename(original_name, new_name)
        
# Creation of a class file for checking the Doppler corrections.
    
    script = []
    # Showing the Doppler corrections.
    for file in output_files:
        script += ['file in {}/{}'.format(output_folder, file)]
        if len(observatory) > 0:
            script += ['set observatory ' + observatory]
        script += ['find /all', 'list']
        for spectrum in output_spectra[file]:
            script += ['get next', 'modify doppler', 'modify doppler *']

    # End of the script.
    script += ['exit']
    sript = [line + '\n' for line in script]
 
    # Writing of the first class file.
    with open('selection-doppler.class', 'w') as file:
        file.writelines(script)

# Running of the class file for checking the Doppler corrections.

    if check_doppler_corrections:
        
        p = subprocess.Popen(['class', '@selection-doppler.class'],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        doppler_text = []        
        prev_line = ''
        for output in p.stdout:
            text_line = output.decode('utf-8').replace('\n','')
            print(text_line)
            if 'Doppler factor' in text_line and '***' not in text_line:
                if (not 'Doppler factor' in prev_line
                        or 'Doppler factor' in prev_line and '***' in prev_line): 
                    doppler_text += [remove_extra_spaces(text_line).split(' ')[-1]]
            if 'I-OBSERVATORY' not in text_line:
                prev_line = text_line
    
        doppler_corr = {}
        if len(doppler_text) == len(all_spectra):
            for i in range(len(doppler_text)):
                spectrum = str(all_spectra[i])
                doppler_corr[spectrum] = doppler_text[i]
        else:
            raise Exception('Error reading Doppler corrections.')
            
    else:
        
        doppler_corr = {}
        for i in range(len(all_spectra)):
            spectrum = str(all_spectra[i])
            doppler_corr[spectrum] = '0'
    
    # Export of the Doppler corrections of each spectrum.
    save_yaml_dict(doppler_corr, './' + exporting_folder
                   + 'doppler_corrections.yaml', default_flow_style=False)
    print()
    print('Saved Doppler corrections in {exporting_folder}doppler_corrections.yaml.')

# Creation of a final file.

    script = []
    sources = []
    for file in output_files:
        sources += [file.split('-')[0]]
    for source in list(np.unique(sources)):
        file_name = output_folder + source + '-' + extra_note + '-all' + ext
        file_name = file_name.replace('--','-')
        script += ['file out ' + file_name + ' m /overwrite']
    # Loop for files created in the data selection.
    for file in output_files:
        source = file.split('-')[0]
        script += ['file in ' + output_folder + file]
        file_name = output_folder + source + '-' + extra_note + '-all' + ext
        file_name = file_name.replace('--','-')
        script += ['file out ' + file_name] 
        script += ['find /all', 'list']
        # Loop for observations of each file.
        for spectrum in output_spectra[file]:
            script += ['get next', 'write']
        file_name = output_folder + file
        shutil.copyfile(file_name, file_name.replace(output_folder,
                                                output_folder + all_subfolder))
            
    # End of the script.
    script += ['exit']
    script = [line + '\n' for line in script]
        
    # Writing of the class file.
    with open('selection-grouping.class', 'w') as file:
        file.writelines(script)
        
    # Running of the class file.
    subprocess.run(['class', '@selection-grouping.class'])
    
    print()
    print(f'Created CLASS files:    (folder {output_folder})')
    for file in output_files:
        print('- ' + file)
    print()


#%% RMS check mode, calculations.

if args.rms_check:
    
    print('\nStarting noise check.\n')
    
    previous_images = glob.glob('*rms-*.png')
    for image in previous_images:
        os.remove(image)

    for option_list in rms_option_list:
        
        plt.close('all')
        os.chdir(config_folder)
    
        rms_params = (options['rms noise check'][option_list]
                      ['source-line-telescopes'])
        rms_freq_ranges = (options['rms noise check'][option_list]
                           ['frequency ranges (GHz)'])
        group_size = options['rms noise check'][option_list]['scans per group']
        num_freq_ranges = len(rms_freq_ranges)
    
        source = list(rms_params.keys())[0]
        line = list(rms_params[source].keys())[0]
        telescopes = rms_params[source][line]
        
        s = ''
        if not only_daily_bad_scans and len(bad_scans) > 0:
            s = '-s'
        
        for telescope in telescopes:
            
            for f, rms_freq_range in enumerate(rms_freq_ranges):
            
                os.chdir(original_folder)
                
                rms_freq_range = 1000*np.array(rms_freq_range)
                range_text = '({:g}-{:g})'.format(*rms_freq_range)
                title = '-'.join([source, line.replace('*',''),
                                  telescope.replace('*','')])
                rms_curve_file = (exporting_folder + 'rms-{}-{}-{}{}.yaml'
                                  .format(title, range_text, group_size, s))
                file_name = ('rms-{}-{}{}-filenames.csv'
                             .format(title, group_size, s))
                
                if not only_rms_plots:
        
                    # Class file for selecting desired observations.
                    script = []
                    
                    for i, file in enumerate(input_files):
                        
                        script += ['file in ' + data_folder + file]
                        ext = '.' + file.split('.')[-1]
                        output_file_all = (output_folder + 'rms_check-all'+ext)
                        script += ['file out ' + output_file_all]
                        if i == 0:
                            script[-1] += ' m /overwrite'
                        # Loop for ignoring the bad scans.
                        bad_scans_i = []
                        if type(bad_scans) == list and not only_daily_bad_scans:
                            bad_scans_i += parse_bad_scans(bad_scans)
                        if 'bad scans' in input_files[file]:
                            bad_scans_i += \
                                parse_bad_scans(input_files[file]['bad scans'])
                        for scan in set(bad_scans_i):
                            script += ['ignore /scan {}'.format(scan)]
                        script += ['set source ' + source]
                        script += ['set line ' + line]
                        script += ['set telescope ' + telescope]
                        # Bad scans.
                        bad_scans_i = []
                        if (type(bad_scans) == dict and not only_daily_bad_scans
                            and source in bad_scans and line in bad_scans[source]
                                and telescope in bad_scans[source][line]):
                            bad_scans_i += parse_bad_scans(bad_scans[source]
                                                           [line][telescope])
                        for scan in set(bad_scans_i):
                            script += ['ignore /scan {}'.format(scan)]
                        # List of observations.
                        script += ['find /all', 'list'] 
                        if fold_spectra:
                            script += ['fold']
                        script += ['set mode x {} {}'.format(*rms_freq_range)]
                        script += ['for i 1 to found', 'get next']
                        script += ['modify source --{}--{}'.format(i+1,source)]
                        script += ['write', 'next']
                    
                    # End of the script.
                    script += ['exit']
                    script = [line + '\n' for line in script]
                 
                    # Writing of the class file.
                    with open('rms_check.class', 'w') as file:
                        file.writelines(script)          
                    
                    os.chdir(original_folder)
                
                    subprocess.run(['class', '@rms_check.class'])
                
                    # Class file for obtaining the information of each
                    # observation.
                    
                    script = []
                    script += ['file in ' + output_file_all]
                    script += ['find /all', 'list']
                    script += ['exit']
                    script = [line + '\n' for line in script]
                    
                    with open('rms_check-info.class', 'w') as file:
                        file.writelines(script)   
                        
                    p = subprocess.Popen(['class', '@rms_check-info.class'],
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT)
                    rms_id = {}
                    list_zone = False
                    file_nums, scan_nums, obs_nums = [], [], []
                    j = 0
                    for text_line in p.stdout:
                        text_line = text_line.decode('utf-8').replace('\n','')
                        print(text_line)
                        if text_line.startswith('I-FIND'):
                            num_obs = int(text_line.replace('I-FIND,','')
                                          .replace('observations found','')
                                          .replace(' ',''))
                        if list_zone:
                            text_line = text_line.replace(';', '; ')
                            params = tuple(remove_extra_spaces(text_line)
                                           .split(' '))
                            column = params[2]
                            elements = column.split('--')
                            if len(elements) >= 2:
                                file_number = elements[1]
                            else:
                                file_number = 0
                            obs_number = tuple(params[0].split(';'))[0]
                            scan_number = params[-2]
                            if (len(obs_nums) == 0
                                or (len(obs_nums) > 0
                                    and [file_number, obs_number, scan_number]
                                    not in rms_id.values()) and j < num_obs):
                                file_nums += [str(file_number)]
                                obs_nums += [str(obs_number)]
                                scan_nums += [str(scan_number)]
                                j += 1
                                rms_id[j] = [file_number, obs_number,
                                             scan_number]    
                        if 'N;V' in text_line and 'Source' in text_line:
                            list_zone = True
                
                    rms_filenames = pd.DataFrame.from_dict(rms_id, orient='index',
                                               columns=['file', 'obs', 'scan'])   
        
                    file_path = os.path.realpath(exporting_folder + file_name)
                    rms_filenames.to_csv(file_path)
                    
                    num_obs = len(obs_nums)
                    
                    # Class file for grouping the files individually and
                    # exporting them.
                    
                    all_rms_spectra = []
                    script = []
                    script += ['file in ' + output_file_all]
                    script += ['file out ' + output_folder + 'rms_check-ind'+ext
                             + ' m /overwrite']
                    script += ['set weight ' + weight_mode]
                    for i in range(0, num_obs, group_size):
                        script += ['set mode x {} {}'.format(*rms_freq_range)]
                        group_obs, group_indices = [], []
                        first = True
                        i_max = min(i + group_size, num_obs)
                        for j in range(i, i_max):
                            script += ['set source --{}--*'.format(file_nums[j])]
                            script += ['set number {} {}'.format(*[obs_nums[j]]*2)]
                            if first:
                                first = False
                                script += ['find /all']
                            else:
                                script += ['find append /all']
                            group_obs += [obs_nums[j]]
                            group_indices += [str(j+1)]
                        group_obs = group_obs[0] + 'to' + group_obs[-1]
                        group_indices = '+'.join(group_indices)
                        script += ['list', 'average /nocheck']
                        script += ['set mode x {} {}'.format(*rms_freq_range)]
                        script += ['modify line ' + group_obs, 'write']
                        spectrum = ('rms-{}-{}-{}-{}-({})-{}'
                                    .format(source, line.replace('*',''),
                                            telescope.replace('*',''), range_text,
                                            group_size, group_indices))
                        spectrum_fits = spectrum + '.fits'
                        all_rms_spectra += [spectrum]
                        script += ['greg ' + exporting_folder + spectrum + '.dat'
                                 + ' /formatted']
                        if os.path.isfile(exporting_folder + spectrum_fits):
                            os.remove(exporting_folder + spectrum_fits)
                        script += ['fits write {} /mode spectrum'
                                  .format(exporting_folder + spectrum_fits)]
                        if check_doppler_corrections:
                            script += ['modify doppler', 'modify doppler *']
                        
                    script += ['exit']
                    script = [line + '\n' for line in script]
                        
                    with open('rms_check-ind.class', 'w') as file:
                        file.writelines(script)
                        
                    p = subprocess.Popen(['class', '@rms_check-ind.class'],
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT)
                    
                    doppler_text = []
                    prev_line = ''
                    for text_line in p.stdout:
                        text_line = text_line.decode('utf-8').replace('\n','')
                        print(text_line)
                        if ('Doppler factor' in text_line and '***' not in text_line):
                            if (not 'Doppler factor' in prev_line
                                    or ('Doppler factor' in prev_line
                                        and '***' in prev_line)):
                                doppler_text += [remove_extra_spaces(text_line)
                                                 .split(' ')[-1].replace('\r','')]
                        if 'I-OBSERVATORY' not in text_line:
                            prev_line = text_line
                    
                    # Class file for grouping the files cumulatively and
                    # exporting them.
                    
                    script = []
                    script += ['file in ' + output_file_all]
                    script += ['file out ' + output_folder + 'rms_check-cum'+ext
                             + ' m /overwrite']
                    script += ['set weight ' + weight_mode]
                    for i in range(0, num_obs, group_size):
                        group_obs, group_indices = [], []
                        first = True
                        i_max = min(i + group_size, num_obs)
                        for j in range(0, i_max):
                            script += ['set source --{}--*'.format(file_nums[j])]
                            script += ['set number {} {}'.format(*[obs_nums[j]]*2)]
                            if first:
                                first = False
                                script += ['find /all']
                            else:
                                script += ['find append /all']
                            group_obs += [obs_nums[j]]
                            group_indices += [str(j+1)]
                        group_obs = 'to' + group_obs[-1]
                        group_indices = 'to{}'.format(group_indices[-1])
                        script += ['list', 'average /nocheck']
                        script += ['set mode x {} {}'.format(*rms_freq_range)]
                        script += ['modify line ' + group_obs, 'write']
                        spectrum = ('rms-{}-{}-{}-{}-({})-{}'
                                    .format(source, line.replace('*',''),
                                            telescope.replace('*',''), range_text,
                                            group_size, group_indices))
                        spectrum_fits = spectrum + '.fits'
                        all_rms_spectra += [spectrum]
                        script += ['greg ' + exporting_folder + spectrum + '.dat'
                                 + ' /formatted']
                        if os.path.isfile(exporting_folder + spectrum_fits):
                            os.remove(exporting_folder + spectrum_fits)
                        script += ['fits write {} /mode spectrum'
                                  .format(exporting_folder + spectrum_fits)]
                        if check_doppler_corrections:
                            script += ['modify doppler', 'modify doppler *']
                        
                    script += ['exit']
                    script = [line + '\n' for line in script]
                        
                    with open('rms_check-cum.class', 'w') as file:
                        file.writelines(script)
                        
                    p = subprocess.Popen(['class', '@rms_check-cum.class'],
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT)
                    
                    prev_line = ''
                    for text_line in p.stdout:
                        text_line = text_line.decode('utf-8').replace('\n','')
                        print(text_line)
                        if 'Doppler factor' in text_line:
                            if ('Doppler factor' in text_line
                                    and '***' not in text_line):
                                if (not 'Doppler factor' in prev_line
                                        or ('Doppler factor' in prev_line
                                            and '***' in prev_line)): 
                                    doppler_text += [remove_extra_spaces(text_line)
                                                     .split(' ')[-1]]
                        if 'I-OBSERVATORY' not in text_line:
                            prev_line = text_line
                        
                    doppler_corr = {}
                    if check_doppler_corrections:
                        if len(doppler_text) == len(all_rms_spectra):
                            for i in range(len(doppler_text)):
                                spectrum = str(all_rms_spectra[i])
                                doppler_corr[spectrum] = doppler_text[i]
                        else:
                            message = 'Error reading Doppler corrections'
                            raise Exception(message)
                    else:
                        for i in range(len(all_rms_spectra)):
                            spectrum = str(all_rms_spectra[i])
                            doppler_corr[spectrum] = '0'
                        
                    save_yaml_dict(doppler_corr, exporting_folder
                                   + 'doppler_corrections.yaml',
                                   default_flow_style=False)
                    print('\nSaved Doppler corrections in ' + exporting_folder
                          + 'doppler_corrections.yaml.\n')
                    
                    # Reduction of spectra in the rms regions.
                    
                    os.chdir(config_folder)
                    arguments = ['classreduction.py', exporting_folder,
                                 ','.join(all_rms_spectra), '-smooth', smooth_factor,
                                 '-plots_folder', config_folder + plots_folder,
                                  '--rms_check']
                    if local_run:
                        os.chdir(original_folder)
                        arguments[0] = codes_folder + arguments[0]
                        arguments = ['python3'] + arguments
                        if args.using_windows_py:
                            arguments[0] = 'py'
                    if check_windows:
                        arguments += ['--check_windows']
                    if save_plots:
                        arguments += ['--save_plots']
                    subprocess.run(arguments)
                    os.chdir(config_folder)
                        
                    # Calibration of the RMS noise and saving of the data.
                    
                    if modify_beam_eff:
                        with open(exporting_folder
                                  + 'frequency_ranges.yaml') as file:
                            frequency_ranges = yaml.safe_load(file)
                    
                    num_rms = len(all_rms_spectra)
                    output_files_rms = ['rms_check-ind' + ext,
                                        'rms_check-cum' + ext]
                    output_spectra_rms = {'rms_check-ind' + ext:
                                          all_rms_spectra[:num_rms//2],
                                          'rms_check-cum' + ext:
                                          all_rms_spectra[num_rms//2:]}
                    
                    with open(exporting_folder + 'rms_noises.yaml') as file:
                        rms_noises = yaml.safe_load(file)  
                    
                    rms_ind, rms_cum = [], []
                    total_spectra = zip(output_spectra_rms['rms_check-ind'+ext],
                                        output_spectra_rms['rms_check-cum'+ext])
                    for spectrum_ind, spectrum_cum in total_spectra:
                        if modify_beam_eff:
                            mean_freq = np.mean(frequency_ranges[spectrum])
                            mean_freq *= to_GHz
                            keys = list(telescope_effs.keys())
                            if keys == ['default']:
                                key = keys[0]
                            else:
                                telescope = spectrum_ind.split('-')[-1]
                                key = 'default'
                                for key_i in telescope_effs:
                                    if 'default' != key_i:
                                        if not '*' in key_i and key_i == telescope:
                                            key = key_i
                                            break
                                        elif ('*' in key_i and
                                              key_i.replace('*','') in telescope):
                                            key = key_i
                                            break
                            if not key in telescope_effs:
                                raise Exception(f'Missing {key} beam efficiency.')
                            beam_eff = np.interp(mean_freq,
                                        telescope_effs[key]['frequency (GHz)'],
                                        telescope_effs[key]['beam efficiency'])
                            beam_eff = float(beam_eff)
                            rms_noises[spectrum_ind] /= beam_eff
                            rms_noises[spectrum_cum] /= beam_eff
                        rms_ind += [rms_noises[spectrum_ind]]
                        rms_cum += [rms_noises[spectrum_cum]]
                        
                    save_yaml_dict(rms_noises, exporting_folder +
                                   'rms_noises.yaml', default_flow_style=False)
                    print(f'Saved calibrated RMS noises in {exporting_folder}'
                          'rms_noises.yaml.')      
                    
                    locs = np.arange(len(rms_ind))
                    labels = output_spectra_rms['rms_check-ind' + ext]
                    for i, label in enumerate(labels):
                        labels[i] = label.split('-')[-1]
                        labels_i = labels[i].split('+')
                        labels[i] = '-'.join([labels_i[0], labels_i[-1]])
                        labels_split = labels[i].split('-')
                        if labels_split[0] == labels_split[1]:
                            labels[i] = labels_split[0]
                    print()
                        
                    rms_curve_dict = {'labels': labels, 'rms_ind': rms_ind,
                                      'rms_cum': rms_cum}
                    save_yaml_dict(rms_curve_dict, rms_curve_file,
                                   default_flow_style=None)
                    print(f'Saved plot points in {rms_curve_file}.')
        
                # Plotting of the rms evolution.
                
                with open(rms_curve_file, 'r') as file:
                    rms_curve_dict = yaml.safe_load(file)
                    
                rms_filenames = pd.read_csv(exporting_folder + file_name)
                rms_scans = rms_filenames[["scan"]].values.flatten()
                file_nums = rms_filenames[["file"]].values.flatten()
        
                labels = rms_curve_dict['labels']
                rms_ind = rms_curve_dict['rms_ind']
                rms_cum = rms_curve_dict['rms_cum']
                locs = np.arange(len(labels))
        
                diff_rms_cum = np.diff(rms_cum)
                metric = diff_rms_cum
                diff_rms_cum = np.concatenate(([0], diff_rms_cum))
                metric = np.concatenate(([0], metric))
                cum_locs = []
                for label in labels:
                    if '-' in label:
                        cum_locs += [float(label.split('-')[1])]
                    else:
                        cum_locs += [float(label)]
                cum_locs = np.array(cum_locs)
                metric[1:] *= np.cumsum(cum_locs[:-1]) / np.sum(cum_locs[:-1])
                       
                tick_size = max([len(text) for text in labels])
                margin = min(1, tick_size/10)
                fontsize = 8 - 5*margin
                color = 'C'+str(f)
                
                num_points = 100
                interval = len(labels) // num_points
                labels_red = copy.copy(labels)
                for i in range(len(labels_red)):
                    if interval != 0 and i%interval != 0:
                        labels_red[i] = ''        
                
                if f+1 == 1:
                    plt.figure(1, figsize=(10+margin*4, 6))
                    plt.clf()
                else:
                    plt.figure(1)
                    
                freqs = range_text[1:-1]
                label_ind = f'{freqs} MHz, individual'
                label_cum = f'{freqs} MHz, cumulative'
                label_rms = f'final noise: {rms_cum[-1]:.2f} mK'
                plt.plot(locs, rms_ind, '+', color=color, label=label_ind)
                plt.plot(locs, rms_cum, '.', color=color, label=label_cum)
                plt.plot([], [], '.', alpha=0, label=label_rms)
                
                for i, metric_i in enumerate(metric):
                    if metric_i > 0:
                        plt.axvline(locs[i], linewidth=1, linestyle='--',
                                    color=color, alpha=0.6)
                for i in range(len(file_nums)-1):
                    if file_nums[i+1] != file_nums[i]:
                        plt.axvline(x=(i+0.5)/group_size, ymin=0, ymax=1,
                                    linestyle='-', color='gray', linewidth=3,
                                    alpha=0.3, clip_on=True)
                        
                if f+1 == num_freq_ranges:
                    plt.margins(x=0.01)
                    plt.xticks(ticks=locs, labels=labels_red, rotation=90,
                               fontsize=fontsize)
                    plt.yscale('log')
                    plt.tick_params(right=True, which='both')
                    plt.gca().yaxis.set_minor_formatter(FuncFormatter(ticks_format))
                    plt.gca().yaxis.set_major_formatter(FuncFormatter(ticks_format))
                    plt.xlabel('observations averaged')
                    plt.ylabel('RMS noise (mK)')
                    plt.title(title, fontweight='bold')
                    plt.legend(fontsize='small')
                    plt.tight_layout()        
                
                if f+1 == 1:
                    plt.figure(2, figsize=(10+margin*4, 6))
                    plt.clf()
                else:
                    plt.figure(2)
                    
                plt.plot(locs, diff_rms_cum, '.', color=color,
                        label=f'{range_text[1:-1]} MHz, RMS variation')
                plt.plot(locs, metric, '+', color=color,
                         label=f'{range_text[1:-1]} MHz, weighted RMS variation')
                plt.axhline(y=0, linestyle='--', linewidth=0.7, color='black')
                for i, metric_i in enumerate(metric):
                    if metric_i > 0:
                        plt.axvline(locs[i], linewidth=1, linestyle='--',
                                    color=color, alpha=0.6)
                for i in range(len(file_nums)-1):
                    if file_nums[i+1] != file_nums[i]:
                        plt.axvline(x=(i+0.5)/group_size, ymin=0, ymax=1,
                                    linestyle='-', color='gray', linewidth=3,
                                    alpha=0.3, clip_on=True)
        
                if f+1 == num_freq_ranges:
                    plt.margins(x=0.01)
                    plt.xticks(ticks=locs, labels=labels_red, rotation=90,
                               fontsize=fontsize)
                    plt.yscale('symlog', linthresh=0.001)
                    plt.xlabel('observations averaged')
                    plt.ylabel('variation of RMS noise (mK)')
                    plt.title(title, fontweight='bold')
                    plt.legend(fontsize='small')
                    plt.tight_layout()
                
                num_points = 100
                num_partial_plots = (len(locs)-1) // num_points + 1
                
                if num_partial_plots > 1:
                    
                    for i in range(num_partial_plots):
                        
                        i1 = i*num_points
                        i2 = min((i+1)*num_points, len(locs))
                        figsize = (8*(i2-i1)/num_points + 4*margin, 8)
                        
                        if f+1 == 1:
                            plt.figure(3+i, figsize=figsize)
                            plt.clf()
                        else:
                            plt.figure(3+i)
                            
                        plt.plot(locs[i1:i2], rms_ind[i1:i2], '+', color=color,
                                 label=label_ind)
                        plt.plot(locs[i1:i2], rms_cum[i1:i2], '.', color=color,
                                 label=label_cum)
                        plt.plot([], [], '.', alpha=0, label=label_rms)
                        diff_rms_cum = np.diff(rms_cum[i1:i2])
                        for i, metric_i in enumerate(diff_rms_cum ):
                            if metric_i > 0:
                                plt.axvline(locs[i1+i+1], linewidth=1,
                                            linestyle='--', color=color,
                                            alpha=0.6)
                                
                        if f+1 == num_freq_ranges:
                            plt.margins(x=0.01)
                            xlims = plt.xlim()
                            for i in range(len(file_nums)-1):
                                if file_nums[i+1] != file_nums[i]:
                                    plt.axvline(x=(i+0.5)/group_size,
                                                ymin=0, ymax=1, linestyle='-',
                                                color='gray', linewidth=3,
                                                alpha=0.3, clip_on=True)
                            plt.xlim(xlims)
                            plt.xticks(ticks=locs[i1:i2], labels=labels[i1:i2],
                                       rotation=90, fontsize=fontsize)
                            plt.yscale('log')
                            plt.tick_params(right=True, which='both')
                            plt.gca().yaxis.set_minor_formatter(
                                FuncFormatter(ticks_format))
                            plt.gca().yaxis.set_major_formatter(
                                FuncFormatter(ticks_format))
                            plt.xlabel('observations averaged')
                            plt.ylabel('RMS noise (mK)')
                            plt.title(title, fontweight='bold')
                            plt.legend(fontsize='small')
                            plt.tight_layout()
                
                if f+1 == num_freq_ranges:
                    
                    os.chdir(config_folder)
                    os.chdir(os.path.realpath(plots_folder))
                
                    plt.figure(1)
                    filename = f'rms-{title}-{group_size}{s}.png'
                    plt.savefig(filename, dpi=400)
                    print(f'Saved full RMS noise evolution in {plots_folder}'
                          f'{filename}.')
                    
                    plt.figure(2)
                    filename = f'rms-metric-{title}-{group_size}{s}.png'
                    plt.savefig(filename, dpi=400)
                    print(f'Saved RMS noise variation evolution in {plots_folder}'
                          f'{filename}. ')
                    
                    if num_partial_plots > 1:
                        for i in range(num_partial_plots):
                            plt.figure(3+i)
                            filename = f'rms-{title}-{group_size}{s}-({i+1}).png'
                            plt.savefig(filename, dpi=300)
                            print(f'Saved partial RMS noise evolution in {plots_folder}'
                                  f'{filename}.')
                            
    plt.close('all')                 
    print()

#%% Line search mode.

if args.line_search:
    print('\nStarting line search.\n')
    files_folder = config_folder + exporting_folder
    if args.use_julia:
        arguments = ['classlinesearch.jl', files_folder, ','.join(all_spectra),
                     '--plots_folder', config_folder + plots_folder,
                     '--width', line_width, '--smooth', smooth_factor,
                     '--threshold', intensity_threshold]
        if local_run:
            os.chdir(original_folder)
            arguments[0] = codes_folder + arguments[0]
            arguments = ['julia'] + arguments
    else:
        arguments = ['classlinesearch.py', files_folder, ','.join(all_spectra),
                     '-plots_folder', config_folder + plots_folder,
                     '-width', line_width, '-smooth', smooth_factor,
                     '-threshold', intensity_threshold]
        if local_run:
            os.chdir(original_folder)
            arguments[0] = codes_folder + arguments[0]
            arguments = ['python3'] + arguments
            if args.using_windows_py:
                arguments[0] = 'py'
    if save_plots:
        arguments += ['--save_plots']
    if rolling_sigma_clip:
        arguments += ['--rolling_sigma']
        
    subprocess.run(arguments)

#%% Reduction mode.

if args.reduction:
    
    print('\nStarting reduction.\n')
    files_folder = config_folder + exporting_folder
    if check_windows:
        args.use_julia = False
    if args.use_julia:
        arguments = ['classreduction.jl', files_folder, ','.join(all_spectra),
                     '--plots_folder', config_folder + plots_folder,
                     '--smooth', smooth_factor,
                     '--rms_margin', rms_margin]
        if local_run:
            os.chdir(original_folder)
            arguments[0] = codes_folder + arguments[0]
            os.chdir(original_folder)
            arguments = ['julia'] + arguments
    else:
        arguments = ['classreduction.py', files_folder, ','.join(all_spectra),
                     '-plots_folder', config_folder + plots_folder,
                     '-smooth', smooth_factor,
                     '-rms_margin', rms_margin]
        if local_run:
            os.chdir(original_folder)
            arguments[0] = codes_folder + arguments[0]
            arguments = ['python3'] + arguments
            if args.using_windows_py:
                arguments[0] = 'py'
    if check_windows:
        arguments += ['--check_windows']
    if save_plots and not (args.line_search and args.reduction):
        arguments += ['--save_plots']
        
    subprocess.run(arguments)

# Grouping of the files in the original file format and calibration.
 
    os.chdir(config_folder)
    script = []
    
    with open(exporting_folder + 'doppler_corrections.yaml') as file:
        doppler_corr = yaml.safe_load(file)
    
    if modify_beam_eff:
        with open(exporting_folder + 'frequency_ranges.yaml') as file:
            frequency_ranges = yaml.safe_load(file)
        with open(exporting_folder + 'rms_noises.yaml') as file:
            rms_noises = yaml.safe_load(file)
        beam_effs = {}
    
    final_files = []
    for red_file in output_files:
        final_files += [red_file.replace(ext, '-r' + ext)]
    for (file, red_file) in zip(output_files, final_files):
        script += ['file out ' + output_folder + red_file + ' m /overwrite']
        for spectrum in output_spectra[file]:
            script += ['fits read ' + exporting_folder + spectrum + '-r.fits']
            if modify_beam_eff:
                mean_freq = np.mean(frequency_ranges[spectrum]) * to_GHz
                noise = rms_noises[spectrum]
                keys = list(telescope_effs.keys())
                if keys == ['default']:
                    key = keys[0]
                else:
                    telescope = spectrum.split('-')[-1]
                    key = 'default'
                    for key_i in telescope_effs:
                        if 'default' != key_i:
                            if not '*' in key_i and key_i == telescope:
                                key = key_i
                                break
                            elif '*' in key_i and key_i.replace('*','') in telescope:
                                key = key_i
                                break
                if not key in telescope_effs:
                    raise Exception(f'Missing {key} beam efficiency.')
                beam_eff = np.interp(mean_freq,
                                     telescope_effs[key]['frequency (GHz)'],
                                     telescope_effs[key]['beam efficiency'])
                rms_noises[spectrum] /= float(beam_eff)
                script += ['modify beam_eff {}'.format(round(beam_eff,2))]
                beam_effs[str(spectrum)] = float(beam_eff)
            script += ['modify doppler ' + doppler_corr[spectrum]]
            script += ['write']
            
    if modify_beam_eff:
        
        save_yaml_dict(rms_noises, exporting_folder + 'rms_noises.yaml',
                       default_flow_style=False)
        print(f'Saved calibrated RMS noises in {exporting_folder}'
              'rms_noises.yaml.')
    
        save_yaml_dict(beam_effs, exporting_folder + 'beam_efficiencies.yaml',
                       default_flow_style=False)
        print(f'Saved beam efficiencies in {exporting_folder}'
              'beam_efficiencies.yaml.')
        print()
    
    # End of the script.
    script += ['exit']
    script = [line + '\n' for line in script]
    # Writing of the class file.
    with open('reduction-grouping-1.class', 'w') as file:
        file.writelines(script)
        
    # Running of the class file.
    subprocess.run(['class', '@reduction-grouping-1.class'])
        
# Creation of a final file.

    script = []
    final_files = []
    for red_file in output_files:
        final_files += [red_file.replace(ext, '-r' + ext)]
    sources = []
    for file in output_files:
        sources += [file.split('-')[0]]
    for source in list(np.unique(sources)):
        file_name = output_folder + source + '-' + extra_note + '-all-r' + ext
        file_name = file_name.replace('--','-')
        script += ['file out ' + file_name + ' m /overwrite']
    # Loop for files created in the data selection. 
    for (file, red_file, source) in zip(output_files, final_files, sources):
        script += ['file in ' + output_folder + red_file]
        file_name = output_folder + source + '-' + extra_note + '-all-r' + ext
        file_name = file_name.replace('--','-')
        script += ['file out ' + file_name] 
        script += ['find /all', 'list']
        # Loop for observations of each file.
        for spectrum in output_spectra[file]:
            script += ['get next', 'write']
            
    # End of the script.
    script += ['exit']
    script = [line + '\n' for line in script]
    # Writing of the class file.
    with open('reduction-grouping-2.class', 'w') as file:
        file.writelines(script)
        
    # Running of the class file.
    subprocess.run(['class', '@reduction-grouping-2.class'])
    
    print(f'\nCreated CLASS files:    (folder {output_folder})')
    for file in output_files:
        print('- ' + file.replace(ext, '-r'+ext))
    print()

#%% Averaging mode.

sources, lines = [], []
for file in output_files:
    sources += [file.split('-')[0]]
    lines += [file.split('-')[1]]

if args.averaging:
    
    print('\nStarting averaging mode.\n')

    config_averaging = {}
    config_averaging['extra note'] = extra_note
    config_averaging['spectra folder'] = exporting_folder
    config_averaging['output folder'] = output_folder
    config_averaging['plots folder'] = plots_folder
    config_averaging['class extension'] = ext
    config_averaging['ghost lines'] = options['averaging']['ghost lines']
    config_averaging['averaged spectra'] = options['averaging']['averaged spectra']
    if 'sources-lines-telescopes' in options['averaging']:
        config_averaging['sources-lines-telescopes'] = \
            options['averaging']['sources-lines-telescopes']
    config_averaging['default telescopes'] = options['default telescopes']
    
    save_yaml_dict(config_averaging, 'config-averaging-auto.yaml',
                   default_flow_style=None)
        
    # Running of the script that joints the overlapping spectra.
    os.chdir(original_folder)
    arguments = ['classaveraging.py', config_folder + sep
                 + 'config-averaging-auto.yaml']
    if local_run:
        arguments[0] = codes_folder + arguments[0]
        arguments = ['python3'] + arguments
        if args.using_windows_py:
            arguments[0] = 'py'
    subprocess.run(arguments)

#%% Merging mode.

sources = []
for file in output_files:
    sources += [file.split('-')[0]]

if args.merging:
    
    if not average_all_input_files:
        raise Exception("Merging mode can only be used with the"
            "'average all input files' option in the configuration file.\n")
    
    with open(exporting_folder + 'frequency_ranges.yaml') as file:
        frequency_ranges = yaml.safe_load(file)

    config_merging = {}
    config_merging['extra note'] = extra_note
    config_merging['spectra folder'] = exporting_folder
    config_merging['output folder'] = output_folder
    config_merging['plots folder'] = plots_folder
    config_merging['input files'] = options['merging']
    
    if options['merging'] == 'auto':
        config_merging['input files'] = {}
        for source in list(np.unique(sources)):
            all_spectra = []
            averaged_spectra = []
            ranges = []
            names = []
            for file in output_files:
                for spectrum in output_spectra[file]:
                    if spectrum.startswith(source+'-'):
                        all_spectra += ['-'.join(spectrum.split('-')[1:])]
                        names += [spectrum]
                        ranges += [frequency_ranges[spectrum]]
            ranges = np.array(ranges)
            names = np.array(names)
            inds = np.argsort(ranges[:,1])
            ranges = ranges[inds,:]
            names = list(names[inds])
            i = 0
            while i < len(ranges) - 1:
                difference = ranges[i+1,0] - ranges[i,1]
                if difference < 0:
                    ranges[i,0] = min(ranges[i,0], ranges[i+1,0])
                    ranges[i,1] = max(ranges[i,1], ranges[i+1,1])
                    ranges = np.delete(ranges, i+1, axis=0)
                    names[i] += ',' + names[i+1]
                    names.pop(i+1)
                else:
                    i += 1
            for spectra_group in names:
                if len(spectra_group.split(',')) > 1:
                    averaged_spectra += [spectra_group.split(',')]
                    for (i, spectrum) in enumerate(averaged_spectra[-1]):
                        averaged_spectra[-1][i] = spectrum.replace(source+'-','')
            file_name = source + '-' + extra_note + '-all-r' + ext
            file_name = file_name.replace('--','-')
            config_merging['input files'][file_name] = {}
            config_merging['input files'][file_name]['all spectra'] = all_spectra
            config_merging['input files'][file_name]['overlapping spectra'] = \
                averaged_spectra

    save_yaml_dict(config_merging, 'config-merging-auto.yaml',
                   default_flow_style=False)
        
    # Running of the script that joints the overlapping spectra.
    os.chdir(original_folder)
    arguments = ['classmerging.py', config_folder +sep+'config-merging-auto.yaml']
    if local_run:
        arguments[0] = codes_folder + arguments[0]
        arguments = ['python3'] + arguments
        if args.using_windows_py:
            arguments[0] = 'py'
    subprocess.run(arguments)

#%% Spectra table mode.

if args.spectra_tables:
    
    print('\nCreating spectra table(s).\n')
    
    with open(exporting_folder + 'frequency_ranges.yaml') as file:
        freq_ranges_or = yaml.safe_load(file)
    with open(exporting_folder + 'frequency_resolutions.yaml') as file:
        freq_resolutions_original = yaml.safe_load(file)
    with open(exporting_folder + 'rms_noises.yaml') as file:
        rms_noises_or = yaml.safe_load(file)
        
    freq_ranges, freq_resolutions, rms_noises, names_length = [], [], [], []
    names = []
    for name in freq_resolutions_original:
        if not name.startswith('rms'):
            names += [name]
        
    for name in names:
         freq_ranges += [freq_ranges_or[name]]
         freq_resolutions += [freq_resolutions_original[name]]
         rms_noises += [rms_noises_or[name]]
         names_length += [len(name)]
    
    all_names = np.array(names)
    all_freq_ranges = np.array(freq_ranges)
    all_freq_resolutions = np.array(freq_resolutions)
    all_rms_noises = np.array(rms_noises)
    order = np.argsort(all_freq_ranges[:,1])
    
    for source in np.unique(sources):
        
        cond = np.char.startswith(all_names[order], source + '-')
        names = all_names[order][cond]
        freq_ranges = all_freq_ranges[order,:][cond,:]
        freq_resolutions = all_freq_resolutions[order][cond]
        rms_noises = all_rms_noises[order][cond]
        num_channels = np.array((freq_ranges[:,1] - freq_ranges[:,0])
                                / freq_resolutions, int)
        
        table = pd.DataFrame({'spectrum': names,
                              'channels': num_channels,
                              'min. frequency (MHz)': freq_ranges[:,0],
                              'max. frequency (MHz)': freq_ranges[:,1],
                              'rms noise (mK)': rms_noises,
                              'resolution (MHz)': freq_resolutions})
        file_name = source + '-' + extra_note + '-table.csv'
        file_name = file_name.replace('--','-')
        table.to_csv(exporting_folder + file_name, sep=',', index=False)
        print(f'Saved file {exporting_folder}{file_name}.')
    
        if average_all_input_files:
        
            some_overlap = False
            
            full_ranges = np.sort(freq_ranges.ravel()).reshape(-1,2)
            i = 0
            while i < len(full_ranges)-1:
                a_i, b_i = full_ranges[i,:]
                a_i1, b_i1 = full_ranges[i+1,:]
                for j in range(len(freq_ranges)):
                    a_j = freq_ranges[j,0]
                    b_j = freq_ranges[j,1]
                    if ((b_i < a_i1) and ((a_j < b_i < b_j)
                                          or (a_j < a_i1 < b_j))):
                        some_overlap = True
                        new_range = np.array([b_i, a_i1])
                        full_ranges = np.insert(full_ranges, i+1, new_range,
                                                axis=0)
                        break
                i += 1
            
            if some_overlap:
                
                N = len(full_ranges)
                full_names = [[] for i in range(N)]
                full_noises = [[] for i in range(N)]
                full_resolutions = [[] for i in range(N)]
                for i in range(len(full_ranges)):
                    mean_freq = np.mean(full_ranges[i,:])
                    for j in range(len(freq_ranges)):
                        if freq_ranges[j,0] < mean_freq < freq_ranges[j,1]:
                            full_names[i] += [names[j]]
                            full_noises[i] += [rms_noises[j]]
                            full_resolutions[i] += [freq_resolutions[j]]
                    full_names[i] = ' + '.join(full_names[i])
                    full_noises[i] = \
                        1 / np.sum(1/np.array(full_noises[i])**2)**0.5
                    full_resolutions[i] = max(full_resolutions[i])
                full_channels = np.array((full_ranges[:,1] - full_ranges[:,0])
                                         / full_resolutions, int)
                
                table = pd.DataFrame({'spectrum': full_names,
                                      'channels': full_channels,
                                      'min. frequency (MHz)': full_ranges[:,0],
                                      'max. frequency (MHz)': full_ranges[:,1],
                                      'rms noise (mK)': full_noises,
                                      'resolution (MHz)': full_resolutions})
                file_name = source + '-' + extra_note + '-table-joint.csv'
                file_name = file_name.replace('--','-')
                table.to_csv(exporting_folder + file_name, sep=',', index=False)
                print(f'Saved file {exporting_folder}{file_name}.')
    
    print()
            
#%% RMS check mode, plots.

if args.check_rms_plots:
    
    print('\nStarting noise check: checking of the plots.\n')
    
    bad_scans = {}
    
    for option_list in rms_option_list:
        
        plt.close('all')
        os.chdir(original_folder)
    
        rms_params = (options['rms noise check'][option_list]
                      ['source-line-telescopes'])
        rms_freq_ranges = (options['rms noise check'][option_list]
                           ['frequency ranges (GHz)'])
        group_size = options['rms noise check'][option_list]['scans per group']
    
        source = list(rms_params.keys())[0]
        line = list(rms_params[source].keys())[0]
        telescopes = rms_params[source][line]
        
        s = ''
        if not only_daily_bad_scans and len(bad_scans) > 0:
            s = '-s'
        
        for telescope in telescopes:
            
            os.chdir(original_folder)
            os.chdir(exporting_folder)
            
            image_prefix = 'rms-spectrum-' + '-'.join([source, line, telescope])
            file_prefix = 'rms-' + '-'.join([source, line, telescope, s])
            
            number_file = glob.glob(file_prefix + '*-filenames.csv')[0]
            number_table = pd.read_csv(number_file)
    
            os.chdir(original_folder)
            os.chdir(plots_folder)
            
            for frequency_range in rms_freq_ranges:
                
                frequency_range = 1000 * np.array(frequency_range)
                rms_freq_text = '({:.0f}-{:.0f})'.format(*frequency_range)
                plot_images = glob.glob('{}*{}*.png'.format(image_prefix,
                                                            rms_freq_text))
                sort_numbers = []
                
                for image in plot_images:
                    scans = image.split(')-')[-1].split('.png')[0]
                    if not scans.startswith('to'):
                        sort_number = float(scans.split('+')[0])
                    else:
                        sort_number = 1e5 + float(scans[2:])
                    sort_numbers += [sort_number]
                
                inds = np.argsort(sort_numbers)
                plot_images = list(np.array(plot_images)[inds])
        
                plt.figure(1)
            
                next_plot = True
                i = 0
                
                while next_plot:
                    
                    i = max(0, i)
                    i = min(i, len(plot_images))
        
                    image = plot_images[i]
                    plt.clf()
                    plt.imshow(plt.imread(image))
                    plt.axis('off')
                    plt.tight_layout()
                    plt.pause(0.1)
                    print(f'File {image}')
                    image_elements = image.replace('rms-spectrum-','').split('-')
                    source = image_elements[0]
                    line = image_elements[1]
                    telescope = f'*{image_elements[2]}*'
                    scans = image_elements[-1].replace('.png','')
                    
                    if not scans.startswith('to'):
                        input_order = \
                            input('Options: Next (Enter), Previous (<), '
                                  + 'Select (x), Skip (s). ')
                    else:
                        input_order = \
                            input('Options: Next (Enter), Previous (<), '
                                  + 'Skip (s). ')
                            
                    if input_order == 'x' and not scans.startswith('to'):   
                        scans = scans.split('+')
                        scan1 = number_table['scan'][int(scans[0])-1]
                        scan2 = number_table['scan'][int(scans[-1])-1]
                        bad_scans_i = f'{scan1}:{scan2}'
                        if source not in bad_scans:
                            bad_scans[source] = {}
                        if line not in bad_scans[source]:
                            bad_scans[source][line] = {}
                        if telescope not in bad_scans[source][line]:
                            bad_scans[source][line][telescope] = []
                        bad_scans[source][line][telescope] += [bad_scans_i]
    
                    if input_order == '' or input_order == 'x':
                        i += 1
                    elif input_order == '<':
                        i -= 1
                    if input_order == 's' or i == len(plot_images):
                        next_plot = False
    
    for source in bad_scans:
        for line in bad_scans[source]:
            for telescope in bad_scans[source][line]:
                bad_scans[source][line][telescope] = \
                    sorted(bad_scans[source][line][telescope])
                
    bad_scans = {'bad scans': bad_scans}
    
    os.chdir(original_folder)
    os.chdir(config_folder.replace('.','') + exporting_folder.replace('./','')) 
    
    # Export of the bad scans.
    save_yaml_dict(bad_scans, 'bad_scans.yaml', default_flow_style=None)
    print()
    print(f'Saved selected bad scans in {exporting_folder}bad_scans.yaml.')
    print()
                
    plt.close('all')
                             
       
#%% Ending.

backup_files = glob.glob(exporting_folder + sep + '*.dat~')
# backup_files = glob.glob(exporting_folder + sep + '*.fits~')
if len(backup_files) != 0:
    for file in backup_files:
        os.remove(file)
        
temp_files = ['selection.class', 'selection-doppler.class', 'selection-grouping.class',
              'rms_check-info.class', 'rms_check-ind.class', 'rms_check-cum.class',
              'reduction-grouping-1.class', 'reduction-grouping-2.class',
              'config-averaging-auto.yaml', 'averaging.class', 'averaging-grouping.class',
              'config-merging-auto-yaml', 'merging.class']
for file in temp_files:
    if os.path.exists(file):
        os.remove(file)
        
time2 = time.time()
total_time = int(time2 - time1)
minutes, seconds = total_time//60, total_time%60
text = f'The processing of the data has finished in {minutes} min + {seconds} s.'
if minutes == 0:
    text = text.replace('0 min + ', '')
    if seconds == 0:
        text = text.replace('0 s', 'less than 1 s')
phases = []

for arg, descr in zip([args.selection, args.line_search, args.reduction,
                       args.merging, args.averaging, args.spectra_tables,
                       args.rms_check, args.check_rms_plots],
                      ['selection', 'line search', 'reduction',
                       'merging', 'averaging', 'spectra table', 'rms check',
                       'checking of rms plots']):
    if arg:
        phases += [descr]
        
text += ' ({})'.format(', '.join(phases))

print(text)
print()

if not args.rms_check:
    plt.show()
    
operating_system = platform.system()
if operating_system == 'Windows' and not local_run:
    input()