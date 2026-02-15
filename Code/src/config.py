# Abnormal Reaction Detection System
# Copyright (C) 2026
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import datetime

# Define Project Root (Assuming this file is in Code/src/)
# Code/src/config.py -> Code/src -> Code -> Root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../'))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'Results')

def get_output_dirs(base_dir=None):
    """
    Creates and returns a dictionary of output directories.
    If base_dir is None, creates a timestamped directory in Results/.
    """
    if base_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(RESULTS_ROOT, timestamp)
    
    dirs = {
        'BASE': base_dir,
        'MODELS': os.path.join(base_dir, 'Models'),
        'DATA': os.path.join(base_dir, 'Data'),
        'FIGURES': os.path.join(base_dir, 'Figures'),
        'REPORTS': os.path.join(base_dir, 'Reports'),
        'LOGS': os.path.join(base_dir, 'Logs')
    }
    
    # Create directories
    for key, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        
    return dirs
