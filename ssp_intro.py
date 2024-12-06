import os

import matplotlib.pyplot as plt
import numpy as np

import mne
from mne.preprocessing import (
    compute_proj_ecg,
    compute_proj_eog,
    create_ecg_epochs,
    create_eog_epochs,
)
