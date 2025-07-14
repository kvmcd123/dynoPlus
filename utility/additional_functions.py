import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import hilbert
import h5py
import math

def moving_average(a, n=10):
  ret = np.cumsum(a, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return ret / n

bord=6
blev=0.05
flt_sos_lp = signal.butter (bord, blev, btype='lowpass', output='sos')
flt_sos_hp = signal.butter (bord, blev, btype='highpass', output='sos')

def my_filter(a, hp=False):
  if hp:
    raw = np.abs(signal.sosfiltfilt (flt_sos_hp, a, padtype=None))
    avg = moving_average(raw, n=30)
    return signal.sosfiltfilt (flt_sos_lp, avg, padtype=None)
#    analytic_signal = hilbert(raw)
#    return np.abs(analytic_signal)
  else:
    return signal.sosfiltfilt (flt_sos_lp, a, padtype=None)

def load_or_zero_signal (grp, key, dlen):
  y = np.zeros(dlen)
  if key in grp:
    grp[key].read_direct (y)
  return y

def process_unbalanced_data():
    inputfiles = ['data/unb3raw.hdf5']
    fout = h5py.File ('c:/data/unb3.hdf5', 'w')
    idx = 0
    for filename in inputfiles:
        fp = h5py.File(filename, 'r')
        for grp_name, grp in fp.items():
            print ('processing', grp_name)
            grpout = fout.create_group ('case{:d}'.format(idx))
            idx += 1
            sigs = {}
            dlen = grp['t'].len()
            for key in ['t', 'Ud', 'Uq', 'Id', 'Iq', 'I0', 'Vd', 'Vq', 'V0']:
                sigs[key] = load_or_zero_signal (grp, key, dlen)
        
            # create the low-pass and high-pass filtered dq0 currents and voltages
            sigs['Idlo'] = my_filter (sigs['Id'], hp=False)
            sigs['Idhi'] = my_filter (sigs['Id'], hp=True)
            sigs['Iqlo'] = my_filter (sigs['Iq'], hp=False)
            sigs['Iqhi'] = my_filter (sigs['Iq'], hp=True)
            sigs['I0lo'] = my_filter (sigs['I0'], hp=False)
            sigs['I0hi'] = my_filter (sigs['I0'], hp=True)

            sigs['Vdlo'] = my_filter (sigs['Vd'], hp=False)
            sigs['Vdhi'] = my_filter (sigs['Vd'], hp=True)
            sigs['Vqlo'] = my_filter (sigs['Vq'], hp=False)
            sigs['Vqhi'] = my_filter (sigs['Vq'], hp=True)
            sigs['V0lo'] = my_filter (sigs['V0'], hp=False)
            sigs['V0hi'] = my_filter (sigs['V0'], hp=True)

            for key in ['Ud', 'Uq', 'Vdlo', 'Vqlo', 'V0lo', 'Vdhi', 'Vqhi', 'V0hi', 'Vrms','Idlo', 'Idhi', 'Iqlo', 'Iqhi', 'I0lo', 'I0hi']:
                grpout.create_dataset (key, data=sigs[key], compression='gzip')

    print ('created {:d} training cases'.format(idx))