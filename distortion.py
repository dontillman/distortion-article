# Copyright 2022, Donald Tillman.  All rights reserved.
# Generate plots for the distortion articles.

from numpy import amax, append, arange, array, clip, ndarray, exp, expand_dims
from numpy import fft, full_like, linspace, logspace, log10, pi, shape, sign
from numpy import sin, sqrt, sum, tanh, vectorize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import os

imgdir = 'images'

# A "tf" (transfer function) provides a nonlinearity, 
# takes an input of -1.0 to +1.0, 
# and nominally maps that to -1.0 to +1.0.

# These tf_ functions build transfer functions from transfer functions
# in a consistent way.  So you use them to wrap each other, without 
# having to make any lambdas.

# Given a transfer function, return a new transfer function
# scaled and translated so the output goes from min_out to max_out
# (typically -1 to +1) as the input goes from -1 to +1.  
# This will remove an inversion if present.
def tf_normalize(tf, min_out=-1, max_out=+1):
    scaled = (max_out - min_out) /(tf(1) - tf(-1))
    offset = min_out - scaled * tf(-1)
    return lambda x: offset + scaled * tf(x)

# Given a transfer function, return a new transfer function
# that's attenuated or overdriven at its input.
def tf_gain(tf, g=1):
    return lambda x: tf(g * x)

# Given a transfer function, return a new transfer function
# of a push-pull pair.
def tf_pushpull(tf, match=1.0):
    return lambda x: match * tf(x) - tf(-x) / match

# Given a transfer function, return a new function 
# of two of these in series.
def tf_series(tf, inv=True):
    if inv:
        return lambda x: -1 * tf(-1 * tf_normalize(tf)(x))
    return lambda x: tf(tf_normalize(tf)(x))

# Given a transfer function, return a new function with a negative
# feedback factor of b.  Since feedback reduces the gain, we
# compensate by adding a gain of 1/(b+1) ahead.
def tf_feedback(tf, b=0):
    ftf = tf_normalize(tf_gain(tf, b + 1))
    def result(x):
        # solve for zero around the feedback loop
        def sfcn(xx):
            return x - b * ftf(xx) - xx
        return ftf(fsolve(sfcn, x))
    return result

# Make a "longtail pair", two transfer function instances 
# sharing a common current source.
# 
# drive is the gain before, you might want to set it to 2.0,
# since there are two of them.
def tf_longtail(tf, drive=1.0):
    # apply drive and normalize function to 0..1
    ftf = tf_gain(tf_normalize(tf, 0, 1), drive)

    def result(vin):
        def sfcn(v0):
            return 1 - ftf(vin - v0) - ftf(-vin - v0)
        v0 = fsolve(sfcn, 0.0 * vin)
        # return tf(drive * vin - v0) - tf(-drive * vin - v0)
        return ftf(vin - v0) - ftf(-vin - v0)
    return result

# Practical nonlinear transfer functions.
# Each of these tf_ functions has a default offset or drive parameter 
# for 10% harmonic distortion.

def tf_exponential(x, drive=0.4): 
    return tf_normalize(lambda xx: exp(drive * xx))(x)

def tf_parabolic(x, offset=2.5):
    return tf_normalize(lambda xx: clip(xx + offset, 0, None) ** 2)(x)

def tf_tanh(x, drive=1.295):
    return tanh(drive * x)

# Vacuum tube equation
# K and mu are for a 12AX7
def tube(vg, vp, K = 1.73e-6, mu = 83.5):
    return K * (clip(mu * vg + vp, 0, None) ** 1.5)

# we can't run this on a 2d arrayfor fsolve() reasons 
def tubecircuit(vg, vb=400, rp=100e+3):
    return fsolve(lambda vp: vp - (vb - rp * tube(vg, vp)),
                  full_like(vg, vb / 2))
    
# 12AX7 circuit, level just below clipping
def tf_tube(x, vb=400, rp=100e+3):
    return tf_normalize(lambda xx: tubecircuit(xx * 2.0 - 2.1))(x)

# Returns a normalized spectrum as array[dc, fund, 2nd, 3rd,...]
# For broadcasting, returns a spectrum for each wave
def dist_spectrum(x):
    spect = fft.rfft(x, norm='forward')
    fund = spect[..., 1]
    if shape(fund):
        return abs(spect / expand_dims(fund, -1))
    else:
        return abs(spect / fund)

def dist_spectrum_thd(s, compensation=False):
    return sqrt(sum(s[..., 2:]**2, axis=-1)) / s[..., 1]

# 6dB/oct boost harmonics above second
def thd_comp(a):
    return a * clip(arange(10), 2, None) / 2

def print_spectrum_dbs(s, harm_max=11):
    print('dBs: ' + ', '.join(f'{n}: {v:.0f}'
                              for n, v in enumerate(db(s)[:harm_max])))
def thd_curve(fcn):
    ls = logspace(-4, 0, 49)

# The inputs, used by the folks below.
xs = linspace(-1, 1, 256, endpoint=False)
vin = sin(xs * pi)

# Plot the transfer function curve on the axis.
# Include a sideways sine on the bottom.
def plot_transfer_function(a, fcn):
    a.set_title('Transfer Function', fontsize=11)
    a.tick_params(labelsize=9)
    a.spines[:].set_color('gray')
    a.set_xlabel('Input Signal')
    a.set_xticks(linspace(-1, 1, 5))    
    a.set_ylabel('Output Signal')
    a.set_yticks(linspace(-1, 1, 5))    
    a.set_ylim((-1.35, 1.15))
    a.plot([-1, 1], [-1, 1], color='lightgray')
    a.axhline(0, linewidth=1, color='lightgreen')
    a.axvline(0, linewidth=1, color='lightgreen')
    a.plot(xs, fcn(xs), color='#333333')
    a.plot(*(bakers(0.15 * xs - 1, vin)[-1::-1]), color='lightgray')

# Plot the distorted sine wave on the axis.
def plot_waveform(a, fcn):
    a.set_title('Distorted Sine Wave', fontsize=11)
    a.tick_params(labelsize=9)    
    a.spines[:].set_color('gray')  
    a.set_xlabel('One cycle')
    a.set_xticks(linspace(-180, 180, 5))
    a.set_ylabel('Output Signal')
    a.set_ylim((-1.35, 1.15))
    a.axhline(0, linewidth=1, color='lightgreen')
    a.axvline(0, linewidth=1, color='lightgreen')
    a.plot(*bakers(180 * xs, vin), color='lightgray')
    a.plot(*bakers(180 * xs, fcn(vin)), color='black')

# Draw a bar graph for the distortion spectrum on the axis.
def plot_dist_spectrum(a, fcn, val_min=1.0e-4, harm_max=10):
    a.set_axisbelow(True)
    a.tick_params(labelsize=9)
    a.spines[:].set_color('gray')
    a.set_title('Harmonic Distortion Spectrum', fontsize=11)
    a.set_xlabel('Harmonic')
    a.set_xlim((1, harm_max))
    a.set_xticks(arange(1, harm_max))    
    a.set_ylabel('dB')
    a.set_ylim((-80, 0))
    a.grid(axis='y', color='#cccccc')

    spect = dist_spectrum(fcn(vin))
    ns = array([n
               for n,h in enumerate(spect[:harm_max  + 1])
               if 2 <= n and val_min <= h])
    if any(ns):
        harm_levels = spect[ns]
        bars = a.bar(ns,
                    80 + db(harm_levels),
                    bottom=-80, 
                    width=.5,
                    color=colors[ns])
        a.bar_label(bars, percent(harm_levels), fontsize=9)
    a.text(9.7, -8, f'{percent(dist_spectrum_thd(spect))} THD', 
           fontsize=9, 
           horizontalalignment='right')     

# Draw a pretty triptych of nonlinear curve, distorted sine,
# and distortion bar graph.
def triptych(fcn, title='Title', filename=None):
    nfcn = tf_normalize(fcn)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3), 
                            dpi=72,
                            facecolor='none',
                            gridspec_kw={'width_ratios': [.55, .7, .7]})

    plot_transfer_function(axs[0], nfcn)
    plot_waveform(axs[1], nfcn)
    plot_dist_spectrum(axs[2], nfcn)
    plt.tight_layout()
    fig.suptitle(title, fontweight='bold')
    fig.subplots_adjust(top=0.82)
    if filename:
        plt.savefig(os.path.join(imgdir, filename))

# Harmonic Distortion over levels
#levels = logspace(0,-4, 49)

# Plot the distortion products as they change with signal level
def plot_harmonics_level(fcn, and_abs=False, title='', filename='', level_max=6, harm_max=10):
    fig, axs = plt.subplots(1, 1, figsize=(4, 4), dpi=72)
    fig.suptitle(title, fontweight='bold')
    axs.set_title('Harmonic Levels With Drive Level')
    axs.set_xlabel('Relative Drive Level dB')
    axs.set_ylabel('Harmonic Level dB')
    axs.set_ylim(-105, 5)
    axs.set_xlim(-105, 5)
    axs.grid(color='#cccccc')
    axs.axvline(0, linewidth=1, color='gray')
    axs.axhline(0, linewidth=1, color='gray')

    levelsdb = linspace(-100.0, level_max, 200)
    vouts = array([fcn(fromdb(lev) * vin)
                   for lev in levelsdb])
    curves = db(dist_spectrum(vouts)[..., 2:harm_max]).T
    for i, curve in enumerate(curves, start=2):        
        if -100 < amax(curve):
            axs.plot(levelsdb, curve, color=colors[i], label=ordinal(i))
            if and_abs:
                axs.plot(levelsdb, curve + levelsdb, color=colors[i], linestyle='dashed')
            axs.legend(loc='upper left')
    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(imgdir, filename))

# Plot the distortion components with level for multiple transfer functions.
def plot_thd_level(fcns):
    fig, axs = plt.subplots(1, 1, figsize=(8, 4), dpi=72)
    axs.set_xlabel('Relative Drive Level dB')
    axs.set_ylabel('THD Level dB')
    axs.set_ylim(-80, 0)
    levelsdb = linspace(-100.0, 6, 200)
    vins = fromdb(levelsdb)[:, None] * vin
    if callable(fcns):
        fcns = [['', fcns]]
    for name, fcn in fcns:
        thd = db(dist_spectrum_thd(dist_spectrum(fcn(vins))))
        axs.plot(levelsdb, thd, label=name)
    axs.legend()

        
# Utilities

# (Close to...) EIA color code
colors = array(('black', 'brown', 'red', 'orange',  '#e0d030', 'green', 'blue', 'violet', 'gray', 'pink', 'black'))

# x to dB
def db(x, low_clip=1e-12):
    return 20 * log10(clip(abs(x), low_clip, None))

def fromdb(x):
    return 10.0 ** (x / 20.0)

def ordinal(n):
    suffixes = { 1: "st", 2: "nd", 3: "rd" }
    nn = n if (n < 20) else (n % 10)
    return f'{n}{suffixes.get(nn, "th")}'

# Given x and y arrays, add a point to the end ("Bakers Dozen")
# to make the drawings prettier.
def bakers(xs, ys):
    return append(xs, 2 * xs[-1] - xs[-2]), append(ys, ys[0])

def percent(x):
    if ndarray == type(x):
        return array(list(map(percent, x)))
    return f'{round(100 * x, 2 if x < .01 else 1)}%'


