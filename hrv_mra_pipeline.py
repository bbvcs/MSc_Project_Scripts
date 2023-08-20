import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
from scipy import fft
from scipy import interpolate
from scipy import signal
from scipy import stats
from scipy.signal import butter, sosfiltfilt, sosfreqz
import pandas as pd

import os
import sys
import json
import time
import datetime
import math
import shelve

import constants

sys.path.append(constants.SPEEDYF_LOCATION)
from speedyf import edf_collate, edf_overlaps, edf_segment

sys.path.append(constants.HRV_PREPROC_LOCATION)
from hrv_preprocessor.hrv_preprocessor import hrv_per_segment, produce_hrv_dataframes, save_hrv_dataframes, load_hrv_dataframes, time_dom_keys, freq_dom_keys

# TODO REMEMBER TO CITE PACKAGES
import emd  # EMD, EEMD
#from PyEMD import EMD
import pywt # Wavelet Transform?
import vmdpy # Variational Mode Decomposition
from pyts import decomposition

from neurodsp.sim import sim_powerlaw
from neurodsp.utils import set_random_seed

ULTRADIAN_COLOR = "#DED5E8"
CIRCADIAN_COLOR ="#CCE5D7"
INFRADIAN_COLOR ="#EED8CD"

ULTRADIAN_MIN, ULTRADIAN_MAX = 0, 20
CIRCADIAN_MIN, CIRCADIAN_MAX = 20, 30
INFRADIAN_MIN, INFRADIAN_MAX = 30, 320#168#72 


HARMONICS_24H = [1/(multiplier*(1/(24*60*60)))/60/60 for multiplier in np.arange(2, 100)] 
#HARMONICS_24H = [48, 16, 8, 4.8, 3.4, 2.8, 2.1, 1.8]

""" UTILITY FUNCTIONS """

def butter_lowpass(cutoff, fs, order=5):
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	sos = butter(order, normal_cutoff, btype="low", analog=False,output="sos")
	return sos

def butter_lowpass_filter(data, cutoff, fs, order=5):
	sos = butter_lowpass(cutoff, fs, order=order)
	y = sosfiltfilt(sos, data)
	return y

def test_butter_lowpass(cutoff=(1/(22*60*60)), fs=1/300, order=5):
	sos = butter_lowpass(cutoff, fs, order)
	w, h = sosfreqz(sos, worN=8000)
	
	fig, ax = plt.subplots()

	ax.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
	#plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
	ax.axvline(cutoff, color="k")
	ax.set_xlim(0, 0.5*fs)
	ax.set_title("Lowpass Filter Frequency Response")
	ax.set_xlabel("Frequency [Hz]")
	ax.grid()
	fig.show()

def butter_bandpass(lowcut, highcut, fs, order=5):    
	sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, analog=False,output="sos")
	return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	sos = butter_bandpass(lowcut, highcut, fs, order)
	y = sosfiltfilt(sos, data) 	
	return y

def test_butter_bandpass(lowcut=(1/(26*60*60)), highcut=(1/(22*60*60)), fs=1/300, order=5):
	sos = butter_bandpass(lowcut, highcut, fs, order)
	w, h = sosfreqz(sos, worN=8000)
	
	fig, ax = plt.subplots()

	ax.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
	#plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
	ax.axvline(lowcut, color="k")
	ax.axvline(highcut, color="k")
	ax.set_xlim(0, 0.5*fs)
	ax.set_title("Bandpass Filter Frequency Response")
	ax.set_xlabel("Frequency [Hz]")
	ax.grid()
	fig.show()


def convert_unixtime_ms_to_datetime(unix_epoch_ms):

	# remove ms from unix epoch
	unix_epoch = np.floor(unix_epoch_ms / 1000)

	# convert to datetime format
	timestamp = datetime.datetime.fromtimestamp(unix_epoch)

	# get ms 
	ms = unix_epoch_ms - (unix_epoch * 1000)

	# add ms
	return timestamp + datetime.timedelta(milliseconds=ms)


def components_plot_old(components, original, out, title, gaps, sharey=True):
	fig, ax = plt.subplots(len(components)+2, 1, sharex=True, sharey=sharey, figsize=(19.20, 19.20))

	x = range(0, len(original))
	alpha = 0.3 # transparency of the interpolated sections
	c = "blueviolet" # color of components

	ax[0].plot(x, original, color="black", alpha=alpha)
	ax[0].plot(x, cut_gaps(original, gaps), color="black")
	ax[0].set_title("Original", loc="left")

	reconstruction = np.sum(components, axis=0)
	ax[1].plot(x, reconstruction, color="black", alpha=alpha)
	ax[1].plot(x, cut_gaps(reconstruction, gaps), color="black")
	ax[1].set_title("Reconstruction", loc="left")	
	
	for i, comp in enumerate(components):
		ax[i+2].plot(x, comp, color=c, alpha=alpha)
		ax[i+2].plot(x, cut_gaps(comp, gaps), color=c)
		ax[i+2].set_title(f"Component #{i}", loc="left")

	fig.suptitle(title)

	plt.subplots_adjust(bottom=0.04, top=0.921, hspace=0.402)
	
	fig.savefig(os.path.join(out, title))
	return fig, ax


def components_plot(timevec, components, original, out, title, gaps, onsets, durations, sharey=True):


	plot_title = title_fullname(title)


	out = os.path.join(out, "components_plots")
	if not os.path.exists(out):
		print("Setting up directory for MRA Output: at '{}'".format(out))
		os.makedirs(out, exist_ok=True)


	""" 1. Plot Components and 2. Plot PSDs of Components"""

	sz = 16
	fig1, ax1 = plt.subplots(len(components)+2, 1, sharex=True, sharey=sharey, figsize=(sz, sz)) # the components
	plt.subplots_adjust(bottom=0.10, top=0.921, hspace=0.402)
	fig2, ax2 = plt.subplots(len(components)+2, 1, sharex=True, sharey=False, figsize=(sz, sz)) # PSDs of the components
	plt.subplots_adjust(bottom=0.10, top=0.921, hspace=0.402)

	#x_original = range(0, len(original))
	#x_imfs = range(0, components.shape[1])
	x_original = timevec
	x_imfs = timevec[0: components.shape[1]]
	alpha = 0.3 # transparency of the interpolated sections
	c = "blueviolet" # color of components
	
	fs = 1/300 # TODO this needs to be param

	ax2_xticks = np.array([0, 1/(24*60*60), 1/(18*60*60),1/(14*60*60),1/(12*60*60),1/(10*60*60),1/(8*60*60),1/(6*60*60),1/(4*60*60),1/(2*60*60),1/(60*60)])	
	ax2_xticklabels = (1/ax2_xticks) # invert to frequency is period (s)
	ax2_xticklabels = (ax2_xticklabels / 60) / 60 # get from s into hours 
	ax2_xticklabels = [np.round(lab, decimals=1) if lab != np.inf else lab for lab in ax2_xticklabels]
	ax2_xlim = [0, 1/(60*60)] # 0 -> 1h period
	ax2_xticklab_rot = 45

	ax1[0].plot(x_original, original, color="black", alpha=alpha)
	ax1[0].plot(x_original, cut_gaps(original, gaps), color="black")
	ax1[0].set_title("Original", loc="left")

	f, Pxx_den = psd(original, fs)
	original_total_power = np.sum(Pxx_den)
	ax2[0].plot(f, Pxx_den, alpha=0.25)
	ax2[0].stem(f, Pxx_den)
	#ax2[0].set_xticks(ticks=f, labels=1/f)
	ax2[0].set_xticks(ticks=ax2_xticks, labels=ax2_xticklabels, rotation=ax2_xticklab_rot)
	#ax2[0].axvline((1/(24*60*60)), c="r", label="Circadian")
	ax2[0].set_xlim(ax2_xlim)
	ax2[0].set_title("Original", loc="left")

	reconstruction = np.sum(components, axis=0)
	ax1[1].plot(x_imfs, reconstruction, color="black", alpha=alpha)
	ax1[1].plot(x_imfs, cut_gaps(reconstruction, gaps), color="black")
	ax1[1].set_title("Reconstruction", loc="left")	
	
	f, Pxx_den = psd(reconstruction, fs)
	reconstruction_total_power = np.sum(Pxx_den)
	ax2[1].plot(f, Pxx_den, alpha=0.25)
	ax2[1].stem(f, Pxx_den)
	#ax2[1].set_xticks(ticks=f, labels=1/f)
	ax2[1].set_xticks(ticks=ax2_xticks, labels=ax2_xticklabels, rotation=ax2_xticklab_rot)
	#ax2[1].axvline((1/(24*60*60)), c="r", label="Circadian")
	ax2[1].set_xlim(ax2_xlim)
	ax2[1].set_title("Reconstruction", loc="left")	

	components_total_powers = []
	#components_to_ignore = []	
	for i, comp in enumerate(components):
		ax1[i+2].plot(x_imfs, comp, color=c, alpha=alpha)
		ax1[i+2].plot(x_imfs, cut_gaps(comp, gaps), color=c)
		ax1[i+2].set_title(f"Component #{i}", loc="left")

		f, Pxx_den = psd(comp, fs)
		component_total_power = np.sum(Pxx_den)
		components_total_powers.append(component_total_power)
		ax2[i+2].plot(f, Pxx_den, alpha=0.25)
		ax2[i+2].stem(f, Pxx_den)
		ax2[i+2].set_xticks(ticks=ax2_xticks, labels=ax2_xticklabels, rotation=ax2_xticklab_rot)
		#ax2[i+2].axvline((1/(24*60*60)), c="r", label="Circadian")
		ax2[i+2].set_xlim(ax2_xlim)
		ax2[i+2].set_title(f"Component #{i}", loc="left")

		#if ((component_total_power/reconstruction_total_power) * 100) <= 1: # if contributes less than 1% to the original total power, ignore this IMF
			#ax1[i+2].set_facecolor("grey") 
			#ax2[i+2].set_facecolor("grey") 
			#components_to_ignore.append(i)

	"""
	components_total_powers_sorted = np.array(sorted(components_total_powers, reverse=True))
	components_cumulative_power = np.cumsum(components_total_powers_sorted / reconstruction_total_power)
	for i, pct in enumerate(components_cumulative_power):
		if pct > 0.9:
			ignored_component =  np.where(components_total_powers == components_total_powers_sorted[i])[0][0]

			ax1[ignored_component+2].set_facecolor("grey") 
			ax2[ignored_component+2].set_facecolor("grey") 
			components_to_ignore.append(ignored_component)
	"""

	""" # TODO try with 0.10?
	components_total_powers_scaled = np.array(components_total_powers) / reconstruction_total_power
	for i, pct in enumerate(components_total_powers_scaled):
		if pct < np.quantile(components_total_powers_scaled, 0.25):
			ax1[i+2].set_facecolor("grey") 
			ax2[i+2].set_facecolor("grey") 
			components_to_ignore.append(i)
	"""		

	
	#print(title)
	#print(np.array(components_total_powers)/original_total_power * 100)
	#print(np.quantile(np.array(components_total_powers)/original_total_power * 100, 0.25))


	soz = None
	fig1.suptitle(plot_title+" Components")
	for ax in ax1:
		ax.xaxis.set_major_formatter(DateFormatter("%d/%m %H:%M:%S"))
		
		for onset, duration in zip(onsets, durations):
			soz = ax.axvspan(onset, onset + datetime.timedelta(seconds=duration), color="r", alpha=0.5)
	
	fig1.supxlabel("Time (d/m h:m:s)")

	
	fig2.suptitle(plot_title+" Components PSDs")
	fig2.supylabel("Power")
	fig2.supxlabel("Period (h)")
	handles, labels = ax2[-1].get_legend_handles_labels() # get from last axis only, to avoid duplicates
	fig2.legend(handles, labels)


	
	greyscale = None
	""" 3. Produce plot where components are stacked, and the peak (maximum) frequency highlighted for each component.""" 
	minimum_rhythmic_period_s = 3000 # 3000s=50m	

	peaks_powers = {}  # dict of peak period -> power
	cmap = matplotlib.colormaps["ocean"]
	fig3, ax3 = plt.subplots(figsize=(10.80, 10.80)) 
	for i, comp in enumerate(components):

		f, Pxx_den = psd(comp, fs) # REPEATED CODE

		peak_idx = np.where(Pxx_den == np.max(Pxx_den))[0][0]
		peak_Pxx = Pxx_den[peak_idx]
		peak_freq = f[peak_idx]
		peak_period_s = 1/peak_freq
		peak_period_h = peak_period_s/60/60
		#peak_period_h = np.round(peak_period_h, decimals=2)
		label = f"Component #{i}, peak period = {peak_period_h}h"
		color = cmap(1/len(components) * i)
	
		ax3.plot(f, Pxx_den, c=color, label=label)
		ax3.fill_between(f, Pxx_den, color=color)
	
		#if not i in components_to_ignore: # TODO THIS IS NO LONGER USED. SWITCHED TO MINIMUM RHYTHMIC PERIOD.	
		if peak_period_s >= minimum_rhythmic_period_s:

			if not peak_period_h in peaks_powers.keys():
				peaks_powers[peak_period_h] = peak_Pxx
			else:
				# if using welch for psd est, common to have duplicate peak periods. 
				if peak_Pxx > peaks_powers[peak_period_h]:
					# take the max power of this period as its power
					peaks_powers[peak_period_h] = peak_Pxx
			

			ax3.scatter(peak_freq, peak_Pxx, marker="x", c="r", zorder=10)
			ax2[i+2].stem(peak_freq, peak_Pxx, markerfmt="r")# zorder=10)
			ylim_min, ylim_max = ax2[i+2].get_ylim()
			ax2[i+2].text(1/(0.98*60*60), np.median([ylim_min,ylim_max]), f"{np.round((1/peak_freq)/60/60, decimals=2)}h", fontsize="x-large")
		else:
			ax1[i+2].set_facecolor("grey") 
			ax2[i+2].set_facecolor("grey") 
			ylim_min, ylim_max = ax2[i+2].get_ylim()
			ax2[i+2].text(1/(0.98*60*60), np.median([ylim_min,ylim_max]), f"{np.round((1/peak_freq)/60/60, decimals=2)}h", fontsize="x-large", color="grey")


	if soz:
		fig1.legend([soz], ["Seizures"])

	fig1.savefig(os.path.join(out, title))
	plt.close(fig1)
	
	fig2.savefig(os.path.join(out, title+"_PSD"))
	plt.close(fig2)

	ax3.set_xscale("log")
	ax3.legend(loc="upper right")
	fig3.suptitle(title+" Components PSDs Overlaid")
	fig3.supxlabel("Frequency (Hz)")
	fig3.savefig(os.path.join(out, title+"_PSD_OVERLAY"))
	plt.close(fig3)

	return peaks_powers


def psd(data, sample_rate):
	# FFT
	data = data - np.mean(data) # remove DC Offset (0Hz Spike)
	yf = fft.rfft(data)
	xf = fft.rfftfreq(len(data), 1/sample_rate)
	return xf, np.abs(yf)
	"""
	
	len_data_m = len(data) * 5
	len_data_h = len_data_m/60

	# Welch
	hours_per_seg = 48 #len_data_h*0.75 #48
	overlap_hours = 40 #hours_per_seg * 0.75 #40
	min_per_seg = hours_per_seg * 60	
	nperseg = (min_per_seg//5) # how many 5min HRV segments in our welch segment
	return signal.welch(data, sample_rate, nperseg=nperseg, noverlap=((overlap_hours*60)//5), scaling="density")	
	
	"""
	


def psd_plot(data, sample_rate): # warn; can't use fft with HRV, irregularly sampled
	fig, ax = plt.subplots()
	xf, yf = psd(data, sample_rate)
	ax.plot(xf, np.abs(yf))

	return fig, ax

def extend(data, out=None, metric=None):
	#print("REMOVE ME NO EXTEND")
	#return data
		
	padlen = len(data) // 2

	data = data.copy()
	
	# 1. 
	m = np.mean(data)
	data = data - m

	# 2. TODO more complicated based on if SLOPE is close to 0 at either end
	#extended = np.pad(data, padlen, mode="median")

	regress_n = 20 # how many points for regression?
	left_x = np.arange(0, regress_n)
	left_y = data[0:regress_n]
	left_res = stats.linregress(left_x, left_y)

	right_x = np.arange(len(data)-regress_n, len(data))
	right_y = data[-regress_n:]
	right_res = stats.linregress(right_x, right_y)



	plotting_x = np.arange(0, (len(data)+(2*padlen)))
	if out != None:
		lc = "mediumpurple"
		rc = "limegreen"
		linewidth = 3

		fig, ax = plt.subplots(5, figsize=(10.8, 10.8))
		plt.subplots_adjust(hspace=0.42)
		ax[0].plot(plotting_x[padlen:padlen+len(data)], data, c="black")
		ax[0].set_title(f"Original Data (Mean Subtracted) with Linear Regression Slopes", loc="left")
		ax[0].set_xlim([padlen, padlen+len(data)])
		ax[0].plot(padlen+left_x, left_res.intercept + left_res.slope * left_x, c=lc, label=left_res.slope, linewidth=linewidth)
		#ax[0].text(padlen+left_x[0], left_y[0]+10, left_res.slope, c="b")
		ax[0].plot(padlen+right_x, right_res.intercept + right_res.slope * right_x, c=rc, label=right_res.slope, linewidth=linewidth)
		#ax[0].text(padlen+right_x[0], right_y[0]+10, right_res.slope, c="r")
		ax[0].legend()

	SLOPE_THRESH = 0.1

	if np.abs(left_res.slope) < SLOPE_THRESH:
		left_mode = "symmetric"
	else:
		left_mode = "wrap"
	extended_left = np.pad(data, padlen, left_mode)
	print(f"Left {left_mode} {left_res.slope}")

	if np.abs(right_res.slope) < SLOPE_THRESH:
		right_mode = "symmetric"
	else:
		right_mode = "wrap"
	extended_right = np.pad(data, padlen, right_mode)
	print(f"Right {right_mode} {right_res.slope}")
	
	extended = np.concatenate((extended_left[0:padlen], data, extended_right[-padlen:]))

	# 3. 
	chi = np.concatenate((np.linspace(0, 1, padlen), np.ones(shape=len(data)), np.linspace(1, 0, padlen))) 
	extended = extended * chi

	# 4.
	extended = extended + m

	if out != None:
		def verbose_mode(mode):
			if mode == "symmetric":
				return f"Symmetric Padding (Slope Below {SLOPE_THRESH})"
			else: 
				return f"Anti-Symmetric (Wrap) Padding (Slope Above {SLOPE_THRESH})"

		ax[1].plot(extended_left, c=lc, alpha=0.2)
		ax[1].plot(plotting_x[0:padlen+len(data)], extended_left[0:padlen+len(data)], c=lc)
		ax[1].set_title(f"Left Extension: {verbose_mode(left_mode)}", loc="left")

		ax[2].plot(extended_right, c=rc, alpha=0.2)
		ax[2].plot(plotting_x[padlen:], extended_right[padlen:], c=rc)
		ax[2].set_title(f"Right Extension: {verbose_mode(right_mode)}", loc="left")

		ax[3].plot(plotting_x[:padlen+1], extended_left[:padlen+1], c=lc)
		ax[3].plot(plotting_x[padlen+len(data):], extended_right[padlen+len(data):], c=rc)
		ax[3].plot(plotting_x[padlen:padlen+len(data)], data, c="black")
		ax[3].set_title(f"Extension Result", loc="left")

		ax[4].plot(plotting_x[padlen:padlen+len(data)], extended[padlen:padlen+len(data)], c="black")
		ax[4].plot(plotting_x[:padlen+1], extended[:padlen+1], c=lc)
		ax[4].plot(plotting_x[padlen+len(data):], extended[padlen+len(data):], c=rc)
		ax[4].set_title(f"Extension Result + Taper + Original Data Mean", loc="left")

		ax[0].sharex(ax[3])

		out = os.path.join(out, "preprocessing_plots")
		if not os.path.exists(out):
			os.makedirs(out, exist_ok=True)

		fig.suptitle("Signal Boundary Extension")
		fig.savefig(os.path.join(out, f"{metric}_extend"))
		plt.close(fig)

	return extended

def remove_extend(data):
	"""Remove data added by extend()"""
	#print("REMOVE ME NO EXTEND (def remove_extend)")
	#return data
	
	padlen = len(data) // 4	 # AS LONG AS padlen = len(data)//2 in extend(), this should work?
	return data[padlen:-padlen]

def get_gaps_positions(data):
	# get idx of runs of NaNs in data
	# (start_idx, end_idx) inclusive; so start:end+1 is the run. data[end] is NaN, data[end+1] shouldn't be.
	gaps = []
	on_NaN = False
	p = 0
	while p < len(data):
	    if np.isnan(data[p]) and not on_NaN:
		    NaN_start = p
		    on_NaN = True
	    elif on_NaN and not np.isnan(data[p]):
		    NaN_end = p-1
		    gaps.append((NaN_start, NaN_end))
		    on_NaN = False

	    if p == len(data)-1:
		    if on_NaN:
			    NaN_end = p
			    gaps.append((NaN_start, NaN_end))

	    p += 1

	return gaps


def cut_gaps(data, gaps):
	
	copy = data.copy()
	
	for gap in gaps:
		copy[gap[0]:gap[1]+1] = np.NaN

	return copy

def interpolate_gaps(data, timevec, ax, out, metric):

	out = os.path.join(out, "preprocessing_plots")
	if not os.path.exists(out):
		os.makedirs(out, exist_ok=True)

	# find outlier spikes
	fig, ax_outliers = plt.subplots(2, 1, figsize=(10.8, 10.8))
	ax_outliers[0].plot(timevec, data, c="black")
	"""
	rolling_mean   = np.zeros(shape=len(data))
	rolling_std    =  np.zeros(shape=len(data))
	rolling_zscore = np.zeros(shape=len(data))
	for j in range(0, len(data)):
		width = 12 # each datapoint represents 5m, so window is 5 * width m long duration
		window = data[max(j-(width//2),0):min(j+(width//2), len(data))]
		rolling_mean[j] = np.nanmean(window)
		rolling_std[j] = np.nanstd(window)
		rolling_zscore[j] = (data[j] - (rolling_mean[j]))/rolling_std[j]
	ax_outliers.plot(timevec, rolling_mean, c="orange", alpha=1, label="Rolling Mean")
	ax_outliers.plot(timevec, rolling_std, c="lime", alpha=1, label="Rolling Std. Dev.")
	ax_outliers.plot(timevec, rolling_zscore, c="deepskyblue", alpha=1, label="Rolling Z-Score")
	"""	
	Z_THRESH = 3

	zscores = np.abs(stats.zscore(data, nan_policy="omit"))	
	ax_outliers[1].plot(timevec, zscores, c="blue")
	ax_outliers[1].axhline(Z_THRESH, c="r", alpha=0.3, label=f"Z-Score Threshold ({Z_THRESH})")
	ax_outliers[1].set_ylabel("Absolute Z-Score")

	outliers = np.where(zscores > Z_THRESH)[0]
	ax_outliers[0].scatter(np.array(timevec)[outliers], data[outliers], marker="x", c="r", label=f"Outliers")
	fig.supxlabel("Time")
		

	"""
	# try to find jumps (spikes, but more prolonged; might be missed by spike detection method)
	J_THRESH = 1.5 * np.nanmean(rolling_mean)
	jumps = np.where(data >= J_THRESH)[0]
	ax_outliers.axhline(J_THRESH, c="g", alpha=0.3, label=f"1.5*Median of Rolling Mean Threshold ({J_THRESH})")
	ax_outliers.scatter(np.array(timevec)[jumps], data[jumps], marker="x", c="g", label="Outliers detected via 2*Mean of Rolling Mean")
	"""	

	fig.legend()	
	fig.suptitle(f"{metric} Outlier Detection via Z-Score Threshold")
	fig.savefig(os.path.join(out, f"{metric}_outlier_spikes"))
	plt.close(fig)

	# remove outlier spikes, will be interpolated
	data[outliers] = np.NaN
	ax[1].plot(timevec, data, c="black")
	ax[1].set_title("Outliers Removed", loc="left")
	ax[1].set_xlim([min(timevec), max(timevec)]) 



	# get gaps (runs of NaN) (includes singular NaN)
	gaps = get_gaps_positions(data)


	# interpolate short gaps
	if len(gaps) > 0:
		short_gap_idx = []
		
		thresh = len(data) * 0.01
		for gap in gaps:
			gap_length = gap[1] - gap[0]
			if gap_length < thresh:
				for gap_idx in range(gap[0], gap[1]+1):
					short_gap_idx.append(gap_idx)
			
		if len(short_gap_idx) > 0:
			x = np.array(range(0, len(data)))[~pd.isnull(data)]
			y = data[~pd.isnull(data)]

			interpolator = interpolate.interp1d(x, y, bounds_error=False, fill_value = np.NaN)#, kind="nearest")
			interpolated = np.array([x if i not in short_gap_idx else interpolator(i) for i, x in enumerate(data)])	
			
			data = interpolated
			
			ax[2].plot(timevec, data, c="black")
			ax[2].set_title("Short Gaps (< 1% Data Length) Interpolated", loc="left")
			ax[2].set_xlim([min(timevec), max(timevec)]) 

			# get new gaps after we've filled in shorter ones 
			gaps = get_gaps_positions(data)

	if len(gaps) > 0:
		# get islands (runs of data between gaps)
		islands = [(gaps[n][1]+1, gaps[n+1][0]) for n in range(0, len(gaps)-1)]
		if gaps[0][0] != 0: 
			islands.insert(0, (0, gaps[0][0]))
			gaps.insert(0, np.NaN) # add in a fake gap to account for fact we've prepended island (so alg works for any gaps besides first) 
		if gaps[-1][1] != len(data)-1:
			islands.append((gaps[-1][1]+1, len(data)))
		
		small_islands_removed=False
		# remove small, isolated islands of data; islands with lengths < 10% of data length, surrounded by gaps with total length > 10% data len
		thresh = len(data) * 0.10
		for n, island in enumerate(islands):
			
			# only consider  

			island_length = island[1] - island[0]
			if island_length < thresh:

				if island[0] == 0:
					total_surrounding_gap_length = gaps[n+1][1]-gaps[n+1][0]
				elif island[1] == len(data):
					total_surrounding_gap_length = gaps[n][1]-gaps[n][0]
				else:
					total_surrounding_gap_length =  (gaps[n][1]-gaps[n][0]) + (gaps[n+1][1]-gaps[n+1][0])

				if total_surrounding_gap_length > thresh:
					data[island[0]:island[1]] = np.NaN
					small_islands_removed=True
		
		if small_islands_removed:

			ax[3].plot(timevec, data, c="black")
			ax[3].set_title("Small Islands (< 10% Data Length, with Surrounding Gaps > 10% Data Length) Removed", loc="left")
			ax[3].set_xlim([min(timevec), max(timevec)]) 
			
			gaps = get_gaps_positions(data)

	# remove gaps that start at start/end at end (will not be interpolated, and will mess up decomposition)
	if len(gaps) > 0:

		if not isinstance(gaps[0], tuple): # if we prepended a NaN in small island removal, remove it
			gaps = gaps[1:]

		NaN_ends_removed = False

		if gaps[-1][1] == len(data)-1:
			data = data[:gaps[-1][0]]
			timevec = timevec[:gaps[-1][0]] # need to ensure timevec corresponds
			NaN_ends_removed = True


		if gaps[0][0] == 0:
			data = data[gaps[0][1]+1:]
			timevec = timevec[gaps[0][1]+1:]
			NaN_ends_removed = True
	
		if NaN_ends_removed:
			ax[4].plot(timevec, data, c="black")
			ax[4].set_title("Runs of NaN at Start/End Removed", loc="left")
			ax[4].set_xlim([min(timevec), max(timevec)]) 
		
	
		gaps = get_gaps_positions(data) # find new indicies of gaps as we may have changed length

	x = np.array(range(0, len(data)))[~pd.isnull(data)]
	y = data[~pd.isnull(data)]

	interpolator = interpolate.interp1d(x, y, bounds_error=False, fill_value = np.NaN)#, kind="nearest")
	interpolated = np.array([x if not pd.isnull(x) else interpolator(i) for i, x in enumerate(data)])	
	
	ax[5].plot(timevec, interpolated, c="black")
	ax[5].set_title("Any Remaining Gaps Interpolated", loc="left")
	ax[5].set_xlim([min(timevec), max(timevec)]) 

	return interpolated, timevec, gaps

def get_seizures(subject):

	severity_table = pd.read_excel("SeverityTable.xlsx")

	subject_seizures = severity_table[severity_table.patient_id == subject]
	# lots of extra useful information in here, not just start

	return subject_seizures["start"], subject_seizures["duration"]


def metric_fullname(metric):
	mapping = {
		"fft_ratio": "LF/HF Ratio",
		"hr_mean": "Mean IHR",
		"tri_index": "Triangular Index",
		"fft_rel_VLF": "Relative VLF Power",
		"fft_rel_LF": "Relative LF Power",
		"fft_rel_HF": "Relative HF Power",
		}
	if metric in mapping.keys():
		return mapping[metric]
	else:
		return str(metric).upper()
	

def method_fullname(method):

	if "DWT" in method:
		
		mapping = {"DWT_bior4-4": "DWT Biorthogonal 4.4 Wavelet", 
			"DWT_coif4": "DWT Coiflet 4", 
			"DWT_db4": "DWT Daubechies 4 Wavelet", 
			"DWT_dmey": "DWT Discrete Meyer Wavelet", 
			"DWT_sym4": "DWT Symlet 4"}

		return mapping[method]
	else:
		return method

def title_fullname(title):

	if "DWT" in title:

		method = title.split("_DWT_")[1]
		method = f"DWT_{method}"
	else:
		method = title.split("_")[-1]

	metric = title.split(f"_{method}")[0]

	return f"{method_fullname(method)} {metric_fullname(metric)}"


""" PIPELINE FUNCTIONS """

def run_speedyf(root, out):

	edf_collate(root, out)

	if len(edf_overlaps.check(root, out, verbose=True)) != 0:
		edf_overlaps.resolve(root, out)


def calculate_hrv_metrics(root, out, rng, forced=False):

	segmenter = edf_segment.EDFSegmenter(root, out, segment_len_s=300, cache_lifetime=1)	

	ecg_channels = [ch for ch in segmenter.get_available_channels() if ("ecg" in ch.lower()) or ("ekg" in ch.lower())]
	if len(ecg_channels) == 0: 
		raise KeyError("No channels with names containing 'ECG' could be found!")
	segmenter.set_used_channels(ecg_channels)


	try:
		load_hrv_dataframes(out)			

		if not forced:
			print("HRV Metrics Dataframes appear to exist, and parameter forced=False, so HRV Metrics will not be re-calculated.") 
			return

	except FileNotFoundError:
		pass

	"""
	for ecg_channel in ecg_channels:

		try:
			if not len(ecg_channels) == 1:
				load_hrv_dataframes(os.path.join(subject_out, ecg_channel))
			else:
				load_hrv_dataframes(out)			

			if not forced:
				print("HRV Metrics Dataframes appear to exist, and parameter forced=False, so HRV Metrics will not be re-calculated.") 
				return

		except FileNotFoundError:
				pass

	
	for ecg_channel in ecg_channels:

		if not len(ecg_channels) == 1:
			print(f"Running for ECG channel {ecg_channel}")
			out = os.path.join(subject_out, ecg_channel)
			if not os.path.exists(out):
				print(f"Setting up directory for {ecg_channel} HRV output (subject has multiple ECG channels): at '{out}'")
				os.makedirs(out, exist_ok=True)
	"""




	freq_dom_hrvs = []
	time_dom_hrvs = []
	modification_reports = []

	print("Gathering Segments...")
	for segment in segmenter:
		print(f"{segment.idx}/{segmenter.get_max_segment_count()-1}")

		#ecg = segment.data[ecg_channel].to_numpy()
		ecg = segment.data.to_numpy()
		# lowpass filter with cutoff of 22Hz to remove high frequency noise in ECG, make QRS detector's job easier
		if len(ecg) != 0:
			if len(ecg.shape) > 1:
				ecg = ecg.T
				for ch in range(0, ecg.shape[0]):
					ecg[ch] = butter_lowpass_filter(ecg[ch], 22, segment.sample_rate, order=4)
			else:
				ecg = butter_lowpass_filter(ecg, 22, segment.sample_rate, order=4)

		#eps = 0.125
		eps = 0.14
		min_samp = 35

		rpeaks, rri, rri_corrected, freq_dom_hrv, time_dom_hrv, modification_report = hrv_per_segment(ecg, segment.sample_rate, 5, segment_idx=segment.idx, save_plots_dir=os.path.join(out, "saved_plots"), save_plots=True, save_plot_filename=segment.idx, use_segmenter="engzee", DBSCAN_RRI_EPSILON_MEAN_MULTIPLIER=eps, DBSCAN_MIN_SAMPLES=min_samp, rng=rng)
		#print(modification_report["notes"])
		
		if not isinstance(freq_dom_hrv, float):
			freq_dom_hrvs.append(np.array(freq_dom_hrv, dtype="object"))
		else:
			freq_dom_hrvs.append(np.full(shape=freq_dom_keys.shape, fill_value=np.NaN))

		if not isinstance(time_dom_hrv, float):
			time_dom_hrvs.append(np.array(time_dom_hrv))
		else:
			time_dom_hrvs.append(np.full(shape=time_dom_keys.shape, fill_value=np.NaN))

		modification_reports.append(modification_report)


	segment_labels = np.array(range(0, segmenter.get_max_segment_count()))
	
	time_dom_df, freq_dom_df, modification_report_df = produce_hrv_dataframes(time_dom_hrvs, freq_dom_hrvs, modification_reports, segment_labels)	
	save_hrv_dataframes(time_dom_df, freq_dom_df, modification_report_df, out)	


def mra(out, metric, timevec, data, gaps, onsets, durations, sharey=False):

	fs = 1/300 # TODO this should be param


	data = extend(data, out, metric)
	
	# # # #
	#print("REMOVE ME MEAN CENteR AND LOWPASS")
	#data = data - np.mean(data)
	#data = butter_lowpass_filter(data, 1/(1*60*60), fs=fs, order=12)
	#data = butter_lowpass_filter(data, 1/(0.5*60*60), fs=fs, order=48)
	# # # #
	

	methods_peaks = {}  # dict of str mode decomposition method -> dict of peak periods of the components identified by those methods and the power WITHIN THE COMPONENT of that period

	# perform EMD
	title = f"{metric}_EMD"
	imfs = emd.sift.sift(data).T
	#emd2 = EMD(); imfs = emd2(data)
	# TODO remove extension from data, imfs
	methods_peaks["EMD"] = components_plot(timevec, np.apply_along_axis(remove_extend, 1, imfs), remove_extend(data), out, title, gaps, onsets, durations, sharey=sharey)

	# perform VMD (using CFSA to determine optimal K)
	title = f"{metric}_VMD"
	# initialise parameters
	alpha = 2000 # bandwidth constraint (2000 = "moderate") 
	tau = 0.      # noise-tolerance (no strict fidelity enforcement)  
	K = len(imfs)         # n of modes to be recovered  
	DC = 0        # no DC part imposed  
	init = 1      # initialize omegas uniformly 
	tol = 1e-7
	
	"""	
	#for alpha in [1, 10, 50, 100, 250, 500, 800, 1000, 1200, 1500, 1800, 1900, 2000, 2250, 2500, 2750, 3000]:	
	ks = []
	ns = []
	same_value_count = 0
	exit_condition = False
	while not exit_condition:
		u, u_hat, omega = vmdpy.VMD(data, alpha, tau, K, DC, init, tol)
		u = np.flipud(u)
		max_centre_frequency = 0
		center_periods = []
		for imf in u:
			f, Pxx_den = psd(imf, fs)
			#imf = imf-np.mean(imf)
			#Pxx_den = np.abs(fft.rfft(imf))
			#f = fft.rfftfreq(len(imf), 1/fs)

			# find center frequency (assumedly frequency with most power?)
			peak_idx = np.where(Pxx_den == np.max(Pxx_den))[0][0]
			peak_Pxx = Pxx_den[peak_idx]
			peak_freq = f[peak_idx] # center frequency
			peak_period_s = 1/peak_freq
			peak_period_h = peak_period_s/60/60
			
			#if peak_period_h < 0.9:
			#	continue	
	
			center_periods.append(peak_period_h)
		
		# produce center period histogram
		#counts, bins = np.histogram(center_periods, bins=len(center_periods))		
		#center_periods_rounded = np.round(center_periods, decimals=0)
		center_periods_rounded = center_periods
		bins = sorted(np.unique(center_periods_rounded))
		counts = [np.sum(center_periods_rounded == val) for val in bins]

		fig, ax = plt.subplots()
		#avg_count = np.mean(counts[counts>0])
		avg_count = np.mean(counts)
		#ax.hist(bins[:-1], bins, weights=counts)
		#ax.hist(center_periods, weights=counts)
		ax.bar(bins, counts)
		ax.axhline(avg_count, c="r")
		

		#N = np.sum(center_periods > np.mean(center_periods))		
		N = np.sum(counts > avg_count) # how many different frequencies are above the mean count of frequencies 
		#N = np.sum(counts[counts >= np.mean(counts[counts>0])])


		ks.append(K)
		ns.append(N)

		
		if (N>0) and (len(ns)>1) and N == ns[-2]:
			same_value_count += 1
		else:
			same_value_count = 0

		
		if (same_value_count >= 20) or (len(ns) > 1) and (N>0) and (N < ns[-2]):#(N <= ns[-2]):
			# N has stopped increasing, so set K to previous N value
			K = N
			#K = ns[-2]
			exit_condition = True
		else:
			K += 1

	"""

	u, u_hat, omega = vmdpy.VMD(data, alpha, tau, K, DC, init, tol)
	u = np.flipud(u)
	methods_peaks["VMD"] = components_plot(timevec, np.apply_along_axis(remove_extend, 1, u), remove_extend(data), out, title, gaps, onsets, durations, sharey=sharey)
	

	# perform EEMD
	title = f"{metric}_EEMD"
	np.random.seed(1905)
	imfs = emd.sift.ensemble_sift(data, nprocesses=1).T
	imfs = imfs[1:]
	methods_peaks["EEMD"] = components_plot(timevec, np.apply_along_axis(remove_extend, 1, imfs), remove_extend(data), out, title, gaps, onsets, durations, sharey=sharey)
	
	# perform DWT MRA
	for wavelet in ["sym4", "db4", "bior4.4", "coif4", "dmey"]:
		data_wt = data if len(data) % 2 == 0 else data[:-1]  # must be odd length
		timevec_wt = timevec if len(timevec) % 2 == 0 else timevec[:-1]
		output = pywt.mra(data_wt, pywt.Wavelet(wavelet), transform="dwt")
		output = np.flip(output, axis=0)
		if '.' in wavelet:
			title = f"{metric}_DWT_{'-'.join(wavelet.split('.'))}" # will mess up saving to file if . present in title
			methods_peaks[f"DWT_{'-'.join(wavelet.split('.'))}"] = components_plot(timevec_wt, np.apply_along_axis(remove_extend, 1, output), remove_extend(data_wt), out, title, gaps, onsets, durations, sharey=sharey)	
		else:
			title = f"{metric}_DWT_{wavelet}"
			methods_peaks[f"DWT_{wavelet}"] = components_plot(timevec_wt, np.apply_along_axis(remove_extend, 1, output), remove_extend(data_wt), out, title, gaps, onsets, durations, sharey=sharey)	

	# perform SSA (Singular Spectrum Analysis)
	title = f"{metric}_SSA"
	#components = decomposition.SingularSpectrumAnalysis(window_size=0.10, groups=K).fit_transform(data.reshape(1, -1))
	components = decomposition.SingularSpectrumAnalysis(window_size=max(2, K)).fit_transform(data.reshape(1, -1))
	components = components[0]
	components = np.flipud(components)
	methods_peaks["SSA"] = components_plot(timevec, np.apply_along_axis(remove_extend, 1, components), remove_extend(data), out, title, gaps, onsets, durations, sharey=sharey)


	# perform CWT 
	methods_peaks["CWT"] = wavelet_transform(timevec, data, gaps, onsets, durations, metric, out)	


	"""
	# filter out the peaks identified	
	methods=["EMD", "EEMD", "VMD", "DWT_coif4"]	
	for method in methods:
		peaks = methods_peaks[method].keys()

		fig, axs = plt.subplots(len(peaks)+2, 1, sharex=True, figsize=(19.20, 19.20)) 
		plt.subplots_adjust(bottom=0.04, top=0.921, hspace=0.402)

		
		fontsize = "smaller"

		axs[0].plot(timevec, remove_extend(data), alpha=0.5, color="black")
		axs[0].plot(timevec, cut_gaps(remove_extend(data), gaps), color="black")
		axs[0].set_title("Original Data", loc="left", fontsize=fontsize)

		filtered_data = []
		for i, peak in enumerate(peaks):

			if peak == np.inf:
				continue

			fs = 1/300 # TODO this should be param
			order=4
		
			if 1/((peak)*60*60) < fs/2:
				lowcut=1/((peak+1)*60*60)
				
				if peak > 1:
					highcut_period = peak-1
				else: 
					highcut_period = peak/2

				highcut = 1/((highcut_period)*60*60) 

				if highcut < fs/2:


					#test_butter_bandpass(lowcut, highcut, fs, order)

					bandpass_filtered = butter_bandpass_filter(data, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
					
					color = "dodgerblue"
					axs[i+1].plot(timevec, remove_extend(bandpass_filtered), alpha=0.5, color=color)
					axs[i+1].plot(timevec, cut_gaps(remove_extend(bandpass_filtered), gaps), color=color)
					axs[i+1].set_title(f"Peak @ {peak}h (Filtered {np.round(highcut_period, decimals=2)}h - {np.round((peak+1), decimals=2)}h)", loc="left", fontsize=fontsize)
					
					filtered_data.append(bandpass_filtered)
				else: print(f"Highcut Frequency {highcut} exceeds Nyquist {fs/2}, so skipping this Peak")
			else: print(f"Peak Frequency {1/(peak*60*60)} exceeds Nyquist {fs/2}, so skipping this Peak")

		if len(peaks) > 0:
			filtered_data = np.sum(np.array(filtered_data), axis=0,)
			axs[-1].plot(timevec, remove_extend(filtered_data), alpha=0.5, color="black")
			axs[-1].plot(timevec, cut_gaps(remove_extend(filtered_data), gaps), color="black")
			axs[-1].set_title("Filtered Signal Summation", loc="left", fontsize=fontsize)

		
		for ax in axs:
			ax.xaxis.set_major_formatter(DateFormatter("%d/%m %H:%M:%S"))
			
			for onset, duration in zip(onsets, durations):
				ax.axvspan(onset, onset + datetime.timedelta(seconds=duration), color="r", alpha=0.5)
		
		fig.supxlabel("Time (d/m h:m:s)")
		fig.suptitle(f"Filtered_Peaks_{method}")
		fig.savefig(os.path.join(out, "components_plots", f"Filtered_Peaks_{method}"))
		plt.close(fig)

	"""
	# convert methods_peaks to df

	unique_peaks = set()
	for method, peaks_powers in methods_peaks.items():
		unique_peaks = unique_peaks.union(peaks_powers.keys()) 
	unique_peaks = sorted(unique_peaks)
	
	methods_peaks_df = pd.DataFrame(columns=unique_peaks, index=methods_peaks.keys())
	for method, peaks_powers in methods_peaks.items():
		for peak, power in peaks_powers.items():
			methods_peaks_df.loc[method][peak] = power

	methods_peaks_df = methods_peaks_df.astype(np.float64)
	methods_peaks_df.index.names = ["method"]
	methods_peaks_df.columns.names = ["cycle_period"]

	return methods_peaks_df


def wavelet_transform(timevec, data, gaps, onsets, durations, metric, out):


	fs = 1/300 # TODO this needs to be param
	#w = 5 # default Omega0 param for morlet2 (5.0). Seems to control frequency of complex sine part?
	

	fig, axs = plt.subplots(3,1,sharex=True, height_ratios=[2, 1, 7], figsize=(19.2, 19.2))

	# remove these 2 lines and code should work as before
	fig, ax = plt.subplots(figsize=(10, 7))
	axs[2] = ax # so axs[2] is now ax

	
	axs[0].plot(timevec, remove_extend(data), color="black", alpha=0.5)
	axs[0].plot(timevec, cut_gaps(remove_extend(data), gaps), color="black")
	axs[0].set_title("Original Data", loc="right")

	data = data - np.mean(data) # remove DC offset, otherwise a lot of power at very low frequencies	

	#freqs = np.array([1/(24*60*60), 1/(18*60*60),1/(14*60*60),1/(12*60*60),1/(10*60*60),1/(8*60*60),1/(6*60*60),1/(4*60*60),1/(2*60*60),1/(1*60*60)])	
	#freqs = np.linspace(freqs[0], freqs[-1], 10000)
	#freqs = np.linspace(0, freqs[-1], 10000)
	#freqs = 
	#freqs = np.linspace(0, fs/2, 10000)

	#some old ways to define periods
	#periods = 1/freqs
	#periods = np.array([days*24*60*60 for days in range(13, 1, -1)] + [hours*60*60 for hours in range(47, 0, -1)])
	#periods = np.array([hours*60*60 for hours in range(288, 0, -1)])
	
	# way to define periods I actually used for a bit
	#periods = np.array([hours*60*60 for hours in np.arange(73, 0, -1)])
	periods = np.array([hours*60*60 for hours in np.arange(200, 0, -1)])
	freqs = 1/periods
	
	#w = np.linspace(30, 3, len(periods)) # omega (for when top was 73)
	w = np.linspace(82, 3, len(periods)) # 200 (new top) / 73 = 2.73972... . So 30 *2.73 gives a new top omega, ~82
	#w = 5

	widths = w * fs / (2 * freqs * np.pi)

	#cwtmatr = signal.cwt(data, signal.morlet2, widths, w=w, dtype=np.complex128)
	cwtmatr = np.empty((len(widths), len(data)), dtype=np.complex128) # adapted from signal.cwt, so that omega (w) can be varied
	for ind, width in enumerate(widths):
		N = np.min([10 * width, len(data)])
		wavelet_data = np.conj(signal.morlet2(N, width, w=w[ind])[::-1])
		cwtmatr[ind] = signal.convolve(data, wavelet_data, mode='same')


	cwtmatr_yflip = cwtmatr	
	#cwtmatr_yflip = np.flipud(cwtmatr) # TODO can keep this, just need to flip y axis labels
	
	# one or the other, as cwtmatr is complex
	#cwtmatr_yflip = np.abs(cwtmatr_yflip) # get magnitude
	cwtmatr_yflip = np.power(np.abs(cwtmatr_yflip), 2) # magnitude **2 : power?
	#cwtmatr_yflip = cwtmatr_yflip.real 

	# remove padding	
	cwtmatr_yflip = np.apply_along_axis(remove_extend, 1, cwtmatr_yflip)

	interpolation = "antialiased"#"none"
	#axs[2].imshow(cwtmatr_yflip, vmax = abs(cwtmatr).max(), vmin = -abs(cwtmatr).max(), aspect="auto", interpolation=interpolation)
	#axs[2].imshow(cwtmatr_yflip, vmax = abs(cwtmatr).max(), vmin = 0, aspect="auto", interpolation=interpolation)
	#pos = axs[2].imshow(cwtmatr_yflip, aspect="auto", interpolation=interpolation, cmap='Greens_r')
	
	pos = axs[2].pcolormesh(timevec, np.round(periods/60/60, decimals=0), cwtmatr_yflip)#, norm=matplotlib.colors.LogNorm(vmin=cwtmatr_yflip.min(), vmax=cwtmatr_yflip.max()))	

	cbar = fig.colorbar(pos, ax=axs[2], label="Magnitude$^2$")#location="bottom", shrink=0.6)
	"""	
	yticks = ax.get_yticks()
	yticks = yticks[yticks>=0]
	yticks[yticks > 0] -= 1
	axs[1].set_yticks(ticks = yticks, labels=periods[np.int32(yticks)])
	"""
	#axs[2].set_yticks(ticks=range(0, len(cwtmatr)), labels=np.floor(periods/60/60))
	axs[2].set_ylabel("Period (log(h))")
	axs[2].set_title("Continuous Wavelet Transform Time-Frequency Plot")

	lowcut=1/(33*60*60)
	highcut=1/(21*60*60) 
	fs=fs 
	order=4
	
	try:
		# IF USING MAGNITUDE:
		circadian_bandpass = butter_bandpass_filter(data, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
		axs[1].plot(timevec, remove_extend(circadian_bandpass), c="r", alpha=0.5)
		axs[1].plot(timevec, cut_gaps(remove_extend(circadian_bandpass), gaps), c="r", label="Circadian Band (21h-33h)")
		axs[1].set_title("Bandpass-filtered Circadian Rhythm (21h-33h)", loc="right")

		"""
		# IF USING REAL PROJECTION:
		period = 24
		circadian_waveletfilt = cwtmatr_yflip[np.where(periods==period*60*60)][0]	
		axs[1].plot(timevec, circadian_waveletfilt, c="r", alpha=0.5)
		axs[1].plot(timevec, cut_gaps(circadian_waveletfilt, gaps), c="r", label="Circadian (24h) Wavelet Filtered")
		axs[1].set_title(f"Circadian Rhythm ({period}h Narrowband Wavelet Filtered)", loc="right")
		"""

	except ValueError as ve:
		print(ve)
		pass


	

	for ax in axs:
		ax.xaxis.set_major_formatter(DateFormatter("%d/%m %H:%M:%S"))
		
		#for onset, duration in zip(onsets, durations):
		#	ax.axvspan(onset, onset + datetime.timedelta(seconds=duration), color="r", alpha=0.5)


	axs[2].set_yscale("log")
	axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=15, ha="right")
	axs[2].set_xlabel("Time (d/m h:m:s)")
	fig.savefig(os.path.join(out, "components_plots", f"{metric}_CWT"))
	plt.close(fig)	


	
	peaks_powers = {}  # dict of peak period -> power
	fig, ax = plt.subplots(figsize=(7, 5))
	
	# terrence and compo
	#wavelet_power_spectum = np.power(cwtmatr_yflip, 2) # pg 65, 3d. NOTE cwtmatr_yflip should be defined as np.abs(cwtmatr_yflip) for this to make sense
	#global_wavelet_power_spectrum = np.mean(wavelet_power_spectrum, axis = 1)
	global_wavelet_power_spectrum = np.mean(cwtmatr_yflip, axis = 1)
	global_wavelet_power_spectrum_total = np.sum(global_wavelet_power_spectrum)

	ax.stem(periods/60/60, global_wavelet_power_spectrum, linefmt="black")

	for peak in signal.find_peaks(global_wavelet_power_spectrum, prominence=100)[0]:
		# if power of peak contributes more than 1% of total power
		#if ((global_wavelet_power_spectrum[peak] / global_wavelet_power_spectrum_total) * 100) > 1:
		period = periods[peak]/60/60
		power = global_wavelet_power_spectrum[peak]
		ax.scatter(period, power, c="r", zorder=4)
		#ax.text(period-2, power + ((ax.get_ylim())[1]*0.05), f"{np.round(period, 2)}h", rotation=90, color="r", weight="bold")
		#ax.text(period + np.log(period), power, f"{np.round(period, 2)}h", rotation=0, color="r", weight="bold")
		ax.text(period, -((ax.get_ylim())[1]*0.03), f"{np.int32(np.round(period, 0))}h", rotation=0, color="r", weight="bold", ha="center", va="center")
		peaks_powers[period] = power

	ax.set_ylabel("Mean Wavelet Power (Magnitude$^2$)")
	ax.set_xscale("log")
	ax.set_xlabel("Period (log(h))")
	ax.set_title("Global Wavelet Power Spectrum")	

	"""
	for n in range(0, cwtmatr_yflip.shape[1]):
		ax.stem(periods/60/60, np.power(np.abs(cwtmatr_yflip[:, n]), 2))
	"""
	
	fig.savefig(os.path.join(out, "components_plots", f"{metric}_CWT_power_spectrum"))
	plt.close(fig)

	return peaks_powers


""" SIMULATED SUBJECTS """

def rhythm(timevec, hours, A=10, phi=0):
	# A = amplitude; sinewave will be centered around 0, with max amplitude at A and minimum at -A 
	# phi = phase - where, in radians, the cycle is at t=0
	return A * np.sin((2*np.pi*(1/(hours*60*60))*timevec) + phi)

def rhythm_chirp(timevec, hours0, t1, hours1, A=10, phi=0, method="linear"):
	# A = amplitude; sinewave will be centered around 0, with max amplitude at A and minimum at -A 
	# phi = phase - where, in radians, the cycle is at t=0
	return A * signal.chirp(timevec, f0=1/(hours0*60*60), t1 = t1, f1=1/(hours1*60*60), method=method, phi=phi)


def rhythm_square(timevec, hours, A=10, phi=0):
	# convolved square wave rhythm
	
	rhythm = signal.square(2*np.pi*(1/(hours))*((timevec/60/60) + phi)) # TODO is phi right here?

	return rhythm * A

def rhythm_square_conv(timevec, hours, A=10, phi=0):
	# convolved square wave rhythm
	
	rhythm = signal.square(2*np.pi*(1/(hours))*((timevec/60/60) + phi)) # TODO is phi right here?
	#rhythm = np.convolve(rhythm, np.hanning(75), "same") # smooth transitions
	rhythm = np.convolve(rhythm, np.bartlett(25), "same") # smooth transitions
	rhythm = stats.zscore((rhythm - np.mean(rhythm))) * A # re-center and set to desired amplitude

	return rhythm



def simulated_subject(code):
	
	segment_len_s = 300
	n_days = 7 # similar length to 909
	n_segments = (n_days * (24*60))/5 # how many segments of len segment_len_s?
	data_length = n_segments * segment_len_s # this is so we can use the same sample rates
	timevec = np.arange(0, data_length, segment_len_s)
 	
	baseline = 85 # add this to sinewave, to shift it from being mean-centered, so it resembles a mean IHR
	
	data = np.zeros(len(timevec))


	rng = np.random.default_rng(1906)

	if code == "A":
		data +=  rhythm_square_conv(timevec, hours=24, A=10, phi=0)
		contains = {24:10}
	
	elif code == "new":
		

		"""
		contains = {
			50.0: 6,
			48.0: 4, 
			24.0: 10, 
			19.0: 5, 
			13.0: 6, 
			9.0: 3, 
			6.5: 2, 
			5.0: 2, 
			4.0: 3, 
			3.2: 2, 
			2.5: 8,
			2.0: 4,
			1.5: 3,
			1.0: 8,
			0.5: 5}
		"""		


		"""
		contains = {
			50.0: 6,
			40.0: 7,
			24.0: 10, 
			19.0: 5, 
			13.0: 6, 
			9.0: 4, 
			6.5: 8, 
			5.0: 4,}
		
			3.5: 4, 
			2.0: 4,
			1.0: 7,
			0.5: 4}
		"""
		

		"""
		contains = {
			168: 10,
			24: 10,
			19: 8,
			14: 9,
			10:7,
			7:8,
			3.7:6,
			}
		"""
		contains = {
			24: 10,
			}



		fig, ax = plt.subplots(len(contains.keys())+2, 1)
		pntr = len(contains.keys())+1	

		for period, amplitude in contains.items(): 
			if period == 24:
				component = rhythm_square(timevec, hours=24, A=amplitude, phi=0)
			else:
				#continue
				component = rhythm(timevec, hours=period, A = amplitude, phi=rng.choice(np.arange(0, 2*np.pi, 0.1)))
			data += component
			contains[period] = amplitude	
			ax[pntr].plot(component)
			pntr -= 1

		spikes = np.zeros(len(data))
		spike_height = 80
		spikes[rng.choice(np.arange(0, len(data)), 3)] = spike_height
		data += spikes

		#ax[0].plot(data, c="black")


	elif code == "909_repro":

		rng = np.random.default_rng(1906)

		repro_dict = {48.0: 4.389601505990354, 24.0: 8.039840623616088, 16.0: 3.7387045650972954, 12.0: 5.881117200600107, 9.600000000000001: 3.5726649952917997, 6.0: 2.2268738774852572, 5.333333333333334: 2.246636380766623, 4.800000000000001: 3.3526555907981552, 4.363636363636364: 1.859224807179114, 3.4285714285714284: 2.1669521632642277, 3.2: 2.22018201747702, 3.0: 2.3274646746726844}

		fig, ax = plt.subplots(len(repro_dict.keys())+1, 1)
		pntr = len(repro_dict.keys())	


		contains = {}
		#for period, proportion in repro_dict.items():
			#power =  (A*proportion)
			#data += rhythm(timevec, hours=period, A = power, phi=0)
			#contains[period] = power	
		for period, amplitude in repro_dict.items(): 
			if period == 24:
				component = rhythm_square(timevec, hours=24, A=amplitude, phi=0)
			else:
				#continue
				component = rhythm(timevec, hours=period, A = amplitude, phi=rng.choice(np.arange(0, 2*np.pi, 0.1)))
			data += component
			contains[period] = amplitude	
			ax[pntr].plot(component)
			pntr -= 1

		spikes = np.zeros(len(data))
		spike_height = 80
		spikes[rng.choice(np.arange(0, len(data)), 3)] = spike_height
		data += spikes

		ax[0].plot(data, c="black")
		plt.close(fig)
	
	elif code == "slides":
		fig, ax = plt.subplots(3)
		data += rhythm(timevec, hours=6, A=10, phi=0)
		ax[1].plot(data)
		ax[1].set_title("High Frequency Component")
		low = (rhythm(timevec, hours=12, A=10, phi=0) * np.hanning(len(timevec)))
		ax[2].plot(low)
		ax[2].set_title("Low Frequency Component")
		data += low
		ax[0].set_title("Summation")
		ax[0].plot(data)

		
		contains = {12:10, 6:10}

	elif code == "B":
		data += rhythm_square_conv(timevec, hours=24, A=10, phi=0)
		data += rhythm(timevec, hours=12, A=8, phi=0)
		contains = {24:10, 12:8}

	elif code == "C":
		data += rhythm(timevec, hours=48, A=8, phi=0)
		data += rhythm_square_conv(timevec, hours=24, A=10, phi=0)
		data += rhythm(timevec, hours=12, A=8, phi=0)		
		contains = {48:8, 24:10, 12:8}

	elif code == "D":
		data +=  rhythm_square_conv(timevec, hours=24, A=10, phi=0)
		data += rhythm(timevec, hours=9, A=8, phi=0)
		contains = {24:10, 9:8}
		
	elif code == "E":
		data +=  rhythm_square_conv(timevec, hours=24, A=10, phi=0)
		data += rhythm(timevec, hours=16, A=10, phi=0)
		data += rhythm(timevec, hours=12, A=8, phi=0)
		data += rhythm(timevec, hours=9, A=7, phi=0)
		data += rhythm(timevec, hours=7, A=6, phi=0)
		data += rhythm(timevec, hours=5, A=4, phi=0)
		data += rhythm(timevec, hours=2, A=2, phi=0)
		contains = {24:10, 16:10, 12:8, 9:7, 7:6, 5:4, 2:2}

	elif code == "F":
		data +=  rhythm_square_conv(timevec, hours=24, A=10, phi=0)
		data += rhythm(timevec, hours=16, A=2, phi=0)
		data += rhythm(timevec, hours=12, A=4, phi=0)
		data += rhythm(timevec, hours=9, A=6, phi=0)
		data += rhythm(timevec, hours=7, A=7, phi=0)
		data += rhythm(timevec, hours=5, A=8, phi=0)
		data += rhythm(timevec, hours=2, A=10, phi=0)
		contains = {24:10, 16:2, 12:4, 9:6, 7:7, 5:8, 2:10}

	elif code == "G":
		data += rhythm(timevec, hours=48, A=8, phi=0)
		data += rhythm(timevec, hours=32, A=8, phi=0)
		data +=  rhythm_square_conv(timevec, hours=24, A=10, phi=0)
		contains = {48:8, 32:8, 24:10}

	elif code == "H":
		data +=  rhythm_square_conv(timevec, hours=24, A=10, phi=0)
		data += rhythm(timevec, hours=9, A=8, phi=0)
		data += rhythm(timevec, hours=8, A=8, phi=0)
		contains = {24:10, 9:8, 8:8}

	elif code == "I":
		data +=  rhythm_square_conv(timevec, hours=24, A=10, phi=0)
		data += rhythm(timevec, hours=9, A=8, phi=0)
		data += rhythm(timevec, hours=2, A=8, phi=0)
		contains = {24:10, 9:8, 2:8}
		
	elif code == "J":
		data +=  rhythm_square_conv(timevec, hours=24, A=10, phi=0)
		data +=  rhythm_square_conv(timevec, hours=12, A=8, phi=0)
		contains = {24:10, 12:8}
	else:
		raise Exception	

	data += baseline
	
	noise_amp = 3
	noise = noise_amp* rng.normal(0, 1, len(timevec))
	#set_random_seed(1905) # neurodsp seed
	exponent = -2  # -2 for brown noise
	#noise = sim_powerlaw(n_seconds=data_length, fs=1/segment_len_s, exponent= exponent)
	noise = noise * noise_amp
	noisy_data = data + noise
	
	
	ax[0].plot(noisy_data, c="black")
	ax[1].plot(noise)
	plt.close(fig)
	
	return contains, data, noisy_data


		

def temp_plot(subjects_methods_peaks, metric, now):

	# set up dir
	logfile_dir = constants.SUBJECT_DATA_OUT.format(subject='LOGS')		
	plots_subdir = f"{now.day}_{now.month}_{now.year}_{now.hour}_{now.minute}_{now.second}_{metric}"	
	plots_dir = os.path.join(logfile_dir, "PLOTS", plots_subdir)
	os.makedirs(plots_dir, exist_ok=True)

	subject_y_mapping = dict(zip(subjects_methods_peaks.keys(), reversed(np.arange(0, len(subjects_methods_peaks.keys())))))

	#methods = subjects_methods_peaks[list(subjects_methods_peaks.keys())[0]].keys()	
	methods = subjects_methods_peaks[list(subjects_methods_peaks.keys())[0]].index	
	methods_y_mapping = dict(zip(methods, np.arange(0, len(methods))))

	figures = []
	axes  = []
	for i in range(0, len(methods)):
		#fig, ax = plt.subplots(figsize=(19.20, 19.20))
		fig, ax = plt.subplots(figsize=(10.8, 10.8))
		figures.append(fig)
		axes.append(ax)	

	#pal = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=False, as_cmap=True)
	#pal = "crest"

	pal = sns.light_palette("black", as_cmap=True)


	subjects_methods_peaks_df = pd.DataFrame(columns = ["subject", "method", "cycle_period", "power"])
	for subject, methods_peaks_df in subjects_methods_peaks.items():

		methods_peaks_df = methods_peaks_df.reset_index().melt(id_vars=["method"])		
		methods_peaks_df = methods_peaks_df.sort_values(by=["method", "cycle_period"])
		methods_peaks_df.insert(0, "subject", subject)
		methods_peaks_df = methods_peaks_df.rename(columns={"value":"power"})

		subjects_methods_peaks_df = pd.concat([subjects_methods_peaks_df, methods_peaks_df], axis=0)
	


	# PLOTS:
	# A: per-subject, per-method peaks power plot, 
	# B: all subjects, per-method scatter plot

	xticks = np.array([1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200, 250], dtype=np.int32)


	for subject in pd.unique(subjects_methods_peaks_df["subject"]):

		if "taVNS" in subject:
			A_title = f"taVNS Subject {subject.split('taVNS')[1]}"
		elif "sim" in subject:
			A_title = f"Simulated Subject {subject}"
		else:
			A_title = f"UCLH Subject {subject}"

		A_title = f"{A_title} ({metric_fullname(metric)})"

		# A
		fig_pp, ax_pp = plt.subplots(figsize=(10.80, 10.80))
		#ax_pp.set_xlabel("Period (log(h))")
		ax_pp.set_xlabel("Period (h)")
		ax_pp.set_ylabel("Signal Decomposition Method")
		
		ultradian_span = ax_pp.axvspan(ULTRADIAN_MIN, ULTRADIAN_MAX, color=ULTRADIAN_COLOR)
		circadian_span = ax_pp.axvspan(CIRCADIAN_MIN, CIRCADIAN_MAX, color=CIRCADIAN_COLOR)
		infradian_span = ax_pp.axvspan(INFRADIAN_MIN, INFRADIAN_MAX, color=INFRADIAN_COLOR)
		fig_pp.legend([ultradian_span, circadian_span, infradian_span], [f"Ultradian ({ULTRADIAN_MIN}h-{ULTRADIAN_MAX}h)", f"Circadian ({CIRCADIAN_MIN}h-{CIRCADIAN_MAX}h)", f"Infradian ({INFRADIAN_MIN}h-$\infty$h)"], loc="upper left")		

		"""	
		for method, peaks_power in subjects_methods_peaks[subject].items():

			# A
			peaks = np.array(list(peaks_power.keys()))
			inf_idx = np.isinf(peaks)
			peaks = peaks[~inf_idx]
			powers = np.array(list(peaks_power.values()))[~inf_idx]
			
			if len(powers) > 1:
				powers_normalised = (powers-np.min(powers))/(np.max(powers)-np.min(powers))
			elif len(powers) == 1:
				powers_normalised = np.array([1])
			#ax_pp.stem(peaks, powers_normalised, label=method)

			powers_normalised = np.log(powers_normalised*100)*100		
			sns.scatterplot(x=peaks, y=methods_y_mapping[method], size=powers, sizes=(20, 200), ax=ax_pp, legend="brief")	

			# B 
			ax = axes[np.where(np.array(list(methods)) == method)[0][0]]
			ax.axhline(subject_y_mapping[subject], c="silver", alpha=0.5)

			

			sns.scatterplot(x=peaks, y=subject_y_mapping[subject], hue=powers_normalised, palette=pal, edgecolor="black", ax=ax, zorder=4)
			#if len(powers_normalised) > 1:	
			#	sns.scatterplot(x=peaks, y=subject_y_mapping[subject], hue=powers_normalised, palette=pal, edgecolor="black", ax=ax, zorder=4)
			#elif len(powers_normalised) == 1:
			#	sns.scatterplot(x=peaks, y=subject_y_mapping[subject], hue=powers_normalised, color="black", edgecolor="black", ax=ax, zorder=4)

		"""

		# A 
		#sns.scatterplot(data=subjects_methods_peaks_df[subjects_methods_peaks_df["subject"] == subject], y="method", x="cycle_period", hue="power", palette=pal, sizes=(10, 300), zorder=4, ax=ax_pp, legend=False)
		sns.scatterplot(data=subjects_methods_peaks_df[subjects_methods_peaks_df["subject"] == subject].dropna(axis=0), y="method", x="cycle_period", c="black", zorder=4, ax=ax_pp, legend=False)
		
		"""
		extra_periods = [2, 12,]
		xt = ax_pp.get_xticks()
		for period in extra_periods:
			xt = np.append(xt, period)
		ax_pp.set_xticks(xt)		
		"""
		harmonics = [(1/(n*(1.15e-5)))/60/60 for n in np.concatenate(([0.25, 0.5], np.arange(1, 20)))]
		for i, harmonic in enumerate(harmonics):
			col="grey"
			decr = 0.05
			alph = max(0, (1 - (decr* np.abs(1-i)))) # decrease alpha of harmonic by decr based on how far it is from central 24h frequency
			ax_pp.axvline(harmonic, c=col, alpha=alph, zorder=3.5)
			ax_pp.text(harmonic, -0.75, f"{np.round(harmonic, decimals=2)}h", c=col, alpha=alph, rotation=90)

		ax_pp.set_xscale("log")
		#ax_pp.autoscale(enable=True, axis='x', tight=True)
		ax_pp.set_xticks(ticks=xticks, labels=xticks)
		#ax_pp.set_yticks(ticks=np.array(list(methods_y_mapping.values())), labels=methods_y_mapping.keys())
		ax_pp.set_xlim([0.8, INFRADIAN_MAX])
		fig_pp.suptitle(A_title)
		fig_pp.savefig(os.path.join(plots_dir, f"{subject}_peaks"))
		plt.close(fig_pp)

	# B 
	for method in methods:	
		ax = axes[np.where(np.array(list(methods)) == method)[0][0]]

		#sns.scatterplot(data=subjects_methods_peaks_df[subjects_methods_peaks_df["method"] == method], y="subject", x="cycle_period", hue="power", palette=pal, edgecolor="black", zorder=4, ax=ax)
		
		points = subjects_methods_peaks_df[subjects_methods_peaks_df["method"] == method]


		for y in ax.get_yticks():
			ax.axhline(y, c="silver", alpha=0.5, zorder=3.5)


	# B
	for i in range(0, len(axes)):

		method = list(methods)[i]

		#axes[i].set_yticks(ticks=np.array(list(subject_y_mapping.values())), labels=subject_y_mapping.keys())

		axes[i].legend([],[], frameon=False)

		axes[i].set_title(f"{method_fullname(method)} ({metric_fullname(metric)})")

		#axes[i].set_xlabel("Period (log(h))")
		axes[i].set_xlabel("Period (h)")
		
		axes[i].set_ylabel("Subject")

		axes[i].axvspan(ULTRADIAN_MIN, ULTRADIAN_MAX, color=ULTRADIAN_COLOR)
		axes[i].axvspan(CIRCADIAN_MIN, CIRCADIAN_MAX, color=CIRCADIAN_COLOR)
		axes[i].axvspan(INFRADIAN_MIN, INFRADIAN_MAX, color=INFRADIAN_COLOR)
		figures[i].legend([ultradian_span, circadian_span, infradian_span], [f"Ultradian ({ULTRADIAN_MIN}h-{ULTRADIAN_MAX}h)", f"Circadian ({CIRCADIAN_MIN}h-{CIRCADIAN_MAX}h)", f"Infradian ({INFRADIAN_MIN}h-$\infty$h)"], loc="upper left")		

		axes[i].set_xlim([0.8, INFRADIAN_MAX])
		axes[i].set_xscale("log")
		axes[i].set_xticks(ticks=xticks, labels=xticks)
		#axes[i].autoscale(enable=True, axis='x', tight=True)

		figures[i].savefig(os.path.join(plots_dir, f"{method}_subjects"))
		plt.close(figures[i])

	return subjects_methods_peaks_df


	

if __name__ == "__main__":
	#subjects = ["909", "902", "931"] 
	UCLH_subjects = ["1005","1055","1097","1119","1167","1182","1284","815","902","934","95", "1006","1064","1109","1149","1178","1200","770","821","909","940","999", "1038","1085","1110","1163","1179","1211","800","852","931","943"]
	taVNS_subjects = ["taVNS001","taVNS002","taVNS003","taVNS004","taVNS005","taVNS006","taVNS007","taVNS008","taVNS009","taVNS010","taVNS011","taVNS012","taVNS013","taVNS014","taVNS015","taVNS017","taVNS018"]

	subjects = UCLH_subjects + taVNS_subjects
	
	subjects = ["sim"]
	
	#subjects = ["1167"]
	#subjects = ["1200"]
	#subjects=["909"]
	#subjects = ["taVNS006"]
	#subjects =["taVNS001"]
	#subjects= ["1284"]
	#subjects = ["1119"]
	#subjects = ["815"]

	#subjects = ["815", "1167", "1200", "909", "1284", "1119", "taVNS001", "taVNS006"]
	#subjects = ["909", "taVNS001"]

	#metrics = ["hr_mean", "fft_ratio", 'sdnn', 'rmssd', 'sdsd', 'nn50', 'pnn50', 'nn20', 'pnn20', 'tri_index', "fft_rel_VLF", "fft_rel_LF", "fft_rel_HF"]
	metrics = ["hr_mean"] 


	start = time.time()	
	now = datetime.datetime.now()


	logfile_dir = constants.SUBJECT_DATA_OUT.format(subject='LOGS')	
	os.makedirs(logfile_dir, exist_ok=True)
	logfile_loc = f"{logfile_dir}/logfile_{now.day}_{now.month}_{now.year}.txt"	

	with open(logfile_loc, "w") as logfile:
		logfile.write(str(now))
		logfile.write("\n")
		logfile.write(f"{len(subjects)}\tsubjects:\t{subjects}")
		logfile.write("\n")
	
	
	if not "sim" in subjects:
	
		for metric in metrics:
			subjects_methods_peaks = {} # dict of subject -> their method peaks results dict

			print(f"-- {metric} --")
			with open(logfile_loc, "a") as logfile:
				logfile.write(f"\n-- {metric} --")

			i = 0
			for subject in subjects:

				plt.close("all")
				print(f"\n$RUNNING FOR SUBJECT: {subject}\n")
			
				try:

					root = constants.SUBJECT_DATA_ROOT.format(subject=subject)
					out = constants.SUBJECT_DATA_OUT.format(subject=subject)

					rng = np.random.default_rng(1905)
					
					if not "taVNS" in subject:	
						# collate data and resolve overlaps
						#run_speedyf(root, out); 
						
						# produce hrv metric dataframes, and save to out/
						calculate_hrv_metrics(root, out, rng)# will not re-calculate if dataframes already present in out

					# load the hrv metric dataframes we just produced
					time_dom_df, freq_dom_df, modification_report_df = load_hrv_dataframes(out)

					# what metric are we interested in?
					metric_is_timedom = metric in time_dom_df.columns
					if metric_is_timedom: 
						data = time_dom_df[metric]
					else:
						data = freq_dom_df[metric]

					data = np.array(data)
				
					if not "taVNS" in subject:	
						# get timestamps corresponding to segments
						timevec = edf_segment.EDFSegmenter(root, out, segment_len_s=300).get_segment_onsets(as_datetime=True)
					else:
						convert = np.vectorize(convert_unixtime_ms_to_datetime)
						timevec = convert(np.array(time_dom_df[time_dom_df.columns[0]]))

					# get timestamps corresponding to seizures and their durations
					onsets, durations = get_seizures(subject)
					
					timevec_old = timevec.copy()

					fig, ax = plt.subplots(6, figsize=(10.8, 10.8))
					plt.subplots_adjust(bottom = 0.05, top=0.93, hspace=0.75)
					fig.suptitle("Data Preprocessing Steps")
					ax[0].plot(timevec_old, data, c="black")
					ax[0].set_title("Original Data", loc="left")
					ax[0].set_xlim([min(timevec), max(timevec)]) 

					# interpolate gaps (runs of NaN) so we can use with signal decomposition. save gap positions for visualisation
					interpolated, timevec, gaps = interpolate_gaps(data, timevec, ax, out, metric) 
				
					fig.supylabel("bpm")	
					fig.savefig(os.path.join(out, "preprocessing_plots", f"{metric}_preprocessing"))
					plt.close(fig)

					# # # #

					ax2_xticks = 1/(np.arange(1, 48)*60*60)
					ax2_xticks = np.concatenate(([0], ax2_xticks)) 
					ax2_xticklabels = (1/ax2_xticks) # invert to frequency is period (s)
					ax2_xticklabels = (ax2_xticklabels / 60) / 60 # get from s into hours 
					ax2_xticklabels = [np.round(lab, decimals=1) if lab != np.inf else lab for lab in ax2_xticklabels]

					f, Pxx_den = psd(interpolated, 1/300)
					fig, ax = plt.subplots()
					ax.stem(f, Pxx_den)
					ax.set_xticks(ticks=ax2_xticks, labels=ax2_xticklabels)
					#fig.show()
					
					max_power = max(Pxx_den)
					repro_dict = {}
					for hz, power in dict(zip(f, Pxx_den)).items():
						period = 1/(hz*60*60)
					

						#proportion = power/max_power 
						proportion = power/np.sum(Pxx_den)

	
						# what % of the maximum power is this power?
						pct = np.round((proportion * 100), 2)
					
						if pct > 1: # keep only those above 1%
							print(f"{np.round(period, 2)}h\t:\t{pct}% ")
							#repro_dict[period] = proportion

							amplitude =np.sqrt(power) / len(Pxx_den) # I guess this is like average of amplitude over duration of signal? 
							repro_dict[period] = amplitude


					print(repro_dict)
					print(np.sum(Pxx_den))

					plt.close(fig)
					
					# # # #


					# perform multi-resolution analysis (signal decomposition)	
					methods_peaks = mra(out, metric, timevec, interpolated, gaps, onsets, durations, sharey=False)	
					subjects_methods_peaks[subject] = methods_peaks
					

			
					with open(logfile_loc, "a") as logfile:
						logfile.write(f"\n{i+1}/{len(subjects)}:\t{subject}:\tSuccess!\tRuntime:{time.time()-start}")

				except Exception as e:
				
					#print("REMOVE ME RAISE EXCEPTION")
					#raise Exception
	
					with open(logfile_loc, "a") as logfile:
						logfile.write(f"\n{i+1}/{len(subjects)}:\t{subject}:\t{e}")


				i += 1


			end = time.time()

			with open(logfile_loc, "a") as logfile:
				logfile.write(f"\nComplete@{str(datetime.datetime.now())}!\tRuntime:{end-start}")
			
			subjects_methods_peaks_df = temp_plot(subjects_methods_peaks, metric, now)

			csvs_dir = constants.SUBJECT_DATA_OUT.format(subject='CSV')	
			os.makedirs(csvs_dir, exist_ok=True)
			subjects_methods_peaks_df.to_csv(os.path.join(csvs_dir, f"{metric}.csv"), index=False)


	
	if "sim" in subjects:
		metric = "hr_mean"
		subjects_methods_peaks = {} # dict of subject -> their method peaks results dict

		simulated_subjects_peaks = {}
		
		#letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
		#letters = ["909_repro"]
		letters = ["new"]
		sim_fig, sim_ax = plt.subplots(max(2, len(letters)), 2, figsize=(19.20, 10.80))

		for i, letter in enumerate(letters):
			
			contains, noisefree_data, noisy_data = simulated_subject(letter)		
			

			sim_ax[i, 0].plot(noisefree_data)
			sim_ax[i, 0].set_title(f"{letter}")
			sim_ax[i, 1].plot(noisy_data)
			sim_ax[i, 1].set_title(f"{letter} (Noisy)")
			
			simulated_subjects_peaks[letter] = contains
			simulated_subjects_peaks[f"{letter}n"] = contains

			for suffix, data in zip(["", "n"], [noisefree_data, noisy_data]):
				rng = np.random.default_rng(1905)
			
				out = constants.SUBJECT_DATA_OUT.format(subject=f"sim{letter}{suffix}")

				# simulate missing data
				data[math.floor(np.quantile(range(0, len(data)), 0.7)): math.ceil(np.quantile(range(0, len(data)), 0.8))] = np.NaN # large gap
				data[rng.choice(len(data), math.floor(len(data) * 0.1))] = np.NaN # smaller gaps randomly throughout

				timevec = [datetime.datetime.fromtimestamp(0) + datetime.timedelta(seconds=(i * 300)) for i in range(0, len(data))]
				
				fig, ax = plt.subplots()
				ax.plot(timevec, data, color="black")
				ax.set_ylabel("Mean Heart Rate per 5 minute segment (bpm)")
				ax.set_xlabel("Time (d/m h:m:s)")
				fig.suptitle("Simulated Heart Rate Data (with missing data, prior to preprocessing)")
				fig.savefig(os.path.join(out, "preprocessing_plots", f"{metric}_viz"))

				onsets = durations = []
			
				fig, ax = plt.subplots(6, figsize=(10.8, 10.8))
				plt.subplots_adjust(bottom = 0.05, top=0.93, hspace=0.75)
				fig.suptitle("Data Preprocessing Steps")
				ax[0].plot(timevec, data, c="black")
				ax[0].set_title("Original Data", loc="left")
				ax[0].set_xlim([min(timevec), max(timevec)]) 

				# interpolate gaps (runs of NaN) so we can use with signal decomposition. save gap positions for visualisation
				interpolated, timevec, gaps = interpolate_gaps(data, timevec, ax, out, metric) 
				
				fig.savefig(os.path.join(out, "preprocessing_plots", f"{metric}_preprocessing"))
				plt.close(fig)
				
				# # # #	
				ax2_xticks = 1/(np.arange(1, 48)*60*60)
				ax2_xticks = np.concatenate(([0], ax2_xticks)) 
				ax2_xticklabels = (1/ax2_xticks) # invert to frequency is period (s)
				ax2_xticklabels = (ax2_xticklabels / 60) / 60 # get from s into hours 
				ax2_xticklabels = [np.round(lab, decimals=1) if lab != np.inf else lab for lab in ax2_xticklabels]

				f, Pxx_den = psd(interpolated, 1/300)
				fig, ax = plt.subplots()
				ax.stem(f, Pxx_den)
				ax.set_xticks(ticks=ax2_xticks, labels=ax2_xticklabels)
				#fig.show()
				# # # #				



				methods_peaks = mra(out, metric, timevec, interpolated, gaps, onsets, durations, sharey=False)
				subjects_methods_peaks[letter+suffix] = methods_peaks

		sim_fig.savefig("simulated_subjects")
		plt.close("all")

		# make a dataframe, like subjects_methods_peaks_df, that has the ACTUAL (not as calc by methods) periods/power for sim sujects
		simulated_subjects_peaks_df = pd.DataFrame(columns = ["subject", "cycle_period", "amplitude"])
		for simulated_subject, contains in simulated_subjects_peaks.items():
			for cycle_period, amplitude in contains.items():
				simulated_subjects_peaks_df = pd.concat([simulated_subjects_peaks_df, pd.DataFrame([{"subject":simulated_subject, "cycle_period": cycle_period, "amplitude": amplitude}])], ignore_index=True)				

		# set up dir (COPIED FROM temp_plot)
		logfile_dir = constants.SUBJECT_DATA_OUT.format(subject='LOGS')		
		plots_subdir = f"{now.day}_{now.month}_{now.year}_{now.hour}_{now.minute}_{now.second}_{metric}_sim"	
		plots_dir = os.path.join(logfile_dir, "PLOTS", plots_subdir)
		os.makedirs(plots_dir, exist_ok=True)

		# plotting
		subjects_methods_peaks_df = temp_plot(subjects_methods_peaks, metric+"_sim", now)

		simulated_subjects = pd.unique(simulated_subjects_peaks_df["subject"])
		methods = pd.unique(subjects_methods_peaks_df["method"])
		
		# A. how many components did each method detect?
		correct_n_comp_df = pd.DataFrame(index=simulated_subjects, columns=methods)
		# B. what percentage of the actual component periods were detected?
		correct_comp_periods_df = correct_n_comp_df.copy()

		# TODO:
			# we have info on amplitude
			# TODO how will significant peak filtering come into this?


		VALID_COLOR = "green"
		INVALID_COLOR = "red"
		HARMONIC_COLOR = "grey"
		VALID_HANDLE = None
		INVALID_HANDLE = None
		HARMONIC_HANDLE = None

	
		for simulated_subject in simulated_subjects:
			
			DP = 1 

			fig, ax = plt.subplots(len(methods)+1, 1, figsize=(10.8, 10.8), sharex=True)
			plt.subplots_adjust(bottom=0.05, top=0.957, hspace=0.402)
			j = 0			
			
			simulated_subject_contains = simulated_subjects_peaks_df[simulated_subjects_peaks_df["subject"] == simulated_subject]				
			actual_periods = np.array(simulated_subject_contains["cycle_period"].values, dtype=np.float32)
		
			barwidth = 10
	
			max_actual = max(simulated_subject_contains["cycle_period"].values)
			#sns.barplot(data=simulated_subject_contains, x="cycle_period", y="amplitude", ax=ax[j])
			VALID_HANDLE = ax[j].bar(actual_periods, simulated_subject_contains["amplitude"], color=VALID_COLOR, zorder=2.5)
			
			ax[j].set_ylabel("Amplitude")
			ax[j].set_title("Simulated Data Specification")
			
			harmonics = simulated_subject_contains[simulated_subject_contains["cycle_period"].astype(float).round(DP).isin(np.unique(np.round(HARMONICS_24H, DP)))]
			ax[j].bar(harmonics["cycle_period"], harmonics["amplitude"], color=HARMONIC_COLOR, zorder=2.55)

			ax[j].axvspan(ULTRADIAN_MIN, ULTRADIAN_MAX, color=ULTRADIAN_COLOR)
			ax[j].axvspan(CIRCADIAN_MIN, CIRCADIAN_MAX, color=CIRCADIAN_COLOR)
			ax[j].axvspan(INFRADIAN_MIN, INFRADIAN_MAX, color=INFRADIAN_COLOR)
			#ax[j].autoscale(enable=True, axis='x', tight=True)
			ax[j].set_xlim([0, max_actual+10])
			#ax[j].set_xscale("log")
			#ax[j].set_xticks(ticks = actual_periods, labels = actual_periods)		
	
			evaluation_df = pd.DataFrame(columns=["method", "tp", "fp", "fn", "sensitivity", "accuracy"])
			
			for method in methods:

				peaks_power_df = subjects_methods_peaks_df[(subjects_methods_peaks_df["subject"] == simulated_subject) & (subjects_methods_peaks_df["method"] == method)]
				
				peaks_power_df =  peaks_power_df.dropna(axis=0)




				j += 1
				#sns.barplot(data=peaks_power_df.dropna(axis=0), x="cycle_period", y="power", ax=ax[j])
				INVALID_HANDLE = ax[j].bar(peaks_power_df["cycle_period"], np.sqrt(peaks_power_df["power"]), color=INVALID_COLOR, zorder=2.5)
			
				# class as correct if in actual data when rounded to N dp
				#correct = peaks_power_df[peaks_power_df["cycle_period"].astype(float).round(DP).isin(simulated_subject_contains["cycle_period"].astype(float).round(DP))]
				
				# class as correct if within N difference in period from rhythm in actual data
				correct = peaks_power_df.copy()	
				diff_to_actuals = []			
				for idx in correct.index:
					condition_met = False
					
					extracted = correct.loc[idx]["cycle_period"]
					if not np.round(extracted, DP) in np.unique(np.round(HARMONICS_24H, DP)):
						diff_to_actual = np.inf


						for actual in simulated_subject_contains["cycle_period"]:
							diff = np.abs(extracted-actual)
							if diff <= 0.17:
								condition_met = True
								if diff < diff_to_actual: 
									diff_to_actual = diff

					if not condition_met:
						correct = correct.drop(idx)			
					else:
						diff_to_actuals.append(diff_to_actual)

				ax[j].bar(correct["cycle_period"], np.sqrt(correct["power"]), color=VALID_COLOR, zorder=2.55)
			
				harmonics = peaks_power_df[peaks_power_df["cycle_period"].astype(float).round(DP).isin(np.unique(np.round(HARMONICS_24H, DP)))]
				HARMONIC_HANDLE = ax[j].bar(harmonics["cycle_period"], np.sqrt(harmonics["power"]), color=HARMONIC_COLOR, zorder=2.55)
				
				
				ax[j].set_ylabel(r"$\sqrt{Power}$") #TODO is this ylabel correct
				#ylim_min, ylim_max = ax[j].get_ylim()
				#ax[j].yaxis.set_label_coords(0, np.median([ylim_min, ylim_max]))
				
				if j == len(methods)+1: ax[j].set_xlabel("Cycle Period")
				ax[j].set_title(method)
	
				ax[j].axvspan(ULTRADIAN_MIN, ULTRADIAN_MAX, color=ULTRADIAN_COLOR)
				ax[j].axvspan(CIRCADIAN_MIN, CIRCADIAN_MAX, color=CIRCADIAN_COLOR)
				ax[j].axvspan(INFRADIAN_MIN, INFRADIAN_MAX, color=INFRADIAN_COLOR)	
				ax[j].set_xlim([0, max_actual+10])
				#ax[j].set_xscale("log")
				#ax[j].set_xticks(ticks = actual_periods, labels = actual_periods)		

				# A. 		
				n_comp_actual = simulated_subject_contains.shape[0]
				comp_detected = peaks_power_df[~pd.isnull(peaks_power_df["power"])]
				n_comp_detected = comp_detected.shape[0]
				n_comp_difference = np.diff([n_comp_actual, n_comp_detected])[0]
				correct_n_comp_df.loc[simulated_subject][method] = n_comp_difference

				# B
				#common_cycle_periods = set(np.round(simulated_subject_contains["cycle_period"].to_numpy(dtype=np.float64), DP)).intersection(set(np.round(comp_detected["cycle_period"].to_numpy(dtype=np.float64), DP)))
				common_cycle_periods = correct["cycle_period"]
				pct_actual_detected = np.round(len(common_cycle_periods) / n_comp_actual * 100, decimals=2)
				correct_comp_periods_df.loc[simulated_subject][method] = pct_actual_detected
			
				n_true_positives  = 0 # how many cycles that we know were present were found?
				n_false_positives = 0 # how many cycles were found that we know were not present?
				n_false_negatives = 0 # how many cycles that we know were present weren't found?
				"""
				for cycle in comp_detected["cycle_period"].astype(float).round(DP).values:	
					if cycle in simulated_subject_contains["cycle_period"].astype(float).round(DP).values:
						n_true_positives += 1
					else:
						n_false_positives += 1

				for cycle in simulated_subject_contains["cycle_period"].astype(float).round(DP).values:
					if cycle not in comp_detected["cycle_period"].astype(float).round(DP).values:
						n_false_negatives += 1
				"""
				for cycle in comp_detected["cycle_period"].astype(float).round(DP).values:	
					if (cycle in common_cycle_periods.astype(float).round(DP).values):# and (np.round(cycle, DP) not in np.unique(np.round(HARMONICS_24H, DP))):
						n_true_positives += 1

					else:
						n_false_positives += 1

				for cycle in simulated_subject_contains["cycle_period"].astype(float).round(DP).values:
					if cycle not in common_cycle_periods.astype(float).round(DP).values:
						n_false_negatives += 1

				print(method)
				print(f"TP: {n_true_positives}")
				print(f"FP: {n_false_positives}")
				print(f"FN: {n_false_negatives}")
			
				sensitivity = n_true_positives / (n_true_positives + n_false_negatives)
				accuracy = np.mean(diff_to_actuals)	
				
				evaluation_df_entry = {"method":method, "tp":n_true_positives, "fp":n_false_positives, "fn":n_false_negatives, "sensitivity":sensitivity, "accuracy":accuracy}	
				evaluation_df.loc[len(evaluation_df)] = evaluation_df_entry

			fig.legend([VALID_HANDLE, INVALID_HANDLE, HARMONIC_HANDLE], ["True Positives", "False Positives", "Probable Harmonics"], loc="center right")
			fig.supxlabel("Period (h)")
			plt.subplots_adjust(hspace=0.529)
			fig.savefig(os.path.join(plots_dir, f"{simulated_subject}_bars"))
			evaluation_df.to_csv(os.path.join(plots_dir, f"sim{simulated_subject}_evaluation.csv"), index=False)		

		import matplotlib.cm as cm
		import matplotlib.colors as mcolors		
				


		# visualise sim result dfs
		# A.
		correct_n_comp_df = correct_n_comp_df.astype(np.int32)
		"""
		fig, ax = plt.subplots(figsize=(10.80, 10.80))
		
		vcenter = 0
		vmin, vmax = correct_n_comp_df.min(axis=None), correct_n_comp_df.max(axis=None)
		vmin = 0-vmax

		normalize = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
		colormap = mcolors.LinearSegmentedColormap.from_list('my_colormap', ["seagreen", "white", "seagreen"])

		sns.heatmap(data=correct_n_comp_df, norm=normalize, cmap=colormap, annot=True, ax=ax, cbar_kws={'label': 'Difference between Detected and Actual Component Count'})
		ax.set_ylabel("Simulated Subject")
		ax.set_xlabel("Method")


		ax_r = ax.secondary_yaxis("right")
		ax_r.set_yticks(ax.get_yticks())
		ax_r.set_yticklabels(correct_n_comp_df.mean(axis=1))
	
		ax_t = ax.secondary_xaxis("top")
		ax_t.set_xticks(ax.get_xticks())
		ax_t.set_xticklabels(correct_n_comp_df.mean(axis=0))
		
		fig.savefig(os.path.join(plots_dir, "correct_n_comp"))	
		plt.close(fig)
		"""
		# B.
		correct_comp_periods_df = correct_comp_periods_df.astype(np.float64)
		fig, ax = plt.subplots(figsize=(10.80, 10.80))
		normalize = mcolors.TwoSlopeNorm(vcenter=50, vmin=0, vmax=100)
		sns.heatmap(data=correct_comp_periods_df, annot=False, cbar=True, norm=normalize, cmap="mako", cbar_kws={"label": "% of Actual Components Detected"}, ax=ax)
		ax.set_ylabel("Simulated Subject")
		ax.set_xlabel("Method")
		
		
		ax_r = ax.secondary_yaxis("right")
		ax_r.set_yticks(ax.get_yticks())
		ax_r.set_yticklabels([f"{val}%" for val in np.round(correct_comp_periods_df.mean(axis=1), decimals=0)])
	
		ax_t = ax.secondary_xaxis("top")
		ax_t.set_xticks(ax.get_xticks())
		ax_t.set_xticklabels([f"{val}%" for val in np.round(correct_comp_periods_df.mean(axis=0), decimals=0)])

		fig.savefig(os.path.join(plots_dir, "correct_comp_periods"))
		plt.close(fig)

