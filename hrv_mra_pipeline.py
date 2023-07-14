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

import constants

sys.path.append(constants.SPEEDYF_LOCATION)
from speedyf import edf_collate, edf_overlaps, edf_segment

sys.path.append(constants.HRV_PREPROC_LOCATION)
from hrv_preprocessor.hrv_preprocessor import hrv_per_segment, produce_hrv_dataframes, save_hrv_dataframes, load_hrv_dataframes

# TODO REMEMBER TO CITE PACKAGES
import emd  # EMD, EEMD
#from PyEMD import EMD
import pywt # Wavelet Transform?
import vmdpy # Variational Mode Decomposition
from pyts import decomposition

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

	out = os.path.join(out, "components_plots")
	if not os.path.exists(out):
		print("Setting up directory for MRA Output: at '{}'".format(out))
		os.makedirs(out, exist_ok=True)


	""" 1. Plot Components and 2. Plot PSDs of Components"""

	fig1, ax1 = plt.subplots(len(components)+2, 1, sharex=True, sharey=sharey, figsize=(19.20, 19.20)) # the components
	fig2, ax2 = plt.subplots(len(components)+2, 1, sharex=False, sharey=True, figsize=(19.20, 19.20)) # PSDs of the components
	plt.subplots_adjust(bottom=0.04, top=0.921, hspace=0.402)

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
	ax2[1].plot(f, Pxx_den, alpha=0.25)
	ax2[1].stem(f, Pxx_den)
	#ax2[1].set_xticks(ticks=f, labels=1/f)
	ax2[1].set_xticks(ticks=ax2_xticks, labels=ax2_xticklabels, rotation=ax2_xticklab_rot)
	#ax2[1].axvline((1/(24*60*60)), c="r", label="Circadian")
	ax2[1].set_xlim(ax2_xlim)
	ax2[1].set_title("Reconstruction", loc="left")	
	
	for i, comp in enumerate(components):
		ax1[i+2].plot(x_imfs, comp, color=c, alpha=alpha)
		ax1[i+2].plot(x_imfs, cut_gaps(comp, gaps), color=c)
		ax1[i+2].set_title(f"Component #{i}", loc="left")

		f, Pxx_den = psd(comp, fs)
		ax2[i+2].plot(f, Pxx_den, alpha=0.25)
		ax2[i+2].stem(f, Pxx_den)
		ax2[i+2].set_xticks(ticks=ax2_xticks, labels=ax2_xticklabels, rotation=ax2_xticklab_rot)
		#ax2[i+2].axvline((1/(24*60*60)), c="r", label="Circadian")
		ax2[i+2].set_xlim(ax2_xlim)
		ax2[i+2].set_title(f"Component #{i}", loc="left")
	
	fig1.suptitle(title)
	for ax in ax1:
		ax.xaxis.set_major_formatter(DateFormatter("%d/%m %H:%M:%S"))
		
		for onset, duration in zip(onsets, durations):
			ax.axvspan(onset, onset + datetime.timedelta(seconds=duration), color="r", alpha=0.5)
	
	fig1.supxlabel("Time (d/m h:m:s)")
	fig1.savefig(os.path.join(out, title))

	
	fig2.suptitle(title+"_PSD")
	fig2.supylabel("")
	fig2.supxlabel("Period (h)")
	handles, labels = ax2[-1].get_legend_handles_labels() # get from last axis only, to avoid duplicates
	fig2.legend(handles, labels)

	fig2.savefig(os.path.join(out, title+"_PSD"))



	""" 3. Produce plot where components are stacked, and the peak (maximum) frequency highlighted for each component.""" 
	
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
	
		if not peak_period_h in peaks_powers.keys():
			peaks_powers[peak_period_h] = peak_Pxx
		else:
			# if using welch for psd est, common to have duplicate peak periods. 
			if peak_Pxx > peaks_powers[peak_period_h]:
				# take the max power of this period as its power
				peaks_powers[peak_period_h] = peak_Pxx
		

		ax3.plot(f, Pxx_den, c=color, label=label)
		ax3.fill_between(f, Pxx_den, color=color)	
		ax3.scatter(peak_freq, peak_Pxx, marker="x", c="r", zorder=10)


	ax3.set_xscale("log")
	fig3.legend(loc="upper right")
	fig3.suptitle(title+" Components PSDs Overlaid")
	fig3.supxlabel("Frequency (Hz)")
	fig3.savefig(os.path.join(out, title+"_PSD_OVERLAY"))

	return peaks_powers


def psd(data, sample_rate):
	""" # FFT
	data = data - np.mean(data) # remove DC Offset (0Hz Spike)
	yf = fft.rfft(data)
	xf = fft.rfftfreq(len(data), 1/sample_rate)
	return xf, np.abs(yf)
	"""
	# Welch
	hours_per_seg = 48#24
	min_per_seg = hours_per_seg * 60	
	nperseg = (min_per_seg//5) # how many 5min HRV segments in our welch segment
	#return signal.welch(data, sample_rate, nperseg=nperseg, noverlap=nperseg//2, scaling="density")	
	return signal.welch(data, sample_rate, nperseg=nperseg, noverlap=((40*60)//5), scaling="density")	


def psd_plot(data, sample_rate): # warn; can't use fft with HRV, irregularly sampled
	fig, ax = plt.subplots()
	xf, yf = psd(data, sample_rate)
	ax.plot(xf, np.abs(yf))

	return fig, ax

def reflect(data):
	# TODO: rename "extend", implement boundary extension method symmetric/antisymmetric depending on phase at boundary
	padlen = len(data) // 2
	return np.pad(data, padlen, mode="median")

def remove_reflect(data):
	"""Remove data added by reflect()"""
	padlen = len(data) // 4	 # AS LONG AS padlen = len(data)//2 in reflect(), this should work?
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

def interpolate_gaps(data, timevec):

	gaps = get_gaps_positions(data)
	# remove gaps that start at start/end at end (will not be interpolated, and will mess up decomposition)
	if len(gaps) > 0:
		if gaps[-1][1] == len(data)-1:
			data = data[:gaps[-1][0]]
			timevec = timevec[:gaps[-1][0]] # need to ensure timevec corresponds


		if gaps[0][0] == 0:
			data = data[gaps[0][1]+1:]
			timevec = timevec[gaps[0][1]+1:]
		
		gaps = get_gaps_positions(data) # find new indicies of gaps as we may have changed length

	x = np.array(range(0, len(data)))[~pd.isnull(data)]
	y = data[~pd.isnull(data)]

	f = interpolate.interp1d(x, y, bounds_error=False, fill_value = np.NaN)
	interpolated = np.array([x if not pd.isnull(x) else f(i) for i, x in enumerate(data)])	

	return interpolated, timevec, gaps


def get_seizures(subject):

	severity_table = pd.read_excel("SeverityTable.xlsx")

	subject_seizures = severity_table[severity_table.patient_id == subject]
	# lots of extra useful information in here, not just start

	return subject_seizures["start"], subject_seizures["duration"]


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

	# TODO keys defined again in produce hrv_dataframes
	time_dom_keys = np.array(['nni_counter', 'nni_mean', 'nni_min', 'nni_max', 'hr_mean', 'hr_min', 'hr_max', 'hr_std', 'nni_diff_mean', 'nni_diff_min', 'nni_diff_max', 'sdnn', 'sdnn_index', 'sdann', 'rmssd', 'sdsd', 'nn50', 'pnn50', 'nn20', 'pnn20', 'nni_histogram', 'tinn_n', 'tinn_m', 'tinn', 'tri_index'])
	freq_dom_keys = np.array(['fft_bands', 'fft_peak', 'fft_abs', 'fft_rel', 'fft_log', 'fft_norm', 'fft_ratio', 'fft_total', 'fft_plot', 'fft_nfft', 'fft_window', 'fft_resampling_frequency', 'fft_interpolation'])

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


def mra(out, timevec, data, gaps, onsets, durations, sharey=False):

	fs = 1/300 # TODO this should be param

	
	#data = data - np.mean(data)
	#data = butter_lowpass_filter(data, 1/(1*60*60), fs=fs, order=12)
	#data = butter_lowpass_filter(data, 1/(0.5*60*60), fs=fs, order=48)

	methods_peaks = {}  # dict of str mode decomposition method -> dict of peak periods of the components identified by those methods and the power WITHIN THE COMPONENT of that period

		
	# perform EMD
	title = f"MeanIHR_EMD"
	imfs = emd.sift.sift(data, sift_thresh=1).T
	#emd2 = EMD(); imfs = emd2(data)
	methods_peaks["EMD"] = components_plot(timevec, imfs, data, out, title, gaps, onsets, durations, sharey=sharey)
	

	# perform VMD (using CFSA to determine optimal K)
	title = f"MeanIHR_VMD"
	# initialise parameters
	alpha = 2000#(1/6) * (300)  # bandwidth constraint (2000 = "moderate") 
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
	methods_peaks["VMD"] = components_plot(timevec, u, data, out, title, gaps, onsets, durations, sharey=sharey)
	

	# perform EEMD
	title = f"MeanIHR_EEMD"
	imfs = emd.sift.ensemble_sift(data, nensembles=4, nprocesses=3, ensemble_noise=1).T
	imfs = imfs[1:]
	methods_peaks["EEMD"] = components_plot(timevec, imfs, data, out, title, gaps, onsets, durations, sharey=sharey)
	
	# perform DWT MRA
	for wavelet in ["sym4", "db4", "bior4.4", "coif4", "dmey"]:
		data_wt = data if len(data) % 2 == 0 else data[:-1]  # must be odd length
		timevec_wt = timevec if len(timevec) % 2 == 0 else timevec[:-1]
		output = pywt.mra(data_wt, pywt.Wavelet(wavelet), transform="dwt")
		output = np.flip(output, axis=0)
		if '.' in wavelet:
			title = f"MeanIHR_DWT_{'-'.join(wavelet.split('.'))}" # will mess up saving to file if . present in title
			methods_peaks[f"DWT_{'-'.join(wavelet.split('.'))}"] = components_plot(timevec_wt, output, data_wt, out, title, gaps, onsets, durations, sharey=sharey)	
		else:
			title = f"MeanIHR_DWT_{wavelet}"
			methods_peaks[f"DWT_{wavelet}"] = components_plot(timevec_wt, output, data_wt, out, title, gaps, onsets, durations, sharey=sharey)	

	# perform SSA (Singular Spectrum Analysis)
	title = f"MeanIHR_SSA"
	#components = decomposition.SingularSpectrumAnalysis(window_size=0.10, groups=K).fit_transform(data.reshape(1, -1))
	components = decomposition.SingularSpectrumAnalysis(window_size=max(2, K)).fit_transform(data.reshape(1, -1))
	components = components[0]
	components = np.flipud(components)
	methods_peaks["SSA"] = components_plot(timevec, components, data, out, title, gaps, onsets, durations, sharey=sharey)


	data = reflect(data)	
	# filter out the peaks identified	
	methods=["EMD", "EEMD", "VMD", "DWT_coif4"]	
	for method in methods:
		peaks = methods_peaks[method].keys()

		fig, axs = plt.subplots(len(peaks)+2, 1, sharex=True, figsize=(19.20, 19.20)) 
		plt.subplots_adjust(bottom=0.04, top=0.921, hspace=0.402)

		
		fontsize = "smaller"

		axs[0].plot(timevec, remove_reflect(data), alpha=0.5, color="black")
		axs[0].plot(timevec, cut_gaps(remove_reflect(data), gaps), color="black")
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
					axs[i+1].plot(timevec, remove_reflect(bandpass_filtered), alpha=0.5, color=color)
					axs[i+1].plot(timevec, cut_gaps(remove_reflect(bandpass_filtered), gaps), color=color)
					axs[i+1].set_title(f"Peak @ {peak}h (Filtered {np.round(highcut_period, decimals=2)}h - {np.round((peak+1), decimals=2)}h)", loc="left", fontsize=fontsize)
					
					filtered_data.append(bandpass_filtered)
				else: print(f"Highcut Frequency {highcut} exceeds Nyquist {fs/2}, so skipping this Peak")
			else: print(f"Peak Frequency {1/(peak*60*60)} exceeds Nyquist {fs/2}, so skipping this Peak")

		if len(peaks) > 0:
			filtered_data = np.sum(np.array(filtered_data), axis=0,)
			axs[-1].plot(timevec, remove_reflect(filtered_data), alpha=0.5, color="black")
			axs[-1].plot(timevec, cut_gaps(remove_reflect(filtered_data), gaps), color="black")
			axs[-1].set_title("Filtered Signal Summation", loc="left", fontsize=fontsize)

		
		for ax in axs:
			ax.xaxis.set_major_formatter(DateFormatter("%d/%m %H:%M:%S"))
			
			for onset, duration in zip(onsets, durations):
				ax.axvspan(onset, onset + datetime.timedelta(seconds=duration), color="r", alpha=0.5)
		
		fig.supxlabel("Time (d/m h:m:s)")
		fig.suptitle(f"Filtered_Peaks_{method}")
		fig.savefig(os.path.join(out, "components_plots", f"Filtered_Peaks_{method}"))


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


def wavelet_transform(timevec, data, gaps, onsets, durations, out):


	fs = 1/300 # TODO this needs to be param
	w = 5 # default Omega0 param for morlet2 (5.0). Seems to control frequency of complex sine part?

	fig, axs = plt.subplots(3,1,sharex=True, height_ratios=[2, 1, 7], figsize=(19.20, 19.20))
	
	axs[0].plot(timevec, data, color="black", alpha=0.5)
	axs[0].plot(timevec, cut_gaps(data, gaps), color="black")
	axs[0].set_title("Original Data", loc="right")

	data = data - np.mean(data) # remove DC offset, otherwise a lot of power at very low frequencies	
	data = reflect(data)
	

	#freqs = np.array([1/(24*60*60), 1/(18*60*60),1/(14*60*60),1/(12*60*60),1/(10*60*60),1/(8*60*60),1/(6*60*60),1/(4*60*60),1/(2*60*60),1/(1*60*60)])	
	#freqs = np.linspace(freqs[0], freqs[-1], 10000)
	#freqs = np.linspace(0, freqs[-1], 10000)
	#freqs = 
	#freqs = np.linspace(0, fs/2, 10000)

	#periods = 1/freqs
	#periods = np.array([days*24*60*60 for days in range(13, 1, -1)] + [hours*60*60 for hours in range(47, 0, -1)])
	#periods = np.array([hours*60*60 for hours in range(288, 0, -1)])
	periods = np.array([hours*60*60 for hours in np.arange(73, 0, -1)])
	freqs = 1/periods
	
	widths = w * fs / (2 * freqs * np.pi)

	cwtmatr = signal.cwt(data, signal.morlet2, widths, w=w, dtype=np.complex128)
	cwtmatr_yflip = cwtmatr	
	#cwtmatr_yflip = np.flipud(cwtmatr) # TODO can keep this, just need to flip y axis labels
	
	# one or the other, as cwtmatr is complex
	cwtmatr_yflip = np.abs(cwtmatr_yflip) # get magnitude
	#cwtmatr_yflip = cwtmatr_yflip.real 

	# remove padding	
	cwtmatr_yflip = np.apply_along_axis(remove_reflect, 1, cwtmatr_yflip)

	interpolation = "antialiased"#"none"
	#axs[2].imshow(cwtmatr_yflip, vmax = abs(cwtmatr).max(), vmin = -abs(cwtmatr).max(), aspect="auto", interpolation=interpolation)
	#axs[2].imshow(cwtmatr_yflip, vmax = abs(cwtmatr).max(), vmin = 0, aspect="auto", interpolation=interpolation)
	#pos = axs[2].imshow(cwtmatr_yflip, aspect="auto", interpolation=interpolation, cmap='Greens_r')
	
	pos = axs[2].pcolormesh(timevec, np.round(periods/60/60, decimals=0), cwtmatr_yflip)#, norm=matplotlib.colors.LogNorm(vmin=cwtmatr_yflip.min(), vmax=cwtmatr_yflip.max()))	

	cbar = fig.colorbar(pos, ax=axs[2], label="Squared Amplitude (Power?) (If Real Projection)", location="bottom", shrink=0.6)
	"""	
	yticks = ax.get_yticks()
	yticks = yticks[yticks>=0]
	yticks[yticks > 0] -= 1
	axs[1].set_yticks(ticks = yticks, labels=periods[np.int32(yticks)])
	"""
	#axs[2].set_yticks(ticks=range(0, len(cwtmatr)), labels=np.floor(periods/60/60))
	axs[2].set_ylabel("Period (h)")
	axs[2].set_title("Wavelet Transform Time-Frequency", loc="right")

	lowcut=1/(33*60*60)
	highcut=1/(21*60*60) 
	fs=fs 
	order=4
	
	try:
		# IF USING MAGNITUDE:
		circadian_bandpass = butter_bandpass_filter(data, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
		axs[1].plot(timevec, remove_reflect(circadian_bandpass), c="r", alpha=0.5)
		axs[1].plot(timevec, cut_gaps(remove_reflect(circadian_bandpass), gaps), c="r", label="Circadian Band (21h-33h)")
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


	fig.supxlabel("Time (d/m h:m:s)")
	fig.savefig(os.path.join(out, "components_plots", f"CWT"))
	#fig.show()	

	#plt.bar(periods/60/60, np.mean(cwtmatr_yflip, axis=1)); # produce FFT-like output


""" SIMULATED SUBJECTS """

def rhythm(timevec, hours, A=10, phi=0):
	# A = amplitude; sinewave will be centered around 0, with max amplitude at A and minimum at -A 
	# phi = phase - where, in radians, the cycle is at t=0
	return A * np.sin((2*np.pi*(1/(hours*60*60))*timevec) + phi)

def rhythm_chirp(timevec, hours0, t1, hours1, A=10, phi=0, method="linear"):
	# A = amplitude; sinewave will be centered around 0, with max amplitude at A and minimum at -A 
	# phi = phase - where, in radians, the cycle is at t=0
	return A * signal.chirp(timevec, f0=1/(hours0*60*60), t1 = t1, f1=1/(hours1*60*60), method=method, phi=phi)

def rhythm_square_conv(timevec, hours, A=10, phi=0):
	# convolved square wave rhythm
	
	rhythm = signal.square(2*np.pi*(1/(hours))*((timevec/60/60) + phi)) # TODO is phi right here?
	rhythm = np.convolve(rhythm, np.hanning(75), "same") # smooth transitions
	rhythm = stats.zscore((rhythm - np.mean(rhythm))) * A # re-center and set to desired amplitude

	return rhythm


def initialise_simulated_subject():
	# define some parameters/components common to all test subjects	

	segment_len_s = 300
	n_days = 3 # similar length to 909
	n_segments = (n_days * (24*60))/5 # how many segments of len segment_len_s?
	data_length = n_segments * segment_len_s # this is so we can use the same sample rates
	timevec = np.arange(0, data_length, segment_len_s)
 	
	baseline = 85 # add this to sinewave, to shift it from being mean-centered, so it resembles a mean IHR
	
	data = np.zeros(len(timevec))

	return timevec, data, baseline

def generate_simulated_noise():
	
	timevec, data, baseline = initialise_simulated_subject()

	noise = 1* np.random.normal(0, 1, len(timevec))
	
	return noise


""" template 
def simulated_subject_A():
	
	timevec, data, baseline = initialise_simulated_subject()

	data += baseline


	noisy_data = data + generate_simulated_noise()

	contains = {}

	return 'A', contains, data, noisy_data

"""
	
def simulated_subject_A():
	
	timevec, data, baseline = initialise_simulated_subject()


	data +=  rhythm_square_conv(timevec, hours=24, A=10, phi=0)


	data += baseline
	
	noisy_data = data + generate_simulated_noise()

	
	contains = {24:10}


	return "A", contains, data, noisy_data

def simulated_subject_B():
	
	timevec, data, baseline = initialise_simulated_subject()


	data += rhythm_square_conv(timevec, hours=24, A=10, phi=0)
	data += rhythm(timevec, hours=12, A=8, phi=0)


	data += baseline
	
	noisy_data = data + generate_simulated_noise()
	
	contains = {24:10, 12:8}
	

	return "B", contains, data, noisy_data

def simulated_subject_C():
	
	timevec, data, baseline = initialise_simulated_subject()

	data += rhythm(timevec, hours=48, A=8, phi=0)
	data += rhythm_square_conv(timevec, hours=24, A=10, phi=0)
	data += rhythm(timevec, hours=12, A=8, phi=0)


	data += baseline
	
	noisy_data = data + generate_simulated_noise()
	
	contains = {48:8, 24:10, 12:8}

	return "C", contains, data, noisy_data

"""
def simulate_data():

	
	segment_len_s = 300

	n_days = 3 # similar length to 909
	n_segments = (n_days * (24*60))/5 # how many segments of len segment_len_s?
	data_length = n_segments * segment_len_s # this is so we can use the same sample rates
	timevec = np.arange(0, data_length, segment_len_s)
 	
	shift = 85 #add this to sinewave, to shift it from being mean-centered, so it resembles a mean IHR
	
	data = np.zeros(len(timevec))
	#circadian1 = rhythym(24, A=12, phi=0); data += circadian1 
	#circadian1 = signal.sawtooth(2*np.pi*(1/(24))*timevec/60/60); data += circadian1

	circadian1_shift = 100
	circadian1_A = 6
	circadian1 = signal.square(2*np.pi*(1/(24))*((timevec/60/60) + circadian1_shift))
	circadian1 = np.convolve(circadian1, np.hanning(75), "same") # smooth transitions
	circadian1 = stats.zscore((circadian1 - np.mean(circadian1))) * circadian1_A # re-center and set to desired amplitude
	data += circadian1
	
	#circadian2 = rhythym(19, A=10, phi=0); data += circadian2
	
	##multidien = rhythym(50, A=2, phi=0);  data += multidien
	
	#ultradian1 = rhythym(12, A=5, phi=1);  data += ultradian1
	ultradian2 = rhythym(3, A=3, phi=0) * np.hanning(len(timevec));  data += ultradian2
	#ultradian3 = rhythym_chirp(8, np.median(timevec), 10, A=5, method="linear"); data+= ultradian3

	noise = 1* np.random.normal(0, 1, len(timevec)); data += noise 
	
	data += shift

	#data = multidien + circadian + (ultradian(1, 20) * np.hanning(n_segments)) + shift


	return data	

"""
		

def temp_plot(subjects_methods_peaks, now):

	# set up dir
	logfile_dir = constants.SUBJECT_DATA_OUT.format(subject='LOGS')		
	plots_subdir = f"{now.day}_{now.month}_{now.year}_{now.hour}_{now.minute}_{now.second}"	
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
		fig, ax = plt.subplots(figsize=(7.20, 7.20))
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

	for subject in pd.unique(subjects_methods_peaks_df["subject"]):

		# A
		fig_pp, ax_pp = plt.subplots(figsize=(10.80, 10.80))
		ax_pp.set_xlabel("Period (h)")
		#ax_pp.set_ylabel("Method")
		#ax_pp.set_xscale("log")
		
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
		sns.scatterplot(data=subjects_methods_peaks_df[subjects_methods_peaks_df["subject"] == subject], y="method", x="cycle_period", size="power", sizes=(10, 300), ax=ax_pp)

		"""
		extra_periods = [2, 12,]
		xt = ax_pp.get_xticks()
		for period in extra_periods:
			xt = np.append(xt, period)
		ax_pp.set_xticks(xt)		
		"""
		harmonics = [(1/(n*(1.15e-5)))/60/60 for n in np.concatenate(([0.5], np.arange(1, 20)))]
		for i, harmonic in enumerate(harmonics):
			col="grey"
			alph = max(0, (1 - (0.1* np.abs(1-i)))) # decrease alpha of harmonic by 0.1 based on how far it is from central 24h frequency
			ax_pp.axvline(harmonic, c=col, alpha=alph)
			ax_pp.text(harmonic, -0.75, f"{np.round(harmonic, decimals=2)}h", c=col, alpha=alph, rotation=90)

		ax_pp.set_yticks(ticks=np.array(list(methods_y_mapping.values())), labels=methods_y_mapping.keys())
		fig_pp.savefig(os.path.join(plots_dir, f"{subject}_peaks"))
	

	# B 
	for method in methods:	
		ax = axes[np.where(np.array(list(methods)) == method)[0][0]]
		#ax.axhline(subject_y_mapping[subject], c="silver", alpha=0.5)
		sns.scatterplot(data=subjects_methods_peaks_df[subjects_methods_peaks_df["method"] == method], y="subject", x="cycle_period", hue="power", palette=pal, edgecolor="black", zorder=4, ax=ax)
		for y in ax.get_yticks():
			ax.axhline(y, c="silver", alpha=0.5, zorder=3.5)


	# B
	for i in range(0, len(axes)):

		method = list(methods)[i]

		axes[i].set_yticks(ticks=np.array(list(subject_y_mapping.values())), labels=subject_y_mapping.keys())

		axes[i].legend([],[], frameon=False)

		axes[i].set_title(method)

		axes[i].set_xlabel("Period (h)")
		
		axes[i].set_ylabel("Subject")

		axes[i].axvspan(0, 20, color="#DED5E8")
		axes[i].axvspan(20, 28, color="#CCE5D7")
		axes[i].axvspan(28, 50, color="#EED8CD")

		axes[i].set_xlim([0, 50])
		#axes[i].autoscale(enable=True, axis='x', tight=True)

		figures[i].savefig(os.path.join(plots_dir, f"{method}_subjects"))
	

if __name__ == "__main__":
	#subjects = ["909", "902", "931"] 
	UCLH_subjects = ["1005","1055","1097","1119","1167","1182","1284","815","902","934","95", "1006","1064","1109","1149","1178","1200","770","821","909","940","999", "1038","1085","1110","1163","1179","1211","800","852","931","943"]
	taVNS_subjects = ["taVNS001","taVNS002","taVNS003","taVNS004","taVNS005","taVNS006","taVNS007","taVNS008","taVNS009","taVNS010","taVNS011","taVNS012","taVNS013","taVNS014","taVNS015","taVNS017","taVNS018"]

	subjects = UCLH_subjects + taVNS_subjects
	#subjects = ["sim"]
	#subjects = ["909"]
	#subjects = ["taVNS018"]
	#subjects =["909", "taVNS001"]

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
	
	subjects_methods_peaks = {} # dict of subject -> their method peaks results dict
	
	i = 0
	for subject in subjects:

		plt.close("all")
		print(f"\n$RUNNING FOR SUBJECT: {subject}\n")
	
		try:

			root = constants.SUBJECT_DATA_ROOT.format(subject=subject)
			out = constants.SUBJECT_DATA_OUT.format(subject=subject)

			rng = np.random.default_rng(1905)


			if not subject=="sim":
			
				if not "taVNS" in subject:	
					# collate data and resolve overlaps
					#run_speedyf(root, out); 
					
					# produce hrv metric dataframes, and save to out/
					calculate_hrv_metrics(root, out, rng)# will not re-calculate if dataframes already present in out

				# load the hrv metric dataframes we just produced
				time_dom_df, freq_dom_df, modification_report_df = load_hrv_dataframes(out)

				# what metric are we interested in?
				metric = "hr_mean"
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
				
				# interpolate gaps (runs of NaN) so we can use with signal decomposition. save gap positions for visualisation
				interpolated, timevec, gaps = interpolate_gaps(data, timevec) 

				# perform multi-resolution analysis (signal decomposition)	
				methods_peaks = mra(out, timevec, interpolated, gaps, onsets, durations, sharey=False)
				subjects_methods_peaks[subject] = methods_peaks

				# perform time-frequency analysis using wavelet_transform
				wavelet_transform(timevec, interpolated, gaps, onsets, durations, out)	
			
			else:

				simulated_subjects_peaks = {}

				for simulate_subject in [simulated_subject_A, simulated_subject_B, simulated_subject_C]:
				#for simulate_subject in [simulated_subject_A]:
					
					letter, contains, noisefree_data, noisy_data = simulate_subject()		
					
					simulated_subjects_peaks[letter] = contains
					simulated_subjects_peaks[f"{letter}n"] = contains

					for suffix, data in zip(["", "n"], [noisefree_data, noisy_data]):
						out = constants.SUBJECT_DATA_OUT.format(subject=f"sim{letter}{suffix}")

						# simulate missing data
						data[math.floor(np.quantile(range(0, len(data)), 0.7)): math.ceil(np.quantile(range(0, len(data)), 0.8))] = np.NaN # large gap
						data[rng.choice(len(data), math.floor(len(data) * 0.1))] = np.NaN # smaller gaps randomly throughout

						timevec = [datetime.datetime.fromtimestamp(0) + datetime.timedelta(seconds=(i * 300)) for i in range(0, len(data))]

						onsets = durations = []
					
						interpolated, timevec, gaps = interpolate_gaps(data, timevec) 

						methods_peaks = mra(out, timevec, interpolated, gaps, onsets, durations, sharey=False)
						subjects_methods_peaks[letter+suffix] = methods_peaks

						wavelet_transform(timevec, interpolated, gaps, onsets, durations, out)	
		
						plt.close("all")	


			with open(logfile_loc, "a") as logfile:
				logfile.write(f"\n{i+1}/{len(subjects)}:\t{subject}:\tSuccess!\tRuntime:{time.time()-start}")

		except Exception as e:
			print("REMOVE RAISE EXCEPION")
			raise e

			with open(logfile_loc, "a") as logfile:
				logfile.write(f"\n{i+1}/{len(subjects)}:\t{subject}:\t{e}")


		i += 1


	end = time.time()

	with open(logfile_loc, "a") as logfile:
		logfile.write(f"\nComplete@{str(datetime.datetime.now())}!\tRuntime:{end-start}")


	temp_plot(subjects_methods_peaks, now)
