import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
from scipy import fft
from scipy import interpolate
from scipy import signal
from scipy.signal import butter, sosfiltfilt, sosfreqz
import pandas as pd

import os
import sys
import json
import datetime

import constants

sys.path.append(constants.SPEEDYF_LOCATION)
from speedyf import edf_collate, edf_overlaps, edf_segment

sys.path.append(constants.HRV_PREPROC_LOCATION)
from hrv_preprocessor.hrv_preprocessor import hrv_per_segment, produce_hrv_dataframes, save_hrv_dataframes, load_hrv_dataframes

# TODO REMEMBER TO CITE PACKAGES
import emd  # EMD, EEMD
import pywt # Wavelet Transform?
import vmdpy # Variational Mode Decomposition

np.random.seed(1905)


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


def components_plot(timevec, components, original, out, title, gaps, sharey=True):

	out = os.path.join(out, "components_plots")
	if not os.path.exists(out):
		print("Setting up directory for MRA Output: at '{}'".format(out))
		os.makedirs(out, exist_ok=True)


	fig1, ax1 = plt.subplots(len(components)+2, 1, sharex=True, sharey=sharey, figsize=(19.20, 19.20))
	fig2, ax2 = plt.subplots(len(components)+2, 1, sharex=False, sharey=True, figsize=(19.20, 19.20))
	plt.subplots_adjust(bottom=0.04, top=0.921, hspace=0.402)

	#x_original = range(0, len(original))
	#x_imfs = range(0, components.shape[1])
	x_original = timevec
	x_imfs = timevec[0: components.shape[1]]
	alpha = 0.3 # transparency of the interpolated sections
	c = "blueviolet" # color of components
	
	fs = 1/300 # TODO this needs to be param
	nperseg = 30 

	ax2_xticks = np.array([0, 1/(24*60*60), 1/(18*60*60),1/(14*60*60),1/(12*60*60),1/(10*60*60),1/(8*60*60),1/(6*60*60),1/(4*60*60),1/(2*60*60),1/(60*60)])	
	ax2_xticklabels = (1/ax2_xticks) # invert to frequency is period (s)
	ax2_xticklabels = (ax2_xticklabels / 60) / 60 # get from s into hours 
	ax2_xticklabels = [np.round(lab, decimals=1) if lab != np.inf else lab for lab in ax2_xticklabels]
	ax2_xlim = [0, 1/(60*60)] # 0 -> 1h period
	ax2_xticklab_rot = 45

	ax1[0].plot(x_original, original, color="black", alpha=alpha)
	ax1[0].plot(x_original, cut_gaps(original, gaps), color="black")
	ax1[0].set_title("Original", loc="left")

	#f, Pxx_den = signal.welch(original, fs, nperseg=nperseg)	
	f, Pxx_den = my_fft(original, fs)
	ax2[0].plot(f, Pxx_den)
	#ax2[0].set_xticks(ticks=f, labels=1/f)
	ax2[0].set_xticks(ticks=ax2_xticks, labels=ax2_xticklabels, rotation=ax2_xticklab_rot)
	ax2[0].axvline((1/(24*60*60)), c="r", label="Circadian")
	ax2[0].set_xlim(ax2_xlim)
	ax2[0].set_title("Original", loc="left")

	reconstruction = np.sum(components, axis=0)
	ax1[1].plot(x_imfs, reconstruction, color="black", alpha=alpha)
	ax1[1].plot(x_imfs, cut_gaps(reconstruction, gaps), color="black")
	ax1[1].set_title("Reconstruction", loc="left")	
	
	#f, Pxx_den = signal.welch(reconstruction, fs, nperseg=nperseg)	
	f, Pxx_den = my_fft(reconstruction, fs)
	ax2[1].plot(f, Pxx_den)
	#ax2[1].set_xticks(ticks=f, labels=1/f)
	ax2[1].set_xticks(ticks=ax2_xticks, labels=ax2_xticklabels, rotation=ax2_xticklab_rot)
	ax2[1].axvline((1/(24*60*60)), c="r", label="Circadian")
	ax2[1].set_xlim(ax2_xlim)
	ax2[1].set_title("Reconstruction", loc="left")	
	
	for i, comp in enumerate(components):
		ax1[i+2].plot(x_imfs, comp, color=c, alpha=alpha)
		ax1[i+2].plot(x_imfs, cut_gaps(comp, gaps), color=c)
		ax1[i+2].set_title(f"Component #{i}", loc="left")

		#f, Pxx_den = signal.welch(comp, fs, nperseg=nperseg)	
		f, Pxx_den = my_fft(comp, fs)
		ax2[i+2].plot(f, Pxx_den)
		#ax2[i+2].set_xticks(ticks=f, labels=1/f)
		#ax2[i+2].set_xticks(ticks=ax2[i+2].get_xticks(), labels=1/ax2[i+2].get_xticks())
		ax2[i+2].set_xticks(ticks=ax2_xticks, labels=ax2_xticklabels, rotation=ax2_xticklab_rot)
		ax2[i+2].axvline((1/(24*60*60)), c="r", label="Circadian")
		ax2[i+2].set_xlim(ax2_xlim)
		ax2[i+2].set_title(f"Component #{i}", loc="left")
	
	fig1.suptitle(title)
	for ax in ax1:
		ax.xaxis.set_major_formatter(DateFormatter("%d/%m %H:%M:%S"))
	fig1.supxlabel("Time (d/m h:m:s)")
	fig1.savefig(os.path.join(out, title))

	
	fig2.suptitle(title+"_PSD")
	fig2.supylabel("")
	fig2.supxlabel("Period (h)")
	handles, labels = ax2[-1].get_legend_handles_labels() # get from last axis only, to avoid duplicates
	fig2.legend(handles, labels)

	fig2.savefig(os.path.join(out, title+"_PSD"))


	peaks = []
	# produce plot of stacked IMFs with peaks highlighted 
	cmap = matplotlib.colormaps["ocean"]
	fig3, ax3 = plt.subplots(figsize=(10.80, 10.80)) 
	for i, comp in enumerate(components):
		f, Pxx_den = my_fft(comp, fs) # REPEATED CODE

		peak_idx = np.where(Pxx_den == np.max(Pxx_den))
		peak_Pxx = Pxx_den[peak_idx]
		peak_freq = f[peak_idx]
		peak_period_s = 1/peak_freq
		peak_period_h = peak_period_s/60/60
		peak_period_h = np.round(peak_period_h, decimals=2)
		label = f"Component #{i}, peak period = {peak_period_h[0]}h"
		color = cmap(1/len(components) * i)
		
		peaks.append(peak_period_h)

		ax3.plot(f, Pxx_den, c=color, label=label)
		ax3.fill_between(f, Pxx_den, color=color)	
		ax3.scatter(peak_freq, peak_Pxx, marker="x", c="r", zorder=10)


	ax3.set_xscale("log")
	fig3.legend(loc="upper right")
	fig3.suptitle(title+" Components PSDs Overlaid")
	fig3.supxlabel("Frequency (Hz)")
	fig3.savefig(os.path.join(out, title+"_PSD_OVERLAY"))

	return peaks


def my_fft(data, sample_rate):

	data = data - np.mean(data) # remove DC Offset (0Hz Spike)

	yf = fft.rfft(data)
	xf = fft.rfftfreq(len(data), 1/sample_rate)

	return xf, np.abs(yf)

def fft_plot(data, sample_rate): # warn; can't use fft with HRV, irregularly sampled
	fig, ax = plt.subplots()
	xf, yf = my_fft(data, sample_rate)
	ax.plot(xf, np.abs(yf))

	return fig, ax

def reflect(data):
	padlen = len(data) // 2
	return np.pad(data, padlen, mode="constant")

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


""" PIPELINE FUNCTIONS """

def run_speedyf(root, out):

	edf_collate(root, out)

	if len(edf_overlaps.check(root, out, verbose=True)) != 0:
		edf_overlaps.resolve(root, out)


def calculate_hrv_metrics(root, out, forced=False):

	subject_out = out

	segmenter = edf_segment.EDFSegmenter(root, out, segment_len_s=300, cache_lifetime=1)	

	ecg_channels = [ch for ch in segmenter.get_available_channels() if "ecg" in ch.lower()]
	if len(ecg_channels) == 0: raise KeyError("No channels with names containing 'ECG' could be found!")
	segmenter.set_used_channels(ecg_channels)


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

		freq_dom_hrvs = []
		time_dom_hrvs = []
		modification_reports = []

		# TODO keys defined again in produce hrv_dataframes
		time_dom_keys = np.array(['nni_counter', 'nni_mean', 'nni_min', 'nni_max', 'hr_mean', 'hr_min', 'hr_max', 'hr_std', 'nni_diff_mean', 'nni_diff_min', 'nni_diff_max', 'sdnn', 'sdnn_index', 'sdann', 'rmssd', 'sdsd', 'nn50', 'pnn50', 'nn20', 'pnn20', 'nni_histogram', 'tinn_n', 'tinn_m', 'tinn', 'tri_index'])
		freq_dom_keys = np.array(['fft_bands', 'fft_peak', 'fft_abs', 'fft_rel', 'fft_log', 'fft_norm', 'fft_ratio', 'fft_total', 'fft_plot', 'fft_nfft', 'fft_window', 'fft_resampling_frequency', 'fft_interpolation'])


		for segment in segmenter:
			print(f"{segment.idx}/{segmenter.get_max_segment_count()-1}")

			ecg = segment.data[ecg_channel].to_numpy()
			if len(ecg) != 0:
				ecg = butter_lowpass_filter(ecg, 22, 512, 4)

			#eps = 0.125
			eps = 0.14

			rpeaks, rri, rri_corrected, freq_dom_hrv, time_dom_hrv, modification_report = hrv_per_segment(ecg, segment.sample_rate, 5, segment_idx=segment.idx, save_plots_dir=os.path.join(out, "saved_plots"), save_plots=True, save_plot_filename=segment.idx, use_segmenter="engzee", DBSCAN_RRI_EPSILON_MEAN_MULTIPLIER=eps, DBSCAN_MIN_SAMPLES=70)
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


def mra(timevec, data, gaps, sharey=False):
		
	# perform EMD
	title = f"MeanIHR_EMD"
	imfs = emd.sift.sift(data).T
	emd_peaks = components_plot(timevec, imfs, data, out, title, gaps, sharey=sharey)
	
	# perform EEMD
	title = f"MeanIHR_EEMD"
	imfs = emd.sift.ensemble_sift(data, nensembles=4, nprocesses=3, ensemble_noise=1).T
	imfs = imfs[1:]
	eemd_peaks = components_plot(timevec, imfs, data, out, title, gaps, sharey=sharey)
	
	# perform VMD
	title = f"MeanIHR_VMD"
	alpha = 2000  # moderate bandwidth constraint  
	tau = 0.      # noise-tolerance (no strict fidelity enforcement)  
	K = 2         # n of modes to be recovered  
	DC = 0        # no DC part imposed  
	init = 1      # initialize omegas uniformly 
	tol = 1e-7
	u, u_hat, omega = vmdpy.VMD(data, alpha, tau, K, DC, init, tol)
	u = np.flipud(u)
	vmd_peaks = components_plot(timevec, u, data, out, title, gaps, sharey=sharey)

	# perform WT MRA
	title = f"MeanIHR_WT"
	#wavelet = pywt.Wavelet("sym4") # sym4 is default wavelet used by MATLAB modwt()
	wavelet = pywt.Wavelet("db4") 
	data_wt = data if len(data) % 2 == 0 else data[:-1]  # must be odd length
	timevec_wt = timevec if len(timevec) % 2 == 0 else timevec[:-1]
	output = pywt.mra(data_wt, wavelet, transform="dwt")
	output = np.flip(output, axis=0)
	wt_peaks = components_plot(timevec_wt, output, data_wt, out, title, gaps, sharey=sharey)	

	# filter out the peaks identified	
	peaks = eemd_peaks # which decomposition method's peaks will we use?
	
	fig, axs = plt.subplots(len(peaks)+2, 1, sharex=True, figsize=(19.20, 19.20)) 
	plt.subplots_adjust(bottom=0.04, top=0.921, hspace=0.402)

	data = reflect(data)
	
	fontsize = "smaller"

	axs[0].plot(timevec, remove_reflect(data), alpha=0.5, color="black")
	axs[0].plot(timevec, cut_gaps(remove_reflect(data), gaps), color="black")
	axs[0].set_title("Original Data", loc="left", fontsize=fontsize)

	filtered_data = []
	for i, peak in enumerate(peaks):

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


				test_butter_bandpass(lowcut, highcut, fs, order)

				bandpass_filtered = butter_bandpass_filter(data, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
				
				color = "dodgerblue"
				axs[i+1].plot(timevec, remove_reflect(bandpass_filtered), alpha=0.5, color=color)
				axs[i+1].plot(timevec, cut_gaps(remove_reflect(bandpass_filtered), gaps), color=color)
				axs[i+1].set_title(f"Peak @ {peak[0]}h (Filtered {np.round(highcut_period[0], decimals=2)}h - {np.round((peak+1)[0], decimals=2)}h)", loc="left", fontsize=fontsize)
				
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
	fig.supxlabel("Time (d/m h:m:s)")

	fig.savefig(os.path.join(out, "Filtered_Peaks"))

def wavelet_transform(timevec, data, gaps):


	fs = 1/300 # TODO this needs to be param
	w = 6.0 # default Omega0 param for morlet2 (5.0). Seems to control frequency of complex sine part?

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
	cwtmatr_yflip = np.abs(cwtmatr_yflip) # this isn't in tutorial (as tehy are using ricker wavelet, not complex morlet, i presume), imshow won't work without
	
	cwtmatr_yflip = np.apply_along_axis(remove_reflect, 1, cwtmatr_yflip)

	#cwtmatr_yflip = np.power(cwtmatr_yflip, 2)

	fig, axs = plt.subplots(3,1,sharex=True, height_ratios=[2, 1, 7])
	interpolation = "antialiased"#"none"
	#axs[2].imshow(cwtmatr_yflip, vmax = abs(cwtmatr).max(), vmin = -abs(cwtmatr).max(), aspect="auto", interpolation=interpolation)
	#axs[2].imshow(cwtmatr_yflip, vmax = abs(cwtmatr).max(), vmin = 0, aspect="auto", interpolation=interpolation)
	#pos = axs[2].imshow(cwtmatr_yflip, aspect="auto", interpolation=interpolation, cmap='Greens_r')
	
	pos = axs[2].pcolormesh(timevec, np.round(periods/60/60, decimals=0), cwtmatr_yflip)	

	#cbar = fig.colorbar(pos, ax=axs[2])
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
		circadian_bandpass = butter_bandpass_filter(data, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
		axs[1].plot(timevec, remove_reflect(circadian_bandpass), c="r", alpha=0.5)
		axs[1].plot(timevec, cut_gaps(remove_reflect(circadian_bandpass), gaps), c="r", label="Circadian Band (21h-33h)")
		axs[1].set_title("Bandpass-filtered Circadian Rhythm (21h-33h)", loc="right")
	except ValueError as ve:
		print(ve)
		pass


	data = remove_reflect(data)
	axs[0].plot(timevec, data, color="black", alpha=0.5)
	axs[0].plot(timevec, cut_gaps(data, gaps), color="black")
	axs[0].set_title("Original Data", loc="right")
	

	for ax in axs:
		ax.xaxis.set_major_formatter(DateFormatter("%d/%m %H:%M:%S"))
	fig.supxlabel("Time (d/m h:m:s)")
	
	fig.show()	

def simulate_data():

	
	segment_len_s = 300

	n_days = 3 # similar length to 909
	n_segments = (n_days * (24*60))/5 # how many segments of len segment_len_s?
	data_length = n_segments * segment_len_s # this is so we can use the same sample rates
	timevec = np.arange(0, data_length, segment_len_s)
 

	def rhythym(hours, A=10, phi=0):
		# A = amplitude; sinewave will be centered around 0, with max amplitude at A and minimum at -A 
		# phi = phase - where, in radians, the cycle is at t=0
		return A * np.sin((2*np.pi*(1/(hours*60*60))*timevec) + phi)

	circadian1 = rhythym(24, 10, 0); data = circadian1
	#circadian2 = rhythym(19, 10, 0); data += circadian2
	#multidien = rhythym(50, 10, 0);  data += multidien
	#ultradian1 = rhythym(5, 10, 0);  data += ultradian1
	ultradian2 = rhythym(3, 100, 0);  data += ultradian2
	
	shift = 85 #add this to sinewave, to shift it from being mean-centered, so it resembles a mean IHR
	data += shift

	#data = multidien + circadian + (ultradian(1, 20) * np.hanning(n_segments)) + shift

	return data	


if __name__ == "__main__":
	subject = "sim"
	root = constants.SUBJECT_DATA_ROOT.format(subject=subject)
	out = constants.SUBJECT_DATA_OUT.format(subject=subject)

	if not subject=="sim":
		# collate data and resolve overlaps
		run_speedyf(root, out); 
		
		# produce hrv metric dataframes, and save to out/
		calculate_hrv_metrics(root, out) # will not re-calculate if dataframes already present in out

		# load the hrv metric dataframes we just produced
		time_dom_df, freq_dom_df, modification_report_df = load_hrv_dataframes(out)

		# what metric are we interested in?
		data = time_dom_df["hr_mean"]
		#data = freq_dom_df["fft_ratio"]
		data = np.array(data)
	
		timevec = edf_segment.EDFSegmenter(root, out, segment_len_s=300).get_segment_onsets(as_datetime=True)
	
	else:
		data = simulate_data()
		timevec = [datetime.datetime.fromtimestamp(0) + datetime.timedelta(seconds=(i * 300)) for i in range(0, len(data))]


	# interpolate gaps (runs of NaN) so we can use with signal decomposition. save gap positions for visualisation
	interpolated, timevec, gaps = interpolate_gaps(data, timevec) 

	# perform multi-resolution analysis (signal decomposition)	
	#mra(timevec, interpolated, gaps, sharey=False)

	# perform time-frequency analysis using wavelet_transform
	wavelet_transform(timevec, interpolated, gaps)	
