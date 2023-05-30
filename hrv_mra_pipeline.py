import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import fft
from scipy import interpolate
from scipy import signal
from scipy.signal import butter, sosfiltfilt, sosfreqz
import pandas as pd

import os
import sys

from speedyf import edf_collate, edf_overlaps, edf_segment

sys.path.append("/home/bcsm/University/stage-3/BSc_Project/summer_proj/VNS_2")
from hrv_preprocessor.hrv_preprocessor import hrv_per_segment, produce_hrv_dataframes, save_hrv_dataframes, load_hrv_dataframes

from filtering import butter_lowpass_filter

# TODO REMEMBER TO CITE PACKAGES
import emd  # EMD, EEMD
import pywt # Wavelet Transform?
import vmdpy # Variational Mode Decomposition

np.random.seed(1905)


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


def components_plot(components, original, out, title, gaps, sharey=True):

	out = os.path.join(out, "components_plots")
	if not os.path.exists(out):
		print("Setting up directory for MRA Output: at '{}'".format(out))
		os.makedirs(out, exist_ok=True)


	fig, ax = plt.subplots(len(components)+2, 1, sharex=True, sharey=sharey, figsize=(19.20, 19.20))
	fig2, ax2 = plt.subplots(len(components)+2, 1, sharex=False, sharey=True, figsize=(19.20, 19.20))

	x_original = range(0, len(original))
	x_imfs = range(0, components.shape[1])
	alpha = 0.3 # transparency of the interpolated sections
	c = "blueviolet" # color of components
	
	fs = 1/300
	nperseg = 30 

	ax2_xticks = np.array([0, 1/(24*60*60), 1/(18*60*60),1/(14*60*60),1/(12*60*60),1/(10*60*60),1/(8*60*60),1/(6*60*60),1/(4*60*60),1/(2*60*60),1/(60*60)])	
	ax2_xticklabels = (1/ax2_xticks) # invert to frequency is period (s)
	ax2_xticklabels = (ax2_xticklabels / 60) / 60 # get from s into hours 
	ax2_xticklabels = [np.round(lab, decimals=1) if lab != np.inf else lab for lab in ax2_xticklabels]
	ax2_xlim = [0, 1/(60*60)] # 0 -> 1h period
	ax2_xticklab_rot = 45

	ax[0].plot(x_original, original, color="black", alpha=alpha)
	ax[0].plot(x_original, cut_gaps(original, gaps), color="black")
	ax[0].set_title("Original", loc="left")

	#f, Pxx_den = signal.welch(original, fs, nperseg=nperseg)	
	f, Pxx_den = my_fft(original, fs)
	ax2[0].plot(f, Pxx_den)
	#ax2[0].set_xticks(ticks=f, labels=1/f)
	ax2[0].set_xticks(ticks=ax2_xticks, labels=ax2_xticklabels, rotation=ax2_xticklab_rot)
	ax2[0].axvline((1/(24*60*60)), c="r", label="Circadian")
	ax2[0].set_xlim(ax2_xlim)
	ax2[0].set_title("Original", loc="left")

	reconstruction = np.sum(components, axis=0)
	ax[1].plot(x_imfs, reconstruction, color="black", alpha=alpha)
	ax[1].plot(x_imfs, cut_gaps(reconstruction, gaps), color="black")
	ax[1].set_title("Reconstruction", loc="left")	
	
	#f, Pxx_den = signal.welch(reconstruction, fs, nperseg=nperseg)	
	f, Pxx_den = my_fft(reconstruction, fs)
	ax2[1].plot(f, Pxx_den)
	#ax2[1].set_xticks(ticks=f, labels=1/f)
	ax2[1].set_xticks(ticks=ax2_xticks, labels=ax2_xticklabels, rotation=ax2_xticklab_rot)
	ax2[1].axvline((1/(24*60*60)), c="r", label="Circadian")
	ax2[1].set_xlim(ax2_xlim)
	ax2[1].set_title("Reconstruction", loc="left")	
	
	for i, comp in enumerate(components):
		ax[i+2].plot(x_imfs, comp, color=c, alpha=alpha)
		ax[i+2].plot(x_imfs, cut_gaps(comp, gaps), color=c)
		ax[i+2].set_title(f"Component #{i}", loc="left")

		#f, Pxx_den = signal.welch(comp, fs, nperseg=nperseg)	
		f, Pxx_den = my_fft(comp, fs)
		ax2[i+2].plot(f, Pxx_den)
		#ax2[i+2].set_xticks(ticks=f, labels=1/f)
		#ax2[i+2].set_xticks(ticks=ax2[i+2].get_xticks(), labels=1/ax2[i+2].get_xticks())
		ax2[i+2].set_xticks(ticks=ax2_xticks, labels=ax2_xticklabels, rotation=ax2_xticklab_rot)
		ax2[i+2].axvline((1/(24*60*60)), c="r", label="Circadian")
		ax2[i+2].set_xlim(ax2_xlim)
		ax2[i+2].set_title(f"Component #{i}", loc="left")
	
	fig.suptitle(title)
	
	fig2.suptitle(title+"_PSD")
	fig2.supylabel("")
	fig2.supxlabel("Period (h)")
	handles, labels = ax2[-1].get_legend_handles_labels() # get from last axis only, to avoid duplicates
	fig2.legend(handles, labels)

	plt.subplots_adjust(bottom=0.04, top=0.921, hspace=0.402)


	fig.savefig(os.path.join(out, title))
	fig2.savefig(os.path.join(out, title+"_PSD"))

	return fig, ax


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


def run_speedyf(root, out):

	edf_collate(root, out)

	if len(edf_overlaps.check(root, out, verbose=True)) != 0:
		edf_overlaps.resolve(root, out)


def calculate_hrv_metrics(root, out, forced=False):


	try:
		load_hrv_dataframes(out)
		
		if not forced:
			print("HRV Metrics Dataframes appear to exist, and parameter forced=False, so HRV Metrics will not be re-calculated.") 
			return

	except FileNotFoundError:
			pass

	segmenter = edf_segment.EDFSegmenter(root, out, segment_len_s=300, cache_lifetime=1)	

	segmenter.set_used_channels(["ECG"])
	
	freq_dom_hrvs = []
	time_dom_hrvs = []
	modification_reports = []


	# TODO keys defined again in produce hrv_dataframes
	time_dom_keys = np.array(['nni_counter', 'nni_mean', 'nni_min', 'nni_max', 'hr_mean', 'hr_min', 'hr_max', 'hr_std', 'nni_diff_mean', 'nni_diff_min', 'nni_diff_max', 'sdnn', 'sdnn_index', 'sdann', 'rmssd', 'sdsd', 'nn50', 'pnn50', 'nn20', 'pnn20', 'nni_histogram', 'tinn_n', 'tinn_m', 'tinn', 'tri_index'])
	freq_dom_keys = np.array(['fft_bands', 'fft_peak', 'fft_abs', 'fft_rel', 'fft_log', 'fft_norm', 'fft_ratio', 'fft_total', 'fft_plot', 'fft_nfft', 'fft_window', 'fft_resampling_frequency', 'fft_interpolation'])


	for segment in segmenter:
		print(f"{segment.idx}/{segmenter.get_max_segment_count()}")

		ecg = segment.data["ECG"].to_numpy()
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

def interpolate_gaps(data):

	gaps = get_gaps_positions(data)
	# remove gaps that start at start/end at end (will not be interpolated, and will mess up decomposition)
	if len(gaps) > 0:
		if gaps[-1][1] == len(data)-1:
			data = data[:gaps[-1][0]]

		if gaps[0][0] == 0:
			data = data[gaps[0][1]+1:]
		
		gaps = get_gaps_positions(data) # find new indicies of gaps as we may have changed length

	x = np.array(range(0, len(data)))[~pd.isnull(data)]
	y = data[~pd.isnull(data)]

	f = interpolate.interp1d(x, y, bounds_error=False, fill_value = np.NaN)
	interpolated = np.array([x if not pd.isnull(x) else f(i) for i, x in enumerate(data)])	

	return interpolated, gaps

def mra(data, gaps, sharey=False):
		
	# perform EMD
	title = f"MeanIHR_EMD"
	imfs = emd.sift.sift(data).T
	fig_emd, ax_emd = components_plot(imfs, data, out, title, gaps, sharey=sharey)
	
	# perform EEMD
	title = f"MeanIHR_EEMD"
	imfs = emd.sift.ensemble_sift(data, nensembles=4, nprocesses=3, ensemble_noise=1).T
	fig_eemd, ax_eemd = components_plot(imfs, data, out, title, gaps, sharey=sharey)
	
	# perform VMD
	title = f"MeanIHR_VMD"
	alpha = 2000  # moderate bandwidth constraint  
	tau = 0.      # noise-tolerance (no strict fidelity enforcement)  
	K = 6         # n of modes to be recovered  
	DC = 0        # no DC part imposed  
	init = 1      # initialize omegas uniformly 
	tol = 1e-7
	u, u_hat, omega = vmdpy.VMD(data, alpha, tau, K, DC, init, tol)
	u = np.flip(u)
	fig_vmd, ax_vmd = components_plot(u, data, out, title, gaps, sharey=sharey)

	# perform WT MRA
	title = f"MeanIHR_WT"
	#wavelet = pywt.Wavelet("sym4") # sym4 is default wavelet used by MATLAB modwt()
	wavelet = pywt.Wavelet("db4") 
	data_wt = data if len(data) % 2 == 0 else data[:-1]
	output = pywt.mra(data_wt, wavelet, transform="dwt")
	output = np.flip(output, axis=0)
	fig_dwt, ax_dwt = components_plot(output, data_wt, out, title, gaps, sharey=sharey)	


def wavelet_transform(data, gaps):

	timevec_h = (np.arange(0, len(data))*5)/24

	fs = 1/300
	w = 10.0 # default Omega0 param for morlet2 (5.0). Seems to control frequency of complex sine part?

	data = data - np.mean(data) # remove DC offset, otherwise a lot of power at very low frequencies
	
	#freqs = np.array([1/(24*60*60), 1/(18*60*60),1/(14*60*60),1/(12*60*60),1/(10*60*60),1/(8*60*60),1/(6*60*60),1/(4*60*60),1/(2*60*60),1/(1*60*60)])	
	#freqs = np.linspace(freqs[0], freqs[-1], 10000)
	#freqs = np.linspace(0, freqs[-1], 10000)
	#freqs = 
	#freqs = np.linspace(0, fs/2, 10000)

	#periods = 1/freqs
	#periods = np.array([days*24*60*60 for days in range(13, 1, -1)] + [hours*60*60 for hours in range(47, 0, -1)])
	#periods = np.array([hours*60*60 for hours in range(288, 0, -1)])
	periods = np.array([hours*60*60 for hours in range(73, 0, -1)])
	freqs = 1/periods
	
	widths = w * fs / (2 * freqs * np.pi)

	cwtmatr = signal.cwt(data, signal.morlet2, widths, w=w)
	cwtmatr_yflip = cwtmatr	
	#cwtmatr_yflip = np.flipud(cwtmatr) # TODO can keep this, just need to flip y axis labels
	cwtmatr_yflip = np.abs(cwtmatr_yflip) # this isn't in tutorial (as tehy are using ricker wavelet, not complex morlet, i presume), imshow won't work without

	fig, axs = plt.subplots(3,1,sharex=True, height_ratios=[2, 1, 7])

	interpolation = "antialiased"#"none"
	#axs[2].imshow(cwtmatr_yflip, vmax = abs(cwtmatr).max(), vmin = -abs(cwtmatr).max(), aspect="auto", interpolation=interpolation)
	#axs[2].imshow(cwtmatr_yflip, vmax = abs(cwtmatr).max(), vmin = 0, aspect="auto", interpolation=interpolation)
	pos = axs[2].imshow(cwtmatr_yflip, aspect="auto", interpolation=interpolation)
	#cbar = fig.colorbar(pos, ax=axs[2])

	"""	
	yticks = ax.get_yticks()
	yticks = yticks[yticks>=0]
	yticks[yticks > 0] -= 1
	axs[1].set_yticks(ticks = yticks, labels=periods[np.int32(yticks)])
	"""
	axs[2].set_yticks(ticks=range(0, len(cwtmatr)), labels=np.floor(periods/60/60))

	axs[2].set_ylabel("Period (h)")

	axs[2].set_title("Wavelet Transform Time-Frequency", loc="right")


	axs[0].plot(range(0, len(data)), data, color="black", alpha=0.5)
	axs[0].plot(range(0, len(data)), cut_gaps(data, gaps), color="black")
	axs[0].set_title("Original Data", loc="right")


	lowcut=1/(33*60*60)
	highcut=1/(21*60*60) 
	fs=fs 
	order=4
	test_butter_bandpass(lowcut, highcut, fs, order)
	axs[1].plot(butter_bandpass_filter(data, lowcut=lowcut, highcut=highcut, fs=fs, order=order), c="r", label="Circadian Band (21h-33h)")

	axs[1].set_title("Bandpass-filtered Circadian Rhythm (21h-33h)", loc="right")

	"""
	original_xticks = axs[0].get_xticks() #sharex=True, so both axs have same xticks
	original_xticks = original_xticks[original_xticks >= 0]
	original_xticks = np.int32(original_xticks)
	original_xticks = [tick for tick in original_xticks if tick < len(data)]
	axs[0].set_xticks(ticks=original_xticks, labels=timevec_h[original_xticks])
	axs[1].set_xticks(ticks=original_xticks, labels=timevec_h[original_xticks])
	"""

	fig.show()	

if __name__ == "__main__":
	subject = "909"
	root = f"/home/bcsm/University/stage-4/MSc_Project/UCLH/{subject}"
	out = f"out/{subject}"

	# collate data and resolve overlaps
	run_speedyf(root, out); 
	
	# produce hrv metric dataframes, and save to out/
	calculate_hrv_metrics(root, out) # will not re-calculate if dataframes already present in out

	# load the hrv metric dataframes we just produced
	time_dom_df, freq_dom_df, modification_report_df = load_hrv_dataframes(out)

	# what metric are we interested in?
	data = time_dom_df["hr_mean"]
	data = np.array(data)

	# interpolate gaps (runs of NaN) so we can use with signal decomposition. save gap positions for visualisation
	interpolated, gaps = interpolate_gaps(data) 

	# perform multi-resolution analysis (signal decomposition)	
	mra(interpolated, gaps, sharey=False)

	# perform time-frequency analysis using wavelet_transform
	wavelet_transform(interpolated, gaps)	
