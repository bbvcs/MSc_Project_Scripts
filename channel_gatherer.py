import pyedflib
import os
import constants

if __name__ == "__main__":
	root = "/data/UCLH/icEEG"	

	edf_files = []
	unreadable_files = []

	# collect edf files under root directory
	for path, subdirs, files in os.walk(root):
		for file in files:
			if file.split(".")[-1].lower() in ["edf", "edf+", "bdf", "bdf+"]:
				edf_files.append(os.path.join(path, file))

	"""
	# read headers and store in dict of filepath -> header
	ecg_channels = set()
	i = 0
	for file in edf_files:
		print(f"{i}/{len(edf_files)}")
		try:
			header = pyedflib.highlevel.read_edf_header(file, read_annotations=False)
			channels = header["channels"]
			channels = [ch for ch in channels if ("ecg" in ch.lower()) or ("ekg" in ch.lower())]
			ecg_channels = ecg_channels.union(set(channels))
		except OSError as e:
			print(f"Could not read {file} ({e}).")

		i += 1
	print(ecg_channels)
	"""

	channel_to_files = {}

	i = 0
	for file in edf_files:
		print(f"{i}/{len(edf_files)}")
		try:
			header = pyedflib.highlevel.read_edf_header(file, read_annotations=False)
			channels = header["channels"]
			ecg_channels = [ch for ch in channels if ("ecg" in ch.lower()) or ("ekg" in ch.lower())]
			for ch in ecg_channels:
				if ch in channel_to_files.keys():
					channel_to_files[ch] = channel_to_files[ch] + [file]
				else:
					channel_to_files[ch] = [file]

		except OSError as e:
			print(f"Could not read {file} ({e}).")

		i += 1

	print(channel_to_files)

