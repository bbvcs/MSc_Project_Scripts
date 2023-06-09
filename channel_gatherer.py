import pyedflib
import os


if __name__ == "__main__":
	root = "../UCLH"

	edf_files = []
	unreadable_files = []

	# collect edf files under root directory
	for path, subdirs, files in os.walk(root):
		for file in files:
			if file.split(".")[-1].lower() in ["edf", "edf+", "bdf", "bdf+"]:
				edf_files.append(os.path.join(path, file))

	# read headers and store in dict of filepath -> header
	channels = set()
	i = 0
	for file in edf_files:
		print(f"{i}/{len(edf_files)}")
		try:
			header = pyedflib.highlevel.read_edf_header(file, read_annotations=False)
			channels = channels.union(set(header["channels"]))
		except OSError as e:
			print(f"Could not read {file} ({e}).")

		i += 1

	print([ch for ch in channels if ("ecg" in ch.lower()) or ("ekg" in ch.lower())])

