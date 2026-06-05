from PIL import Image
import numpy as np

def extract_metrics(image_path):
	"""Compute simple per-image metrics for analysis/demo purposes."""
	img = Image.open(image_path).convert('L')
	arr = np.asarray(img, dtype=np.float32) / 255.0
	height, width = arr.shape
	mean_intensity = float(arr.mean())
	std_intensity = float(arr.std())
	min_intensity = float(arr.min())
	max_intensity = float(arr.max())
	return {
		'path': image_path,
		'width': width,
		'height': height,
		'mean': mean_intensity,
		'std': std_intensity,
		'min': min_intensity,
		'max': max_intensity,
	}

if __name__ == '__main__':
	import sys
	for p in sys.argv[1:]:
		m = extract_metrics(p)
		print(m)

