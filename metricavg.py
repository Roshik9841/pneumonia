import os
from collections import defaultdict
from metrics import extract_metrics

def analyze_dataset(directories):
	"""Walk given directories and compute summary metrics."""
	metrics_summary = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf'), 'sum': 0.0, 'count': 0})
	num_images = 0
	for directory in directories:
		for root, _, files in os.walk(directory):
			for fname in files:
				if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
					path = os.path.join(root, fname)
					m = extract_metrics(path)
					num_images += 1
					for key in ['mean', 'std', 'min', 'max']:
						value = m[key]
						ms = metrics_summary[key]
						ms['min'] = min(ms['min'], value)
						ms['max'] = max(ms['max'], value)
						ms['sum'] += value
						ms['count'] += 1
	return metrics_summary, num_images

def print_results(metrics_summary, num_images):
	if num_images == 0:
		print('No images found.')
		return
	print(f"Analyzed {num_images} images")
	print(f"{'Metric':<20}{'Min':>10}{'Max':>10}{'Avg':>10}")
	for metric, values in metrics_summary.items():
		avg = values['sum'] / max(values['count'], 1)
		print(f"{metric.replace('_', ' ').title():<20} {values['min']:>10.2f} {values['max']:>10.2f} {avg:>10.2f}")

# Dataset directories
train_dir = 'C:/xampp/htdocs/Pneumonia/archive/train'
val_dir = 'C:/xampp/htdocs/Pneumonia/archive/val'
directories = [train_dir, val_dir]

# Analyze dataset and print results
metrics_summary, num_images = analyze_dataset(directories)
print_results(metrics_summary, num_images)

