# config.py
import os.path


# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

ray512 = {
    'num_classes': 3,
    'lr_steps': (60000, 65000, 70000),
    'max_iter': 75000,
    'feature_maps': [64, 32, 16, 8, 4, 2],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256],
    'min_sizes': [51, 102, 189, 276, 363, 450],
    'max_sizes': [102, 189, 276, 363, 450, 537],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'RAY',
}

ray = {
    'num_classes': 3,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 10,  # 120000
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'RAY',
}
