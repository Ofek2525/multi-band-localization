import os

# main folders
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This is your Project Root
CODE_DIR = os.path.join(ROOT_DIR, 'python_code')
RESOURCES_DIR = os.path.join(ROOT_DIR, 'resources')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

# subfolders
RAYTRACING_DIR = os.path.join(RESOURCES_DIR, 'raytracing')
ALLBSs_DIR = os.path.join(RESOURCES_DIR, 'all_BSs')
