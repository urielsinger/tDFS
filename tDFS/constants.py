import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PROJECT_NAME = os.path.basename(PROJECT_ROOT)
REPO_DIR = os.path.dirname(PROJECT_ROOT)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
CACHE_PATH = os.path.join(PROJECT_ROOT, 'cache')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(CACHE_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)