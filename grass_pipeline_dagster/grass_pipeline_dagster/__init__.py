from .definitions import defs

import sys
import os
# grass_pipeline_dagster/__init__.py


# Ensure project root is in sys.path so Dagster can import cpp_processing
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
