import os
import re
from datetime import datetime
from pathlib import Path



def resolve_model_path(model_name_or_path):
    """
    Resolves a model path, checking for a direct path
    before the default models directory.
    """

    path_candidate = Path(model_name_or_path)
    if path_candidate.exists():
        return path_candidate
    else:
        return Path(f'{shared.args.model_dir}/{model_name_or_path}')
