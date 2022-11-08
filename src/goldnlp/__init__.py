import os

import nltk
from goldnlp.spacy_pipeline import create_document

# Add package nltk_data to nltk search path on init
ROOT_PATH = f"{os.path.abspath(os.path.dirname(__file__))}/"
nltk.data.path.append(f"{ROOT_PATH}nltk_data")
