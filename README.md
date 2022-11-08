# GoldNLP: Golden's NLP/NLU python package and framework for the the Golden Knowledge Graph Protocol.

<a href="https://pypi.org/project/goldnlp" target="_blank">
    <img src="https://img.shields.io/pypi/v/goldnlp?logo=pypi">
</a>
<a href="https://github.com/goldenrecursion/goldnlp/tree/master/.github/workflows" target="_blank">
    <img src="https://img.shields.io/github/workflow/status/goldenrecursion/goldnlp/Docker%20Compose%20CI?logo=github">
</a>

## Overview

GoldNLP is Golden's open-source NLP/NLU package and framework for running and creating AI/ML-based models and tools that can support ML practitioners and developers contributing to Golden's protocol community.

This repository will also include API services and end-to-end ML training/inference framworks for tools like search/disambiguation, named entity recognition, texts classification, relationship extraction, and more. GoldNLP is built with [Spacy](https://spacy.io/), [Transformers](https://github.com/huggingface/transformers), [FastNN](https://github.com/aychang95/fastnn), and [FastAPI](https://fastapi.tiangolo.com/).

Our models will also be hosted in in our [golden-models](https://github.com/goldenrecursion/golden-models) repository and [Hugging Face's Model Hub](https://huggingface.co/Golden-AI) for pulling and installation.

## Installation

GoldNLP requires Python 3.7+

```
pip install goldnlp
```

## Quickstart

First we'll need to load our models so run the following to install some of Golden's and Spacy models:

```
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.1.0/en_core_web_md-3.1.0-py3-none-any.whl
pip install https://github.com/goldenrecursion/golden-models/releases/download/en_funding_round_model-0.0.1/en_funding_round_model-0.0.1-py3-none-any.whl
pip install https://github.com/goldenrecursion/golden-models/releases/download/en_ceo_model-0.0.1/en_ceo_model-0.0.1-py3-none-any.whl
```

Now we can get started with running inference on some of Golden's custom trained named-entity recognition models.

```python
from goldnlp import create_document

text = "Tim Cook is the CEO of Apple."
doc = create_document(text=text, model="default")
entities = [(spn.text, spn.start_char, spn.end_char, spn.label_) for spn in doc.ents]
print(entities)
```

## Docker Setup

Run `docker-compose build` and then `docker-compose up`.

Go to `localhost:8888` to run our tutorials in jupyter lab.


## Contact

For all things related to `goldnlp` and development, please contact the maintainer Andrew Chang at andrew@golden.co or [@achang1618](https://twitter.com/achang1618) for any quesions or comments.

For all other support, please reach out to support@golden.co.

Follow [@golden](https://twitter.com/Golden) to keep up with additional news!

## License

This project is licensed under the terms of the Apache 2.0 license