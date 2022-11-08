import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from goldnlp.spacy_pipeline import create_document
from data_models import NERRequest, NERResponse, SimpleEntity

app = FastAPI()


############
### CORS ###
############

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#####################
### Initialization###
#####################

# Initialize Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level="INFO",
    format="%(process)d-%(levelname)s-%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Event Handling
@app.on_event("startup")
async def initialize_nlp_task_modules():
    doc = create_document(text="Hello world", model="default")


###################
### GoldNLP API ###
###################
@app.get("/")
async def root():
    return {"message": "Welcome to GoldNLP"}


@app.post("/api/named_entity_recognition", response_model=NERResponse)
async def named_entity_recognition(ner_request: NERRequest):
    text = ner_request.text
    doc = create_document(text=text, model="default")
    entities = [
        SimpleEntity(
            text=spn.text, start_char=spn.start_char, end_char=spn.end_char, type=spn.label_
        )
        for spn in doc.ents
    ]

    payload = NERResponse(text=text, entities=entities)
    return payload
