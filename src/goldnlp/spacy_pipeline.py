import time
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import islice

import spacy
from spacy import Language
import en_funding_round_model
import en_ceo_model
import nltk
nltk.download('stopwords')
nltk.download('names')
from nltk import FreqDist

from goldnlp.constants import (
    ACQUISITION_PATTERNS,
    BOARD_PATTERNS,
    CEO_PATTERNS,
    CFO_PATTERNS,
    CTO_PATTERNS,
    DATELINE_PATTERNS,
    EDUCATED_PATTERNS,
    FOUNDER_OF_PATTERNS,
    FUNDING_ROUND_PATTERNS,
    GENERIC_EVENT_PATTERNS,
    LANGUAGE_CHECK_WINDOW,
    LOCATED_IN_PATTERNS,
    TICKER_PATTERNS,
)
from goldnlp.events import (
    fuzzy_parse,
    preprocess_dates,
    pub_date_native,
    title_as_doc,
    to_timeline,
)
from goldnlp.nlpdocument import NLPDocument
from goldnlp.quality_check import buzzword_score, quality_check
from goldnlp.spacy_utils import (
    dependency_visualization,
    entity_visualization,
    make_matches_greedy,
    matcher_wrapper,
    relabel_ent,
)
from goldnlp.utils import check_language


def load_with_timing(model):
    print(f"loading spacy model `{model}`...")
    t0 = time.time()
    model = spacy.load(model)
    print(f"...model loaded in {time.time() - t0:.1f}s")
    return model


spacy_nlp, funding_round_nlp, ceo_nlp = (
    None,
    None,
    None,
)  
PRELOAD_MODELS = {
    "spacy_nlp": "en_core_web_md",
    "funding_round_nlp": "en_funding_round_model",
    "ceo_nlp": "en_ceo_model",
}
for varname, model in PRELOAD_MODELS.items():
    globals()[varname] = load_with_timing(model)


MODELS = {
    "default": spacy_nlp,
    "funding_round": funding_round_nlp,
    "ceo": ceo_nlp,
}


def create_document(
    url=None,
    pub_date=None,
    title="",
    text="",
    slate=None,
    token_metadata=None,
    source_task=None,
    nlpdocument=None,
    model="default",
    search_hosts=None,
    language_check=True,
    language_check_kwargs=None,
    allow_blank=False,
    **kwargs,
):
    """[summary]

    Args:
        url ([type], optional): [description]. Defaults to None.
        pub_date ([type], optional): [description]. Defaults to None.
        title (str, optional): [description]. Defaults to "".
        text (str, optional): [description]. Defaults to "".
        slate ([type], optional): [description]. Defaults to None.
        token_metadata ([type], optional): [description]. Defaults to None.
        source_task ([type], optional): [description]. Defaults to None.
        nlpdocument ([type], optional): [description]. Defaults to None.
        model (str, optional): [description]. Defaults to "default".
        search_hosts ([type], optional): [description]. Defaults to None.
        language_check (bool, optional): [description]. Defaults to True.
        language_check_kwargs ([type], optional): [description]. Defaults to None.
        allow_blank (bool, optional): [description]. Defaults to False.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if (text and slate) or ((not text and not slate) and not allow_blank):
        raise ValueError("must pass either slate or text")
    if slate is not None:
        nlpdocument = NLPDocument(slate, auto_heuristic=False, auto_word_features=False)
        text = nlpdocument.text
    language_check_kwargs = language_check_kwargs or {}
    if language_check and not check_language(text, **language_check_kwargs):
        print(f"doesn't look like English: `{text[:LANGUAGE_CHECK_WINDOW]}`")
    doc = MODELS[model](text)
    doc._.language_check = language_check
    doc._.language_check_kwargs = language_check_kwargs
    doc._.url = url
    doc._.pub_date = pub_date
    doc._.title = title
    doc._.source_task = source_task
    doc._.slate = slate
    doc._.nlpdocument = nlpdocument
    doc._.token_metadata = token_metadata
    if search_hosts:
        doc._.search_hosts = search_hosts
    return doc


def create_documents(data, language_check=True, language_check_kwargs=None, **kwargs):
    """[summary]

    Args:
        data ([type]): [description]
        language_check (bool, optional): [description]. Defaults to True.
        language_check_kwargs ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    # use bulk pipelines for efficient spacy multiprocessing
    language_check_kwargs = language_check_kwargs or {}
    if language_check and any(
        d["text"] for d in data if not check_language(d["text"], **language_check_kwargs)
    ):
        print(
            "received at least one text that doesn't look like english, "
            "please filter your data through check_language ahead of time."
        )
    docs = list(spacy_nlp.pipe([d["text"] for d in data]))
    for data, doc in zip(data, docs):
        for k, v in data:
            if doc.has_extension(k):
                doc._.set(k, v)
    return docs


def create_subdocument(doc, text, inherit_dateline_date=True):
    """[summary]

    Args:
        doc ([type]): [description]
        text ([type]): [description]
        inherit_dateline_date (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    if not isinstance(text, str):
        start = text[0].idx
        text = str(text)
        end = start + len(text)
    else:
        try:
            start = str(doc).index(text)
            end = start + len(text)
        except ValueError:
            start = 0
            end = -1
    realigned_token_metadata = []
    for meta in doc._._token_metadata_raw:
        if (
            meta["start_char_index"] >= start
            and meta["start_char_index"] + len(meta["text"]) <= end
        ):
            realigned = meta.copy()
            realigned["start_char_index"] -= start
            realigned_token_metadata.append(realigned)
    subdoc = create_document(
        url=doc._.url,
        pub_date=doc._.pub_date,
        title="",
        text=text,
        source_task=doc._.source_task,
        slate=doc._.slate,
        nlpdocument=doc._.nlpdocument,
        token_metadata=realigned_token_metadata,
        search_hosts=doc._.search_hosts,
        allow_blank=True,
        language_check=doc._.language_check,
        language_check_kwargs=doc._.language_check_kwargs,
    )
    if inherit_dateline_date:
        subdoc._.dateline_date = doc._.dateline_date
    return subdoc


@Language.component("sentence_newline_boundaries")
def sentence_newline_boundaries(doc):
    """[summary]

    Args:
        doc ([type]): [description]

    Returns:
        [type]: [description]
    """
    if len(doc) == 0:
        return doc
    graf_index = 0
    doc[0]._.is_graf_start = True
    for token in doc[:-1]:
        token._.graf = graf_index
        if token.text == "\n":
            doc[token.i + 1].is_sent_start = True
            doc[token.i + 1]._.is_graf_start = True
            graf_index += 1
    doc[-1]._.graf = graf_index
    return doc


@Language.component("set_sent_indices")
def set_sent_indices(doc):
    """[summary]

    Args:
        doc ([type]): [description]

    Returns:
        [type]: [description]
    """
    for i, s in enumerate(doc.sents):
        for t in s:
            t._.sent_idx = i
    return doc


def get_sent(doc, idx):
    try:
        sent = next(s for i, s in enumerate(doc.sents) if i == idx)
    except StopIteration:
        raise LookupError("sentence {idx} out of range")
    else:
        return sent


@Language.component("set_dateline")
def set_dateline(doc):
    """[summary]

    Args:
        doc ([type]): [description]

    Returns:
        [type]: [description]
    """
    if doc._.dateline is not None:
        return doc
    grafs = islice(doc._.iter_paragraphs(), 2)
    for g in grafs:
        subdoc = g.as_doc()
        if not len(subdoc):
            continue
        subdoc[0]._.is_graf_start = True
        offset = g.start
        dateline = make_matches_greedy(tuple(dateline_matcher(subdoc)))
        if dateline and len(dateline) == 1:
            start, end = dateline[0][1], dateline[0][2]
            doc._.dateline = doc[start + offset : end + offset]
            for token in doc._.dateline:
                token._.is_dateline = True
            break
    if doc._.dateline is None:
        doc._.dateline = False
    return doc


@Language.component("set_dateline_date")
def set_dateline_date(doc):
    """[summary]

    Args:
        doc ([type]): [description]

    Returns:
        [type]: [description]
    """
    if doc._.dateline_date is not None:
        return doc

    dateline_date = [
        fuzzy_parse(e, settings={"RELATIVE_BASE": doc._.pub_date_native})
        for e in doc._.preprocess_dates(token_filter=("is_dateline", True))
    ]
    doc._.dateline_date = next(filter(None, dateline_date), False)
    return doc


@lru_cache()
def get_paragraph(doc, index):
    """[summary]

    Args:
        doc ([type]): [description]
        index ([type]): [description]

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    tokens = [t for t in doc if t._.graf == index]
    if not tokens:
        raise ValueError(f"no paragraph {index} found in doc")
    start, end = tokens[0].i, tokens[-1].i
    return spacy.tokens.span.Span(doc, start, end + 1)


def iter_paragraphs(doc):
    """[summary]

    Args:
        doc ([type]): [description]

    Yields:
        [type]: [description]
    """
    i = 0
    while True:
        try:
            yield doc._.get_paragraph(i)
            i += 1
        except ValueError:
            break


def set_token_metadata(doc, metadata):
    """[summary]

    Args:
        doc ([type]): [description]
        metadata ([type]): [description]
    """
    """
    expects a list of dictionaries representing tokens with keys `text, start_char_index, metadata`
    """
    if not metadata:
        return
    doc._._token_metadata_raw = metadata
    for d in metadata:
        start_char = d["start_char_index"]
        end_char = start_char + len(d["text"]) - 1
        start_token = doc._.char_index_to_token.get(start_char)
        end_token = doc._.char_index_to_token.get(end_char)
        if not (start_token and end_token):
            continue
        for token in range(start_token.i, end_token.i + 1):
            doc[token]._.metadata = d["metadata"]
            doc._._token_metadata[token] = d["metadata"]


def get_token_metadata(doc):
    return doc._._token_metadata


def get_meta_chunks(doc):
    if not doc._.token_metadata:
        return []
    last_key = None
    last_meta = None
    current_chunk = []
    chunks = []
    for k in sorted(list(doc._.token_metadata.keys())):
        if (last_key is None and last_meta is None) or (
            k == last_key + 1 and doc._.token_metadata[k] == last_meta
        ):
            current_chunk.append(k)
        else:
            start = min(current_chunk)
            end = max(current_chunk)
            chunks.append((start, end, last_meta))
            current_chunk = [k]
        last_key = k
        last_meta = doc._.token_metadata[k]
    # don't forget to add the last chunk at the end...
    if last_meta and current_chunk:
        chunks.append((min(current_chunk), max(current_chunk), last_meta))
    return [(doc[start : end + 1], meta) for start, end, meta in chunks]


@Language.component("map_tokens")
def map_tokens(doc):
    for t in doc:
        start = t.idx
        end = t.idx + len(str(t))
        for i in range(start, end):
            doc._.char_index_to_token[i] = t
    return doc


def ent_counter(doc):
    if not doc._._ent_counter:
        doc._._ent_counter = Counter([f"{(e.text.strip(), e.label_)}" for e in doc.ents])
    return doc._._ent_counter


ROUND_LABEL = spacy_nlp.vocab.strings["ROUND"]
TICKER_LABEL = spacy_nlp.vocab.strings["TICKER"]


def round_annotation(matcher, doc, i, matches):
    match = matches[i]
    if match not in make_matches_greedy(tuple(matches)):
        return
    _, start, end = match
    # FIXME
    # some data we bring in doesn't contain a specific funding round
    # designation, but does contain a MONEY entity near a funding BASE
    # (see constants.py). in order to annotate, we need to preserve the
    # MONEY alongside some kind of round designation. so, this "reverses"
    # the MONEY_PHRASES patterns to get just the BASE part and any
    # modifiers (WILDCARD part) e.g "big beautiful funding".
    # this is an ugly hacky solution and should get revisited later.
    filtered_start = next(
        (t for t in doc[start : end + 1] if t.ent_type_ != "MONEY" and t.pos_ != "ADP"),
        None,
    )
    if not filtered_start:
        return
    start = filtered_start.i
    entity = ROUND_LABEL, start, end
    relabel_ent(doc, entity)


def ceo_annotation(matcher, doc, i, matches):
    match = matches[i]
    if match not in make_matches_greedy(tuple(matches)):
        return
    _, start, end = match
    if doc[start]._.sent_idx == doc[end - 1]._.sent_idx:
        doc._._ceo_annotations.append((start, end))


def cto_annotation(matcher, doc, i, matches):
    match = matches[i]
    if match not in make_matches_greedy(tuple(matches)):
        return
    _, start, end = match
    if doc[start]._.sent_idx == doc[end - 1]._.sent_idx:
        doc._._cto_annotations.append((start, end))


def cfo_annotation(matcher, doc, i, matches):
    match = matches[i]
    if match not in make_matches_greedy(tuple(matches)):
        return
    _, start, end = match
    if doc[start]._.sent_idx == doc[end - 1]._.sent_idx:
        doc._._cfo_annotations.append((start, end))


def board_annotation(matcher, doc, i, matches):
    match = matches[i]
    if match not in make_matches_greedy(tuple(matches)):
        return
    _, start, end = match
    if doc[start]._.sent_idx == doc[end - 1]._.sent_idx:
        doc._._board_annotations.append((start, end))


def educated_annotation(matcher, doc, i, matches):
    match = matches[i]
    if match not in make_matches_greedy(tuple(matches)):
        return
    _, start, end = match
    if doc[start]._.sent_idx == doc[end - 1]._.sent_idx:
        doc._._educated_annotations.append((start, end))


def founded_by_annotation(matcher, doc, i, matches):
    match = matches[i]
    if match not in make_matches_greedy(tuple(matches)):
        return
    _, start, end = match
    if doc[start]._.sent_idx == doc[end - 1]._.sent_idx:
        doc._._founded_by_annotations.append((start, end))


def located_in_annotation(matcher, doc, i, matches):
    match = matches[i]
    if match not in make_matches_greedy(tuple(matches)):
        return
    _, start, end = match
    if doc[start]._.sent_idx == doc[end - 1]._.sent_idx:
        doc._._located_in_annotations.append((start, end))


def ticker_annotation(matcher, doc, i, matches):
    match = matches[i]
    if match not in make_matches_greedy(tuple(matches)):
        return
    _, start, end = match
    start, end = start + 1, end - 1  # take off the parens
    entity = TICKER_LABEL, start, end
    relabel_ent(doc, entity)


@lru_cache()
def noun_chunk_fd(doc):
    items, reverse = [], defaultdict(list)
    for chunk in doc.noun_chunks:
        item = tuple(t.lemma_.lower() for t in chunk if not t.is_stop and not t.is_punct)
        if len(item) == 0:
            continue
        reverse[item].append(chunk)
        items.append(item)
    doc._._reverse_noun_chunk_fd = reverse
    return FreqDist(items)


def reverse_noun_chunk_fd(doc):
    if not doc._._reverse_noun_chunk_fd:
        # initialize it
        doc._.noun_chunk_fd
    return doc._._reverse_noun_chunk_fd




###########################################
#     pipeline setup below this point     #
###########################################
#               MATCHING                  #
###########################################
acquisition_event_matcher = spacy.matcher.Matcher(spacy_nlp.vocab)
dateline_matcher = spacy.matcher.Matcher(spacy_nlp.vocab)
generic_event_matcher = spacy.matcher.Matcher(spacy_nlp.vocab)
funding_round_matcher = spacy.matcher.Matcher(spacy_nlp.vocab)
ticker_matcher = spacy.matcher.Matcher(spacy_nlp.vocab)
ceo_matcher = spacy.matcher.Matcher(spacy_nlp.vocab)
cto_matcher = spacy.matcher.Matcher(spacy_nlp.vocab)
cfo_matcher = spacy.matcher.Matcher(spacy_nlp.vocab)
board_matcher = spacy.matcher.Matcher(spacy_nlp.vocab)
educated_matcher = spacy.matcher.Matcher(spacy_nlp.vocab)
founded_by_matcher = spacy.matcher.Matcher(spacy_nlp.vocab)
located_in_matcher = spacy.matcher.Matcher(spacy_nlp.vocab)

acquisition_event_matcher.add("acquisition", ACQUISITION_PATTERNS)
funding_round_matcher.add("funding_round", FUNDING_ROUND_PATTERNS, on_match=round_annotation)
ticker_matcher.add("ticker", TICKER_PATTERNS, on_match=ticker_annotation)
ceo_matcher.add("ceo", CEO_PATTERNS, on_match=ceo_annotation)
cto_matcher.add("cto", CTO_PATTERNS, on_match=cto_annotation)
cfo_matcher.add("cfo", CFO_PATTERNS, on_match=cfo_annotation)
board_matcher.add("board", BOARD_PATTERNS, on_match=board_annotation)
educated_matcher.add("educated", EDUCATED_PATTERNS, on_match=educated_annotation)
founded_by_matcher.add("founded_by", FOUNDER_OF_PATTERNS, on_match=founded_by_annotation)
located_in_matcher.add("located_in", LOCATED_IN_PATTERNS, on_match=located_in_annotation)
dateline_matcher.add("dateline", DATELINE_PATTERNS)
generic_event_matcher.add("event", GENERIC_EVENT_PATTERNS)

# Create Language components for matchers
Language.component("funding_round_matcher")(matcher_wrapper(funding_round_matcher, "funding_round"))
Language.component("ticker_matcher")(matcher_wrapper(ticker_matcher, "ticker"))
Language.component("ceo_matcher")(matcher_wrapper(ceo_matcher, "ceo"))
Language.component("cto_matcher")(matcher_wrapper(cto_matcher, "cto"))
Language.component("cfo_matcher")(matcher_wrapper(cfo_matcher, "cfo"))
Language.component("board_matcher")(matcher_wrapper(board_matcher, "board"))
Language.component("educated_matcher")(matcher_wrapper(educated_matcher, "educated"))
Language.component("founded_by_matcher")(matcher_wrapper(founded_by_matcher, "founded_by"))
Language.component("located_in_matcher")(matcher_wrapper(located_in_matcher, "located_in"))

###########################################
#             DEFAULT MODEL               #
###########################################
# split sentences by newlines before parse
spacy_nlp.add_pipe("sentence_newline_boundaries", before="parser")
# populate some mappings and indices to make life easier
spacy_nlp.add_pipe("set_sent_indices", after="parser")
spacy_nlp.add_pipe("map_tokens")
# manage datelines
spacy_nlp.add_pipe("set_dateline")
spacy_nlp.add_pipe("set_dateline_date")
# label funding rounds based on matcher rules
spacy_nlp.add_pipe("funding_round_matcher")
spacy_nlp.add_pipe("ticker_matcher")
spacy_nlp.add_pipe("ceo_matcher")
spacy_nlp.add_pipe("cto_matcher")
spacy_nlp.add_pipe("cfo_matcher")
spacy_nlp.add_pipe("board_matcher")
spacy_nlp.add_pipe("educated_matcher")
spacy_nlp.add_pipe("founded_by_matcher")
spacy_nlp.add_pipe("located_in_matcher")

###########################################
#          FUNDING ROUND MODEL            #
###########################################
# funding_round_nlp.add_pipe(sentence_newline_boundaries, before="ner")
# funding_round_nlp.add_pipe(map_tokens)
###########################################
#               CEO MODEL                 #
###########################################
# ceo_nlp.add_pipe(sentence_newline_boundaries, before="ner")
# ceo_nlp.add_pipe(map_tokens)
###########################################
#            DOC EXTENSIONS               #
###########################################
spacy.tokens.doc.Doc.set_extension("nlpdocument", default=None)
spacy.tokens.doc.Doc.set_extension("slate", default=None)
spacy.tokens.doc.Doc.set_extension("preprocess_dates", method=preprocess_dates)
spacy.tokens.doc.Doc.set_extension("ent_viz", method=entity_visualization)
spacy.tokens.doc.Doc.set_extension("dep_viz", method=dependency_visualization)
spacy.tokens.doc.Doc.set_extension("url", default="")
spacy.tokens.doc.Doc.set_extension("title", default="")
spacy.tokens.doc.Doc.set_extension("pub_date", default={})
spacy.tokens.doc.Doc.set_extension("source_task", default=None)
spacy.tokens.doc.Doc.set_extension("pub_date_native", getter=pub_date_native)
spacy.tokens.doc.Doc.set_extension("dateline", default=None)
spacy.tokens.doc.Doc.set_extension("to_timeline", method=to_timeline)
spacy.tokens.doc.Doc.set_extension("title_as_doc", getter=title_as_doc)
spacy.tokens.doc.Doc.set_extension("dateline_date", default=None)
spacy.tokens.doc.Doc.set_extension("create_subdocument", method=create_subdocument)
spacy.tokens.doc.Doc.set_extension("get_paragraph", method=get_paragraph)
spacy.tokens.doc.Doc.set_extension("iter_paragraphs", method=iter_paragraphs)
spacy.tokens.doc.Doc.set_extension("quality_check", method=quality_check)
spacy.tokens.doc.Doc.set_extension("buzzword_score", method=buzzword_score)
spacy.tokens.doc.Doc.set_extension("char_index_to_token", default={})
spacy.tokens.doc.Doc.set_extension("_token_metadata", default={})
spacy.tokens.doc.Doc.set_extension("_token_metadata_raw", default={})
spacy.tokens.doc.Doc.set_extension(
    "token_metadata", setter=set_token_metadata, getter=get_token_metadata
)
spacy.tokens.doc.Doc.set_extension("chunked_metadata", getter=get_meta_chunks)
spacy.tokens.doc.Doc.set_extension("_ent_counter", default=None)
spacy.tokens.doc.Doc.set_extension("ent_counter", getter=ent_counter)
spacy.tokens.doc.Doc.set_extension("chunked_metadata", getter=get_meta_chunks, force=True)
spacy.tokens.doc.Doc.set_extension("_reverse_noun_chunk_fd", default=None, force=True)
spacy.tokens.doc.Doc.set_extension("noun_chunk_fd", getter=noun_chunk_fd, force=True)
spacy.tokens.doc.Doc.set_extension(
    "reverse_noun_chunk_fd", getter=reverse_noun_chunk_fd, force=True
)
spacy.tokens.doc.Doc.set_extension("get_sent", method=get_sent)
spacy.tokens.doc.Doc.set_extension("secondary_model_ents", default=[])
spacy.tokens.doc.Doc.set_extension("language_check", default=True)
spacy.tokens.doc.Doc.set_extension("language_check_kwargs", default={})
spacy.tokens.doc.Doc.set_extension("_ceo_annotations", default=[])
spacy.tokens.doc.Doc.set_extension("_cto_annotations", default=[])
spacy.tokens.doc.Doc.set_extension("_cfo_annotations", default=[])
spacy.tokens.doc.Doc.set_extension("_board_annotations", default=[])
spacy.tokens.doc.Doc.set_extension("_educated_annotations", default=[])
spacy.tokens.doc.Doc.set_extension("_founded_by_annotations", default=[])
spacy.tokens.doc.Doc.set_extension("_located_in_annotations", default=[])
###########################################
#            TOKEN EXTENSIONS             #
###########################################
spacy.tokens.Token.set_extension("is_sent_start", getter=lambda t: t.is_sent_start or False)
spacy.tokens.Token.set_extension("is_graf_start", default=False)
spacy.tokens.Token.set_extension("graf", default=None)
spacy.tokens.Token.set_extension("sent_idx", default=None)
spacy.tokens.Token.set_extension("is_dateline", default=False)
spacy.tokens.Token.set_extension("metadata", default={})
