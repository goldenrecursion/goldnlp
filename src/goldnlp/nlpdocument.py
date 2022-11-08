import json
import random
import re
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache, reduce

import nltk
import numpy as np
from nltk.collocations import (
    BigramCollocationFinder,
    QuadgramCollocationFinder,
    TrigramCollocationFinder,
)
from simhash import Simhash
from sklearn.metrics import mean_squared_error

from goldnlp.constants import NAMES, PENN_TAGS, STOPWORDS
from goldnlp.utils import adjust_type, jaccard


class NLPDocument(object):
    """
    preprocessor for Golden entity content
    """

    # DRY on filters for common tasks e.g. word frequency analysis
    freq_filters = ("punctuation", "lower", "stopword")

    def __init__(
        self,
        text,
        slug=None,
        name="",
        ev_id=None,
        binary_ne=True,
        auto_heuristics=True,
        auto_word_features=True,
        describe_tags=False,
        stem_mode="regex",
        **kwargs,
    ):
        self.slug = slug
        self.name = name
        self.ev_id = ev_id

        # define filter functions to make available to get_tokens
        # this is important because functions themselves arent hashable -> cant be used as args to the cached chain
        # therefore we need to be able to easily refer to the filter functions using strings, and this seems
        # cleaner to me than using locals() or getattr
        self.token_filters = {
            "lower": lambda tokens: [(i, t.lower()) for i, t in tokens],
            "punctuation": lambda tokens: list(
                filter(
                    lambda tup: tup[1],
                    [(i, "".join(re.findall(r"[\w\s\d]+", t))) for i, t in tokens],
                )
            ),
            "stopword": lambda tokens: list(
                filter(
                    lambda tup: tup[1],
                    [(i, t) for i, t in tokens if t.lower() not in STOPWORDS],
                )
            ),
            "stem": lambda tokens: list(
                filter(lambda tup: tup[1], [(i, self.stemmer.stem(t)) for i, t in tokens])
            ),
            "unique": lambda tokens: set([t for i, t in tokens]),
            "pos": lambda tokens: list(
                zip([i for i, t in tokens], nltk.pos_tag([t for i, t in tokens]))
            ),
        }

        # defer loading of stemmer (see NLPDocument.stemmer)
        self.stem_mode = stem_mode
        self._stemmer = None

        self.raw = text
        if isinstance(text, dict):
            # pull data out of slate nodes
            try:
                self.text, self.node_map = self._nodes_to_text()
            except KeyError:
                self.text = ""
                self.node_map = {}
        else:
            self.text = self.replace_overwrought_chars(self.raw)
            self.node_map = None

        # word-level features
        if auto_word_features:
            self.wordcount = len(self.get_tokens(filters=("punctuation",), idx=False))
            self.vocabulary = self.get_freqdist(
                filters=("punctuation", "lower", "stopword", "stem")
            )

        # complexity heuristics
        if auto_heuristics:
            # lexical diversity
            self.ttr_raw = self.get_ttr()
            self.ttr_d = self.get_d()

        # tagging/chunking
        self.tags = self.get_tokens(by_sentence=True, filters=("pos",), idx=False)
        self.nnp_chunks = [nltk.RegexpParser(r"NNP: {<NNP>+}").parse(s) for s in self.tags]
        self.ne_chunks = [nltk.ne_chunk(s, binary=binary_ne) for s in self.tags]
        self.proper_noun_phrases = self._get_chunk_content(self.nnp_chunks, ["NNP"])
        self.entities = self._get_chunk_content(
            self.ne_chunks, ["NE", "GPE", "PERSON", "ORGANIZATION"]
        )

        # include tag descriptions if requested
        if describe_tags:
            self.tag_descriptions = PENN_TAGS

    @classmethod
    def from_sibling(cls, instance):
        new = cls(**instance.__dict__)
        for attr in dir(instance):
            if not attr.startswith("__"):
                try:
                    setattr(new, attr, getattr(instance, attr))
                except AttributeError:
                    continue
        return new

    @property
    def stemmer(self):
        """
        defer loading of stemmer until needed
        """
        if self._stemmer is None:
            if self.stem_mode == "porter":
                from nltk.stem.porter import PorterStemmer

                self._stemmer = PorterStemmer()
            else:
                from nltk.stem.regexp import RegexpStemmer

                self._stemmer = RegexpStemmer("s$|ies$")
        return self._stemmer

    @stemmer.setter
    def set_stem_mode(self, mode):
        self._stemmer = None
        self.stem_mode = mode

    @lru_cache(maxsize=128)
    def get_tokens(self, text=None, by_sentence=False, filters=(), idx=True):
        """
        :param text: text to tokenize (default None, uses self.text)
        :param by_sentence: if True, group tokens by sentences
        :param filters: iterable of filter names (needs to be hashable! tuple recommended)
        :param idx: include index of characters for the token in the original text
        :return: list of idx, tokens tuples (or just tokens if idx=False) after all filters have been applied
                 (or list-of-lists, if by_sentence=True)
        """
        if text is None:
            text = self.text

        if by_sentence:
            tokens = []
            offset = 0
            for s in nltk.sent_tokenize(text):
                sent_tokens = nltk.word_tokenize(s)
                # include span info to find tokens in text
                sent_tokens = list(self.get_token_span(sent_tokens, text, offset))
                tokens += [self._cached_chain(sent_tokens, filters)]
                offset = sent_tokens.pop()[0][
                    1
                ]  # prevent re-matching previously seen tokens from other sents
            if idx:
                return tokens
            else:
                return [[t for _, t in s] for s in tokens]
        else:
            tokens = list(self.get_token_span(nltk.word_tokenize(text), text))
            tokens = self._cached_chain(tokens, filters)
            if idx:
                return tokens
            else:
                return [t for i, t in tokens]

    @staticmethod
    def replace_overwrought_chars(string):
        if not string:
            return string
        # slightly adapted copypasta from golden web repo
        trying_too_hard = {
            "”": '"',
            "''": '"',
            "“": '"',
            "‘": "'",
            "’": "'",
            "‐": "-",
            "‑": "-",
            "‒": "-",
            "–": "-",
            "—": "-",
            "―": "-",
            "⸺": "-",
            "⸻": "-",
            "﹘": "-",
            "﹣": "-",
            "－": "-",
            "\u200b": " ",
            "\u200c": " ",
            "\u200d": " ",
            "\u200e": " ",
            "\u200f": " ",
        }
        for fancy, simple in trying_too_hard.items():
            string = string.replace(fancy, simple)
        return string

    def get_token_span(self, tokens, text=None, offset=0):
        """
        :param tokens: list of tokens to get span indices for
        :param text: text to search for those tokens (if None, self.text)
        :param offset: initial text offset for search (needed to make get_tokens work if by_sentences==True)
        :return: generator of ((start, end), token) tuples
        """
        if text is None:
            text = self.text
        for token in tokens:
            if token in ["``", "''"]:
                token = '"'
            start = text[offset:].index(token) + offset
            end = start + len(token)
            offset = end
            yield ((start, end), token)

    def get_ngram_span(self, ngram_tuple, indexed_tokens=None):
        """
        :param ngram_tuple: the ngram to search for, as a tuple of tokens
        :param indexed_tokens: the indexed token list as produced by get_tokens with idx=True
        :return: generator for tuples of start/stop indices where the ngram can be found
        """
        if indexed_tokens is None:
            indexed_tokens = self.get_tokens()
        n = len(ngram_tuple)
        for i in range(len(indexed_tokens) - n + 1):
            _slice = indexed_tokens[i : i + n]
            if [t for idx, t in _slice] == [w for w in ngram_tuple]:
                start = _slice[0][0][0]
                end = _slice[n - 1][0][1]
                yield (start, end)

    @lru_cache(maxsize=128)
    def get_collocations(self, filters=(), n=2):
        """
        :param filters: filters to apply before finding collocations (through get_tokens)
        :param n: n-gram size for collocations (currently support 2-, 3-, 4-grams)
        :return: NLTK CollocationFinder for n-gram size
        """
        if n == 2:
            return BigramCollocationFinder.from_words(self.get_tokens(filters=filters, idx=False))
        if n == 3:
            return TrigramCollocationFinder.from_words(self.get_tokens(filters=filters, idx=False))
        if n == 4:
            return QuadgramCollocationFinder.from_words(self.get_tokens(filters=filters, idx=False))
        else:
            raise ValueError("n must be between 2 and 4 (inclusive)")

    @lru_cache(maxsize=128)
    def get_freqdist(self, filters=()):
        """
        :param filters: filters to apply before making freqdist (through get_tokens)
        :return: nltk.FreqDist
        """
        return nltk.FreqDist(self.get_tokens(filters=filters, idx=False))

    def get_ttr(self):
        """
        :return: uncorrected type-token ratio of lower-cased word stems
        """
        ttr_input = self.get_tokens(filters=("punctuation", "lower", "stem"), idx=False)
        if len(ttr_input) > 0:
            return len(set(ttr_input)) / len(ttr_input)
        else:
            return 0

    def get_d(self, iterations=5, n_range=range(35, 51), d_range=range(1, 201), n_samples=100):
        """
        A document-length-corrected measure of lexical diversity, D, adapted from
        Richards & Malvern, 2000, http://www.leeds.ac.uk/educol/documents/00001541.htm
        :return: estimated D score (or None if not enough tokens)
        """
        d_curve = {
            d: [(d / n) * (((1 + (2 * (n / d))) ** 0.5) - 1) for n in n_range] for d in d_range
        }
        tokens = self.get_tokens(filters=("punctuation", "lower"), idx=False)
        # D is undefined if the text is shorter than the maximum sample size
        if len(tokens) < max(n_range):
            return 0

        d_results = []
        for i in range(iterations):
            # compute empirical TTR curve
            ttr_empirical = []
            for n in n_range:
                samples = [random.sample(tokens, n) for _ in range(n_samples)]
                mean = np.mean([len(set(s)) / len(s) for s in samples])
                ttr_empirical.append(mean)

            # find D parameter that best fits data
            d_scores = {}
            y = np.array(ttr_empirical)
            for d in d_range:
                yhat = d_curve[d]
                d_scores[d] = mean_squared_error(y, yhat)

            mse_best = min(d_scores.values())
            d_best = min(
                [d for d in d_scores.keys() if d_scores[d] == mse_best]
            )  # resolve ties by erring low
            d_results.append(d_best)

        return np.mean(d_results)

    @staticmethod
    def join_sentences(sentences):
        out = []
        for sentence in sentences:
            start_idx = sentence[0][0][0]
            end_idx = sentence[-1][0][1]
            tokens = [t for _, t in sentence]
            text = " ".join(tokens)
            out.append(((start_idx, end_idx), tokens, text))
        return out

    def indices_to_nodes(self, start, end):
        """
        :param start, end: indices as provided by get_tokens
        :return: sufficient information to find that text in slate
        """
        if not self.node_map:
            raise RequiresSlate("no nodes to map to without slate input")
        text_range = (self.node_map[start][-1], self.node_map[end][-1])
        node_path = self.node_map[start][:-1]
        while node_path:
            obj = reduce(lambda o, k: o[k], [self.raw] + node_path)
            # TODO: no re-links inside existing links (currently managed on web app)
            if isinstance(obj, dict):
                if obj.get("kind") == "block" or obj.get("object") == "block":
                    break
            node_path.pop()

        # adjust text range based on other leaves of this parent node
        text_offset = 0
        for path, text, _ in extract_text_from_slate(
            [obj], idxs=node_path[:-1], enum_offset=node_path[-1]
        ):
            if path == self.node_map[start][:-1]:
                break
            text_offset += len(text)
        text_range = tuple(i + text_offset for i in text_range)
        text_content = self.text[start:end]

        return {
            "text_range": text_range,
            "node": obj,
            "path": node_path,
            "text_content": text_content,
        }

    def compare_to(
        self,
        doc,
        mode="word_jaccard",
        filters=None,
        fd_n=None,
        colloc_n=2,
        sample_n=None,
        **kwargs,
    ):
        """
        :param doc: another NLPDocument instance
        :param mode: how to do the comparison ("sentence_jaccard", "word_jaccard", "entity_jaccard", ...)
        :return: overall similarity measure + list of sentences the documents share in common
        """
        filters = filters or (
            "punctuation",
            "lower",
        )

        def maxmean_seqm(A, B, quick=False):
            # take the mean value of the maximum-ratio sentence for each sentence
            skip, eval = 0, 0
            vals, seen = [], []
            A, B = adjust_type(list, A, B)
            seqm = SequenceMatcher()
            # i know this seems backwards but sequencematcher caches sequence b
            if sample_n:
                B = random.choice(B, sample_n)
            for j, b in enumerate(B):
                seqm.set_seq2(b)
                maxrat = 0
                for i, a in enumerate(A):
                    if (j, i) in seen:
                        skip += 1
                        continue
                    eval += 1
                    seen.append((i, j))
                    seqm.set_seq1(a)
                    if quick:
                        rat = seqm.quick_ratio()
                    else:
                        rat = seqm.ratio()
                    maxrat = max(rat, maxrat)
                vals.append(maxrat)
            return np.mean(vals)

        def maxmean_jaccard(A, B):
            # take the mean value of the maximum-word-jaccard sentence for each sentence
            skip, eval = 0, 0
            vals, seen = [], []
            if sample_n:
                A = random.choice(A, sample_n)
            for i, a in enumerate(A):
                maxjac = 0
                for j, b in enumerate(B):
                    if (j, i) in seen:
                        skip += 1
                        continue
                    seen.append((i, j))
                    eval += 1
                    maxjac = max(jaccard(a, b), maxjac)
                vals.append(maxjac)
            return np.mean(vals)

        def document_seqm(quick=False):
            if quick:
                return SequenceMatcher(None, self.text, doc.text).quick_ratio()
            return SequenceMatcher(None, self.text, doc.text).ratio()

        def sentence(score, **kwargs):
            own = self.join_sentences(self.get_tokens(filters=filters, by_sentence=True))
            other = doc.join_sentences(doc.get_tokens(filters=filters, by_sentence=True))
            own_text = {text for _, _, text in own}
            other_text = {text for _, _, text in other}
            if score == "jaccard":
                return jaccard(own_text, other_text)
            elif score == "maxmean_jaccard":
                own_tokens = {tuple(tokens) for _, tokens, _ in own}
                other_tokens = {tuple(tokens) for _, tokens, _ in other}
                return maxmean_jaccard(own_tokens, other_tokens)
            elif score == "maxmean_seqm":
                return maxmean_seqm(own_text, other_text, **kwargs)
            else:
                raise ValueError(
                    "score metric must be either jaccard, maxmean_jaccard, or maxmean_seqm"
                )

        def word_jaccard():
            return jaccard(
                self.get_tokens(idx=False, filters=filters),
                doc.get_tokens(idx=False, filters=filters),
            )

        def entity_jaccard():
            return jaccard(
                {k[1] for k in self.entities.most_common(fd_n)},
                {k[1] for k in doc.entities.most_common(fd_n)},
            )

        def collocation_jaccard():
            return jaccard(
                self.get_collocations(filters=filters, n=colloc_n).ngram_fd.most_common(fd_n),
                doc.get_collocations(filters=filters, n=colloc_n).ngram_fd.most_common(fd_n),
            )

        def colloc_feature_extractor(d):
            return [
                "".join(k)
                for k, v in d.get_collocations(filters=filters, n=colloc_n).ngram_fd.most_common(
                    fd_n
                )
            ]

        def entity_feature_extractor(d):
            return ["".join(k) for k, v in d.entities.most_common(fd_n)]

        def simhash(feature_fn="colloc"):
            if isinstance(feature_fn, str):
                fn_map = {
                    "colloc": colloc_feature_extractor,
                    "entity": entity_feature_extractor,
                }
                if feature_fn not in fn_map:
                    raise ValueError(
                        f"feature_fn either must be a function that takes "
                        f"an NLPDocument as its sole argument or one of "
                        f"{list(fn_map.keys())}"
                    )
                feature_fn = fn_map[feature_fn]
            own_features = feature_fn(self)
            other_features = feature_fn(doc)
            distance = Simhash(own_features).distance(Simhash(other_features))
            return -1 * distance

        modemap = {
            "sentence_jaccard": lambda: sentence("jaccard"),
            "sentence_maxmean_jaccard": lambda: sentence("maxmean_jaccard"),
            "sentence_maxmean_seqm": lambda: sentence("maxmean_seqm"),
            "sentence_maxmean_seqm_quick": lambda: sentence("maxmean_seqm", quick=True),
            "word_jaccard": word_jaccard,
            "entity_jaccard": entity_jaccard,
            "collocation_jaccard": collocation_jaccard,
            "document_seqm": document_seqm,
            "document_seqm_quick": lambda: document_seqm(True),
            "simhash": lambda: simhash(**kwargs),  # TODO - dont do this
        }
        return modemap[mode]()

    @staticmethod
    def _get_chunk_content(chunks, labels):
        """
        search through the chunked trees looking for labels
        :return: a counter of how many times each chunk is mentioned
        """
        content = []
        for tree in chunks:
            for st in tree.subtrees():
                if st.label() in labels:
                    content.append((st.label(), " ".join([leaf[0] for leaf in st.leaves()])))

        return Counter(content)

    @lru_cache(maxsize=1024)
    def _cached_chain_call(self, function_name, data):
        """
        cache results of filtering functions to improve performance of get_tokens
        """
        f = self.token_filters[function_name]
        return f(data)

    def _cached_chain(self, data, function_names):
        """
        chain filtering functions on data, using a cache (see _cached_chain_call)
        """
        for function_name in function_names:
            try:  # arguments to cached functions must be hashable
                hash(data)
            except TypeError:
                data = tuple(data)
            data = self._cached_chain_call(function_name, data)

        return data

    def _nodes_to_text(self):
        """
        :return: extracted text, map to recover content node from text character index
        """
        full_text = ""
        node_map = []
        try:
            nodes = self.raw["value"]["nodes"]
        except (TypeError, AttributeError):
            pass
        else:
            for node_idx, node_text, end_of_block in extract_text_from_slate(
                nodes, idxs=["value", "nodes"]
            ):
                new_text = self.replace_overwrought_chars(node_text)
                if end_of_block:
                    new_text += "\n"
                full_text += new_text
                for local_idx in range(len(new_text)):
                    node_map.append(node_idx + [local_idx])
        finally:
            return full_text, node_map

    def to_json(self):
        exclusions = [
            "stemmer",
            "node_map",
            "raw",
            "token_filters",
        ]
        # prepopulate all attributes that don't start with _ and are not callable and not excluded
        data = {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not attr.startswith("_")
            and not callable(getattr(self, attr))
            and attr not in exclusions
        }
        # override specific attributes to adjust as necessary e.g. for chunk trees
        data["ne_chunks"] = [str(c) for c in self.ne_chunks]
        data["nnp_chunks"] = [str(c) for c in self.nnp_chunks]
        data["entities"] = {
            "{} <{}>".format(k[1], k[0]) if k[0] != "NE" else k[1]: v
            for k, v in self.entities.items()
        }
        data["proper_noun_phrases"] = {k[1]: v for k, v in self.proper_noun_phrases.items()}
        try:
            data["vocabulary"] = dict(self.vocabulary)
        except AttributeError:
            data["vocabulary"] = None

        # return json
        return json.dumps(data)


def extract_text_from_slate(nodes, idxs=[], enum_offset=0, parent=None, last_graf_node=None):
    for i, node in enumerate(nodes):
        # exclude results from headings or subheadings
        if node.get("type") in ["heading", "subheading"]:
            continue
        if node.get("type") == "paragraph":
            last_graf_node = node["nodes"][-1]
        active_idx = idxs + [i + enum_offset]
        for ntype in ["nodes", "leaves"]:
            if node.get(ntype):
                active_idx += [ntype]
                for text in extract_text_from_slate(
                    node[ntype], active_idx, parent=node, last_graf_node=last_graf_node
                ):
                    yield text
        if node.get("text"):
            active_idx += ["text"]
            text = node["text"]
            end_of_block = parent == last_graf_node
            yield active_idx, text, end_of_block


class RequiresSlate(Exception):
    pass
