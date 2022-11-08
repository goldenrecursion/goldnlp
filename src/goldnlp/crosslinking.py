import re
from collections import defaultdict, deque
from difflib import SequenceMatcher
from itertools import combinations
from string import punctuation

from nltk import BigramAssocMeasures, TokenSearcher, TrigramAssocMeasures
from scipy.stats import t as t_distribution

from goldnlp.nlpdocument import NAMES, NLPDocument, extract_text_from_slate

MIN_TOKENS = 15  # minimum words to process; can be overriden by kwarg


class CrosslinkingDocument(NLPDocument):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_tokens = kwargs.get("min_tokens", MIN_TOKENS)
        self.high_freq_1grams = [
            stem
            for stem, count in self.get_freqdist(filters=self.freq_filters).most_common(5)
            if count > 1
        ]
        # entity names (and other proper nouns?)
        raw_names = list(self.entities.keys())  # + list(self.proper_noun_phrases.keys())
        self.names = [
            " ".join(self.get_tokens(k[1], filters=("punctuation", "lower"), idx=False))
            for k in raw_names
        ]

    def get_crosslink_candidates(self, expand_terms=True):
        """
        :return: dict of words to search as likely crosslinks, grouped into 'stemmed' and 'unstemmed' sets
        """
        if not self.text:
            return {}
        if len(self.get_tokens(filters=("punctuation",), idx=False)) < self.min_tokens:
            print(
                "not enough text to crosslink (min. {} non-punctuation tokens)".format(
                    self.min_tokens
                )
            )
            return {}

        candidates = {"stemmed": [], "unstemmed": []}
        print("found names: {}".format(self.names))
        candidates["unstemmed"] += self.names

        # italicized terms
        # TODO: redo for Slate format
        # candidates['unstemmed'] += [' '.join(self.get_tokens(
        #    filters=('punctuation', 'lower',), text=e.text_content(), idx=False))
        #    for e in self.etree.xpath('//em')]

        # frequent terms and collocations
        print("found high freq unigrams: {}".format(self.high_freq_1grams))
        candidates["stemmed"] += self.high_freq_1grams

        colloc_data = [
            (
                self.get_collocations(n=2, filters=self.freq_filters),
                BigramAssocMeasures,
            ),
            (
                self.get_collocations(n=3, filters=self.freq_filters),
                TrigramAssocMeasures,
            ),
        ]
        for finder, metric_cls in colloc_data:
            # run a t-test on each n-gram frequency; include if p < 0.1
            df = len(finder.ngram_fd) - 1
            t = metric_cls.student_t
            for words, count in dict(finder.ngram_fd).items():
                if count <= 2:  # require at least 3 cases
                    continue
                t_stat = finder.score_ngram(t, *words)
                p = t_distribution.sf(t_stat, df)
                if p < 0.1:
                    print("sig collocation: {} (p={:.3})".format(" ".join(words), p))
                    candidates["stemmed"].append(" ".join(words))

        if expand_terms:
            filters = {
                "unstemmed": ("punctuation", "lower", "stopword"),
                "stemmed": ("punctuation", "lower", "stopword", "stem"),
            }

            # expand to larger phrases 2 words in either direction from all candidates
            for qtype, qlist in candidates.copy().items():
                qset = set(qlist)
                token_searcher = TokenSearcher(self.get_tokens(filters=filters[qtype], idx=False))
                for term in qset:
                    termtag = r" ".join([r"<" + t + r">" for t in term.split()])
                    back_expansions = set(
                        [" ".join(r) for r in token_searcher.findall(r"<\w+> " + termtag)]
                    )
                    back_expansions = back_expansions.union(
                        set([" ".join(r) for r in token_searcher.findall(r"<\w+>{2} " + termtag)])
                    )
                    forward_expansions = set(
                        [" ".join(r) for r in token_searcher.findall(termtag + r" <\w+>")]
                    )
                    forward_expansions = forward_expansions.union(
                        set([" ".join(r) for r in token_searcher.findall(termtag + r" <\w+>{2}")])
                    )
                    sym_expansions = set(
                        [
                            " ".join(r)
                            for r in token_searcher.findall(r"<\w+> " + termtag + r" <\w+>")
                        ]
                    )
                    expansions = list(sym_expansions | back_expansions | forward_expansions)
                    candidates[qtype] += expansions

            print("expanded to {} queries".format(sum([len(v) for v in candidates.values()])))

        # deduplicate
        for qtype, qlist in candidates.items():
            if expand_terms:
                # now that we've seeded expansions, kill plain unigrams
                culled_candidates = [w for w in self.high_freq_1grams if w not in self.names]
                print(f"dropped unremarkable {qtype} unigrams: {culled_candidates}")
                qlist = [q for q in qlist if q not in culled_candidates]
            candidates[qtype] = set(qlist)

        # try to prevent self-referencing terms being searched, we dont want to link them
        filters = {
            "unstemmed": ("punctuation", "lower"),
            "stemmed": ("punctuation", "lower", "stem"),
        }
        for qtype in filters.keys():
            name_tokens = self.get_tokens(filters=filters[qtype], text=self.name, idx=False)
            if not name_tokens:
                continue
            # remove personal names when checking for self-naming
            name_tokens = set(name_tokens) - set(NAMES)
            for query in candidates[qtype].copy():
                query_tokens = set(query.split()) - set(NAMES)
                if query_tokens == name_tokens:
                    print("dropped apparent self-reference: {}".format(query))
                    candidates[qtype].remove(query)

        return candidates

    def clean_crosslink_suggestions(self, suggestions, result_slice_size=10, max_results=1):
        """
        :param suggestions: dictionary like {(query_type, query): results} from celeryapp.crosslink()
        :return: a dictionary like {query: slug}, picking only the best results
        """
        if not suggestions:
            return None

        cleaned_suggestions = defaultdict(list)
        filters = {
            "unstemmed": ("punctuation", "lower"),
            "stemmed": ("punctuation", "lower", "stem"),
        }
        ambig_count = 0
        for (qtype, query), results in suggestions.items():
            seqmatcher = SequenceMatcher()
            seqmatcher.set_seq2(
                query
            )  # sequence 2 is cached, much better performance for many comparisons
            for result in results[:result_slice_size]:
                # no self-links
                if result["slug"] == self.slug:
                    print("dropped self-link: {}".format(query))
                    continue

                # only take the first link for a given target
                if result.meta.id in cleaned_suggestions.values():
                    print("dropped repeated target: {}".format(query))
                    continue

                # dont match on this name in other categories
                if result["name"].split(" (")[0] == self.name.split(" (")[0]:
                    print("dropped equivalent name from other disambig context: {}".format(query))
                    continue

                # dont match an acronym to a normal name
                if (
                    query.upper() in self.get_tokens(idx=False)
                    and not query.upper() in result["name"].split(" (")[0]
                ):
                    print(
                        "dropped apparent acronym {} matching on non-acronym: {}".format(
                            query, result["name"]
                        )
                    )
                    continue

                if query.upper() in self.text and not [
                    c
                    for c in query.upper()
                    if c not in [w[0].upper() for w in self.name.split(" (")[0].split()]
                ]:
                    print("dropped an apparent self-referencing acronym: {}".format(query))
                    continue

                # ignore results that dont have enough context to disambiguate
                if not result["description"] and "(" not in result["name"]:
                    ambig_count += 1
                    continue

                result_name_raw = result["name"].split("(")[0]
                result_name_cleaned = " ".join(
                    self.get_tokens(filters=filters[qtype], text=result_name_raw, idx=False)
                )
                # avoid weird results like "Attack!!!" and "Finally!!" and "Yes!"
                if [
                    c for c in result_name_raw if c in punctuation and c not in "-"
                ] and result_name_raw not in self.text:
                    print("dropped unmatched punctuated term: {}".format(result_name_raw))
                    continue

                # apply increasingly lax heuristics until we find a good result
                # TODO: handle acronyms better
                if result_name_cleaned == query or result_name_cleaned.split("(")[0] == query:
                    if result_name_raw.isupper() and result_name_raw.split("(")[
                        0
                    ] not in self.get_tokens(idx=False):
                        print(
                            "dropped to avoid mistaking normal terms for acronyms: {} -> {}".format(
                                query, result_name_raw
                            )
                        )
                        continue  # no more weird acronym results pls
                    print("found top-quality match: {} -> {}".format(query, result["slug"]))
                    cleaned_suggestions[(qtype, query)].append(result.meta.id)
                    if len(cleaned_suggestions[(qtype, query)]) == max_results:
                        break

                elif result_name_cleaned.startswith(query):
                    seqmatcher.set_seq1(result_name_cleaned)
                    if seqmatcher.ratio() > 0.95:
                        print("found near-match: {} -> {}".format(query, result["slug"]))
                        cleaned_suggestions[(qtype, query)].append(result.meta.id)
                        if len(cleaned_suggestions[(qtype, query)]) == max_results:
                            break

                    elif query.upper() in re.findall(r"\((\w+)\)", result_name_raw) and not [
                        c
                        for c in query.upper()
                        if c not in [w[0].upper() for w in result_name_raw.split("(")[0]]
                    ]:
                        print("found acronym match: {} -> {}".format(query, result["slug"]))
                        cleaned_suggestions[(qtype, query)].append(result.meta.id)
                        if len(cleaned_suggestions[(qtype, query)]) == max_results:
                            break

        print("dropped {} ambiguous results during search".format(ambig_count))

        # filter nested names e.g. no 'series' if there's 'series a' and 'series b' in the matched suggestions
        nested_names = set()
        sugg_queries = [query for qtype, query in cleaned_suggestions.keys()]
        for c1, c2 in combinations(sugg_queries, 2):
            set1, set2 = set(c1.split()), set(c2.split())
            # make order right
            if len(set1) > len(set2):
                set1, set2 = set2, set1
                c1, c2 = c2, c1
            if set1.issubset(set2):
                nested_names.add(c1)
                print("dropped nested names: {} / {}".format(c1, c2))
        cleaned_suggestions = {
            k: v for k, v in cleaned_suggestions.items() if k[1] not in nested_names
        }
        print("ready to apply suggestions: {}".format(list(cleaned_suggestions.keys())))
        return cleaned_suggestions

    def get_crosslink_upgrades(self, suggestions, max_results=1, simple_results=False):
        """
        :param suggestions: dictionary like {(querytype, query): results} from celeryapp.crosslink()
        :param max_results: # of possible entity targets to return
        :param simple_results: basic text -> entity output useful for when you dont care about making slate patches
        :return: list of possible crosslinks in a format matching the web app upgrades module
        """

        cleaned_suggestions = self.clean_crosslink_suggestions(suggestions, max_results=max_results)
        if not cleaned_suggestions:
            return None

        # TODO - stop repeating filters everywhere, make everything consistent
        filters = {
            "unstemmed": ("punctuation", "lower"),
            "stemmed": ("punctuation", "lower", "stem"),
        }

        sentence_starts = {
            qtype: [
                s[0][0][0] for s in self.get_tokens(filters=filters[qtype], by_sentence=True) if s
            ]
            for qtype in filters.keys()
        }

        for (qtype, query), ids in cleaned_suggestions.items():
            # now that we're applying the links to the text, we have some more cleaning to do...
            query_recleaned = self.get_tokens(text=query, filters=filters["stemmed"], idx=False)

            query_spans = deque(
                self.get_ngram_span(query_recleaned, self.get_tokens(filters=filters["stemmed"]))
            )

            own_name_cleaned = self.get_tokens(
                text=self.name, filters=filters["stemmed"], idx=False
            )

            if not own_name_cleaned:
                own_name_spans = []
            else:
                own_name_spans = list(
                    self.get_ngram_span(
                        own_name_cleaned, self.get_tokens(filters=filters["stemmed"])
                    )
                )

            start = end = None
            while query_spans:
                start, end = query_spans.popleft()
                # if we cross a sentence boundary or this instance is inside the topic name, keep looping
                crossed_bounds = [s for s in sentence_starts[qtype] if start < s < end]
                name_embedded = bool(
                    [
                        s
                        for s in own_name_spans
                        if start in range(s[0], s[1] + 1) and end in range(s[0], s[1] + 1)
                    ]
                )
                if name_embedded:
                    print(
                        "{} ({}, {}) skipped - embedded in the topic name".format(query, start, end)
                    )
                elif crossed_bounds:
                    print(
                        "{} ({}, {}) crossed sentence boundaries: {}".format(
                            query, start, end, crossed_bounds
                        )
                    )
                else:
                    break
                start = end = None

            if start is None:
                print("no usable spans for {} ({})".format(query, qtype))
                continue

            if simple_results:
                yield {"text": self.text[start:end], "target_entity": ids}
                continue

            if self.node_map[start][:-1] != self.node_map[end][:-1]:
                continue

            result = self.indices_to_nodes(start, end)
            result.update(
                {
                    "upgrade_type": "crosslink",
                    "entity_version": self.ev_id,
                    "target_entity": ids[0],
                }
            )
            yield result
