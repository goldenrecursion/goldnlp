from itertools import chain

import numpy as np
from nltk.sentiment import vader
from pyphen import Pyphen

from goldnlp.nlpdocument import NLPDocument


class ClassifyingDocument(NLPDocument):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sentiment_analyzer = None
        self._syllable_analyzer = None

    @property
    def sentiment_analyzer(self):
        if self._sentiment_analyzer is None:
            self._sentiment_analyzer = vader.SentimentIntensityAnalyzer()
        return self._sentiment_analyzer

    @property
    def syllable_analyzer(self):
        if self._syllable_analyzer is None:
            self._syllable_analyzer = Pyphen(lang="en")
        return self._syllable_analyzer

    def get_sentiment(self, sentences=None, aggregate=True):
        if sentences is None:
            sentences = self.get_tokens(by_sentence=True)
        scores = [
            self.sentiment_analyzer.polarity_scores(self.text[s[0][0][0] : s[-1][0][1]])
            for s in sentences
        ]
        if not aggregate:
            return scores
        return {
            k: {
                "mean": np.mean([score[k] for score in scores]),
                "median": np.median([score[k] for score in scores]),
                "std": np.std([score[k] for score in scores]),
            }
            for k in ["pos", "neg", "neu", "compound"]
        }

    def get_fres(self):
        # Flesch Reading Ease Score
        # these constants come directly from the flesch model, dont ask me how we got here
        # TODO: use a better syllable counting library
        syllable_count = sum(
            [
                len(self.syllable_analyzer.inserted(t).split("-"))
                for t in self.get_tokens(filters=("punctuation",), idx=False)
            ]
        )
        sentence_count = len(self.get_tokens(by_sentence=True))
        return (
            206.835
            - (1.015 * self.wordcount / sentence_count)
            - (84.6 * syllable_count / self.wordcount)
        )

    def get_featureset(self):
        sentiment = self.get_sentiment()
        tagged_tokens = list(chain(*self.tags))
        nonpunct_tokens = self.get_tokens(filters=("punctuation",), idx=False)
        return {
            "sentiment_pos_mean": sentiment["pos"]["mean"],
            "sentiment_neg_mean": sentiment["neg"]["mean"],
            "sentiment_neu_mean": sentiment["neu"]["mean"],
            "sentiment_cpd_mean": sentiment["compound"]["mean"],
            "sentiment_pos_std": sentiment["pos"]["std"],
            "sentiment_neg_std": sentiment["neg"]["std"],
            "sentiment_neu_std": sentiment["neu"]["std"],
            "sentiment_cpd_std": sentiment["compound"]["std"],
            "sentiment_pos_median": sentiment["pos"]["median"],
            "sentiment_neg_median": sentiment["neg"]["median"],
            "sentiment_neu_median": sentiment["neu"]["median"],
            "sentiment_cpd_median": sentiment["compound"]["median"],
            "ttr_raw": self.ttr_raw,
            "ttr_d": self.ttr_d,
            "fres": self.get_fres(),
            "sentence_length_mean": np.mean([len(s) for s in self.get_tokens(by_sentence=True)]),
            "word_length_mean": np.mean([len(t) for t in nonpunct_tokens]),
            "punctuation_proportion": (len(tagged_tokens) - len(nonpunct_tokens))
            / len(tagged_tokens),
            "entities_proportion": len(self.entities) / len(tagged_tokens),
            "quotation_mark_proportion": (
                len([c for c in self.get_tokens(idx=False) if c in ["``", '"']]) / len(self.text)
            ),
            "personal_pronoun_proportion": (
                len([_ for _, pos in tagged_tokens if pos.startswith("PRP")]) / len(tagged_tokens)
            ),
            "modal_proportion": (
                len([_ for _, pos in tagged_tokens if pos == "MD"]) / len(tagged_tokens)
            ),
            "modifier_proportion": (
                len([_ for _, pos in tagged_tokens if pos.startswith("RB") or pos.startswith("JJ")])
                / len(tagged_tokens)
            ),
            # TOOD: rare words vs big corpus e.g. reuters ??
        }
