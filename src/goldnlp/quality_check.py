import pandas as pd
from nltk import ngrams
from numpy import inf, log
from scipy.stats import norm

from goldnlp.constants import FIRST_PERSON_PRONOUNS, PUNCTUATION, ROOT_PATH, SECOND_PERSON_PRONOUNS

# computed from a sample of ~200k tokens / ~1m chars
# from 33 wikipedia pages on big companies.
# in the future, learn this from golden data!
# min/max ranges can be adjusted to change sensitivity
HEURISTIC_STATS = {
    "sentence_length": {
        "mean": 23.669,
        "std": 3.697,
        "min_z": -2,
        "max_z": 2,
    },
    "word_length": {
        "mean": 5.340,
        "std": 0.115,
        "min_z": -3,
        "max_z": 3,
    },
    "conversational_pronoun_rate": {
        "mean": 0.00189,
        "std": 0.00203,
        "min_z": -inf,
        "max_z": 2,
    },
    "all_caps_rate": {
        "mean": 0.0188,
        "std": 0.0153,
        "min_z": -inf,
        "max_z": 2,
    },
    "punct_rate": {
        "mean": 0.0310,
        "std": 0.00295,
        "min_z": -5,
        "max_z": 2,
    },
    "url_rate": {
        "mean": 0.00114,
        "std": 0.00160,
        "min_z": -inf,
        "max_z": 2,
    },
    "buzzword_index": {
        "mean": 0.943,
        "std": 0.511,
        "min_z": -inf,
        "max_z": 2,
    },
}


def quality_check(doc, warnings_only=False, heuristic_stats=None):
    if heuristic_stats is None:
        heuristic_stats = HEURISTIC_STATS
    summary = {}
    chars = len(str(doc))
    tokens = len(doc)
    sentences = len(list(doc.sents))
    if not sentences or not tokens or not chars:
        return {"not_processed": f"{sentences} sentences, {tokens} tokens, {chars} chars"}
    buzzword_info = doc._.buzzword_score()
    conversational_pronouns = len(
        [t for t in doc if t.lower_ in FIRST_PERSON_PRONOUNS + SECOND_PERSON_PRONOUNS]
    )
    all_caps_tokens = len([t for t in doc if all(c.isupper() for c in t.text)])
    punct_chars = len([c for t in doc for c in t.orth_ if c in PUNCTUATION])
    likely_urls = [str(t) for t in doc if t.like_url]
    features = {
        "chars": chars,
        "tokens": tokens,
        "sentences": sentences,
        "sentence_length": tokens / sentences,
        "word_length": chars / tokens,
        "conversational_pronouns": conversational_pronouns,
        "conversational_pronoun_rate": conversational_pronouns / tokens,
        "all_caps_tokens": all_caps_tokens,
        "all_caps_rate": all_caps_tokens / tokens,
        "punct_chars": punct_chars,
        "punct_rate": punct_chars / chars,
        "likely_urls": likely_urls,
        "url_rate": len(likely_urls) / tokens,
        "buzzword_index": buzzword_info["total"] / sentences,
    }
    for feature, value in features.items():
        stats = heuristic_stats.get(feature)
        if not stats:
            if not warnings_only:
                summary[feature] = {"raw": value}
            continue
        z = (value - stats["mean"]) / stats["std"]
        out_of_range = not (stats["min_z"] < z < stats["max_z"])
        if warnings_only and not out_of_range:
            continue
        summary[feature] = {
            "level": z_description(z),
            "raw": value,
            "z": z,
            "normal_z_range": [stats["min_z"], stats["max_z"]],
            "p_one_tailed": norm.cdf(abs(z)),
            "log_prob": log(norm.pdf(z)),
            "sample_mean": stats["mean"],
            "sample_std": stats["std"],
        }
        if not warnings_only:
            summary[feature].update(
                {
                    "out_of_range": out_of_range,
                }
            )
    if not warnings_only or "buzzword_index" in summary:
        summary["buzzword_index"].update(
            {
                "details": buzzword_info,
            }
        )
    return summary


def z_description(z):
    if abs(z) <= 2:
        return "normal"
    if abs(z) <= 4:
        return "high" if z > 0 else "low"
    if abs(z) <= 6:
        return "very high" if z > 0 else "very low"
    return "extremely high" if z > 0 else "extremely low"


def buzzword_score(doc):
    # stored weights are -1 * the token log probability given
    # by the the _large_ spacy english model (not medium).
    # phrases got the average of their members, minus stopwords.
    # this procedure can definitely be improved in the future.
    # terms came from http://www.adamsherk.com/public-relations/most-overused-press-release-buzzwords/
    df = pd.read_json(ROOT_PATH + "buzzwords_weighted.json")
    score = 0
    by_sentence = []
    buzzwords_seen = set()
    for sent in doc.sents:
        sent_as_string = str(sent)
        sent_as_slate = (
            doc._.nlpdocument.indices_to_nodes(sent[0].idx, sent[-1].idx) if doc._.slate else None
        )
        sent = [t.lower_ for t in sent if not t.is_punct]
        phrases = [ngram for i in range(2, 5) for ngram in ngrams(sent, i)]
        tokens = [token for token in sent if token not in phrases]
        candidates = [" ".join(phrase) for phrase in phrases] + tokens
        filtered_df = df[df.term.isin(candidates)]
        local_score = filtered_df.weight.sum()
        buzzwords_seen.update(filtered_df.index)
        score += local_score
        if local_score:
            by_sentence.append(
                {
                    "sentence": sent_as_string,
                    "node_info": sent_as_slate,
                    "score": local_score,
                    "tokens": len(sent),
                    "chars": len(sent_as_string),
                }
            )
    return {
        "total": score,
        "buzzwords_seen": list(buzzwords_seen),
        "by_sentence": by_sentence,
    }
