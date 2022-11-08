from numpy import inf
from pandas import Series
from scipy.stats import ttest_ind_from_stats

from goldnlp.nlpdocument import NLPDocument

AUTO_INCLUDE_THRESHOLDS = {  # min non-punctuation tokens for auto-include per unit
    "sentence": 5,
    "paragraph": 15,
}


class SummarizingDocument(NLPDocument):
    def __init__(self, *args, **kwargs):
        if "stem_mode" not in kwargs:
            kwargs["stem_mode"] = "porter"  # default to better stemming for summaries
        super().__init__(*args, **kwargs)
        self.freq_filters = (
            "punctuation",
            "lower",
            "stopword",
            "stem",
        )

    def summarize(
        self,
        unit="paragraph",
        n_units=5,
        preserve_article_order=True,
        score_mode="t",
        auto_include_threshold=None,
        simple_format=False,
    ):
        """
        use word frequency scoring to determine the n_units
        most "important" units in a document. unit can be
        "sentence" "paragraph" "word" or "debug".
        if n_units is a fraction, it is interpreted as a % of
        all units (e.g. 0.85 -> return 85% of paragraphs).
        score_mode can be used to select summed-frequency ("sum")
        or mean-token-frequency t-stat ("t", default) as the
        evaluative metric for unit importance. score_mode has
        no effect when unit is 'word'.
        TODO: use t stat to drop all units below a lower bound
        TODO: and return the rest?
        """
        fd = self.get_freqdist(self.freq_filters)
        if auto_include_threshold is None:
            auto_include_threshold = AUTO_INCLUDE_THRESHOLDS[unit]
        unit_threshold_met = True  # all units default off unless at least n_units exist
        output = []
        mode_map = {
            "sum": "scores",
            "t": "t_scores",
        }
        if unit not in ["word", "sentence", "paragraph", "debug"]:
            raise NotImplementedError('unit must be "word" "sentence" "paragraph" or "debug"')

        if score_mode not in mode_map.keys():
            raise NotImplementedError(f"score_mode must be one of: {list(mode_map.keys())}")

        def slice_scores(
            sorted_scores,
            n_units=n_units,
            preserve_article_order=preserve_article_order,
        ):
            if 0 < n_units < 1:  # support fractional units (%age to return)
                n_units = round(len(sorted_scores) * n_units) or 1
            top = sorted_scores[:n_units]
            if preserve_article_order:
                top = sorted(top)
            return top

        def get_min_length(data, quantile=0.1, floor=5):
            s = Series(data)
            try:
                min_length = max(
                    int(s.quantile(quantile), floor)
                )  # very small n makes t-test weird
            except (TypeError, ValueError):
                min_length = floor
            return min_length

        document_series = Series(
            [fd[word] for word in self.get_tokens(filters=self.freq_filters, idx=False)]
        )

        def score_series(local_series, min_length, document_series=document_series):
            # dont interpret the t-test for very short units
            if len(local_series) < min_length or not document_series.any():
                return local_series.sum(), -inf, inf
            # we are definitely violating the independence assumption but if it works...
            ttest = ttest_ind_from_stats(
                local_series.mean(),
                local_series.std(),
                len(local_series),
                document_series.mean(),
                document_series.std(),
                len(document_series),
                equal_var=False,
            )
            return local_series.sum(), ttest.statistic, ttest.pvalue

        def should_include(text, unit_threshold_met):
            if not unit_threshold_met:
                return False
            return (
                len(self.get_tokens(text, filters=("punctuation",), idx=False))
                > auto_include_threshold
            )

        if unit == "sentence" or unit == "debug":
            scores = {}
            t_scores = {}
            p_values = {}
            sentences = list(
                filter(None, self.get_tokens(filters=self.freq_filters, by_sentence=True))
            )
            min_length = get_min_length([len(s) for s in sentences])
            # FIXME use full sentence w/o filters for output (and length counting?)
            for i, s in enumerate(sentences):
                local_series = Series([fd[word] for _, word in s])
                scores[i], t_scores[i], p_values[i] = score_series(local_series, min_length)

            score_dict = locals()[mode_map[score_mode]]
            sorted_scores = sorted(score_dict, key=lambda k: score_dict[k], reverse=True)

            if unit == "debug":
                output.append(
                    {
                        "sentence": [
                            (
                                i,
                                scores[i],
                                t_scores[i],
                                p_values[i],
                                " ".join([t for _, t in sentences[i]]),
                            )
                            for i in sorted_scores
                        ]
                    }
                )
            else:
                top = slice_scores(sorted_scores)
                if len(top) < n_units:
                    unit_threshold_met = False
                for i in top:
                    start = sentences[i][0][0][0]
                    end = sentences[i][-1][0][1] + 1
                    output.append(
                        {
                            "content": self.text[start:end],
                            "auto_include": should_include(
                                self.text[start:end], unit_threshold_met
                            ),
                        }
                    )

        if unit == "paragraph" or unit == "debug":
            scores = {}
            t_scores = {}
            p_values = {}
            grafs = [g for g in self.text.split("\n") if g]
            min_length = get_min_length([len(g) for g in grafs])

            for i, graf in enumerate(grafs):
                graf_tokens = self.get_tokens(graf, filters=self.freq_filters, idx=False)
                local_series = Series([fd[word] for word in graf_tokens])
                scores[i], t_scores[i], p_values[i] = score_series(local_series, min_length)

            score_dict = locals()[mode_map[score_mode]]
            sorted_scores = sorted(score_dict, key=lambda k: score_dict[k], reverse=True)

            if unit == "debug":
                output.append(
                    {
                        "paragraph": [
                            (i, scores[i], t_scores[i], p_values[i], grafs[i])
                            for i in sorted_scores
                        ]
                    }
                )
            else:
                top = slice_scores(sorted_scores)
                if len(top) < n_units:
                    unit_threshold_met = False
                for i in top:
                    output.append(
                        {
                            "content": grafs[i],
                            "auto_include": should_include(grafs[i], unit_threshold_met),
                        }
                    )

        if unit == "word" or unit == "debug":
            if unit == "debug":
                output.append({"word": fd.most_common()})
            else:
                # TODO: gross
                if 0 < n_units < 1:  # support fractional units (%age to return)
                    n_units = round(len(sorted_scores) * n_units) or 1
                return fd.most_common(n_units)

        if simple_format:
            return "\n".join([u["content"] for u in output if u["auto_include"]])

        return {"summary": output}
