from collections import defaultdict

from nltk.sem import relextract

from goldnlp.nlpdocument import NLPDocument


def relation_dicts(pairs, window=5):
    """
    rage copypasta.
    in relextract this same basic behavior is called "semi_rel2reldict"
    which is a hideous name and also the default behavior forces ugly
    slashed tags into your results.
    """
    result = []

    while len(pairs) > 2:
        reldict = defaultdict(str)
        reldict["lcon"] = relextract._join(pairs[0][0][-window:])
        reldict["subjclass"] = pairs[0][1].label()
        reldict["subjtext"] = relextract._join(pairs[0][1].leaves(), untag=True)
        reldict["subjsym"] = relextract.list2sym(pairs[0][1].leaves())
        reldict["filler"] = relextract._join(pairs[1][0])
        reldict["untagged_filler"] = relextract._join(pairs[1][0], untag=True)
        reldict["objclass"] = pairs[1][1].label()
        reldict["objtext"] = relextract._join(pairs[1][1].leaves(), untag=True)
        reldict["objsym"] = relextract.list2sym(pairs[1][1].leaves())
        reldict["rcon"] = relextract._join(pairs[2][0][:window])
        result.append(reldict)
        pairs = pairs[1:]
    return result


class ExtractingDocument(NLPDocument):
    def __init__(self, *args, **kwargs):
        # force nonbinary NER so stupid relextract stuff works
        # FIXME: fork relextract
        kwargs["binary_ne"] = False
        super().__init__(*args, **kwargs)

    def get_relations(self, min_rel_len=2, max_rel_len=30):
        relations = []
        for i, tree in enumerate(self.ne_chunks):
            rds = relation_dicts(relextract.tree2semi_rel(tree))
            for rd in rds:
                if (
                    not rd["untagged_filler"]
                    or not min_rel_len <= len(rd["untagged_filler"]) <= max_rel_len
                ):
                    continue
                context_start = self.get_tokens(by_sentence=True)[i][0][0][0]
                context_end = self.get_tokens(by_sentence=True)[i][-1][0][1]
                context = self.text[
                    context_start : context_end + 1
                ]  # try to get trailing puncutation for sentence
                relations.append(
                    {
                        "subject": rd["subjtext"],
                        "relation": rd["untagged_filler"],
                        "object": rd["objtext"],
                        "context": context,
                    }
                )
        return {"relations": relations}
