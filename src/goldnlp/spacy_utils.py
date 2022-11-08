from functools import lru_cache

import spacy


def spacy_ws_join(tokens):
    return "".join(t.text_with_ws for t in tokens).strip()


def dependency_visualization(doc):
    dep_viz_opts = {
        "compact": True,
        "collapse_punct": True,
        "collapse_phrases": True,
    }
    return spacy.displacy.render(doc.sents, style="dep", minify=True, options=dep_viz_opts)


def entity_visualization(doc):
    return spacy.displacy.render(
        doc,
        style="ent",
        minify=True,
        options={
            "colors": {
                "ROUND": "#f1b61b",
                "TICKER": "#e2c881",
                "FUNDED_COMPANY": "#03adfc",
                "INVESTOR": "#03fca1",
            }
        },
    )


def trim_tree(entity):
    # TODO this doesnt work so great, fix it up
    try:
        subject = [
            t
            for t in entity.sent
            if t.dep_ in ["nsubj"] and t.head == entity.root.head and t != entity.root
        ][0]
    except IndexError:
        print(
            f"cannot find subject of verb with entity as object, returning full sentence: {entity.sent}"
        )
        return str(entity.sent)
    reduced = (
        spacy_ws_join(subject.subtree)
        + entity.sent.root.text_with_ws
        + spacy_ws_join(entity.subtree)
    )
    if reduced[0].islower():
        reduced = reduced[0].upper() + reduced[1:]
    return reduced


@lru_cache()
def make_matches_greedy(matches):
    """
    :param matches: spacy.matcher.Matcher matches
    :return: only the longest matches (properly greedy)
    """
    sorted_matches = sorted(matches)
    longest = []

    for match1 in sorted_matches:
        if any(True for match2 in longest if match2[1] <= match1[1] and match2[2] >= match1[2]):
            continue
        longest.append(
            sorted(
                [match2 for match2 in sorted_matches if match2[1] <= match1[1]],
                key=lambda m: m[2],
                reverse=True,
            )[0]
        )
    return longest


def relabel_ent(doc, entity):
    label, start, end = entity
    doc.ents = tuple(
        e
        for e in doc.ents
        # remove ents that overlap
        if not any(True for i in range(start, end) if i in range(e.start, e.end))
    ) + (entity,)


def matcher_wrapper(matcher, pipe_name, *args, **kwargs):
    def inner(doc, *args, **kwargs):
        matcher(doc, *args, **kwargs)
        return doc

    # if you dont do this, spacy gets mad bc
    # the pipeline component names repeat
    inner.__name__ = pipe_name
    return inner
