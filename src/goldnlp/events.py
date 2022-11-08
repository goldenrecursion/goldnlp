import calendar
import datetime
from collections import OrderedDict, deque
from uuid import uuid4

import dateparser
import pytz

from goldnlp.spacy_utils import relabel_ent, spacy_ws_join

def to_timeline(doc):
    events = list(
        filter(
            lambda e: e and e["date"] and e["contents"],
            [e.as_dict() for e in doc._.get_events()],
        )
    )
    return [
        {
            "date": dict(ts),
            "contents": list(
                OrderedDict.fromkeys((e["contents"] for e in events if e["date"] == dict(ts)))
            ),
            "event_ids": [e["uuid"] for e in events if e["date"] == dict(ts)],
        }
        for ts in list(OrderedDict.fromkeys(tuple(e["date"].items()) for e in events))
    ]


def pub_date_native(doc):
    date_dict = doc._.pub_date
    if not date_dict or "year" not in date_dict:
        return datetime.datetime.now()
    if "month" not in date_dict:
        date_dict["month"] = 1
    if "day" not in date_dict:
        date_dict["day"] = 1
    return datetime.datetime(**date_dict)


def title_as_doc(doc):
    return doc._.create_subdocument(doc._.title)


def preprocess_dates(doc, token_filter=None):
    if token_filter:
        tokens = [t for t in doc if getattr(t._, token_filter[0]) == token_filter[1]]
        if not tokens:
            return []
        doc = doc[tokens[0].i : tokens[-1].i].as_doc()

    dates = [e for e in doc.ents if e.label_ in ["DATE", "TIME"]]
    # TODO include "TIME" label once partial precision is accounted for
    # so we can make sure we pick out the date-ish times that get mis-classed

    # remove adjectives e.g. "late 2019"
    cleaned_dates = []
    for e in dates:
        cleaned_dates.append(spacy_ws_join([t for t in doc[e.start : e.end] if t.pos_ != "ADJ"]))

    # transform e.g. "Tuesday" to "today" if pub_date is a Tuesday
    # otherwise dateparse wants it to mean the previous Tuesday
    weekday = doc._.pub_date_native.weekday()
    for i, date in enumerate(cleaned_dates):
        if date.lower() in [
            calendar.day_name[weekday].lower(),
            calendar.day_abbr[weekday].lower(),
        ]:
            cleaned_dates[i] = "today"
    return cleaned_dates


def fuzzy_parse(string, settings=None):
    parser = dateparser.date.DateDataParser(languages=["en"], settings=settings)
    result = parser.get_date_data(string)
    if result["date_obj"]:
        return FuzzyDate(result["date_obj"], precision_period=result["period"])


class FuzzyDate:
    def __init__(self, datetime_obj, precision_period="day"):
        if datetime_obj is None:
            self.datetime_obj, self.fuzzy = None, None
        else:
            fuzzy_date = {}
            for precision_level in ["year", "month", "day"]:
                fuzzy_date[precision_level] = getattr(datetime_obj, precision_level)
                if precision_level == precision_period:
                    break
            localized = datetime_obj if datetime_obj.tzinfo else pytz.utc.localize(datetime_obj)
            self.datetime_obj = localized
            self.fuzzy = fuzzy_date

    # when comparing act like datetime
    # TODO what about when it's None? never seems to come up but...?
    def __gt__(self, other):
        return self.datetime_obj > other

    def __ge__(self, other):
        return self.datetime_obj >= other

    def __lt__(self, other):
        return self.datetime_obj < other

    def __le__(self, other):
        return self.datetime_obj <= other

    def __eq__(self, other):
        return self.datetime_obj == other

    def __ne__(self, other):
        return self.datetime_obj != other


def make_fuzzy(datetime_obj):
    return {
        "year": datetime_obj.year,
        "month": datetime_obj.month,
        "day": datetime_obj.day,
    }
