import os
from string import punctuation

import nltk
from nltk.corpus import names, stopwords

ROOT_PATH = f"{os.path.abspath(os.path.dirname(__file__))}/"

PUNCTUATION = punctuation

STOPWORDS = stopwords.words("english") + [
    "might",
    "could",
    "may",
    "would",
    "must",
    "should",
    "nt",
    "n't",
]

NAMES = [name.lower() for name in names.words()]

TITLES = ["Mr.", "Ms.", "Mrs.", "Dr.", "Prof.", "Rev."]
TITLES += [t[:-1] for t in TITLES]

LANGUAGE_CHECK_WINDOW = 100

PENN_TAGS = {
    "CC": "Coordinating conjunction",
    "CD": "Cardinal number",
    "DT": "Determiner",
    "EX": "Existential there",
    "FW": "Foreign word",
    "IN": "Preposition or subordinating conjunction",
    "JJ": "Adjective",
    "JJR": "Adjective, comparative",
    "JJS": "Adjective, superlative",
    "LS": "List item marker",
    "MD": "Modal",
    "NN": "Noun, singular or mass",
    "NNS": "Noun, plural",
    "NNP": "Proper noun, singular",
    "NNPS": "Proper noun, plural",
    "PDT": "Predeterminer",
    "POS": "Possessive ending",
    "PRP": "Personal pronoun",
    "PRP$": "Possessive pronoun",
    "RB": "Adverb",
    "RBR": "Adverb, comparative",
    "RBS": "Adverb, superlative",
    "RP": "Particle",
    "SYM": "Symbol",
    "TO": "to",
    "UH": "Interjection",
    "VB": "Verb, base form",
    "VBD": "Verb, past tense",
    "VBG": "Verb, gerund or present participle",
    "VBN": "Verb, past participle",
    "VBP": "Verb, non-3rd person singular present",
    "VBZ": "Verb, 3rd person singular present",
    "WDT": "Wh-determiner",
    "WP": "Wh-pronoun",
    "WP$": "Possessive wh-pronoun",
    "WRB": "Wh-adverb ",
}

FIRST_PERSON_PRONOUNS = [
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
]

SECOND_PERSON_PRONOUNS = ["you", "your", "yours", "yourself", "yourselves"]


def sentence_limited_entity_pattern(entity_label):
    """
    if you don't add the sentence boundary constraint and just use a "+",
    weird things happen when there's a WILDCARD after these entities and
    the matches will end up crossing sentences in ways you wouldn't expect
    """
    return [
        {"ENT_TYPE": entity_label},
        {"ENT_TYPE": entity_label, "_": {"is_sent_start": False}, "OP": "*"},
    ]


WILDCARD = {"_": {"is_sent_start": False}, "OP": "*"}


################################
#    FUNDING ROUND PATTERNS    #
################################
GENERIC_DESCRIPTORS = [
    [
        {"LOWER": "series", "OP": "?"},
        {"LOWER": "pre", "OP": "?"},
        {"ORTH": "-", "OP": "?"},
        {"LOWER": "seed"},
    ],
    [{"LOWER": "growth"}],
    [{"LOWER": "mezzanine"}],
    [{"LOWER": "angel"}],
    [{"LOWER": "equity"}],
    [{"LOWER": "debt"}],
    [{"LOWER": "crowd"}],
    [{"LOWER": "round"}, {"LOWER": "of"}],
]

SERIES_DESCRIPTORS = [
    [
        {"LOWER": "series"},
        {"ORTH": "-", "OP": "?"},
        {"ORTH": letter},
        {"ORTH": "-", "OP": "?"},
        {"IS_DIGIT": True, "OP": "?"},
    ]
    for letter in [chr(i) for i in range(ord("A"), ord("M"))]
]
SERIES_DESCRIPTORS += [
    [
        {"LOWER": "pre"},
        {"ORTH": "-"},
        {"ORTH": letter},
        {"ORTH": "-", "OP": "?"},
        {"IS_DIGIT": True, "OP": "?"},
    ]
    for letter in [chr(i) for i in range(ord("A"), ord("M"))]
]
SERIES_DESCRIPTORS += [
    [{"LOWER": "series"}, {"ORTH": "-", "OP": "?"}, {"ORTH": f"{letter}-{number}"}]
    for letter in [chr(i) for i in range(ord("A"), ord("M"))]
    for number in range(1, 4)
]
SERIES_DESCRIPTORS += [
    [{"LOWER": "series"}, {"ORTH": "-", "OP": "?"}, {"ORTH": f"{letter}{number}"}]
    for letter in [chr(i) for i in range(ord("A"), ord("M"))]
    for number in range(1, 4)
]

BASES = [
    [{"LOWER": "funding"}],
    [{"LOWER": "fundraising"}],
    [{"LOWER": "fund"}, {"ORTH": "-", "OP": "?"}, {"LOWER": "raising"}],
    [{"LOWER": "financing"}],
    [{"LOWER": "investment"}],
    [{"LOWER": "crowdfunding"}],
]

ORDINAL_PHRASES = [
    [{"ENT_TYPE": "ORDINAL"}, {"ORTH": "-", "OP": "?"}]
    + [{"LOWER": "round", "OP": "?"}, {"LOWER": "of", "OP": "?"}]
    + base
    + [{"LOWER": "round", "OP": "?"}]
    for base in BASES
]

COMPOUND_PHRASES = [
    descriptor
    + [{"LOWER": "round", "OP": "?"}, {"LOWER": "of", "OP": "?"}]
    + base
    + [{"LOWER": "round", "OP": "?"}]
    for descriptor in GENERIC_DESCRIPTORS + SERIES_DESCRIPTORS
    for base in BASES
]

MONEY_PHRASES = [
    sentence_limited_entity_pattern("MONEY")
    + [{"POS": "ADP", "OP": "?", "_": {"is_sent_start": False}}, WILDCARD]
    + base
    for base in BASES
]

FUNDING_ROUND_PATTERNS = SERIES_DESCRIPTORS + ORDINAL_PHRASES + COMPOUND_PHRASES + MONEY_PHRASES


################################
#     ACQUISITION PATTERNS     #
################################
ACQUISITION_TERMS = [
    [{"LOWER": "acquisition"}],
    [{"LEMMA": "acquire"}],
    [{"LEMMA": "buy"}],
    [{"LEMMA": "purchase"}],
    [{"LEMMA": "take"}, {"LOWER": "over"}],
]

ORG_OR_PERSON = [
    [{"ENT_TYPE": "PERSON"}],
    [{"ENT_TYPE": "ORG"}],
]

ACQUISITION_PATTERNS = [
    acquirer + [WILDCARD] + [{"LEMMA": "have", "OP": "?"}] + word + [WILDCARD] + acquired
    for acquirer in ORG_OR_PERSON
    for acquired in ORG_OR_PERSON
    for word in ACQUISITION_TERMS
]

################################
#        CHIEF PATTERNS        #
################################
PERSON = sentence_limited_entity_pattern("PERSON")
ORG = sentence_limited_entity_pattern("ORG")

OPT_COMMA = [{"LOWER": ",", "OP": "?"}]
OPT_AP_S = [{"LOWER": "'s", "OP": "?"}]
OPT_THE = [{"LOWER": "the", "OP": "?"}]
OPT_PREP = [{"POS": "ADP", "OP": "?"}]
OPT_BE = [{"LEMMA": "be", "OP": "?"}]

WILDCARD_PO = [
    {
        "_": {"is_sent_start": False},
        "ENT_TYPE": {"NOT_IN": ["PERSON", "ORG"]},
        "OP": "*",
    }
]
PREPS = [[{"LOWER": prep}] for prep in ["of", "at", "for"]]


def chief_patterns(role_name):
    character = role_name[0]

    CHIEF_TERMS = [
        [{"LOWER": "c" + character + "o"}],
        [{"LOWER": "c." + character + ".o."}],
        [
            {"LOWER": "c"},
            {"LOWER": character},
            {"LOWER": "o"},
        ],
        [
            {"LOWER": "c."},
            {"LOWER": character + "."},
            {"LOWER": "o."},
        ],
        [
            {"LOWER": "c"},
            {"LOWER": "."},
            {"LOWER": character},
            {"LOWER": "."},
            {"LOWER": "o"},
            {"LOWER": "."},
        ],
        [{"LOWER": "chief"}, {"LOWER": role_name}, {"LOWER": "officer", "OP": "?"}],
    ]

    CHIEF_ACTIONS = [
        [{"LEMMA": "bring"}, {"POS": "ADP"}],
        [{"LEMMA": "hire"}],
        [{"LEMMA": "add"}],
        [{"LEMMA": "name"}],  # could add POS: VERB here but it's error-prone
    ]

    CHIEF_PATTERNS = (
        [ORG + OPT_AP_S + WILDCARD_PO + c + PERSON for c in CHIEF_TERMS]
        + [chief + prep + ORG + PERSON for chief in CHIEF_TERMS for prep in PREPS]
        + [PERSON + OPT_COMMA + OPT_THE + WILDCARD_PO + ORG + WILDCARD_PO + c for c in CHIEF_TERMS]
        + [PERSON + OPT_COMMA + ORG + OPT_AP_S + c for c in CHIEF_TERMS]
        + [
            PERSON + OPT_COMMA + OPT_THE + WILDCARD_PO + c + prep + ORG
            for c in CHIEF_TERMS
            for prep in PREPS
        ]
        + [
            ORG + action + PERSON + [{"LOWER": "as"}] + OPT_THE + c
            for action in CHIEF_ACTIONS
            for c in CHIEF_TERMS
        ]
        + [
            PERSON + OPT_BE + OPT_THE + WILDCARD_PO + c + prep + ORG
            for c in CHIEF_TERMS
            for prep in PREPS
        ]
    )

    return CHIEF_PATTERNS


CEO_PATTERNS = chief_patterns("executive")

CTO_PATTERNS = chief_patterns("technology")

CFO_PATTERNS = chief_patterns("financial")

################################
#    BOARD MEMBER PATTERNS     #
################################
BOARD_PERSON = sentence_limited_entity_pattern("PERSON")
BOARD_ORG = sentence_limited_entity_pattern("ORG")

BOARD_WILDCARD = [
    {
        "_": {"is_sent_start": False},
        "ENT_TYPE": {"NOT_IN": ["PERSON", "ORG"]},
        "OP": "*",
    }
]

# board
# board member
# board of directors
# member of board of directors
BOARD_TERMS = [
    [
        {"LOWER": "company", "OP": "?"},
        {"LOWER": "board"},
        {"LOWER": "member", "OP": "?"},
    ],
    [
        {"LOWER": "member", "OP": "?"},
        {"LOWER": "of", "OP": "?"},
        {"LOWER": "the", "OP": "?"},
        {"LOWER": "board"},
        {"LOWER": "of"},
        {"LOWER": "directors"},
    ],
]

# brought on the BOARD_COMPANY
# joined the BOARD_COMPANY
# added to the BOARD_COMPANY
# elected to the BOARD_COMPANY
# appointed to the BOARD_COMPANY
# named to the BOARD_COMPANY
BOARD_ACTIONS = [
    [{"LEMMA": "bring"}, {"POS": "ADP"}],
    [{"LEMMA": "join"}],
    [{"LEMMA": "add"}],
    [{"LEMMA": "elect"}],
    [{"LEMMA": "appoint"}],
    [{"LEMMA": "name"}],  # could add POS: VERB here but it's error-prone
]

# Comp board Name
# board of Comp Name
# Name, (the) Comp board
# Name, (the) Comp's board
# Name, (the) board of Comp
# Comp Action Name as board member
# Comp Action Name as member of the board
# Comp's board Action Name
# Name to the board of Comp
BOARD_PATTERNS = (
    [BOARD_ORG + OPT_AP_S + BOARD_WILDCARD + board + BOARD_PERSON for board in BOARD_TERMS]
    + [board + prep + BOARD_ORG + BOARD_PERSON for board in BOARD_TERMS for prep in PREPS]
    + [
        BOARD_PERSON + OPT_COMMA + OPT_THE + BOARD_WILDCARD + BOARD_ORG + BOARD_WILDCARD + board
        for board in BOARD_TERMS
    ]
    + [BOARD_PERSON + OPT_COMMA + BOARD_ORG + OPT_AP_S + board for board in BOARD_TERMS]
    + [
        BOARD_PERSON + OPT_COMMA + OPT_THE + BOARD_WILDCARD + board + prep + BOARD_ORG
        for board in BOARD_TERMS
        for prep in PREPS
    ]
    + [
        BOARD_ORG + action + BOARD_PERSON + [{"LOWER": "as"}] + OPT_THE + board
        for action in BOARD_ACTIONS
        for board in BOARD_TERMS
    ]
    + [
        BOARD_PERSON + OPT_BE + OPT_THE + BOARD_WILDCARD + board + prep + BOARD_ORG
        for board in BOARD_TERMS
        for prep in PREPS
    ]
)


################################
#     EDUCATED AT PATTERNS     #
################################
def degree_patterns(name):
    return [
        {"LOWER": "a", "OP": "?"},
        {"LOWER": name},
        {"LOWER": "'s", "OP": "?"},
        {"LOWER": "degree", "OP": "?"},
        {"POS": "ADP"},  # of, in
        {"POS": "noun", "OP": "*"},
    ]


EDUCATED_DEGREE_TERMS = [degree_patterns(n) for n in ["associate", "bachelor", "master", "doctor"]]

EDUCATED_TERMS = [
    [{"LOWER": "a", "OP": "?"}, {"LOWER": "degree"}],
    [{"LOWER": "education"}],
    [{"LOWER": "training"}],
    [{"LOWER": "alumni"}],
    [{"LOWER": "alum"}],
    [{"LOWER": "alma"}, {"LOWER": "mater"}],
] + EDUCATED_DEGREE_TERMS

EDUCATED_ACTIONS = [
    [{"LEMMA": "enrol"}],
    [{"LEMMA": "train"}],
    [{"LEMMA": "receive"}],
    [{"LEMMA": "earn"}],
    [{"POS": "VERB", "OP": "?"}, {"LEMMA": "educate"}],
    [{"LEMMA": "attend"}],
]

# PERSON educated at ORG
# PERSON trained at ORG
# ORG alumni PERSON
# PERSON's alma mater ORG
EDUCATED_PATTERNS = (
    [PERSON + action + OPT_PREP + ORG for action in EDUCATED_ACTIONS]
    + [
        PERSON + action + term + OPT_PREP + ORG
        for action in EDUCATED_ACTIONS
        for term in EDUCATED_TERMS
    ]
    + [PERSON + OPT_AP_S + term + ORG for term in EDUCATED_TERMS]
    + [ORG + OPT_AP_S + term + PERSON for term in EDUCATED_TERMS]
    + [ORG + [{"LOWER": "-", "OP": "?"}] + action + PERSON for action in EDUCATED_ACTIONS]
)

################################
#      FOUNDED BY PATTERNS     #
################################
# FOUNDED_DATE = sentence_limited_entity_pattern("DATE")
#
# def founders(n):
#     p = []
#     for i in range(1, n + 1):
#         p.append(PERSON)
#         if n > 1:
#             p.append({"LOWER": ",", "OPT": "?"})
#             p.append({"LOWER": "and", "OPT": "?"})
#     return p
#
# FOUNDER_PEOPLE = founders(1) #[founders(i) for i in range(1,2)]

FOUNDED_TERMS = [
    [{"LOWER": "founder"}],
    [{"LOWER": "establisher"}],
    [{"LOWER": "conciever"}],
    [{"LOWER": "initiator"}],
    [{"LOWER": "institutor"}],
    [{"LOWER": "originator"}],
]

FOUNDED_ACTIONS = [
    # [{"LOWER": "co", "OP": "?"}, {"LOWER": "-", "OP": "?"},{"LEMMA": "found"}],
    [{"LEMMA": "cofound"}],
    [{"LEMMA": "found"}],
    [{"LEMMA": "incorporate"}],
    [{"LEMMA": "start"}],
    [{"LEMMA": "establish"}],
    [{"LEMMA": "begin"}],
    [{"LEMMA": "concieve"}],
    [{"LEMMA": "initiate"}],
    [{"LEMMA": "institute"}],
    [{"LEMMA": "set"}, {"LOWER": "up"}],
]

# Apple was founded by Steve Jobs.
# [co-]founded, incorporated, started, established, begun, conceived,
# initiated, instituted, set up,

FOUNDER_OF_PATTERNS = (
    [ORG + OPT_BE + action + [{"LOWER": "by"}] + PERSON for action in FOUNDED_ACTIONS]
    + [PERSON + OPT_COMMA + OPT_BE + action + OPT_PREP + ORG for action in FOUNDED_ACTIONS]
    + [PERSON + OPT_COMMA + OPT_BE + OPT_THE + term + OPT_PREP + ORG for term in FOUNDED_TERMS]
    + [ORG + OPT_COMMA + term + PERSON for term in FOUNDED_TERMS]
)
################################
#      LOCATED PATTERNS     #
################################
LOCATIONS = [
    sentence_limited_entity_pattern("GPE"),
    sentence_limited_entity_pattern("LOC"),
]

LOCATED_ACTIONS = [
    [{"LEMMA": "headquarter"}],
    [{"LEMMA": "base"}],
    [{"LEMMA": "locate"}],
    [{"LEMMA": "situate"}],
]

LOCATED_IN_PATTERNS = (
    [ORG + OPT_COMMA + OPT_PREP + location for location in LOCATIONS]
    + [
        ORG + OPT_COMMA + OPT_BE + action + OPT_PREP + location
        for action in LOCATED_ACTIONS
        for location in LOCATIONS
    ]
    + [
        ORG
        + OPT_COMMA
        + OPT_BE
        + [{"LOWER": "a", "OP": "?"}]
        + location
        + [{"LOWER": "-", "OP": "?"}]
        + action
        for action in LOCATED_ACTIONS
        for location in LOCATIONS
    ]
    + [
        location + [{"LOWER": "-", "OP": "?"}] + action + ORG
        for action in LOCATED_ACTIONS
        for location in LOCATIONS
    ]
)


################################
#    GENERIC EVENT PATTERNS    #
################################
MONEY_WITH_DATE = [
    [
        {"ENT_TYPE": "MONEY"},
        {"_": {"is_sent_start": False}, "OP": "*"},
        {"ENT_TYPE": "DATE"},
    ],
    [
        {"ENT_TYPE": "DATE"},
        {"_": {"is_sent_start": False}, "OP": "*"},
        {"ENT_TYPE": "MONEY"},
    ],
    [
        {"ENT_TYPE": "MONEY"},
        {"_": {"is_sent_start": False}, "OP": "*"},
        {"ENT_TYPE": "TIME"},
    ],
    [
        {"ENT_TYPE": "TIME"},
        {"_": {"is_sent_start": False}, "OP": "*"},
        {"ENT_TYPE": "MONEY"},
    ],
]

AGENTIVE_ENTITIES = [
    [{"ENT_TYPE": "PERSON"}],
    [{"ENT_TYPE": "ORG"}],
    [{"ENT_TYPE": "GPE"}],
    [{"ENT_TYPE": "NORP"}],
]

ANNOUNCE_TERMS = [
    # the lemmatizer isn't so good on headlines...
    [{"LOWER": "announce"}],
    [{"LOWER": "announcement"}],
    [{"LOWER": "announces"}],
    [{"LOWER": "announced"}],
    [{"LOWER": "announcing"}],
    [{"LOWER": "release", "POS": "VERB"}],
    [{"LOWER": "releases", "POS": "VERB"}],
    [{"LOWER": "released"}],
    [{"LOWER": "releasing"}],
    [{"LOWER": "publish"}],
    [{"LOWER": "publishes"}],
    [{"LOWER": "published"}],
    [{"LOWER": "publishing"}],
    [{"LOWER": "launch", "POS": "VERB"}],
    [{"LOWER": "launches", "POS": "VERB"}],
    [{"LOWER": "launched"}],
    [{"LOWER": "launching"}],
]

# FOUNDING_DATES = [
#     [{"LOWER": "founded"}, {"LOWER": "in"}, {"ENT_TYPE": "DATE"}],
#     [{"LOWER": "founded"}, {"LOWER": "on"}, {"ENT_TYPE": "DATE"}],
# ]

# NATIVE_EVENTS = [
#     [{"ENT_TYPE": "EVENT"}]
# ]

ENTITY_ANNOUNCEMENTS = [
    entity + [{"_": {"is_sent_start": False}, "OP": "*"}] + announce_term
    for entity in AGENTIVE_ENTITIES
    for announce_term in ANNOUNCE_TERMS
]
ENTITY_ANNOUNCEMENTS += [
    announce_term + [{"_": {"is_sent_start": False}, "OP": "*"}] + entity
    for entity in AGENTIVE_ENTITIES
    for announce_term in ANNOUNCE_TERMS
]

DATED_ANNOUNCEMENTS = [
    [{"ENT_TYPE": "DATE"}, {"_": {"is_sent_start": False}, "OP": "*"}] + announce_term
    for announce_term in ANNOUNCE_TERMS
]
DATED_ANNOUNCEMENTS += [
    announce_term + [{"_": {"is_sent_start": False}, "OP": "*"}, {"ENT_TYPE": "DATE"}]
    for announce_term in ANNOUNCE_TERMS
]

GENERIC_EVENT_PATTERNS = (
    MONEY_WITH_DATE
    # + FOUNDING_DATES
    # + NATIVE_EVENTS
    + ENTITY_ANNOUNCEMENTS
    + DATED_ANNOUNCEMENTS
)


################################
#    TICKER SYMBOL PATTERNS    #
################################

TICKER_PATTERNS = [
    [
        {"ORTH": "("},
        {},
        {"ORTH": ":"},
        {"IS_UPPER": True},
        {"ORTH": ")"},
    ],
    [
        {"ORTH": "("},
        {},
        {"ORTH": ":"},
        {"IS_DIGIT": True},  # TYO uses numbers as symbols
        {"ORTH": ")"},
    ],
]


################################
#      DATELINE PATTERNS       #
################################
LOCATIONS = [
    [
        {
            "_": {"is_graf_start": True},
            "ENT_TYPE": {"IN": ["LOCATION", "GPE"]},
            "OP": "+",
        }
    ],
    [
        {"_": {"is_graf_start": True}, "IS_PUNCT": True, "OP": "+"},
        {"IS_UPPER": True, "OP": "+"},
    ],
    [{"_": {"is_graf_start": True}, "IS_UPPER": True, "OP": "+"}],
]

TERMINATORS = [[{"ORTH": {"IN": ["-", "--"]}}]]

DATELINE_PATTERNS = [
    location + [{"OP": "*", "ORTH": {"NOT_IN": ["-", "--"]}}] + terminator
    for location in LOCATIONS
    for terminator in TERMINATORS
]
