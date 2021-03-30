# -*- coding: utf-8 -*-
# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

from tagger import Tagger
import spacy
from spacy.matcher import PhraseMatcher
import pandas as pd


def test_tagger():
    # TODO write a real test
    tagger = Tagger()
    assert True


def test_list_sentences():
    tagger = Tagger(
        pd.DataFrame({"text": [float("nan")]}),
        None,
        "text",
        "en",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    tagger.nlp_dict["en"] = spacy.load("en_core_web_sm")
    tagger.text_df["splitted_sentences"] = tagger.text_df.apply(
        tagger._list_sentences, axis=1
    )


def test_missing_keyword_in_ontology():

    tagger = Tagger(
        None,
        pd.DataFrame(
            {"tag": ["tag1", "tag2", "tag3"], "keyword": [float("nan"), "keyword2", ""]}
        ),
        "text",
        "en",
        None,
        "tag",
        None,
        "keyword",
        None,
        None,
        None,
        None,
    )
    tagger.nlp_dict["en"] = spacy.load("en_core_web_sm")
    tagger.nlp_dict["en"].add_pipe("sentencizer")
    tagger._get_patterns()
    tagger._match_no_category("en")