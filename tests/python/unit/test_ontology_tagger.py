# -*- coding: utf-8 -*-
# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

from spacy.matcher import PhraseMatcher
import pandas as pd
from ontology_tagging.ontology_tagger import Tagger
from utils.language_support import SUPPORTED_LANGUAGES_SPACY


def test_create_matcher_missing_keywords():
    """Test behavior with Nan/empty strings values in ontology"""
    ontology_df = pd.DataFrame(
        {"tag": ["tag1", "tag2", "tag3"], "keyword": [float("nan"), "keyword2", ""]}
    )
    tagger = Tagger(
        ontology_df=ontology_df,
        tag_column="tag",
        category_column=None,
        keyword_column="keyword",
        language="en",
    )
    tagger._initialize_tokenizer([tagger.language])
    keywords = ontology_df["keyword"].values.tolist()
    tags = ontology_df["tag"].values.tolist()
    tagger._match_no_category(tags, keywords)
    assert len(tagger._matcher_dict[tagger.language]) == 1


def test_keywords_tokenization():
    """Test equality between keywords in the matcher and keywords in keyword_to_tag dictionary"""
    ontology_df = pd.DataFrame(
        {
            "tag": ["tag1", "tag2", "tag3", "tag4"],
            "keyword": ["keyword", "keyword two", "N.Y", "1.1.1.1"],
        }
    )
    tagger = Tagger(
        ontology_df=ontology_df,
        tag_column="tag",
        category_column=None,
        keyword_column="keyword",
        language="en",
    )
    tags = ontology_df["tag"].values.tolist()
    keywords = ontology_df["keyword"].values.tolist()
    tagger._initialize_tokenizer([tagger.language])
    tagger._match_no_category(tags, keywords)
    matcher = tagger._matcher_dict[tagger.language]
    patterns = tagger._tokenize_keywords(tagger.language, tags, keywords)
    for elt in patterns:
        assert elt.text in tagger._keyword_to_tag["en"]


def test_initialize_tokenizer():
    """Test content of each tokenizer"""
    ontology_df = pd.DataFrame({"tag": ["tag1"], "keyword": ["keyword1"]})
    tagger = Tagger(
        ontology_df=ontology_df,
        tag_column="tag",
        category_column=None,
        keyword_column="keyword",
        language="language_column",
    )
    tagger._initialize_tokenizer(SUPPORTED_LANGUAGES_SPACY.keys())
    for language in tagger.tokenizer.spacy_nlp_dict:
        assert tagger.tokenizer.spacy_nlp_dict[language].pipe_names == ["sentencizer"]


def test_matching_in_lowercase():
    """Test matching for the option 'normalize_case'"""
    ontology_df = pd.DataFrame(
        {"tag": ["tag1", "tag2"], "keyword": ["My KeYword", "other keyword"]}
    )
    text_df = pd.DataFrame(
        {
            "text": [
                "I have my keyword in this sentence. I have an oTHer keyWord in the second sentence."
            ]
        }
    )
    tagger = Tagger(
        ontology_df=ontology_df,
        tag_column="tag",
        category_column=None,
        keyword_column="keyword",
        language="en",
        normalize_case=True,
    )
    df = tagger.tag_and_format(
        text_df=text_df,
        text_column="text",
        output_format="one_row_per_match",
        languages=["en"],
    )
    assert len(df["tag_keyword"]) == 2 == len(df["tag_sentence"]) == len(df["tag"])
    
def test_matching_normalize_diacritics():
    """Test matching for the option 'normalize_diacritics'"""
    ontology_df = pd.DataFrame(
            {"tag": ["tag1"], "keyword": ["ÄâêËùûôçèîÏìàñ"]}
    )
    text_df = pd.DataFrame(
        {
            "text": [
                "The keyword is AaeEuuoceiIian."
            ]
        }
    )
    tagger = Tagger(
        ontology_df=ontology_df,
        tag_column="tag",
        category_column=None,
        keyword_column="keyword",
        language="en",
        normalize_diacritics=True,
    )
    df = tagger.tag_and_format(
        text_df=text_df,
        text_column="text",
        output_format="one_row_per_match",
        languages=["en"],
    )
    assert len(df["tag_keyword"]) == 1 == len(df["tag_sentence"]) == len(df["tag"])