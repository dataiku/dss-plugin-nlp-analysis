# -*- coding: utf-8 -*-
# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

from ontology_tagger import Tagger
import spacy
from spacy.matcher import PhraseMatcher
import pandas as pd
from language_support import SUPPORTED_LANGUAGES_SPACY
from spacy_tokenizer import MultilingualTokenizer
from plugin_io_utils import replace_nan_values


def test_list_sentences():
    ontology_df = pd.DataFrame({"tag": ["tag1"], "keyword": ["keyword1"]})
    tagger = Tagger(
        ontology_df=ontology_df,
        tag_column=None,
        category_column=None,
        keyword_column=None,
        language="en",
        lemmatization=None,
        case_insensitivity=None,
        normalization=None,
    )
    text_df = pd.DataFrame({"text": [float("nan")]})
    tagger._create_pipelines([tagger.language])
    text_df = tagger._add_column_of_splitted_sentences(
        text_df=text_df, text_column="text", language_column=None
    )
    assert text_df[tagger.splitted_sentences_column].iloc[0] == []


def test_missing_keyword_in_ontology():
    ontology_df = pd.DataFrame(
        {"tag": ["tag1", "tag2", "tag3"], "keyword": [float("nan"), "keyword2", ""]}
    )
    tagger = Tagger(
        ontology_df=ontology_df,
        tag_column="tag",
        category_column=None,
        keyword_column="keyword",
        language="en",
        lemmatization=None,
        case_insensitivity=None,
        normalization=None,
    )
    tagger._create_pipelines([tagger.language])
    keywords = ontology_df["keyword"].values.tolist()
    tags = ontology_df["tag"].values.tolist()
    patterns = tagger._get_patterns(keywords)
    tagger._match_no_category(tagger.language, tags, keywords)
    assert len(tagger.matcher_dict[tagger.language]) == 1


def test_keyword_tokenization():
    from ontology_tagger import Tagger
    import spacy
    from spacy.matcher import PhraseMatcher
    import pandas as pd

    ontology_df = pd.DataFrame(
        {
            "tag": ["tag1", "tag2", "tag3", "tag4"],
            "keyword": ["keyword", "keyword two", "N.Y", "1.1.1.1"],
            "category": ["category1", "category2", "category3", "category4"],
        }
    )
    tagger = Tagger(
        ontology_df=ontology_df,
        tag_column="tag",
        category_column="category",
        keyword_column="keyword",
        language="en",
        lemmatization=None,
        case_insensitivity=None,
        normalization=None,
    )
    tags = ontology_df["tag"].values.tolist()
    keywords = ontology_df["keyword"].values.tolist()
    patterns = tagger._get_patterns(keywords)
    tagger._create_pipelines([tagger.language])
    tagger._match_with_category(patterns, tags, keywords)
    ruler = tagger.nlp_dict[tagger.language].get_pipe("entity_ruler")
    for elt in ruler.patterns:
        assert elt["pattern"] in tagger.keyword_to_tag["en"]


def test_pipeline_components():
    ontology_df = pd.DataFrame({"tag": ["tag1"], "keyword": ["keyword1"]})
    tagger = Tagger(
        ontology_df=ontology_df,
        tag_column="tag",
        category_column=None,
        keyword_column="keyword",
        language="language_column",
        lemmatization=None,
        case_insensitivity=None,
        normalization=None,
    )
    tagger._create_pipelines(SUPPORTED_LANGUAGES_SPACY.keys())
    for language in tagger.nlp_dict:
        assert tagger.nlp_dict[language].pipe_names == ["sentencizer"]