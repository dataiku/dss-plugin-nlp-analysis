# -*- coding: utf-8 -*-
# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

from ontology_tagger import Tagger
import spacy
from spacy.matcher import PhraseMatcher
import pandas as pd

    
def test_list_sentences():
    ontology_df=pd.DataFrame({"tag":["tag1"],"keyword":["keyword1"]})
    tagger = Tagger(
        ontology_df=ontology_df,
        tag_column=None,
        category_column=None,
        keyword_column=None,
        language="en",
        lemmatization=None,
        case_insensitive=None,
        normalization=None,
        output_format=None,
    )
    assert isinstance(tagger,Tagger)
    text_df = pd.DataFrame({"text":[float("nan")]})
    text_column = "text"
    tagger.nlp_dict[tagger.language] = spacy.load("en_core_web_sm")
    text_df["splitted_sentences"] = text_df.apply(
        tagger._list_sentences,args=[text_column], axis=1
    )
    assert text_df["splitted_sentences"].iloc[0] == []
    
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
        case_insensitive=None,
        normalization=None,
        output_format=None,
    )
    assert isinstance(tagger,Tagger)
    tagger.nlp_dict[tagger.language] = spacy.load("en_core_web_sm")
    tagger.nlp_dict[tagger.language].add_pipe("sentencizer")
    tagger._get_patterns()
    assert(len(tagger.patterns)==1==len(ontology_df))
    tagger._match_no_category(tagger.language)
    assert isinstance(tagger.matcher_dict[tagger.language],PhraseMatcher)
    assert len(tagger.matcher_dict[tagger.language])==1