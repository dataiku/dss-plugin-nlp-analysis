# -*- coding: utf-8 -*-
# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

from ontology_tagger import Tagger
import spacy
from spacy.matcher import PhraseMatcher
import pandas as pd

    
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
    assert isinstance(tagger, Tagger)
    text_df = pd.DataFrame({"text": [float("nan")]})
    text_column = "text"
    tagger.nlp_dict[tagger.language] = spacy.load("en_core_web_sm")
    text_df["splitted_sentences"] = text_df.apply(
        tagger._list_sentences, args=[text_column], axis=1
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
        case_insensitivity=None,
        normalization=None,
    )
    assert isinstance(tagger, Tagger)
    tagger.nlp_dict[tagger.language] = spacy.load("en_core_web_sm")
    tagger.nlp_dict[tagger.language].add_pipe("sentencizer")
    keywords = ontology_df["keyword"].values.tolist()
    tags = ontology_df["tag"].values.tolist()
    patterns = tagger._get_patterns(keywords)
    assert len(patterns) == 1 == len(ontology_df)
    tagger._match_no_category(tagger.language,tags,keywords)
    assert isinstance(tagger.matcher_dict[tagger.language], PhraseMatcher)
    assert len(tagger.matcher_dict[tagger.language]) == 1

def test_tokenization():
    from ontology_tagger import Tagger
    import spacy
    from spacy.matcher import PhraseMatcher
    import pandas as pd

    ontology_df = pd.DataFrame(
        {"tag": ["tag1", "tag2", "tag3","tag4"], "keyword": ["keyword", "keyword two", "N.Y","1.1.1.1"],"category":["category1","category2","category3","category4"]}
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
    tagger.nlp_dict[tagger.language] = spacy.load("en_core_web_sm")
    tagger.nlp_dict[tagger.language].add_pipe("sentencizer")
    tagger._match_with_category(patterns,tags,keywords)
    ruler = tagger.nlp_dict[tagger.language].get_pipe("entity_ruler")
    assert len(ruler) == len(tagger.keyword_to_tag["en"])
    for elt in ruler.patterns:
        assert elt["pattern"] in tagger.keyword_to_tag["en"]
    