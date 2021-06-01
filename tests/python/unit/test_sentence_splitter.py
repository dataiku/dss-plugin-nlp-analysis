import pandas as pd
from nlp.ontology_tagging.ontology_tagger import Tagger


def test_split_sentences_nan_values():
    """Test behavior when splitting sentences with NaN value in the text column"""
    ontology_df = pd.DataFrame({"tag": ["tag1"], "keyword": ["keyword1"]})
    tagger = Tagger(
        ontology_df=ontology_df,
        tag_column=None,
        category_column=None,
        keyword_column=None,
        language="en",
    )
    text_df = pd.DataFrame({"text": [float("nan")]})
    tagger._initialize_tokenizer([tagger.language])
    text_df, text_column_tokenized = tagger._sentence_splitting(text_df, "text")
    assert text_df[text_column_tokenized].iloc[0] == []


def test_split_sentences_linebreaks():
    """Test behavior when splitting sentences with linebreaks in the text column"""
    ontology_df = pd.DataFrame({"tag": ["first", "second", "third"]})
    tagger = Tagger(
        ontology_df=ontology_df,
        tag_column="tag",
        category_column="None",
        keyword_column="tag",
        language="en",
    )
    text_df = pd.DataFrame(
        {
            "text": [
                'first line with carriage return\rsecond line with two linebreaks\n\nthird line with parenthesis)\nLast line'
            ]
        }
    )
    tagger._initialize_tokenizer([tagger.language])
    text_df, text_column_tokenized = tagger._sentence_splitting(text_df, "text")
    assert len(text_df[text_column_tokenized].tolist()[0]) == 4
    