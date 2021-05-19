import pandas as pd
from ontology_tagging.ontology_tagger import Tagger


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