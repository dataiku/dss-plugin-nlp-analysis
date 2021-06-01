# -*- coding: utf-8 -*-
# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

import os

import pytest

import pandas as pd

from nlp.ontology_tagging.spacy_tokenizer import MultilingualTokenizer


def test_tokenize_df_english():
    input_df = pd.DataFrame({"input_text": ["I hope nothing. I fear nothing. I am free. 💩 😂 #OMG"]})
    tokenizer = MultilingualTokenizer()
    output_df = tokenizer.tokenize_df(df=input_df, text_column="input_text", language="en")
    tokenized_document = output_df[tokenizer.tokenized_column][0]
    assert len(tokenized_document) == 15


def test_tokenize_df_japanese():
    input_df = pd.DataFrame({"input_text": ["期一会。 異体同心。 そうです。"]})
    tokenizer = MultilingualTokenizer()
    output_df = tokenizer.tokenize_df(df=input_df, text_column="input_text", language="ja")
    tokenized_document = output_df[tokenizer.tokenized_column][0]
    assert len(tokenized_document) == 9


def test_tokenize_df_multilingual():
    input_df = pd.DataFrame(
        {
            "input_text": [
                "I hope nothing. I fear nothing. I am free.",
                " Les sanglots longs des violons d'automne",
                "子曰：“學而不思則罔，思而不學則殆。”",
                "期一会。 異体同心。 そうです。",
            ],
            "language": ["en", "fr", "zh", "ja"],
        }
    )
    tokenizer = MultilingualTokenizer()
    output_df = tokenizer.tokenize_df(df=input_df, text_column="input_text", language_column="language")
    tokenized_documents = output_df[tokenizer.tokenized_column]
    tokenized_documents_length = [len(doc) for doc in tokenized_documents]
    assert tokenized_documents_length == [12, 8, 19, 9]


def test_tokenize_df_long_text():
    input_df = pd.DataFrame({"input_text": ["Long text"]})
    tokenizer = MultilingualTokenizer(max_num_characters=1)
    with pytest.raises(ValueError):
        tokenizer.tokenize_df(df=input_df, text_column="input_text", language="en")

        
def test_create_spacy_tokenizer_no_model():
    input_df = pd.DataFrame({"input_text": ["I hope nothing. I fear nothing. I am free. 💩 😂 #OMG"]})
    tokenizer = MultilingualTokenizer(add_pipe_components=["sentencizer"],enable_pipe_components=["sentencizer"],use_models=False)
    nlp = tokenizer._create_spacy_tokenizer("en")
    assert nlp.pipe_names == ["sentencizer"]
    
    
def test_create_spacy_tokenizer_model():
    input_df = pd.DataFrame({"input_text": ["I hope nothing. I fear nothing. I am free. 💩 😂 #OMG"]})
    tokenizer = MultilingualTokenizer(add_pipe_components=["sentencizer"],enable_pipe_components=["tagger","sentencizer"],use_models=True)
    nlp = tokenizer._create_spacy_tokenizer("en")
    assert nlp.pipe_names == ["tagger","sentencizer"]    
    
def test_activate_components_to_lemmatize_no_model():
    input_df = pd.DataFrame({"input_text": ["I hope nothing. I fear nothing. I am free. 💩 😂 #OMG"]})
    tokenizer = MultilingualTokenizer(enable_pipe_components=["sentencizer","tagger"],use_models=False)
    tokenizer.spacy_nlp_dict["en"] = tokenizer._create_spacy_tokenizer("en")
    tokenizer._activate_components_to_lemmatize("en")
    assert tokenizer.spacy_nlp_dict["en"].pipe_names == ["lemmatizer"]
    

def test_activate_components_to_lemmatize_model():
    input_df = pd.DataFrame({"input_text": ["I hope nothing. I fear nothing. I am free. 💩 😂 #OMG"]})
    tokenizer = MultilingualTokenizer(enable_pipe_components=["sentencizer","tagger"],use_models=True)
    tokenizer.spacy_nlp_dict["en"] = tokenizer._create_spacy_tokenizer("en")
    tokenizer._activate_components_to_lemmatize("en")
    assert tokenizer.spacy_nlp_dict["en"].pipe_names == ["tok2vec","tagger","attribute_ruler","lemmatizer"]
    