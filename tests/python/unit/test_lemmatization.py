# -*- coding: utf-8 -*-
# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information
from ontology_tagger import Tagger
import pandas as pd


def lemmatize_keywords(keyword, language):
    tagger = Tagger(
        ontology_df=pd.DataFrame({"tag": ["verb"], "keyword": [keyword]}),
        tag_column="tag",
        keyword_column="keyword",
        category_column=None,
        language=language,
        lemmatization=True,
    )
    tagger._initialize_tokenizer([tagger.language])
    tag = tagger.ontology_df["tag"].tolist()
    keyword = tagger.ontology_df["keyword"].tolist()
    tokenized_keyword = tagger._tokenize_keywords(tagger.language, tag, keyword)
    assert len([token.lemma_ for token in tokenized_keyword[0].sents]) > 0


def test_lemmatize_en():
    lemmatize_keywords(keyword="worked", language="en")


def test_lemmatize_es():
    lemmatize_keywords(keyword="trabajó", language="es")


def test_lemmatize_nb():
    lemmatize_keywords(keyword="jobbet", language="nb")


def test_lemmatize_fr():
    lemmatize_keywords(keyword="travaillé", language="fr")


def test_lemmatize_de():
    lemmatize_keywords(keyword="hat funktioniert", language="de")


def test_lemmatize_ru():
    lemmatize_keywords(keyword="работал", language="ru")


def test_lemmatize_pl():
    lemmatize_keywords(keyword="zadziałało", language="pl")


def test_lemmatize_ca():
    lemmatize_keywords(keyword="treballat", language="ca")


def test_lemmatize_cs():
    lemmatize_keywords(keyword="pracoval", language="cs")


def test_lemmatize_da():
    lemmatize_keywords(keyword="arbejdede", language="da")


def test_lemmatize_hr():
    lemmatize_keywords(keyword="radio", language="hr")


def test_lemmatize_hu():
    lemmatize_keywords(keyword="dolgozott", language="hu")


def test_lemmatize_id():
    lemmatize_keywords(keyword="bekerja", language="id")


def test_lemmatize_it():
    lemmatize_keywords(keyword="lavorato", language="it")


def test_lemmatize_lb():
    lemmatize_keywords(keyword="geschafft", language="lb")


def test_lemmatize_lt():
    lemmatize_keywords(keyword="dirbo", language="lt")


def test_lemmatize_pt():
    lemmatize_keywords(keyword="trabalhado", language="pt")


def test_lemmatize_ro():
    lemmatize_keywords(keyword="a lucrat", language="ro")


def test_lemmatize_sr():
    lemmatize_keywords(keyword="радио", language="sr")


def test_lemmatize_tl():
    lemmatize_keywords(keyword="nagtrabaho", language="tl")


def test_lemmatize_tr():
    lemmatize_keywords(keyword="çalıştı", language="tr")


def test_lemmatize_ur():
    lemmatize_keywords(keyword="کام کیا", language="ur")


def test_lemmatize_bn():
    lemmatize_keywords(keyword="কাজ করছে", language="bn")


def test_lemmatize_el():
    lemmatize_keywords(keyword="δούλεψε", language="el")


def test_lemmatize_fa():
    lemmatize_keywords(keyword="کار کرد", language="fa")


def test_lemmatize_mk():
    lemmatize_keywords(keyword="работел", language="mk")


def test_lemmatize_nl():
    lemmatize_keywords(keyword="werkte", language="nl")


def test_lemmatize_sv():
    lemmatize_keywords(keyword="arbetade", language="sv")
