# -*- coding: utf-8 -*-
# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information
from ontology_tagging.ontology_tagger import Tagger
import pandas as pd
import pytest


@pytest.mark.parametrize(
    "keyword,language,lemma",
    [
        ("worked", "en","work"),
        ("trabajó", "es","trabajar"),
        ("jobbet", "nb","jobbet"),
        ("travaillé", "fr","travailler"),
        ("hat funktioniert", "de","haben funktionieren"),
        ("работал", "ru","работать"),
        ("zadziałało", "pl","zadziałać"),
        ("treballat", "ca","treballar"),
        ("pracoval", "cs","pracovat"),
        ("arbejdede", "da","arbejde"),
        ("radio", "hr","raditi"),
        ("dolgozott", "hu","dolgozik"),
        ("bekerja", "id","kerja"),
        ("lavorato", "it","lavorare"),
        ("geschafft", "lb","schaffen"),
        ("dirbo", "lt","dirbti"),
        ("trabalhado", "pt","trabalhar"),
        ("a lucrat", "ro","avea lucra"),
        ("радио", "sr","радити"),
        ("nagtrabaho", "tl","nagtrabaho"),
        ("çalıştı", "tr","çalış"),
        ("کام کیا", "ur","کام کَیا"),
        ("কাজ করছে", "bn","কাজ করছে"),
        ("δούλεψε", "el","δούλεψε"),
        ("کار کرد", "fa","کار کرد"),
        ("работел", "mk","работел"),
        ("werkte", "nl","werkte"),
        ("arbetade", "sv","arbeta"),
    ],
)
def test_lemmatize_keywords(keyword, language, lemma):
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
    assert "".join([token.lemma_ for token in tokenized_keyword[0].sents]) == lemma