# -*- coding: utf-8 -*-
# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information
from ontology_tagger import Tagger
import pandas as pd
import pytest 

@pytest.mark.parametrize("keyword,language", [("worked","en"),("trabajó","es"),("jobbet","nb"),("travaillé","fr"),("hat funktioniert","de"),("работал","ru"), ("zadziałało","pl"),("treballat", "ca"),("pracoval","cs"),("arbejdede","da"),("radio","hr"),("dolgozott", "hu"),("bekerja","id"),("lavorato","it"),("geschafft","lb"),("dirbo","lt"),("trabalhado","pt"),("a lucrat","ro"),("радио","sr"),("nagtrabaho","tl"),("çalıştı","tr"),("کام کیا","ur"),("কাজ করছে","bn"),("δούλεψε","el"),("کار کرد","fa"),("работел","mk"),("werkte","nl"),("arbetade","sv")])
def test_lemmatize_keywords(keyword, language):
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