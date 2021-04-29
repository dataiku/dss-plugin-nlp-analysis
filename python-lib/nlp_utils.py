from typing import AnyStr, Union
from spacy.tokens import Span, Doc


def get_keyword(text: AnyStr, normalize_case: bool) -> AnyStr:
    """Return text in its wanted-case form"""
    return text.lower() if normalize_case else text


def get_sentence(span: Span, normalize_case: bool) -> Union[Span, Doc]:
    """Return Span object as a Doc if case_insensitive is set to True"""
    return span if normalize_case else span.as_doc()


def get_attr(lemmatize: bool) -> AnyStr:
    """Return the right attribute to pass to spaCy Matcher"""
    return "LEMMA" if lemmatize else "TEXT"


def get_keyword_lemma(pattern, case):
    text = " ".join([x.lemma_ for x in pattern])
    print(text)
    # text = "".join([sent.lemma_ for sent in pattern.sents])
    return get_keyword(text, case)


def get_tag(case, lemma, keyword):
    if lemma:
        return keyword.lemma_
    return get_keyword(keyword.text, case)
