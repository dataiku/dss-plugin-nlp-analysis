from typing import AnyStr, Union
from spacy.tokens import Span, Doc


def get_text_case(text: AnyStr, normalize_case: bool) -> AnyStr:
    """Return text in its wanted-case form"""
    return text.lower() if normalize_case else text


def get_attribute(lemmatize: bool) -> AnyStr:
    """Return the right attribute to pass to spaCy Matcher"""
    return "LEMMA" if lemmatize else "TEXT"


def get_text_normalized(text: Doc, normalize_case: bool) -> AnyStr:
    """Lemmatize text and return it lowercase if normalize_case is True"""
    text = " ".join([word.lemma_ for word in text])
    return get_text_case(text, normalize_case)


def get_keyword(normalize_case: bool, lemmatize: bool, keyword: Span) -> AnyStr:
    """Return text after normalizing it if needed

    Args:
        normalize_case (bool): if True, keyword will be lowercased
        lemmatize (bool): if True, return the keyword lemma
        keyword (spacy.tokens.Span): the text you want to process

    Returns:
        The keyword normalized as needed
    """
    if lemmatize:
        return keyword.lemma_
    return get_text_case(keyword.text, normalize_case)
