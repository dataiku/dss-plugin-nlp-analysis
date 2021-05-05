from typing import AnyStr, Union, List
from spacy.tokens import Span, Doc
import unicodedata


def normalize_case(text: AnyStr, lowercase: bool) -> AnyStr:
    """Return text in its wanted-case form"""
    return text.lower() if lowercase else text


def get_token_attribute(lemmatize: bool) -> AnyStr:
    """Return the attribute to pass to spaCy Matcher"""
    return "LEMMA" if lemmatize else "TEXT"


def lemmatize(text: Doc, lowercase: bool) -> AnyStr:
    """Lemmatize text. The lemma is lowercased if 'lowercase' is True"""
    text = " ".join([word.lemma_ for word in text])
    return normalize_case(text, lowercase)


def normalize(
    keyword: Span, lowercase: bool, lemmatize: bool
) -> AnyStr:
    """Normalize keyword. Available normalizations : lemmatizing / lowercasing

    Args:
        keyword (spacy.tokens.Span): the text to process
        lowercase (bool): if True, keyword will be lowercased
        lemmatize (bool): if True, return the keyword lemma

    Returns:
        The keyword lemmatized or lowercased
    """
    if lemmatize:
        return keyword.lemma_
    return normalize_case(keyword.text, lowercase)


def unicode_normalize_text(texts: List[AnyStr]) -> List[AnyStr]:
    """Return a list of texts NFD normalized"""
    return [unicodedata.normalize("NFD", text) for text in texts]
