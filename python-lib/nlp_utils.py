from typing import AnyStr, Union, List
from spacy.tokens import Span, Doc
import unicodedata


def normalize_case_text(text: AnyStr, lowercase: bool) -> AnyStr:
    """Return text in its wanted-case form"""
    return text.lower() if lowercase else text


def get_token_attribute(lemmatize: bool) -> AnyStr:
    """Return the attribute to pass to spaCy Matcher"""
    return "LEMMA" if lemmatize else "TEXT"


def lemmatize_doc(doc: Doc) -> AnyStr:
    """Lemmatize SpaCy.tokens.Doc object.

    Args:
        doc (SpaCy.tokens.Doc): Text to lemmatize

    Returns:
        str: Text in its lemmatized form

    """
    return " ".join([span.lemma_ for span in doc])


def normalize_span(span: Span, lowercase: bool, lemmatize: bool) -> AnyStr:
    """Normalize SpaCy.tokens.Span object.
    Available normalizations : lemmatizing / lowercasing

    Args:
        span (spacy.tokens.Span): Text to process
        lowercase (bool): if True, the text will be lowercased
        lemmatize (bool): if True, return the text lemmatized

    Returns:
        str: Text lemmatized or lowercased
        
    """
    if lemmatize:
        return span.lemma_
    return normalize_case_text(span.text, lowercase)


def unicode_normalize_text(texts: List[AnyStr]) -> List[AnyStr]:
    """Return a list of texts NFD normalized"""
    return [unicodedata.normalize("NFD", text) for text in texts]
