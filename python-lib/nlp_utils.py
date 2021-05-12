from typing import AnyStr, Union, List
from spacy.tokens import Span, Doc
import unicodedata


def normalize_case_text(text: AnyStr, lowercase: bool) -> AnyStr:
    """Return text in its wanted case form"""
    return text.lower() if lowercase else text


def get_phrase_matcher_attr(lemmatize: bool) -> AnyStr:
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


def get_span_text(span: Span, lemmatize: bool) -> AnyStr:
    """Return the Span text, or the Span lemma if 'lemmatize' is True

    Args:
        span (spacy.tokens.Span): Text to process
        lemmatize (bool): if True, return the text lemmatized

    Returns:
        str: Text or lemma associated to the span

    """
    return span.lemma_ if lemmatize else span.text


def unicode_normalize_text(texts: List[AnyStr]) -> List[AnyStr]:
    """Return a list of texts NFD normalized"""
    return [unicodedata.normalize("NFD", text) for text in texts]
