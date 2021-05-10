from typing import AnyStr, Union, List
from spacy.tokens import Span, Doc
import unicodedata


def normalize_case_text(text: AnyStr, lowercase: bool) -> AnyStr:
    """Return text in its wanted case form"""
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


def lemmatize_span(span: Span, lemmatize: bool) -> AnyStr:
    """Lemmatize SpaCy.tokens.Span object if 'lemmatize' is True.

    Args:
        span (spacy.tokens.Span): Text to process
        lemmatize (bool): if True, return the text lemmatized

    Returns:
        str: Text or lemma associated to the span

    """
    if lemmatize:
        return span.lemma_
    return span.text


def remove_diacritics_text(text: AnyStr) -> AnyStr:
    """Remove diacritics from text, e.g accents, cedillas, tildes."""
    return "".join([c for c in text if not unicodedata.combining(c)])


def normalize_nfd_text(text, normalize_diacritics: bool) -> AnyStr:
    """Apply NFD normalization to text and remove its diacritics if 'normalize_diacritics' is True."""
    nfd_form = unicodedata.normalize("NFD", text)
    if normalize_diacritics:
        nfd_form = remove_diacritics_text(nfd_form)
    return nfd_form
