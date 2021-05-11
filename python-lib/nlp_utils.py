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


def unicode_normalize_text(
    text: AnyStr, use_nfc: bool = False, normalize_diacritics: bool = False
):
    """Apply unicode_normalization to text

    Args:
        text (str): Text to normalize with NFD or NFC norm.
        use_nfc (bool): Apply NFC norm if True, NFD otherwise.
        normalize_diacritics(bool): if True, remove diacritics after unicode normalizing.

    Returns:
        str: Text normalized with NFC or NFD norm.

    """
    norm = "NFC" if use_nfc else "NFD"
    text = unicodedata.normalize(norm, text)
    if normalize_diacritics:
        text = "".join([c for c in text if not unicodedata.combining(c)])
    return text
