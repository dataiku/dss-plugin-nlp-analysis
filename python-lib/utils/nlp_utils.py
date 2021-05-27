import unicodedata
from enum import Enum
from typing import AnyStr, Union, List
from spacy.tokens import Span, Doc


class UnicodeNormalization(Enum):
    NFC = "NFC"
    NFD = "NFD"


def lowercase_if(text: AnyStr, lowercase: bool) -> AnyStr:
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


def unicode_normalize_text(
    text: AnyStr, use_nfc: bool = False, ignore_diacritics: bool = False
):
    """Apply unicode_normalization to text

    Args:
        text (str): Text to normalize with NFD or NFC norm.
        use_nfc (bool): Apply NFC norm if True, NFD otherwise.
        ignore_diacritics(bool): if True, remove diacritics after unicode normalizing.

    Returns:
        str: Text normalized with NFC or NFD norm.

    """
    norm = UnicodeNormalization.NFC.value if use_nfc else UnicodeNormalization.NFD.value
    text = unicodedata.normalize(norm, text)
    if ignore_diacritics:
        text = "".join([c for c in text if not unicodedata.combining(c)])
    return text
