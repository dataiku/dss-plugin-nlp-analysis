from typing import AnyStr, Union
from spacy.tokens import Span, Doc


def get_keyword(text: AnyStr, case_insensitive: bool, normalization: bool) -> AnyStr:
    """Return text in its wanted-case form"""
    if case_insensitive:
        text = text.lower()
    if normalization:
        text = remove_accents(text)
    return text


def get_sentence(span: Span, normalize_case: bool) -> Union[Span, Doc]:
    """Return Span object as a Doc if case_insensitive is set to True"""
    return span if normalize_case else span.as_doc()


def remove_accents(input_str):
    nfd_form = unicodedata.normalize("NFD", input_str)
    return u"".join([c for c in nfd_form if not unicodedata.combining(c)])
