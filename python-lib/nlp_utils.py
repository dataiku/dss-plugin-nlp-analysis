from typing import AnyStr, Union
from spacy.tokens import Span, Doc


def get_keyword(text: AnyStr, normalize_case: bool) -> AnyStr:
    """Return text in its wanted-case form"""
    return text.lower() if normalize_case else text


def get_sentence(span: Span, normalize_case: bool) -> Union[Span, Doc]:
    """Return Span object as a Doc if case_insensitive is set to True"""
    return span if normalize_case else span.as_doc()
