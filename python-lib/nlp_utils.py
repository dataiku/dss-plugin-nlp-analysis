from typing import AnyStr, Union
from spacy.tokens import Span, Doc


def get_keyword(text: AnyStr, case_insensitive: bool) -> AnyStr:
    """Return text in its wanted-case form"""
    return text.lower() if case_insensitive else text


def get_sentence(span: Span, case_insensitive: bool) -> Union[Span, Doc]:
    """Return Span object as a Doc if case_insensitive is set to True"""
    return span if case_insensitive else span.as_doc()
