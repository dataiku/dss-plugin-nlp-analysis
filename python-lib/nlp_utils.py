from typing import AnyStr, Union, List
from spacy.tokens import Span, Doc
import unicodedata


def get_keyword(text: AnyStr, normalize_case: bool) -> AnyStr:
    """Return text in its wanted-case form"""
    return text.lower() if normalize_case else text


def get_sentence(span: Span, normalize_case: bool) -> Union[Span, Doc]:
    """Return Span object as a Doc if case_insensitive is set to True"""
    return span if normalize_case else span.as_doc()


def normalize_text(texts: List[AnyStr]) -> List[AnyStr]:
    """Return a list of texts NFD normalized"""
    return [unicodedata.normalize("NFD", text) for text in texts]