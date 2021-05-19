from enum import Enum
from typing import AnyStr

from .ontology_tagging_formatting import FormatterByDocumentJson
from .ontology_tagging_formatting import FormatterByDocument
from .ontology_tagging_formatting import FormatterByTag


class OutputFormat(Enum):
    ONE_ROW_PER_TAG = "one_row_per_tag"
    ONE_ROW_PER_DOCUMENT = "one_row_per_doc"
    ONE_ROW_PER_DOCUMENT_JSON = "one_row_per_doc_json"


class FormatterInstanciator:
    
    INSTANCES = {
        OutputFormat.ONE_ROW_PER_DOCUMENT.value: FormatterByDocument,
        OutputFormat.ONE_ROW_PER_DOCUMENT_JSON.value: FormatterByDocumentJson,
        OutputFormat.ONE_ROW_PER_TAG.value: FormatterByTag,
    }

    TAG_COLUMNS = {
        OutputFormat.ONE_ROW_PER_DOCUMENT.value: {
            "category": ["tag_keywords", "tag_sentences"],
            "no_category": ["tag_list", "tag_keywords", "tag_sentences"],
        },
        OutputFormat.ONE_ROW_PER_DOCUMENT_JSON.value: {
            "category": ["tag_json_categories", "tag_json_full"],
            "no_category": ["tag_json_full"],
        },
        OutputFormat.ONE_ROW_PER_TAG.value: {
            "category": ["tag_category", "tag", "tag_keyword", "tag_sentence"],
            "no_category": ["tag", "tag_keyword", "tag_sentence"],
        },
    }

    def get_formatter(self, config: dict, format: AnyStr, category: AnyStr):
        """get the right formatting instance and returns it"""
        formatter = self.INSTANCES[format](**config)
        formatter.tag_columns = self.TAG_COLUMNS[format][category]
        return formatter
