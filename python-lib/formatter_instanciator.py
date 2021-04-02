from enum import Enum
from ontology_tagging_formatting import (
    FormatterByDocumentJson,
    FormatterByDocument,
    FormatterByTag,
)


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
            "no_category": ["tag_keywords", "tag_sentences", "tag_list"],
        },
        OutputFormat.ONE_ROW_PER_DOCUMENT_JSON.value: {
            "category": ["tag_json_full", "tag_json_categories"],
            "no_category": ["tag_json_full"],
        },
        OutputFormat.ONE_ROW_PER_TAG.value: {
            "category": ["tag_keyword", "tag_sentence", "tag_category", "tag"],
            "no_category": ["tag_keyword", "tag_sentence", "tag"],
        },
    }

    def get_formatter(self, config, format, category):
        formatter = self.INSTANCES[format](**config)
        formatter.tag_columns = self.TAG_COLUMNS[format][category]
        return formatter
