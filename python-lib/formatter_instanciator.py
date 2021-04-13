from enum import Enum
from typing import AnyStr
from ontology_tagging_formatting import (
    FormatterByDocumentJson,
    FormatterByDocument,
    FormatterByTag,
)

# names of all additional columns depending on the output_format
COLUMNS_DESCRIPTION = {
    "tag_keywords": "Matched keywords",
    "tag_sentences": "Sentences with keywords",
    "tag_json_full": "Detailed tag column",
    "tag_json_categories": "category → tags",
    "tag_list": "List of all found tags",
    "tag": "Found tag",
    "tag_keyword": "Matched keyword",
    "tag_sentence": "Sentence with keywords",
    "tag_category": "Category of tag",
}


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
            "category": ["tag_sentences", "tag_keywords"],
            "no_category": ["tag_list", "tag_sentences", "tag_keywords"],
        },
        OutputFormat.ONE_ROW_PER_DOCUMENT_JSON.value: {
            "category": ["tag_json_categories", "tag_json_full"],
            "no_category": ["tag_json_full"],
        },
        OutputFormat.ONE_ROW_PER_TAG.value: {
            "category": ["tag", "tag_category", "tag_sentence", "tag_keyword"],
            "no_category": ["tag", "tag_sentence", "tag_keyword"],
        },
    }

    def get_formatter(self, config: dict, format: AnyStr, category: AnyStr):
        """get the right formatting instance and returns it"""
        formatter = self.INSTANCES[format](**config)
        formatter.tag_columns = self.TAG_COLUMNS[format][category]
        return formatter
