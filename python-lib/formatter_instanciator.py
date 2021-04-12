from enum import Enum
from ontology_tagging_formatting import (
    FormatterByDocumentJson,
    FormatterByDocument,
    FormatterByTag,
)

COLUMNS_DESCRIPTION = {
    "tag_keywords": "All found keywords",
    "tag_sentences": "All sentences containing keywords",
    "tag_json_full": "Full tags informations",
    "tag_json_categories": "Category <-> tags",
    "tag_list": "List of all found tags",
    "tag": "Assigned tag",
    "tag_keyword": "Matched keyword",
    "tag_sentence": "Sentence containing keyword(s)",
    "tag_category": "Category of the tag",
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

    def get_formatter(self, config, format, category):
        formatter = self.INSTANCES[format](**config)
        formatter.tag_columns = self.TAG_COLUMNS[format][category]
        return formatter
