from enum import Enum
from typing import AnyStr

from .formatter_by_document import FormatterByDocumentJson
from .formatter_by_document import FormatterByDocument
from .formatter_by_match import FormatterByMatch


class OutputFormat(Enum):
    ONE_ROW_PER_MATCH = "one_row_per_match"
    ONE_ROW_PER_DOCUMENT = "one_row_per_doc"
    ONE_ROW_PER_DOCUMENT_JSON = "one_row_per_doc_json"


INSTANCES = {
    OutputFormat.ONE_ROW_PER_DOCUMENT.value: FormatterByDocument,
    OutputFormat.ONE_ROW_PER_DOCUMENT_JSON.value: FormatterByDocumentJson,
    OutputFormat.ONE_ROW_PER_MATCH.value: FormatterByMatch,
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
    OutputFormat.ONE_ROW_PER_MATCH.value: {
        "category": ["tag_category", "tag", "tag_keyword", "tag_sentence"],
        "no_category": ["tag", "tag_keyword", "tag_sentence"],
    },
}

"""Dictionary of names of the columns to create. The names of the columns depend on the output format chosen in the input parameters
of the recipe. 
Each key of TAG_COLUMNS is an output format name. Each format has two values: 'category' (when there are categories in the ontology)
and 'no_category' (when there is not), and an associated list of column names, defined as follow: 

Format 'one_row_per_doc':
    -'tag_keywords' (List): List of matched keywords
    -'tag_sentences' (List): Sentences containing matched keywords
    Plus, if there are categories: 
        For each category in the ontology, a column:
            -tag_list_'category_name' (List) : List of all assigned tags that have for category 'category_name' in a document
    Otherwise:
        -tag_list (List): List of all assigned tags in a document 

Format 'one_row_per_doc_json':
    -'tag_json_full' (dict dumped as JSON): Detailed tag column: List of matched keywords per tag and category, count of occurrences, sentences containing matched keywords
    Plus, if there are categories: 
        -'tag_json_categories' (dict dumped as JSON): List of tags per category

Format 'one_row_per_match':
    -'tag' (str): Assigned tag
    -'tag_keyword' (str): Matched keyword
    -'tag_sentence'(str): Sentence containing the matched keyword
    Plus, if there are categories: 
        -'tag_category'(str): Category of tag

"""


class FormatterInstanciator:
    @staticmethod
    def get_formatter(config: dict, format: AnyStr, category: AnyStr):
        """get the right formatting instance and returns it

        Args:
            config (dict): class attributes for the Formatter instance to instanciate
            format (AnyStr): name of the format to instanciate
            category (AnyStr): 'category' if we want the list of tag columns when there are categories in the ontology , 'no category' if we don't.

        Returns:
            The instance of the formatter"""
        formatter = INSTANCES[format](
            tag_columns=TAG_COLUMNS[format][category], **config
        )
        return formatter
