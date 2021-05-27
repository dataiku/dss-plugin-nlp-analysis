import pandas as pd
from tqdm import tqdm
from fastcore.utils import store_attr

from typing import AnyStr
from typing import List

from utils.plugin_io_utils import move_columns_after
from utils.plugin_io_utils import unique_list
from utils.plugin_io_utils import generate_unique_columns
from utils.nlp_utils import lowercase_if
from utils.nlp_utils import unicode_normalize_text

from ontology_tagging.spacy_tokenizer import MultilingualTokenizer


# names of all additional columns depending on the output_format
COLUMN_DESCRIPTION = {
    "tag_keywords": "List of matched keywords",  # List
    "tag_sentences": "Sentences containing matched keywords",  # String
    "tag_json_full": "Detailed tag column: list of matched keywords per tag and category, count of occurrences, sentences containing matched keywords",  # nested json
    "tag_json_categories": "List of tags per category",  # nested json
    "tag_list": "List of all assigned tags",  # List
    "tag": "Assigned tag",  # string
    "tag_keyword": "Matched keyword",  # string
    "tag_sentence": "Sentence containing the matched keyword",  # string
    "tag_category": "Category of tag",  # string
}


class FormatterBase:
    """
    Base class to write the output dataframe depending on the output format
    The subclasses are called by the Tagger class where the tokenization, sentence splitting and Matcher instanciation has been done

    Attributes:
        language (string): language: Language code in ISO 639-1 format, cf. https://spacy.io/usage/models#languages
            Used if there is only one language to treat.
            Use the argument 'language_column' for passing a language column name in 'write_df' methods otherwise.
        tokenizer (MultilingualTokenizer): Tokenizer instance to create the tokenizers for each language
        category_column (string): Name of the column in the Ontology. Contains the category of each tag to assign.
        ignore_case (bool): If True, match on lowercased forms. Default is False.
        lemmatization (bool): If True , match on lemmatized forms. Default is False.
        ignore_diacritics (bool): If True, ignore diacritic marks e.g., accents, cedillas, tildes. Default is False.
        text_column_tokenized (string): Name of the column which contains the text splitted by sentences

    """

    def __init__(
        self,
        language: AnyStr,
        tokenizer: MultilingualTokenizer,
        category_column: AnyStr,
        ignore_case: bool,
        lemmatization: bool,
        ignore_diacritics: bool,
        text_column_tokenized: AnyStr,
        _use_nfc: bool,
        tag_columns: List[AnyStr],
        _keyword_to_tag: dict = None,
        _matcher_dict: dict = None,
    ):
        store_attr()
        self.output_df = (
            pd.DataFrame()
        )  # pandas.DataFrame with new columns concerning the found tags
        tqdm.pandas(miniters=1, mininterval=5.0)
        self.column_descriptions = {}
        """Dictionary of new columns to add in the dataframe (key) and their descriptions (value)
        It is filled in _generate_columns_names"""

    def _generate_columns_names(self, text_df: pd.DataFrame) -> None:
        """Create unique names for tag columns and store their descriptions"""
        new_tag_columns = generate_unique_columns(text_df, self.tag_columns)
        for tag_column, new_tag_column in zip(self.tag_columns, new_tag_columns):
            self.column_descriptions[new_tag_column] = COLUMN_DESCRIPTION[tag_column]
        self.tag_columns = new_tag_columns

    def _get_document_language(
        self, row: pd.Series, language_column: AnyStr = None
    ) -> AnyStr:
        """Return the language of the document in the row"""
        return row[language_column] if language_column else self.language

    def _get_document_to_match(self, row: pd.Series, language) -> List:
        """Return the document to match tokenized as list of sentences,
        after applying the desired normalization steps (lowercasing, unicode_normalization, diacritic removal)"""
        return list(
            self.tokenizer.spacy_nlp_dict[language].pipe(
                [
                    unicode_normalize_text(
                        text=lowercase_if(text=sentence, lowercase=self.ignore_case),
                        use_nfc=self._use_nfc,
                        ignore_diacritics=self.ignore_diacritics,
                    )
                    for sentence in row[self.text_column_tokenized]
                ]
            )
        )

    def _set_columns_order(
        self, input_df: pd.DataFrame, output_df: pd.DataFrame, text_column: AnyStr
    ) -> pd.DataFrame:
        """Concatenate the input_df with the new one,reset its columns in the right order, and return it"""
        input_df = input_df.drop(columns=self.text_column_tokenized)
        df = pd.concat([input_df, output_df], axis=1)
        df = df.drop_duplicates()
        return move_columns_after(
            df=df,
            columns_to_move=self.tag_columns,
            after_column=text_column,
        )
