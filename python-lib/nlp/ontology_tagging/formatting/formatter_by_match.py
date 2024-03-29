import numpy as np
import pandas as pd
import logging
from time import perf_counter

from typing import AnyStr, List

from spacy.tokens import Doc

from nlp.utils import get_span_text

from .base import FormatterBase


class FormatterByMatch(FormatterBase):
    """Class to write a dataframe which contains one row per document per matched keyword"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contains_match = False
        self._duplicate_df = pd.DataFrame()

    def write_df_category(
        self,
        input_df: pd.DataFrame,
        text_column: AnyStr,
        language_column: AnyStr = None,
    ) -> pd.DataFrame:
        """
        Calls write_df to write the output dataframe
        Args:
            input_df (pd.DataFrame): The input dataframe.
            text_column (AnyStr): Name of the column which contains the text to match with.
            language_column (AnyStr): If not None, name of the column with contains the language of each document

        Returns:
               The input dataframe enriched withdataframe with new columns of tag, keyword, sentence and category

        """
        return self.write_df(input_df, text_column, language_column)

    def write_df(
        self,
        input_df: pd.DataFrame,
        text_column: AnyStr,
        language_column: AnyStr = None,
    ) -> pd.DataFrame:
        """
        Write a dataframe which contains one row per document per matched keyword
        Args:
            input_df (pd.DataFrame): The input dataframe.
            text_column (AnyStr): Name of the column which contains the text to match with.
            language_column (AnyStr): If not None, name of the column with contains the language of each document

        Returns:
               The dataframe with all columns from the input, plus new columns of tag, keyword and sentence

        """
        start = perf_counter()
        self._generate_columns_names(input_df)
        input_df.progress_apply(self._write_row, args=[language_column], axis=1)
        logging.info(
            f"Tagging {len(input_df)} documents : Done in {perf_counter() - start:.2f} seconds."
        )
        return self._set_columns_order(self._duplicate_df, self.output_df, text_column)

    def _write_row(self, row: pd.Series, language_column: AnyStr = None) -> None:
        """
        Called by write_df on each row
        Update the output dataframes which will be concatenated after that:
        - self.output_df is filled with columns with the tag columns
        - self._duplicate_df is filled with the original rows of the Document Dataset, with copies
        There are as many copies of a document as there are keywords in this document
        Args:
            row: pandas.Series from text_df
            language_column: if not None, matcher will apply with the given language of the row
        """
        self.contains_match = False
        language = self._get_document_language(row, language_column)
        matches = []
        document_to_match = self._get_document_to_match(row, language)
        empty_row = {column: np.nan for column in self.tag_columns}
        if not self.category_column:
            matches = [
                (
                    self._matcher_dict[language](sentence, as_spans=True),
                    row[self.text_column_tokenized][idx],
                )
                for idx, sentence in enumerate(document_to_match)
            ]  # list of tuples (list of found keywords, list of associated sentences)
            self._get_tags_in_row(matches, row, language)
        else:
            self._get_tags_in_row_category(document_to_match, row, language)
        if not self.contains_match:
            self.output_df = self.output_df.append(empty_row, ignore_index=True)
            self._duplicate_df = self._duplicate_df.append(
                pd.DataFrame([row]), ignore_index=True
            )

    def _get_tags_in_row(self, matches: List, row: pd.Series, language: AnyStr) -> None:
        """
        Called by _write_row
        Create new rows from the input one, containing the tag, keyword, and sentence columns for each match, and append them to the output_df.
        """
        # For FormatterByMatch with no category, the self.tag_columns are ordered as: [tag, keyword, sentence]
        for match, sentence in matches:
            tag_rows = [
                {
                    self.tag_columns[0]: self._keyword_to_tag[language][
                        get_span_text(span=keyword, lemmatize=self.lemmatization)
                    ],
                    self.tag_columns[1]: keyword.text,
                    self.tag_columns[2]: sentence,
                }
                for keyword in match
            ]
            self._update_df(tag_rows, row)

    def _get_tags_in_row_category(
        self, document_to_match: List, row: pd.Series, language: AnyStr
    ) -> None:
        """
        Called by _write_row_category
        Create the list of new rows with infos about the tags and gives it to _update_df function
        """
        original_document = list(row[self.text_column_tokenized])
        # For FormatterByMatch with categories, the self.tag_columns are ordered as: [category, tag, keyword, sentence]
        for idx, sentence in enumerate(document_to_match):
            tag_rows = [
                {
                    self.tag_columns[0]: keyword.label_,
                    self.tag_columns[1]: keyword.ent_id_,
                    self.tag_columns[2]: keyword.text,
                    self.tag_columns[3]: original_document[idx],
                }
                for keyword in sentence.ents
            ]
            self._update_df(tag_rows, row)

    def _update_df(self, tag_rows: List[dict], row: pd.Series) -> None:
        """
        Appends:
        - row with tagging info (tag, keyword, sentence, category if available) to output_df
        - duplicated initial row from the Document dataframe to df.duplicated_lines
        """
        if tag_rows:
            self.output_df = self.output_df.append(tag_rows, ignore_index=True)
            self._duplicate_df = self._duplicate_df.append(
                pd.DataFrame([row] * len(tag_rows)), ignore_index=True
            )
            self.contains_match = True
