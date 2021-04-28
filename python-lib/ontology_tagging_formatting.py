"""
Module to write the output dataframe depending on the output format
This module inherits from Tagger where the tokenization, sentence splitting and Matcher instanciation has been done
"""
from fastcore.utils import store_attr
import pandas as pd
from collections import defaultdict
from spacy.tokens import Span, Doc
from typing import AnyStr, Dict, List, Tuple
import numpy as np
from time import perf_counter
import logging
import json
from plugin_io_utils import move_columns_after, unique_list
from spacy_tokenizer import MultilingualTokenizer


class Formatter:
    def __init__(
        self,
        language: AnyStr,
        splitted_sentences_column: AnyStr,
        tokenizer: MultilingualTokenizer,
        matcher_dict: dict,
        keyword_to_tag: dict,
        category_column: AnyStr,
    ):
        store_attr()
        self.output_df = pd.DataFrame()

    def _apply_matcher(
        self, row: pd.Series, language_column: AnyStr = None
    ) -> Tuple[AnyStr, List]:
        """Apply matcher to the document and returns it
        Args:
            row: pandas.Series, document from text_df
            language_column: if not None, document is matched with the language_column indication
                             else, document is matched with the formatter language attribute
        Returns: language of the document and tagged document
        """
        language = (
            row[language_column]
            if self.language == "language_column"
            else self.language
        )
        document = list(
            self.tokenizer.spacy_nlp_dict[language].pipe(
                row[self.splitted_sentences_column]
            )
        )
        return language, document

    def _set_columns_order(
        self, input_df: pd.DataFrame, output_df: pd.DataFrame, text_column: AnyStr
    ) -> pd.DataFrame:
        """Concatenate the input_df with the new one,reset its columns in the right order, and return it"""
        input_df = input_df.drop(columns=[self.splitted_sentences_column])
        df = pd.concat([input_df, output_df], axis=1)
        return move_columns_after(
            input_df=input_df,
            df=df,
            columns_to_move=self.tag_columns,
            after_column=text_column,
        )


class FormatterByTag(Formatter):
    def __init__(self, *args, **kwargs):
        super(FormatterByTag, self).__init__(*args, **kwargs)
        self.contains_match = False
        self.duplicate_df = pd.DataFrame()

    def write_df_category(
        self,
        input_df: pd.DataFrame,
        text_column: AnyStr,
        language_column: AnyStr = None,
    ) -> pd.DataFrame:
        return self.write_df(input_df, text_column, language_column)

    def write_df(
        self,
        input_df: pd.DataFrame,
        text_column: AnyStr,
        language_column: AnyStr = None,
    ) -> pd.DataFrame:
        """Write the output dataframe for one_row_per_tag format (with or without categories)"""
        start = perf_counter()
        input_df.apply(self._write_row, args=[language_column], axis=1)
        logging.info(
            f"Tagging {len(input_df)} documents : Done in {perf_counter() - start:.2f} seconds."
        )
        return self._set_columns_order(self.duplicate_df, self.output_df, text_column)

    def _write_row(self, row: pd.Series, language_column: AnyStr = None) -> None:
        """
        Called by write_df on each row
        Update the output dataframes which will be concatenated after :
        -> output_df contains the columns with informations about the tags
        -> df_duplicated_lines contains the original rows of the Document Dataset, with copies
        There are as many copies of a document as there are keywords in this document
        Args:
            row: pandas.Series from text_df
            language_column: if not None, matcher will apply with the given language of the row
        """
        self.contains_match = False
        language, document = super()._apply_matcher(row, language_column)
        matches = []
        empty_row = {column: np.nan for column in self.tag_columns}
        if not self.category_column:
            matches = [
                (
                    self.matcher_dict[language](sentence, as_spans=True),
                    sentence,
                )
                for sentence in document
            ]
            self._get_tags_in_row(matches, row, language)
        else:
            self._get_tags_in_row_category(document, row, language)
        if not self.contains_match:
            self.output_df = self.output_df.append(empty_row, ignore_index=True)
            self.duplicate_df = self.duplicate_df.append(
                pd.DataFrame([row]), ignore_index=True
            )

    def _get_tags_in_row(self, matches: List, row: pd.Series, language: AnyStr) -> None:
        """
        Called by _write_row
        Create the list of new rows with infos about the tags and gives it to _update_df
        """
        values = []
        for match, sentence in matches:
            values = [
                self._list_to_dict(
                    [
                        self.keyword_to_tag[language][keyword.text],
                        keyword.text,
                        sentence.text,
                    ]
                )
                for keyword in match
            ]
            self._update_df(match, values, row)

    def _get_tags_in_row_category(
        self, document: List, row: pd.Series, language: AnyStr
    ) -> None:
        """
        Called by _write_row_category
        Create the list of new rows with infos about the tags and gives it to _update_df function
        """
        tag_rows = []
        for sentence in document:
            tag_rows = [
                self._list_to_dict(
                    [
                        keyword.label_,
                        self.keyword_to_tag[language][keyword.text],
                        keyword.text,
                        sentence.text,
                    ]
                )
                for keyword in sentence.ents
            ]
            self._update_df(tag_rows, tag_rows, row)

    def _update_df(self, match: List, values: List[dict], row: pd.Series) -> None:
        """
        Appends:
        -row with infos about the founded tags to output_df
        -duplicated initial row from the Document dataframe to df.duplicated_lines
        """
        if match:
            self.output_df = self.output_df.append(values, ignore_index=True)
            self.duplicate_df = self.duplicate_df.append(
                pd.DataFrame([row for i in range(len(values))]), ignore_index=True
            )
            self.contains_match = True

    def _list_to_dict(self, tag_infos: List[AnyStr]) -> dict:
        """Returns dictionary containing a new row with tag datas"""
        return {
            column_name: tag_info
            for column_name, tag_info in zip(self.tag_columns, tag_infos)
        }


class FormatterByDocument(Formatter):
    def __init__(self, *args, **kwargs):
        super(FormatterByDocument, self).__init__(*args, **kwargs)
        self.tag_sentences, self.tag_keywords = [], []

    def _fill_tags(self, condition: bool, value: dict):
        """Dump the dictionary 'value' if 'condition' is True"""
        return json.dumps(value, ensure_ascii=False) if condition else np.nan

    def write_df(
        self,
        input_df: pd.DataFrame,
        text_column: AnyStr,
        language_column: AnyStr = None,
    ) -> pd.DataFrame():
        """Write the output dataframe for One row per document format (without categories)"""
        start = perf_counter()
        input_df.apply(self._write_row, args=[language_column], axis=1)
        logging.info(
            f"Tagging {len(input_df)} documents : Done in {perf_counter() - start:.2f} seconds."
        )
        return self._merge_df_columns(input_df, text_column)

    def _write_row(self, row: pd.Series, language_column: AnyStr = None) -> None:
        """Called by write_df on each row
        Append columns of sentences,keywords and tags to the output dataframe
        Args:
            row: pandas.Series from text_df
            language_column: if not None, matcher will apply with the given language of the row
        """
        language, document = super()._apply_matcher(row, language_column)
        tags_in_document, keywords_in_document, matched_sentences = [], [], []
        for sentence in document:
            (
                tags_in_document,
                keywords_in_document,
                matched_sentences,
            ) = self._get_tags_in_row(
                sentence=sentence,
                tags_in_document=tags_in_document,
                keywords_in_document=keywords_in_document,
                matched_sentences=matched_sentences,
                language=language,
            )
        if tags_in_document != []:
            line = {
                self.tag_columns[0]: self._fill_tags(
                    True, unique_list(tags_in_document)
                ),
                self.tag_columns[1]: self._fill_tags(
                    True, unique_list(keywords_in_document)
                ),
                self.tag_columns[2]: "".join(unique_list(matched_sentences)),
            }
        else:
            line = {column: np.nan for column in self.tag_columns}
        self.output_df = self.output_df.append(line, ignore_index=True)

    def _get_tags_in_row(
        self,
        sentence: Doc,
        tags_in_document: List,
        keywords_in_document: List,
        matched_sentences: List,
        language: AnyStr,
    ) -> Tuple[List[AnyStr], List[AnyStr], List[AnyStr]]:
        """
        Called by _write_row on each sentence
        Return the tags, sentences and keywords linked to the given sentence
        """
        tags_in_sentence = []
        matches = self.matcher_dict[language](sentence, as_spans=True)
        for match in matches:
            keyword = match.text
            tag = self.keyword_to_tag[language][keyword]
            tags_in_document.append(tag)
            keywords_in_document.append(keyword)
            matched_sentences.append(sentence.text + "\n")
        return tags_in_document, keywords_in_document, matched_sentences

    def write_df_category(
        self,
        input_df: pd.DataFrame,
        text_column: AnyStr,
        language_column: AnyStr = None,
    ) -> pd.DataFrame:
        """Write the output dataframe for One row per document with category"""
        start = perf_counter()
        input_df.apply(self._write_row_category, args=[False, language_column], axis=1)
        logging.info(
            f"Tagging {len(input_df)} documents : Done in {perf_counter() - start:.2f} seconds."
        )
        return self._merge_df_columns_category(input_df, text_column)

    def _write_row_category(
        self, row: pd.Series, one_row_per_doc_json: bool, language_column: AnyStr = None
    ) -> None:
        """
        Called by write_df_category
        Append columns to the output dataframe, depending on the output format
        Args:
            row: pandas.Series , document from text_df
            one_row_per_doc_json: Bool to know if the format is JSON
            language_column: if not None, matcher will apply with the given language of the row
        """
        language, document = super()._apply_matcher(row, language_column)
        matched_sentence, keyword_list = [], []
        tag_columns_for_json, line, line_full = (
            defaultdict(),
            defaultdict(list),
            defaultdict(defaultdict),
        )
        for sentence in document:
            for keyword in sentence.ents:
                line, line_full = self._get_tags_in_row_category(
                    match=keyword,
                    line=line,
                    line_full=line_full,
                    sentence=sentence,
                    language=language,
                )
                keyword_list.append(keyword.text + " ")
                matched_sentence.append(sentence.text + "\n")
            tag_columns_for_json["tag_json_categories"] = self._fill_tags(
                condition=(line and one_row_per_doc_json), value=dict(line)
            )
            tag_columns_for_json["tag_json_full"] = self._fill_tags(
                condition=(line and one_row_per_doc_json),
                value={
                    column_name: dict(value) for column_name, value in line_full.items()
                },
            )
        for col in line:
            line[col] = self._fill_tags(True, line[col])
        if keyword_list:
            self.tag_keywords.append(self._fill_tags(True, unique_list(keyword_list)))
        else:
            self.tag_keywords.append(np.nan)
        self.tag_sentences.append(" ".join(unique_list(matched_sentence)))
        self.output_df = (
            self.output_df.append(tag_columns_for_json, ignore_index=True)
            if one_row_per_doc_json
            else self.output_df.append(
                line,
                ignore_index=True,
            )
        )

    def _get_tags_in_row_category(
        self, match: Span, line: dict, line_full: dict, sentence: Doc, language: AnyStr
    ) -> Tuple[dict, dict]:
        """
        Called by _write_row_category
        Args:
            line: dictionary {category: tag}
            line_full: dictionary with full informations about the found tags
            sentence: Doc object containing the keyword
            language: string
        Returns:
            line, line_full
        """
        keyword = match.text
        tag = self.keyword_to_tag[language][keyword]
        category = match.label_
        sentence = sentence.text
        if tag not in line_full[category]:
            line_full[category][tag] = {
                "count": 1,
                "sentences": [sentence],
                "keywords": [keyword],
            }

            line[category].append(tag)
        else:
            line_full[category][tag]["count"] += 1
            if sentence not in line_full[category][tag]["sentences"]:
                line_full[category][tag]["sentences"].append(sentence)
            if keyword not in line_full[category][tag]["keywords"]:
                line_full[category][tag]["keywords"].append(keyword)
        return line, line_full

    def _merge_df_columns_category(
        self, input_df: pd.DataFrame, text_column: AnyStr
    ) -> pd.DataFrame:
        """
        Called by _write_row_category,when the format is one row per document.
        Insert columns tag_keywords and tag_sentences
        Returns the complete output dataframe after setting the columns in the right order
        """
        output_df_copy = self.output_df.copy().add_prefix("tag_list_")
        tag_list_columns = output_df_copy.columns.tolist()
        output_df_copy.insert(
            len(self.output_df.columns),
            self.tag_columns[0],
            self.tag_keywords,
            True,
        )
        output_df_copy.insert(
            len(self.output_df.columns),
            self.tag_columns[1],
            self.tag_sentences,
            True,
        )
        self.tag_columns = tag_list_columns + self.tag_columns
        return self._set_columns_order(input_df, output_df_copy, text_column)

    def _merge_df_columns(
        self, input_df: pd.DataFrame, text_column: AnyStr
    ) -> pd.DataFrame:
        """
        Called by write_df
        Insert columns tag_sentences and tag_keywords
        Return the complete output dataframe after setting the columns in the right order
        """
        for column in self.tag_columns[::-1]:
            self.output_df.set_index(column, inplace=True)
            self.output_df.reset_index(inplace=True)
        return self._set_columns_order(input_df, self.output_df, text_column)


class FormatterByDocumentJson(FormatterByDocument):
    def __init__(self, *args, **kwargs):
        super(FormatterByDocumentJson, self).__init__(*args, **kwargs)

    def write_df(
        self,
        input_df: pd.DataFrame,
        text_column: AnyStr,
        language_column: AnyStr = None,
    ) -> pd.DataFrame:
        """
        Write the output dataframe for the Json Format without category
        """
        start = perf_counter()
        input_df.apply(self._write_row, args=[language_column], axis=1)
        logging.info(
            f"Tagging {len(input_df)} documents : Done in {perf_counter() - start:.2f} seconds."
        )
        return self._set_columns_order(input_df, self.output_df, text_column)

    def _write_row(self, row: pd.Series, language_column: AnyStr = None) -> None:
        """
        Called by write_df on each row
        Update column tag_json_full with the found tags
        Args:
            row: pandas.Series from text_df
            language_column: if not None, matcher will apply with the given language of the row
        """
        language, document = super()._apply_matcher(row, language_column)
        line_full, tag_column_for_json = defaultdict(defaultdict), {}
        for sentence in document:
            matches = self.matcher_dict[language](sentence, as_spans=True)
            for keyword in matches:
                line_full = self._get_tags_in_row(
                    match=keyword,
                    line_full=line_full,
                    sentence=sentence,
                    language=language,
                )

        tag_column_for_json["tag_json_full"] = super()._fill_tags(
            condition=line_full,
            value={
                column_name: dict(value) for column_name, value in line_full.items()
            },
        )

        self.output_df = self.output_df.append(tag_column_for_json, ignore_index=True)

    def write_df_category(
        self,
        input_df: pd.DataFrame,
        text_column: AnyStr,
        language_column: AnyStr = None,
    ) -> pd.DataFrame:
        """
        Write the output dataframe for the Json Format with category :
        """
        start = perf_counter()
        input_df.apply(
            super()._write_row_category, args=[True, language_column], axis=1
        )
        logging.info(
            f"Tagging {len(input_df)} documents: Done in {perf_counter() - start:.2f} seconds."
        )
        return self._set_columns_order(input_df, self.output_df, text_column)

    def _get_tags_in_row(
        self, match: Span, line_full: dict, sentence: Doc, language: AnyStr
    ) -> dict:
        """
        Called by _write_row on each sentence
        Return a dictionary containing precisions about each tag
        """
        keyword = match.text
        tag = self.keyword_to_tag[language][keyword]
        sentence = sentence.text
        if tag not in line_full.keys():
            line_full[tag] = {
                "count": 1,
                "sentences": [sentence],
                "keywords": [keyword],
            }
        else:
            line_full[tag]["count"] += 1
            if sentence not in line_full[tag]["sentences"]:
                line_full[tag]["sentences"].append(sentence)
            if keyword not in line_full[tag]["keywords"]:
                line_full[tag]["keywords"].append(keyword)

        return line_full