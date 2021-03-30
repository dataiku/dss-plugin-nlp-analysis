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


class Formatter:
    def __init__(
        self,
        input_df,
        splitted_sentences,
        nlp_dict,
        matcher_dict,
        text_column,
        language,
        keyword_to_tag,
        category_column,
    ):
        store_attr()
        self.output_df = pd.DataFrame()
        self.duplicate_df = pd.DataFrame()
        self.contains_match = False
        self.tag_columns = []
        self.tag_keywords, self.tag_sentences = [], []

    def _arrange_columns_order(
        self, df: pd.DataFrame, columns: List[AnyStr]
    ) -> pd.DataFrame:
        """Put columns in the right order in the Output dataframe"""
        for column in columns:
            df.set_index(column, inplace=True)
            df.reset_index(inplace=True)
        return df


class FormatterByTag(Formatter):
    def __init__(self, *args, **kwargs):
        super(FormatterByTag, self).__init__(*args, **kwargs)

    def write_df(self) -> pd.DataFrame:
        """Write the output dataframe for one_row_per_tag format (with or without categories)"""
        start = perf_counter()
        self.input_df.apply(self._write_row, axis=1)
        logging.info(
            f"Tagging {len(self.input_df)} documents : Done in {perf_counter() - start:.2f} seconds."
        )
        self.output_df.reset_index(drop=True, inplace=True)
        self.duplicate_df.reset_index(drop=True, inplace=True)
        return super()._arrange_columns_order(
            pd.concat(
                [
                    self.output_df,
                    self.duplicate_df.drop(columns=[self.splitted_sentences]),
                ],
                axis=1,
            ),
            self.tag_columns + [self.text_column],
        )

    def _write_row(self, row: pd.Series) -> None:
        """
        Called by write_df on each row
        Updates the output dataframes which will be concatenated after :
        -> output_df contains the columns with informations about the tags
        -> df_duplicated_lines contains the original rows of the Document Dataset, with copies
        There are as many copies of a document as there are keywords in this document
        """
        self.contains_match = False
        document = list(self.nlp_dict[self.language].pipe(row[self.splitted_sentences]))
        matches = []
        empty_row = {column: np.nan for column in self.tag_columns}
        if not self.category_column:
            matches = [
                (
                    self.matcher_dict[self.language](sentence, as_spans=True),
                    sentence,
                )
                for sentence in document
            ]
            self._get_tags_in_row(matches, row)
        else:
            self._get_tags_in_row_category(document, row)
        if not self.contains_match:
            self.output_df = self.output_df.append(empty_row, ignore_index=True)
            self.duplicate_df = self.duplicate_df.append(row)

    def _get_tags_in_row(self, matches: List, row: pd.Series) -> None:
        """
        Called by _write_row
        Creates the list of new rows with infos about the tags and gives it to _update_output_df function
        """
        values = []
        for match, sentence in matches:
            values = [
                self._list_to_dict(
                    [
                        keyword.text,
                        sentence.text,
                        self.keyword_to_tag[keyword.text],
                    ]
                )
                for keyword in match
            ]
            self._update_df(match, values, row)

    def _get_tags_in_row_category(self, document: List, row: pd.Series) -> None:
        """
        Called by _write_row_category
        Creates the list of new rows with infos about the tags and gives it to _update_df function
        """
        tag_rows = []
        for sentence in document:
            tag_rows = [
                self._list_to_dict(
                    [
                        keyword.text,
                        sentence.text,
                        keyword.label_,
                        self.keyword_to_tag[keyword.text],
                    ]
                )
                for keyword in sentence.ents
            ]
            self._update_df(tag_rows, tag_rows, row)

    def _update_df(self, match: List, values: List[dict], row: pd.Series) -> None:
        """
        Appends:
        -row with infos about the founded tags to output_df
        -duplicated initial row from the Document dataframe(input_df) to df.duplicated_lines
        """
        if match:
            self.output_df = self.output_df.append(values, ignore_index=True)
            self.duplicate_df = self.duplicate_df.append(
                [row for i in range(len(values))]
            )
            self.contains_match = True

    def _list_to_dict(self, tag_infos):
        return {
            column_name: tag_info
            for column_name, tag_info in zip(self.tag_columns, tag_infos)
        }


class FormatterByDocument(Formatter):
    def __init__(self, *args, **kwargs):
        super(FormatterByDocument, self).__init__(*args, **kwargs)

    def _fill_tags(self, condition, value)  #TODO put in an utility py file later
        return value if condition else np.nan

    def write_df(self) -> pd.DataFrame():
        """Write the output dataframe for One row per document format (without categories)"""
        start = perf_counter()
        self.input_df.apply(self._write_row, axis=1)
        logging.info(
            f"Tagging {len(self.input_df)} documents : Done in {perf_counter() - start:.2f} seconds."
        )
        return self._merge_df_columns()

    def _write_row(self, row: pd.Series) -> None:
        """Called by write_df on each row
        Appends columns of sentences,keywords and tags to the output dataframe"""
        document = list(self.nlp_dict[self.language].pipe(row[self.splitted_sentences]))
        tags_in_document = []
        list_matched_tags = []
        string_sentence, string_keywords = "", ""
        for sentence in document:
            (
                tags_in_document,
                string_sentence,
                string_keywords,
            ) = self._get_tags_in_row(
                sentence,
                list_matched_tags,
                tags_in_document,
                string_sentence,
                string_keywords,
            )
        self.tag_sentences.append(string_sentence)
        self.tag_keywords.append(string_keywords)
        tag_list = tags_in_document if tags_in_document != [] else np.nan
        self.output_df = self.output_df.append(
            {self.tag_columns[2]: tag_list}, ignore_index=True
        )

    def _get_tags_in_row(
        self,
        sentence: Doc,
        list_matched_tags: List,
        tags_in_document: Dict[AnyStr, List],
        string_sentence: AnyStr,
        string_keywords: AnyStr,
    ) -> Tuple[Dict[AnyStr, List], AnyStr, AnyStr]:
        """
        Called by _write_row on each sentence
        Returns the tags, sentences and keywords linked to the given sentence
        """
        tags_in_sentence = []
        matches = self.matcher_dict[self.language](sentence, as_spans=True)
        for match in matches:
            keyword = match.text
            tag = self.keyword_to_tag[keyword]
            if tag not in list_matched_tags:
                list_matched_tags.append(tag)
                tags_in_sentence.append(tag)
            string_keywords = string_keywords + " " + keyword
            string_sentence = string_sentence + sentence.text

        if tags_in_sentence != []:
            tags_in_document.extend(tags_in_sentence)
        return tags_in_document, string_sentence, string_keywords

    def write_df_category(self) -> pd.DataFrame:
        """
        Write the output dataframe for One row per document with category :
        format one_row_per_doc_tag_lists
        """
        start = perf_counter()
        self.input_df.apply(self._write_row_category, args=[False], axis=1)
        logging.info(
            f"Tagging {len(self.input_df)} documents : Done in {perf_counter() - start:.2f} seconds."
        )
        return self._merge_df_columns_category()

    def _write_row_category(self, row: pd.Series, one_row_per_doc_json) -> None:
        """
        Called by write_df_category
        Appends columns to the output dataframe, depending on the output format
        """
        document = list(self.nlp_dict[self.language].pipe(row[self.splitted_sentences]))
        string_sentence, string_keywords = "", ""
        tag_columns_for_json, line, line_full = (
            defaultdict(),
            defaultdict(list),
            defaultdict(defaultdict),
        )
        for sentence in document:
            for keyword in sentence.ents:
                line, line_full = self._get_tags_in_row_category(
                    keyword, line, line_full, sentence
                )
                string_keywords = string_keywords + " " + keyword.text
                string_sentence += sentence.text
            tag_columns_for_json["tag_json_categories"] = self._fill_tags(
                (line and one_row_per_doc_json), dict(line)
            )
            tag_columns_for_json["tag_json_full"] = self._fill_tags(
                (line and one_row_per_doc_json),
                {column_name: dict(value) for column_name, value in line_full.items()},
            )
        self.tag_sentences.append(string_sentence)
        self.tag_keywords.append(string_keywords)
        self.output_df = (
            self.output_df.append(tag_columns_for_json, ignore_index=True)
            if one_row_per_doc_json
            else self.output_df.append(line, ignore_index=True)
        )

    def _get_tags_in_row_category(
        self, match: Span, line: dict, line_full: dict, sentence: Doc
    ) -> Tuple[dict, dict]:
        """
        Called by _write_row_category
        Writes the needed informations about founded tags:
        -line is a dictionary {category:tag}
        -line_full is a dictionary containing full information about the founded tags
        """
        keyword = match.text
        tag = self.keyword_to_tag[keyword]
        category = match.label_
        sentence = sentence.text
        if tag not in line_full[category]:
            line_full[category][tag] = {
                "occurence": 1,
                "sentences": [sentence],
                "keywords": [keyword],
            }

            line[category].append(tag)
        else:
            line_full[category][tag]["occurence"] += 1
            line_full[category][tag]["sentences"].append(sentence)
            line_full[category][tag]["keywords"].append(keyword)
        return line, line_full

    def _merge_df_columns_category(self) -> pd.DataFrame:
        """
        Called by _write_row_category,
        when the format is one row per document.
        Insert columns tag_sentences and tag_keywords, and returns the complete output dataframe
        """
        output_df_copy = self.output_df.copy().add_prefix("tag_list_")
        output_df_copy.insert(
            len(self.output_df.columns), self.tag_columns[0], self.tag_keywords, True
        )
        output_df_copy.insert(
            len(self.output_df.columns), self.tag_columns[1], self.tag_sentences, True
        )
        output_df_copy = pd.concat(
            [
                output_df_copy,
                self.input_df.drop(columns=[self.splitted_sentences]),
            ],
            axis=1,
        )
        output_df_copy.set_index(self.text_column, inplace=True)
        output_df_copy.reset_index(inplace=True)
        return output_df_copy

    def _merge_df_columns(self) -> pd.DataFrame:
        """
        Called by write_df
        insert columns tag_sentences and tag_keywords
        returns the complete output dataframe
        """
        self.output_df.insert(0, self.tag_columns[1], self.tag_sentences, True)
        self.output_df.insert(1, self.tag_columns[0], self.tag_keywords, True)
        self.output_df = pd.concat(
            [
                self.input_df.drop(columns=[self.splitted_sentences]),
                self.output_df,
            ],
            axis=1,
        )
        return super()._arrange_columns_order(
            self.output_df,
            self.tag_columns + [self.text_column],
        )


class FormatterByDocumentJson(FormatterByDocument):
    def __init__(self, *args, **kwargs):
        super(FormatterByDocumentJson, self).__init__(*args, **kwargs)

    def write_df(self) -> pd.DataFrame:
        """
        Writes the output dataframe for the json format without category
        """
        start = perf_counter()
        self.input_df.apply(self._write_row, axis=1)
        logging.info(
            f"Tagging {len(self.input_df)} documents : Done in {perf_counter() - start:.2f} seconds."
        )
        return super()._arrange_columns_order(
            pd.concat(
                [
                    self.output_df,
                    self.input_df.drop(columns=[self.splitted_sentences]),
                ],
                axis=1,
            ),
            self.tag_columns + [self.text_column],
        )

    def _write_row(self, row: pd.Series) -> None:
        """
        Called by write_df on each row
        Updates column tag_json_full with informations about the founded tags
        """
        document = list(self.nlp_dict[self.language].pipe(row[self.splitted_sentences]))
        line_full, tag_column_for_json = defaultdict(defaultdict), {}
        for sentence in document:
            matches = self.matcher_dict[self.language](sentence, as_spans=True)
            for keyword in matches:
                line_full = self._get_tags_in_row(keyword, line_full, sentence)

        tag_column_for_json["tag_json_full"] = self._fill_tags(
            line_full,
            {column_name: dict(value) for column_name, value in line_full.items()},
        )

        self.output_df = self.output_df.append(tag_column_for_json, ignore_index=True)

    def write_df_category(self) -> pd.DataFrame:
        """
        Write the output dataframe for One row per document with category :
        format one_row_per_doc_json
        """
        start = perf_counter()
        self.input_df.apply(super()._write_row_category, args=[True], axis=1)
        logging.info(
            f"Tagging {len(self.input_df)} documents : Done in {perf_counter() - start:.2f} seconds."
        )
        return super()._arrange_columns_order(
            pd.concat(
                [
                    self.output_df,
                    self.input_df.drop(columns=[self.splitted_sentences]),
                ],
                axis=1,
            ),
            self.tag_columns + [self.text_column],
        )

    def _get_tags_in_row(self, match: Span, line_full: dict, sentence: Doc) -> dict:
        """
        Called by _write_row on each sentence
        Returns a dictionary containing precisions about each tag
        """
        keyword = match.text
        tag = self.keyword_to_tag[keyword]
        sentence = sentence.text
        if tag not in line_full.keys():
            line_full[tag] = {
                "occurence": 1,
                "sentences": [sentence],
                "keywords": [keyword],
            }

        else:
            line_full[tag]["occurence"] += 1
            line_full[tag]["sentences"].append(sentence)
            line_full[tag]["keywords"].append(keyword)

        return line_full
