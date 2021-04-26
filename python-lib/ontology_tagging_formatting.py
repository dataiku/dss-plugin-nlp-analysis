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
from plugin_io_utils import move_columns_after, unique_list, get_keyword, get_sentence, get_tag
from spacy_tokenizer import MultilingualTokenizer


class Formatter:
    def __init__(
        self,
        language: AnyStr,
        tokenizer: MultilingualTokenizer,
        category_column: AnyStr,
        case_insensitivity: bool,
        lemmatization: bool,
        text_column_tokenized: AnyStr,
        text_lower_column_tokenized: AnyStr = None,
        keyword_to_tag: dict = None,
        matcher_dict: dict = None,
    ):
        store_attr()
        self.output_df = pd.DataFrame()

    def _get_document_language(
        self, row: pd.Series, language_column: AnyStr = None
    ) -> AnyStr:
        """Return the language of the document in the row"""
        return row[language_column] if language_column else self.language

    def _get_document_to_match(self, row: pd.Series) -> List:
        """Return the original document (as list of sentences) or, the lowercase one"""
        if self.case_insensitivity:
            return row[self.text_lower_column_tokenized]
        else:
            return row[self.text_column_tokenized]

    def _columns_to_drop(self) -> None:
        """Return the names of the column(s) to drop from the final output dataset, i.e column(s) of splitted sentences"""
        return (
            [self.text_column_tokenized, self.text_lower_column_tokenized]
            if self.case_insensitivity
            else [self.text_column_tokenized]
        )

    def _set_columns_order(
        self, input_df: pd.DataFrame, output_df: pd.DataFrame, text_column: AnyStr
    ) -> pd.DataFrame:
        """Concatenate the input_df with the new one,reset its columns in the right order, and return it"""
        input_df = input_df.drop(columns=self._columns_to_drop())
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
        ruler=None,
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
        language = super()._get_document_language(row, language_column)
        matches = []
        document_to_match = super()._get_document_to_match(row)
        empty_row = {column: np.nan for column in self.tag_columns}
        if not self.category_column:
            matches = [
                (
                    self.matcher_dict[language](sentence, as_spans=True),
                    row[self.text_column_tokenized][
                        idx
                    ],  # original sentence (not lowercased)
                )
                for idx, sentence in enumerate(document_to_match)
            ]
            self._get_tags_in_row(matches, row, language)
        else:
            ruler = self.tokenizer.spacy_nlp_dict[language].get_pipe("entity_ruler")
            self._get_tags_in_row_category(document_to_match, row, language, ruler)
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
            print(match,sentence)
            values = [
                self._list_to_dict(
                    [
                        self.keyword_to_tag[language][get_tag(self.case_insensitivity,self.lemmatization,keyword)]
                        ,
                        sentence.text,
                        keyword.text,
                    ]
                )
                for keyword in match
            ]
            self._update_df(match, values, row)

    def _get_tags_in_row_category(
        self, document_to_match: List, row: pd.Series, language: AnyStr, ruler
    ) -> None:
        """
        Called by _write_row_category
        Create the list of new rows with infos about the tags and gives it to _update_df function
        """
        tag_rows = []
        for idx, sentence in enumerate(document_to_match):
            tag_rows = [
                self._list_to_dict(
                    [
                        keyword.ent_id_,
                        keyword.label_,
                        list(row[self.text_column_tokenized])[idx].text,
                        keyword.text,
                    ]
                )
                for keyword in ruler(
                    get_sentence(sentence, self.case_insensitivity)
                ).ents
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
        language = super()._get_document_language(row, language_column)
        document_to_match = super()._get_document_to_match(row)
        tags_in_document, keywords_in_document, matched_sentences = [], [], []
        for idx, sentence in enumerate(document_to_match):
            (
                tags_in_document,
                keywords_in_document,
                matched_sentences,
            ) = self._get_tags_in_row(
                sentence=sentence,
                original_sentence=row[self.text_column_tokenized][idx],
                tags_in_document=tags_in_document,
                keywords_in_document=keywords_in_document,
                matched_sentences=matched_sentences,
                language=language,
            )

        if tags_in_document != []:
            line = {
                self.tag_columns[0]: unique_list(tags_in_document),
                self.tag_columns[1]: "".join(unique_list(matched_sentences)),
                self.tag_columns[2]: ", ".join(unique_list(keywords_in_document)),
            }
        else:
            line = {column: np.nan for column in self.tag_columns}
        self.output_df = self.output_df.append(line, ignore_index=True)

    def _get_tags_in_row(
        self,
        sentence: Doc,
        original_sentence: Span,
        tags_in_document: List,
        keywords_in_document: List,
        matched_sentences: List,
        language: AnyStr,
    ) -> Tuple[List[AnyStr], List[AnyStr], List[AnyStr]]:
        """
        Called by _write_row on each sentence
        Return the tags, sentences and keywords linked to the given sentence
        """
        matches = self.matcher_dict[language](sentence, as_spans=True)
        for match in matches:
            keyword = match.text
            tag = self.keyword_to_tag[language][
                get_keyword(keyword, self.case_insensitivity)
            ]
            tags_in_document.append(tag)
            keywords_in_document.append(keyword)
            matched_sentences.append(original_sentence.text + " ")
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
        language = super()._get_document_language(row, language_column)
        document_to_match = super()._get_document_to_match(row)
        ruler = self.tokenizer.spacy_nlp_dict[language].get_pipe("entity_ruler")
        matched_sentence, keyword_list = [], []
        tag_columns_for_json, line, line_full = (
            defaultdict(),
            defaultdict(list),
            defaultdict(defaultdict),
        )
        for idx, sentence in enumerate(document_to_match):
            for keyword in ruler(get_sentence(sentence, self.case_insensitivity)).ents:
                line, line_full = self._get_tags_in_row_category(
                    match=keyword,
                    line=line,
                    line_full=line_full,
                    sentence=row[self.text_column_tokenized][idx].text,
                    language=language,
                )
                keyword_list.append(keyword.text + " ")
                matched_sentence.append(row[self.text_column_tokenized][idx].text + " ")
            tag_columns_for_json["tag_json_categories"] = self._fill_tags(
                condition=(line and one_row_per_doc_json), value=dict(line)
            )
            tag_columns_for_json["tag_json_full"] = self._fill_tags(
                condition=(line and one_row_per_doc_json),
                value={
                    column_name: dict(value) for column_name, value in line_full.items()
                },
            )
        self.tag_keywords.append(", ".join(unique_list(keyword_list)))
        self.tag_sentences.append(" ".join(unique_list(matched_sentence)))
        self.output_df = (
            self.output_df.append(tag_columns_for_json, ignore_index=True)
            if one_row_per_doc_json
            else self.output_df.append(line, ignore_index=True)
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
        tag = match.ent_id_
        category = match.label_
        sentence = sentence
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
            len(self.output_df.columns), self.tag_columns[1], self.tag_keywords, True
        )
        output_df_copy.insert(
            len(self.output_df.columns), self.tag_columns[0], self.tag_sentences, True
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
        language = super()._get_document_language(row, language_column)
        document_to_match = super()._get_document_to_match(row)
        line_full, tag_column_for_json = defaultdict(defaultdict), {}
        for idx, sentence in enumerate(document_to_match):
            matches = self.matcher_dict[language](sentence, as_spans=True)
            for keyword in matches:
                line_full = self._get_tags_in_row(
                    match=keyword,
                    line_full=line_full,
                    sentence=sentence,
                    original_sentence=row[self.text_column_tokenized][idx],
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
        self,
        match: Span,
        line_full: dict,
        sentence: Doc,
        original_sentence: Doc,
        language: AnyStr,
    ) -> dict:
        """
        Called by _write_row on each sentence
        Return a dictionary containing precisions about each tag
        """
        keyword = match.text
        tag = self.keyword_to_tag[language][
            get_keyword(keyword, self.case_insensitivity)
        ]
        sentence = original_sentence.text
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