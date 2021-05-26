import json
import numpy as np
import pandas as pd
import logging
from time import perf_counter

from typing import AnyStr
from typing import List
from typing import Tuple
from collections import defaultdict

from spacy.tokens import Span
from spacy.tokens import Doc

from utils.plugin_io_utils import unique_list
from utils.plugin_io_utils import generate_unique_columns
from utils.nlp_utils import get_span_text
from utils.nlp_utils import unicode_normalize_text

from .ontology_tagging_formatting import FormatterBase


class FormatterByDocument(FormatterBase):
    """Class to write a dataframe which contains one row per document, with associated tags and keywords stored as array columns."""

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
        """Write the output dataframe for format 'one row per document (array columns)' when there is no category in the ontology

        Returns:
            pd.DataFrame : a DataFrame with the following columns:
            - all columns from the input dataframe input_df
            - a column with the list of keywords
            - a column with the list of tags
            - a column with concatenated matched sentences

        """
        start = perf_counter()
        self._generate_columns_names(input_df)
        input_df.progress_apply(self._write_row, args=[language_column], axis=1)
        logging.info(
            f"Tagging {len(input_df)} documents : Done in {perf_counter() - start:.2f} seconds."
        )
        return self._merge_df_columns(input_df, text_column)

    def _write_row(self, row: pd.Series, language_column: AnyStr = None) -> None:
        """Called by write_df on each row
        Append columns 'tag_sentences', 'tag_keywords' and 'tag_list' to the output dataframe self.output_df
        Args:
            row: pandas.Series from text_df
            language_column: if not None, the matcher will be applied with the given language of the row
        """
        language = self._get_document_language(row, language_column)
        document_to_match = self._get_document_to_match(row, language)
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
        original_sentence: Span,
        tags_in_document: List,
        keywords_in_document: List,
        matched_sentences: List,
        language: AnyStr,
    ) -> Tuple[List[AnyStr], List[AnyStr], List[AnyStr]]:
        """
        Called by _write_row on each sentence

        Args:
            sentence (Doc): the sentence to search matches in (possibly previously normalized by _get_document_to_match)
            original_sentence (Span): the original sentence from the input dataframe
            tags_in_document (List): List of tags found in the current document
            keywords_in_document (List): List of keywords tags found in the current document
            matched_sentences (List): List of matched sentences in the current document
            language (AnyStr): Language of the current document

        Returns:
            Lists of tags, keywords and sentences enriched with 'sentence' matches

        """
        matches = self._matcher_dict[language](sentence, as_spans=True)
        for match in matches:
            keyword = match.text
            tag = self._keyword_to_tag[language][
                get_span_text(span=match, lemmatize=self.lemmatization)
            ]
            tags_in_document.append(tag)
            keywords_in_document.append(keyword)
            matched_sentences.append(original_sentence + "\n")
        return tags_in_document, keywords_in_document, matched_sentences

    def write_df_category(
        self,
        input_df: pd.DataFrame,
        text_column: AnyStr,
        language_column: AnyStr = None,
    ) -> pd.DataFrame:
        """Write the output dataframe for format 'one row per document (array columns)' when there are categories in the ontology

        Returns:
         pd.DataFrame: a dataframe with the following columns:
             - all columns from the input dataframe input_df
             - a column with lists of keywords
             - one column by category, with lists of tags
             - a column with concatenated matched sentences

        """
        start = perf_counter()
        self._generate_columns_names(input_df)
        input_df.progress_apply(
            self._write_row_category, args=[False, language_column], axis=1
        )
        logging.info(
            f"Tagging {len(input_df)} documents : Done in {perf_counter() - start:.2f} seconds."
        )
        return self._merge_df_columns_category(input_df, text_column)

    def _write_row_category(
        self, row: pd.Series, one_row_per_doc_json: bool, language_column: AnyStr = None
    ) -> None:
        """
        Called by FormatterByDocument.write_df_category or FormatterByDocumentJson.write_df_category
        Append the tag columns to the output dataframe self.output_df
        If one_row_per_doc_json is True, new columns are:
        - 'tag_json_categories' : dictionary of category (keys) and tags (value)
        - 'tag_json_full': dictionary of category (keys).
            Each category has for value a dictionary of tags and their associated datas (keywords, sentences, occurences of the tag)
        If one_row_per_doc_json is False, new columns are :
            - a column with lists of keywords
            - one column by category, with lists of tags
            - a column with concatenated matched sentences

        Args:
            row: pandas.Series , document from text_df
            one_row_per_doc_json: Bool to know if the format is nested JSON
            language_column: if not None, matcher will apply with the given language of the row

        """
        language = self._get_document_language(row, language_column)
        document_to_match = self._get_document_to_match(row, language)
        matched_sentence, keyword_list = [], []
        tag_columns_for_json, line, line_full = (
            defaultdict(),
            defaultdict(list),
            defaultdict(defaultdict),
        )
        for idx, sentence in enumerate(document_to_match):
            original_sentence = row[self.text_column_tokenized]
            for keyword in sentence.ents:
                line, line_full = self._get_tags_in_row_category(
                    match=keyword,
                    line=line,
                    line_full=line_full,
                    sentence=original_sentence[idx],
                    language=language,
                )
                keyword_list.append(keyword.text + " ")
                matched_sentence.append(original_sentence[idx] + "\n")
            tag_columns_for_json["tag_json_categories"] = self._fill_tags(
                condition=(line and one_row_per_doc_json), value=dict(line)
            )
            tag_columns_for_json["tag_json_full"] = self._fill_tags(
                condition=(line and one_row_per_doc_json),
                value={
                    column_name: dict(value) for column_name, value in line_full.items()
                },
            )
        line = {column: self._fill_tags(True, line[column]) for column in line}
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
        Enrich 'line' and 'line_full' dictionaries with new found tags

        Args:
            line (dict): dictionary of category (keys) and tags (values)
            line_full (dict): dictionary of category (keys).
                Each category has for value a dictionary of tags and their associated datas (keywords, sentences, occurences of the tag)
            sentence (Doc): sentence that matched with keyword(s)
            language (AnyStr): language of the sentence

        Returns:
            - Tuple(dict,dict) : The enriched dictionaries 'line' and 'line_full'

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
        Called by _write_row_category, when the format is 'one row per document (array columns)' and there are categories.
        Insert columns 'tag_keywords' and 'tag_sentences' into self.output_df

        Returns:
            pd.DataFrame: the complete output dataframe after setting the columns in the right order

        """
        category_columns = self.output_df.columns.tolist()
        category_columns_unique = generate_unique_columns(
            df=input_df,
            columns=[
                unicode_normalize_text(text=column) for column in category_columns
            ],
            prefix="tag_list",
        )
        self.output_df.columns = category_columns_unique
        for category, category_unique in zip(category_columns, category_columns_unique):
            self.column_descriptions[category_unique] = f"List of '{category}' tags"
        self.output_df.insert(
            len(self.output_df.columns),
            self.tag_columns[0],
            self.tag_keywords,
            True,
        )
        self.output_df.insert(
            len(self.output_df.columns),
            self.tag_columns[1],
            self.tag_sentences,
            True,
        )
        self.tag_columns = category_columns_unique + self.tag_columns
        return self._set_columns_order(input_df, self.output_df, text_column)

    def _merge_df_columns(
        self, input_df: pd.DataFrame, text_column: AnyStr
    ) -> pd.DataFrame:
        """
        Called by write_df
        Insert columns 'tag_sentences' and 'tag_keywords' into self.output_df and there is no category

        Returns:
            pd.DataFrame: the complete output dataframe after setting the columns in the right order

        """
        for column in self.tag_columns[::-1]:
            self.output_df.set_index(column, inplace=True)
            self.output_df.reset_index(inplace=True)
        return self._set_columns_order(input_df, self.output_df, text_column)


class FormatterByDocumentJson(FormatterByDocument):
    """Class to write a dataframe which contains one row per document, with tag datas in nested json columns"""

    def __init__(self, *args, **kwargs):
        super(FormatterByDocumentJson, self).__init__(*args, **kwargs)

    def write_df(
        self,
        input_df: pd.DataFrame,
        text_column: AnyStr,
        language_column: AnyStr = None,
    ) -> pd.DataFrame:
        """Write the output dataframe for format 'one row per document (nested JSON)' when there is no category in the ontology

        Returns:
            pd.DataFrame : a dataframe with the following columns:
            - all columns from the input dataframe input_df
            - a column with a dictionary of tags (keys). Each tag has for value a dictionary with:
                -key 'count' : number of occurences of the keywords in the document
                -key 'sentences' : list of the sentences that matched keywords in the document
                -key 'keywords' : list of the keywords in the document

        """
        start = perf_counter()
        self._generate_columns_names(input_df)
        input_df.progress_apply(self._write_row, args=[language_column], axis=1)
        logging.info(
            f"Tagging {len(input_df)} documents : Done in {perf_counter() - start:.2f} seconds."
        )
        return self._set_columns_order(input_df, self.output_df, text_column)

    def _write_row(self, row: pd.Series, language_column: AnyStr = None) -> None:
        """
        Called by write_df on each row
        Update the column 'tag_json_full' with the found tags

        Args:
            row: pandas.Series from text_df
            language_column: if not None, matcher will be applied with the given language of the row; uses self.language otherwise

        """
        language = self._get_document_language(row, language_column)
        document_to_match = self._get_document_to_match(row, language)
        line_full, tag_column_for_json = defaultdict(defaultdict), {}
        for idx, sentence in enumerate(document_to_match):
            matches = self._matcher_dict[language](sentence, as_spans=True)
            for keyword in matches:
                line_full = self._get_tags_in_row(
                    match=keyword,
                    line_full=line_full,
                    original_sentence=row[self.text_column_tokenized][idx],
                    language=language,
                )

        tag_column_for_json["tag_json_full"] = self._fill_tags(
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
        """Write the output dataframe for format 'one row per document (nested JSON)' when there are categories in the ontology
        Returns:
            pd.DataFrame : a dataframe with the following columns:
            - all columns from the input dataframe input_df
            - a column with a dictionary of categories (keys)
                Each category has for value a dictionary of tags (keys).
                Each tag has for value a dictionary with the following schema:
                    -key 'count' : number of occurences of the keywords in the document
                    -key 'sentences' : list of the sentences that matched keywords in the document
                    -key 'keywords' : list of the keywords in the document

        """
        start = perf_counter()
        self._generate_columns_names(input_df)
        input_df.progress_apply(
            self._write_row_category, args=[True, language_column], axis=1
        )
        logging.info(
            f"Tagging {len(input_df)} documents: Done in {perf_counter() - start:.2f} seconds."
        )
        return self._set_columns_order(input_df, self.output_df, text_column)

    def _get_tags_in_row(
        self,
        match: Span,
        line_full: dict,
        original_sentence: Doc,
        language: AnyStr,
    ) -> dict:
        """
        Called by _write_row on each sentence
        Returns:
            dict:  Dictionary of tags(keys). Each tag has for value a dictionary defined by the following keys:
                -key 'count' : number of occurences of the the keywords in the document
                -key 'sentences' : list of the sentences that matched keywords in the document
                -key 'keywords' : list of the keywords in the document with details about each tags
        """
        keyword = match.text
        tag = self._keyword_to_tag[language][
            get_span_text(span=match, lemmatize=self.lemmatization)
        ]
        if tag not in line_full.keys():
            line_full[tag] = {
                "count": 1,
                "sentences": [original_sentence],
                "keywords": [keyword],
            }
        else:
            line_full[tag]["count"] += 1
            if original_sentence not in line_full[tag]["sentences"]:
                line_full[tag]["sentences"].append(original_sentence)
            if keyword not in line_full[tag]["keywords"]:
                line_full[tag]["keywords"].append(keyword)

        return line_full