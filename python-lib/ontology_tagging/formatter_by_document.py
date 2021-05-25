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
        super().__init__(*args, **kwargs)
        self.tag_sentences, self.tag_keywords = [], []

    @staticmethod
    def _fill_tags(value: dict, not_empty: bool = True):
        """Dump the dictionary 'value' if 'not_empty' is True"""
        return json.dumps(value, ensure_ascii=False) if not_empty else np.nan

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
        return self._set_columns_order(input_df, self.output_df, text_column)

    def _write_row(self, row: pd.Series, language_column: AnyStr = None) -> None:
        """Called by write_df on each row
        Append columns 'tag_sentences', 'tag_keywords' and 'tag_list' to the output dataframe self.output_df
        Args:
            row: pandas.Series from text_df
            language_column: if not None, the matcher will be applied with the given language of the row
        """
        language = self._get_document_language(row, language_column)
        document_to_match = self._get_document_to_match(row, language)
        tags_in_document = []
        keywords_in_document = []
        matched_sentences = []
        for idx, sentence in enumerate(document_to_match):
            original_sentence = row[self.text_column_tokenized][idx]
            tags_in_sentence, keywords_in_sentence = self._get_tags_in_sentence(
                sentence=sentence,
                original_sentence=original_sentence,
                language=language,
            )
            tags_in_document.extend(tags_in_sentence)
            keywords_in_document.extend(keywords_in_sentence)
            if tags_in_sentence:
                matched_sentences.append(original_sentence + "\n")

        if tags_in_document:
            tag_datas = {
                self.tag_columns[0]: self._fill_tags(unique_list(tags_in_document)),
                self.tag_columns[1]: self._fill_tags(unique_list(keywords_in_document)),
                self.tag_columns[2]: "".join(matched_sentences),
            }
        else:
            tag_datas = {column: np.nan for column in self.tag_columns}
        self.output_df = self.output_df.append(tag_datas, ignore_index=True)

    def _get_tags_in_sentence(
        self,
        sentence: Doc,
        original_sentence: Span,
        language: AnyStr,
    ) -> Tuple[List[AnyStr], List[AnyStr]]:
        """
        Called by _write_row on each sentence.

        Args:
            sentence (Doc): the sentence to search matches in (possibly previously normalized by _get_document_to_match)
            original_sentence (Span): the original sentence from the input dataframe
            language (AnyStr): Language of the current document

        Returns:
            Lists of tags and keywords found in a sentence

        """
        matches = self._matcher_dict[language](sentence, as_spans=True)
        tags_in_sentence = []
        keywords_in_sentence = []
        for match in matches:
            keyword = match.text
            tag = self._keyword_to_tag[language][
                get_span_text(span=match, lemmatize=self.lemmatization)
            ]
            tags_in_sentence.append(tag)
            keywords_in_sentence.append(keyword)
        return tags_in_sentence, keywords_in_sentence

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
        matched_sentence = []
        keyword_list = []
        categories_and_tags = defaultdict(list)
        for idx, sentence in enumerate(document_to_match):
            original_sentence = row[self.text_column_tokenized][idx]
            for keyword in sentence.ents:
                tag = keyword.ent_id_
                category = keyword.label_
                if tag not in categories_and_tags[category]:
                    categories_and_tags[category].append(tag)
                if keyword.text not in keyword_list:
                    keyword_list.append(keyword.text)
                if original_sentence + "\n" not in matched_sentence:
                    matched_sentence.append(original_sentence + "\n")
        categories_and_tags = {
            column: self._fill_tags(categories_and_tags[column])
            for column in categories_and_tags
        }
        if keyword_list:
            self.tag_keywords.append(self._fill_tags(keyword_list))
        else:
            self.tag_keywords.append(np.nan)
        self.tag_sentences.append(" ".join(matched_sentence))
        self.output_df = self.output_df.append(
            categories_and_tags,
            ignore_index=True,
        )

    def _merge_df_columns_category(
        self, input_df: pd.DataFrame, text_column: AnyStr
    ) -> pd.DataFrame:
        """
        Called by _write_row_category, when the format is 'one row per document (array columns)' and there are categories.
        Insert columns 'tag_keywords' and 'tag_sentences' into self.output_df

        Returns:
            pd.DataFrame: the complete output dataframe after setting the columns in the right order

        """
        tag_list_columns = self.output_df.columns.tolist()
        tag_list_columns_unique = generate_unique_columns(
            df=self.output_df,
            columns=[
                unicode_normalize_text(text=column) for column in tag_list_columns
            ],
            prefix="tag_list",
        )
        self.output_df.columns = tag_list_columns_unique
        for column, column_unique in zip(tag_list_columns, tag_list_columns_unique):
            category = column.split("_")[-1]
            self.column_descriptions[column_unique] = f"List of '{category}' tags"
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
        self.tag_columns = tag_list_columns_unique + self.tag_columns
        return self._set_columns_order(input_df, self.output_df, text_column)


class FormatterByDocumentJson(FormatterByDocument):
    """Class to write a dataframe which contains one row per document, with tag datas in nested json columns"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        tags_full, tag_column = defaultdict(defaultdict), {}
        for idx, sentence in enumerate(document_to_match):
            matches = self._matcher_dict[language](sentence, as_spans=True)
            for keyword in matches:
                original_sentence = row[self.text_column_tokenized][idx]
                tag = self._keyword_to_tag[language][
                    get_span_text(span=keyword, lemmatize=self.lemmatization)
                ]
                if tag not in tags_full.keys():
                    tags_full[tag] = {
                        "count": 1,
                        "sentences": [original_sentence],
                        "keywords": [keyword.text],
                    }
                else:
                    tags_full[tag]["count"] += 1
                    if original_sentence not in tags_full[tag]["sentences"]:
                        tags_full[tag]["sentences"].append(original_sentence)
                    if keyword.text not in tags_full[tag]["keywords"]:
                        tags_full[tag]["keywords"].append(keyword.text)
        tag_column["tag_json_full"] = self._fill_tags(
            not_empty=bool(tags_full),
            value={
                column_name: dict(value) for column_name, value in tags_full.items()
            },
        )
        self.output_df = self.output_df.append(tag_column, ignore_index=True)

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

    def _write_row_category(
        self, row: pd.Series, one_row_per_doc_json: bool, language_column: AnyStr = None
    ) -> None:
        language = self._get_document_language(row, language_column)
        document_to_match = self._get_document_to_match(row, language)
        tag_columns = defaultdict()
        categories_and_tags = defaultdict(list)
        categories_and_tags_full = defaultdict(defaultdict)
        for idx, sentence in enumerate(document_to_match):
            original_sentence = row[self.text_column_tokenized]
            for keyword in sentence.ents:
                tag = keyword.ent_id_
                category = keyword.label_
                if tag not in categories_and_tags_full[category]:
                    categories_and_tags_full[category][tag] = {
                        "count": 1,
                        "sentences": [original_sentence[idx]],
                        "keywords": [keyword.text],
                    }
                    categories_and_tags[category].append(tag)
                else:
                    categories_and_tags_full[category][tag]["count"] += 1
                    if (
                        original_sentence[idx]
                        not in categories_and_tags_full[category][tag]["sentences"]
                    ):
                        categories_and_tags_full[category][tag]["sentences"].append(
                            original_sentence[idx]
                        )
                    if (
                        keyword.text
                        not in categories_and_tags_full[category][tag]["keywords"]
                    ):
                        categories_and_tags_full[category][tag]["keywords"].append(
                            keyword.text
                        )

            tag_columns["tag_json_categories"] = self._fill_tags(
                not_empty=bool(categories_and_tags), value=dict(categories_and_tags)
            )
            tag_columns["tag_json_full"] = self._fill_tags(
                not_empty=bool(categories_and_tags),
                value={
                    column_name: dict(value)
                    for column_name, value in categories_and_tags_full.items()
                },
            )
        self.output_df = self.output_df.append(tag_columns, ignore_index=True)
