from fastcore.utils import store_attr
import pandas as pd
from collections import defaultdict
from spacy.tokens import Span, Doc
from typing import AnyStr, Dict, List, Tuple
import numpy as np
from time import perf_counter
import logging
import json
from plugin_io_utils import move_columns_after, unique_list, generate_unique_columns
from nlp_utils import get_span_text, unicode_normalize_text
from spacy_tokenizer import MultilingualTokenizer
from tqdm import tqdm

# names of all additional columns depending on the output_format
COLUMN_DESCRIPTION = {
    "tag_keywords": "List of matched keywords",
    "tag_sentences": "Sentences containing matched keywords",
    "tag_json_full": "Detailed tag column: list of matched keywords per tag and category, count of occurrences, sentences containing matched keywords",
    "tag_json_categories": "List of tags per category",
    "tag_list": "List of all assigned tags",
    "tag": "Assigned tag",
    "tag_keyword": "Matched keyword",
    "tag_sentence": "Sentence containing the matched keyword",
    "tag_category": "Category of tag",
}


class Formatter:
    """
    Write the output dataframe depending on the output format
    This class is called by the Tagger class where the tokenization, sentence splitting and Matcher instanciation has been done
    """

    def __init__(
        self,
        language: AnyStr,
        tokenizer: MultilingualTokenizer,
        category_column: AnyStr,
        normalize_case: bool,
        lemmatization: bool,
        text_column_tokenized: AnyStr,
        _keyword_to_tag: dict = None,
        _matcher_dict: dict = None,
    ):
        store_attr()
        self.output_df = pd.DataFrame()
        tqdm.pandas(miniters=1, mininterval=5.0)
        self._column_descriptions = COLUMN_DESCRIPTION

    def _generate_columns_names(self, text_df: pd.DataFrame) -> None:
        """Create unique names for tag columns and store their descriptions"""
        tag_columns_names = generate_unique_columns(text_df, self.tag_columns)
        for tag_column, tag_column_name in zip(self.tag_columns, tag_columns_names):
            self._column_descriptions[tag_column_name] = COLUMN_DESCRIPTION[tag_column]
        self.tag_columns = tag_columns_names

    def _get_document_language(
        self, row: pd.Series, language_column: AnyStr = None
    ) -> AnyStr:
        """Return the language of the document in the row"""
        return row[language_column] if language_column else self.language

    def _get_document_to_match(self, row: pd.Series, language) -> List:
        """Return the original document (as list of sentences) or, the lowercase one"""
        if self.normalize_case:
            return list(
                self.tokenizer.spacy_nlp_dict[language].pipe(
                    [sentence.lower() for sentence in row[self.text_column_tokenized]]
                )
            )
        else:
            return list(
                self.tokenizer.spacy_nlp_dict[language].pipe(
                    row[self.text_column_tokenized]
                )
            )

    def _set_columns_order(
        self, input_df: pd.DataFrame, output_df: pd.DataFrame, text_column: AnyStr
    ) -> pd.DataFrame:
        """Concatenate the input_df with the new one,reset its columns in the right order, and return it"""
        input_df = input_df.drop(columns=self.text_column_tokenized)
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
        self._generate_columns_names(input_df)
        input_df.progress_apply(self._write_row, args=[language_column], axis=1)
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
        document_to_match = super()._get_document_to_match(row, language)
        empty_row = {column: np.nan for column in self.tag_columns}
        if not self.category_column:
            matches = [
                (
                    self._matcher_dict[language](sentence, as_spans=True),
                    row[self.text_column_tokenized][idx],
                )
                for idx, sentence in enumerate(document_to_match)
            ]
            self._get_tags_in_row(matches, row, language)
        else:
            self._get_tags_in_row_category(document_to_match, row, language)
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
                        self._keyword_to_tag[language][
                            get_span_text(
                                span=keyword, lemmatize=self.lemmatization
                            )
                        ],
                        keyword.text,
                        sentence,
                    ]
                )
                for keyword in match
            ]
            self._update_df(match, values, row)

    def _get_tags_in_row_category(
        self, document_to_match: List, row: pd.Series, language: AnyStr
    ) -> None:
        """
        Called by _write_row_category
        Create the list of new rows with infos about the tags and gives it to _update_df function
        """
        tag_rows = []
        original_document = list(row[self.text_column_tokenized])
        for idx, sentence in enumerate(document_to_match):
            tag_rows = [
                self._list_to_dict(
                    [
                        keyword.label_,
                        keyword.ent_id_,
                        keyword.text,
                        original_document[idx],
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
        self._generate_columns_names(input_df)
        input_df.progress_apply(self._write_row, args=[language_column], axis=1)
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
        document_to_match = super()._get_document_to_match(row, language)
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
        Return the tags, sentences and keywords linked to the given sentence
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
        """Write the output dataframe for One row per document with category"""
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
        Called by write_df_category
        Append columns to the output dataframe, depending on the output format
        Args:
            row: pandas.Series , document from text_df
            one_row_per_doc_json: Bool to know if the format is JSON
            language_column: if not None, matcher will apply with the given language of the row
        """
        language = super()._get_document_language(row, language_column)
        document_to_match = super()._get_document_to_match(row, language)
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
        tag_list_columns = self.output_df.columns.tolist()
        tag_list_columns_unique = generate_unique_columns(
            df=self.output_df,
            columns=unicode_normalize_text(tag_list_columns),
            prefix="tag_list",
        )
        self.output_df.columns = tag_list_columns_unique
        for column, column_unique in zip(tag_list_columns, tag_list_columns_unique):
            category = column.split("_")[-1]
            self._column_descriptions[column_unique] = f"List of '{category}' tags"
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
        self._generate_columns_names(input_df)
        input_df.progress_apply(self._write_row, args=[language_column], axis=1)
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
        document_to_match = super()._get_document_to_match(row, language)
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
        self._generate_columns_names(input_df)
        input_df.progress_apply(
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
        original_sentence: Doc,
        language: AnyStr,
    ) -> dict:
        """
        Called by _write_row on each sentence
        Return a dictionary containing precisions about each tag
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
