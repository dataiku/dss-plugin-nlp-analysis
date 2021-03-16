"""
Module to write the output dataframe depending on the output format
This module inherits from Tagger where the tokenization, sentence splitting and Matcher instanciation has been done
"""
from tagger import Tagger
from collections import defaultdict
import pandas as pd, numpy as np
from enum import Enum


class OutputFormat(Enum):
    ONE_ROW_PER_DOC = "one_row_per_doc"
    ONE_ROW_PER_DOC_JSON = "one_row_per_doc_json"
    ONE_ROW_PER_TAG = "one_row_per_tag"


class TaggerFormatter:
    def __init__(self, tagger_instance):
        self.tagger_instance = tagger_instance
        self.df_duplicated_lines = pd.DataFrame()
        self.output_df = pd.DataFrame()
        self.tag_sentences = []
        self.tag_keywords = []
        self.matches = False
        self.output_dataset_columns = self._get_output_dataset_columns()

    def tagging_procedure(self):
        """
        Public function called by recipe.py
        Returns the output dataframe depending on the given output format
        """

        if self.tagger_instance.mode == OutputFormat.ONE_ROW_PER_TAG.value:
            return self._row_per_keyword()
        elif self.tagger_instance.mode == OutputFormat.ONE_ROW_PER_DOC_JSON.value:
            if self.tagger_instance.category_column:
                return self._row_per_document_category()
            else:
                return self._row_per_document_json()
        else:
            if (
                self.tagger_instance.category_column
            ):  # one_row_per_doc with category /  one_row_per_doc_json
                self._row_per_document_category()
                return self._row_per_document_category_output()
            else:
                return self._row_per_document()

    def _row_per_document_json(self):
        """
        Writes the output dataframe for the json format without category
        """
        self.tagger_instance.text_df.apply(self._write_row_per_document_json, axis=1)
        return self._arrange_columns_order(
            pd.concat(
                [
                    self.output_df,
                    self.tagger_instance.text_df.drop(
                        columns=[self.tagger_instance.splitted_sentences_column]
                    ),
                ],
                axis=1,
            ),
            self.output_dataset_columns + [self.tagger_instance.text_column],
        )

    def _write_row_per_document_json(self, x):
        """
        Called by _row_per_document_json on each row
        Updates column tag_json_full with informations about the founded tags
        """
        document = list(
            self.tagger_instance.nlp_dict[self.tagger_instance.language].pipe(
                x[self.tagger_instance.splitted_sentences_column]
            )
        )
        line_full, tag_column_for_json = defaultdict(defaultdict), {}
        for sentence in document:
            matches = self.tagger_instance.matcher_dict[self.tagger_instance.language](
                sentence, as_spans=True
            )
            for match in matches:
                line_full = self._get_tags_row_per_document_json(
                    match, line_full, sentence
                )
        if line_full:
            tag_column_for_json["tag_json_full"] = {
                column_name: dict(value) for column_name, value in line_full.items()
            }
        self.output_df = self.output_df.append(tag_column_for_json, ignore_index=True)

    def _row_per_document(self):
        """Write the output dataframe for One row per document format (without categories)"""
        self.tagger_instance.text_df.apply(self._write_row_per_document, axis=1)
        return self._merge_df_columns()

    def _write_row_per_document(self, x):
        """Called by _row_per_document
        Appends columns of sentences,keywords and tags to the output dataframe"""
        document = list(
            self.tagger_instance.nlp_dict[self.tagger_instance.language].pipe(
                x[self.tagger_instance.splitted_sentences_column]
            )
        )
        tag_columns_in_document = defaultdict(list)
        list_matched_tags = []
        string_sentence, string_keywords = "", ""
        for sentence in document:
            (
                tag_columns_in_document,
                string_sentence,
                string_keywords,
            ) = self._get_tags_row_per_document(
                sentence,
                list_matched_tags,
                tag_columns_in_document,
                string_sentence,
                string_keywords,
            )
        self.tag_sentences.append(string_sentence)
        self.tag_keywords.append(string_keywords)
        self.output_df = self.output_df.append(
            tag_columns_in_document, ignore_index=True
        )

    def _row_per_document_category(self):
        """
        Write the output dataframe for One row per document with category :
        format one_row_per_doc_tag_lists / format JSON
        """
        self.tagger_instance.text_df.apply(
            self._write_row_per_document_category, axis=1
        )
        if self.tagger_instance.mode == OutputFormat.ONE_ROW_PER_DOC_JSON.value:
            return self._arrange_columns_order(
                pd.concat(
                    [
                        self.output_df,
                        self.tagger_instance.text_df.drop(
                            columns=[self.tagger_instance.splitted_sentences_column]
                        ),
                    ],
                    axis=1,
                ),
                self.output_dataset_columns + [self.tagger_instance.text_column],
            )
        return self._row_per_document_category_output()

    def _write_row_per_document_category(self, x):
        """
        Called by _row_per_document category
        Appends columns to the output dataframe, depending on the output format
        """
        document = list(
            self.tagger_instance.nlp_dict[self.tagger_instance.language].pipe(
                x[self.tagger_instance.splitted_sentences_column]
            )
        )
        string_sentence, string_keywords = "", ""
        tag_columns_for_json, line, line_full = (
            defaultdict(),
            defaultdict(list),
            defaultdict(defaultdict),
        )
        for sentence in document:
            for pattern in sentence.ents:
                line, line_full = self._get_tags_row_per_document_category(
                    pattern, line, line_full, sentence
                )
                string_keywords = string_keywords + " " + pattern.text
                string_sentence += sentence.text
            if (
                line
                and self.tagger_instance.mode == OutputFormat.ONE_ROW_PER_DOC_JSON.value
            ):
                tag_columns_for_json["tag_json_categories"] = dict(line)
                tag_columns_for_json["tag_json_full"] = {
                    column_name: dict(value) for column_name, value in line_full.items()
                }
        self.tag_sentences.append(string_sentence)
        self.tag_keywords.append(string_keywords)
        self.output_df = (
            self.output_df.append(tag_columns_for_json, ignore_index=True)
            if self.tagger_instance.mode == OutputFormat.ONE_ROW_PER_DOC_JSON.value
            else self.output_df.append(line, ignore_index=True)
        )

    def _row_per_document_category_output(self):
        """
        Called by _row_per_document_category,
        when the format is one row per document.
        Insert columns tag_sentences and tag_keywords, and returns the complete output dataframe
        """
        output_df_copy = self.output_df.copy().add_prefix("tag_list_")
        output_df_copy.insert(
            len(self.output_df.columns), "tag_keywords", self.tag_keywords, True
        )
        output_df_copy.insert(
            len(self.output_df.columns), "tag_sentences", self.tag_sentences, True
        )
        output_df_copy = pd.concat(
            [
                output_df_copy,
                self.tagger_instance.text_df.drop(
                    columns=[self.tagger_instance.splitted_sentences_column]
                ),
            ],
            axis=1,
        )
        output_df_copy.set_index(self.tagger_instance.text_column, inplace=True)
        output_df_copy.reset_index(inplace=True)
        return output_df_copy

    def _row_per_keyword(self):
        """Write the output dataframe for one_row_per_tag format (with or without categories)"""
        self.tagger_instance.text_df.apply(self._write_row_per_keyword, axis=1)
        self.output_df.reset_index(drop=True, inplace=True)
        self.df_duplicated_lines.reset_index(drop=True, inplace=True)
        return self._arrange_columns_order(
            pd.concat(
                [
                    self.output_df,
                    self.df_duplicated_lines.drop(
                        columns=[self.tagger_instance.splitted_sentences_column]
                    ),
                ],
                axis=1,
            ),
            self.output_dataset_columns + [self.tagger_instance.text_column],
        )

    def _write_row_per_keyword(self, x):
        """
        Called by _row_per_keyword on each row
        Updates the output dataframes which will be concatenated after :
        -> output_df contains the columns with informations about the tags
        -> df_duplicated_lines contains the original rows of the Document Dataset, with copies
        There are as many copies of a document as there are keywords in this document
        """
        self.matches = False
        document = list(
            self.tagger_instance.nlp_dict[self.tagger_instance.language].pipe(
                x[self.tagger_instance.splitted_sentences_column]
            )
        )
        matches = []
        empty_row = {column: np.nan for column in self.output_dataset_columns}
        if not self.tagger_instance.category_column:
            matches = [
                (
                    self.tagger_instance.matcher_dict[self.tagger_instance.language](
                        sentence, as_spans=True
                    ),
                    sentence,
                )
                for sentence in document
            ]
            self._get_tags_row_per_keyword(document, matches, x)
        else:
            self._get_tags_row_per_keyword_category(document, x)
        if not self.matches:
            self.output_df = self.output_df.append(empty_row, ignore_index=True)
            self.df_duplicated_lines = self.df_duplicated_lines.append(x)

    def _get_tags_row_per_document_json(self, match, line_full, sentence):
        """
        Called by _row_per_document_json on each sentence
        Returns a dictionary containing precisions about each tag
        """
        keyword = match.text
        tag = self.tagger_instance.keyword_to_tag[keyword]
        sentence = sentence.text
        if tag not in line_full.keys():
            json_dictionary = {
                "occurence": 1,
                "sentences": [sentence],
                "keywords": [keyword],
            }
            line_full[tag] = json_dictionary
        else:
            line_full[tag]["occurence"] += 1
            line_full[tag]["sentences"].append(sentence)
            line_full[tag]["keywords"].append(keyword)

        return line_full

    def _get_tags_row_per_document(
        self,
        sentence,
        list_matched_tags,
        tag_columns_in_document,
        string_sentence,
        string_keywords,
    ):
        """
        Called by _row_per_document on each sentence
        Returns the tags, sentences and keywords linked to the given sentence
        """
        tag_columns_in_sentence = defaultdict(list)
        matches = self.tagger_instance.matcher_dict[self.tagger_instance.language](
            sentence, as_spans=True
        )
        for span in matches:
            keyword = span.text
            tag = self.tagger_instance.keyword_to_tag[keyword]
            if tag not in list_matched_tags:
                list_matched_tags.append(tag)
                tag_columns_in_sentence[self.output_dataset_columns[2]].append(tag)
            string_keywords = string_keywords + " " + keyword
            string_sentence = string_sentence + sentence.text

        if tag_columns_in_sentence != {}:
            tag_columns_in_document[self.output_dataset_columns[2]].extend(
                tag_columns_in_sentence[self.output_dataset_columns[2]]
            )
        return tag_columns_in_document, string_sentence, string_keywords

    def _get_tags_row_per_document_category(self, pattern, line, line_full, sentence):
        """
        Called by _row_per_document_category
        Writes the needed informations about founded tags:
        -line is a dictionary {category:tag}
        -line_full is a dictionary containing full information about the founded tags
        """
        keyword = pattern.text
        tag = self.tagger_instance.keyword_to_tag[keyword]
        category = pattern.label_
        sentence = sentence.text
        if tag not in line_full[category]:
            json_dictionary = {
                "occurence": 1,
                "sentences": [sentence],
                "keywords": [keyword],
            }
            line_full[category][tag] = json_dictionary
            line[category].append(tag)
        else:
            line_full[category][tag]["occurence"] += 1
            line_full[category][tag]["sentences"].append(sentence)
            line_full[category][tag]["keywords"].append(keyword)
        return line, line_full

    def _get_tags_row_per_keyword(self, document, matches, x):
        """
        Called by _row_per_keyword
        Creates the list of new rows with infos about the tags and gives it to _update_output_df function
        """
        values = []
        for match, sentence in matches:
            values = [
                {
                    "tag_keyword": span.text,
                    "tag_sentence": sentence.text,
                    "tag": self.tagger_instance.keyword_to_tag[span.text],
                }
                for span in match
            ]
            self._update_output_df(match, values, x)

    def _get_tags_row_per_keyword_category(self, document, x):
        """
        Called by _row_per_keyword_category
        Creates the list of new rows with infos about the tags and gives it to _update_output_df function
        """
        tag_rows = []
        for sentence in document:
            tag_rows = [
                {
                    "tag_keyword": ent.text,
                    "tag_sentence": sentence.text,
                    "tag_category": ent.label_,
                    "tag": self.tagger_instance.keyword_to_tag[ent.text],
                }
                for ent in sentence.ents
            ]
            self._update_output_df(tag_rows, tag_rows, x)

    def _update_output_df(self, match, vals, x):
        """
        Called by _get_tags_row_per_keyword_category and _get_tags_row_per_keyword
        Appends:
        -row with infos about the founded tags to output_df
        -duplicated initial row from the Document dataframe(text_df) to df.duplicated_lines
        """
        if match:
            self.output_df = self.output_df.append(vals, ignore_index=True)
            self.df_duplicated_lines = self.df_duplicated_lines.append(
                [x for i in range(len(vals))]
            )
            self.matches = True

    def _merge_df_columns(self):
        """
        Called by _row_per_document
        insert columns tag_sentences and tag_keywords
        returns the complete output dataframe
        """
        self.output_df.insert(
            0, self.output_dataset_columns[1], self.tag_sentences, True
        )
        self.output_df.insert(
            1, self.output_dataset_columns[0], self.tag_keywords, True
        )
        self.output_df = pd.concat(
            [
                self.tagger_instance.text_df.drop(
                    columns=[self.tagger_instance.splitted_sentences_column]
                ),
                self.output_df,
            ],
            axis=1,
        )
        return self._arrange_columns_order(
            self.output_df,
            self.output_dataset_columns + [self.tagger_instance.text_column],
        )

    def _arrange_columns_order(self, df, columns):
        """Put columns in the right order in the Output dataframe"""
        for column in columns:
            df.set_index(column, inplace=True)
            df.reset_index(inplace=True)
        return df

    def _get_output_dataset_columns(self):
        """Returns the list of additional columns for the output dataframe"""
        if self.tagger_instance.category_column:
            if self.tagger_instance.mode == OutputFormat.ONE_ROW_PER_DOC_JSON.value:
                return ["tag_json_full", "tag_json_categories"]
            if self.tagger_instance.mode == OutputFormat.ONE_ROW_PER_DOC.value:
                return ["tag_keywords", "tag_sentences"]
            if self.tagger_instance.mode == OutputFormat.ONE_ROW_PER_TAG.value:
                return ["tag_keyword", "tag_sentence", "tag_category", "tag"]
        else:
            if self.tagger_instance.mode == OutputFormat.ONE_ROW_PER_DOC_JSON.value:
                return ["tag_json_full"]
            if self.tagger_instance.mode == OutputFormat.ONE_ROW_PER_DOC.value:
                return ["tag_keywords", "tag_sentences", "tag_list"]
            if self.tagger_instance.mode == OutputFormat.ONE_ROW_PER_TAG.value:
                return ["tag_keyword", "tag_sentence", "tag"]