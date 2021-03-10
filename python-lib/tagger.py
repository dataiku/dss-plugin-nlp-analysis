from tagger_init import TaggerInitializer
from collections import defaultdict
import pandas as pd, numpy as np

"""Module to write the output dataframe depending on the output format"""


class Tagger(TaggerInitializer):
    def __init__(self, settings):
        super().__init__(settings)
        self.df_duplicated_lines = pd.DataFrame()
        self.output_df = pd.DataFrame()
        self.tag_sentences = []
        self.tag_keywords = []
        self.no_matches = False

    def tagging_procedure(self):
        """
        Public function called by recipe.py
        Returns the output dataframe depending on the given output format
        """
        self.get_patterns()
        matcher = self.nlp_pipeline()

        if self.mode == "one_row_per_tag":
            return self._row_per_keyword(matcher)
        elif self.mode == "one_row_per_doc_json":
            if bool(self.category_column):
                return self._row_per_document_category()
            else:
                raise NotImplementedError(
                    "JSON output without category has not been implemented yet."
                )
        else:
            if bool(
                self.category_column
            ):  # one_row_per_doc with category /  one_row_per_doc_json
                self._row_per_document_category()
                return self._row_per_document_category_output()
            else:
                return self._row_per_document(matcher)

    def _row_per_document(self, matcher):
        """Write the output dataframe for One row per document format (without categories) """
        for document in self.text_df["list-sentences"]:
            document = list(self.nlp.pipe(document))
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
                    matcher,
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
        return self._merge_df_columns()

    def _row_per_document_category(self):
        """
        Write the output dataframe for One row per document with category :
        format one_row_per_doc_tag_lists / format JSON
        """
        for document in self.text_df["list-sentences"]:
            document = list(self.nlp.pipe(document))
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
                if bool(line) and self.mode == "one_row_per_doc_json":
                    tag_columns_for_json["tag_json_categories"] = dict(line)
                    tag_columns_for_json["tag_json_full"] = {
                        column_name: dict(value)
                        for column_name, value in line_full.items()
                    }
            self.tag_sentences.append(string_sentence)
            self.tag_keywords.append(string_keywords)
            tag_columns_in_document = (
                tag_columns_for_json if self.mode == "one_row_per_doc_json" else line
            )
            self.output_df = self.output_df.append(
                tag_columns_in_document, ignore_index=True
            )
        if self.mode == "one_row_per_doc_json":
            return self._arrange_columns_order(
                pd.concat([self.output_df, self.text_df], axis=1),
                self.output_dataset_columns + [self.text_column],
            )
        return self._row_per_document_category_output()

    def _row_per_document_category_output(self):
        """
        Called by _row_per_document_category,
        when the format is one_row_per_doc_tag_lists.
        Insert columns tag_sentences and tag_keywords, and returns the complete output dataframe
        """
        output_df_copy = self.output_df.copy().add_prefix("tag_list_")
        output_df_copy.insert(
            len(self.output_df.columns), "tag_keywords", self.tag_keywords, True
        )
        output_df_copy.insert(
            len(self.output_df.columns), "tag_sentences", self.tag_sentences, True
        )

        output_df_copy = pd.concat([output_df_copy, self.text_df], axis=1)
        output_df_copy.set_index(self.text_column, inplace=True)
        output_df_copy.reset_index(inplace=True)
        return output_df_copy

    def _row_per_keyword(self, matcher):
        """Write the output dataframe for one_row_per_tag format (with or without categories)"""
        self.text_df.apply(self._write_row_per_keyword, args=[matcher], axis=1)
        self.output_df.reset_index(drop=True, inplace=True)
        self.df_duplicated_lines.reset_index(drop=True, inplace=True)
        return self._arrange_columns_order(
            pd.concat([self.output_df, self.df_duplicated_lines], axis=1),
            self.output_dataset_columns + [self.text_column],
        )

    def _write_row_per_keyword(self, x, matcher):
        """
        Called by _row_per_keyword on text_df dataframe
        """
        self.no_matches = False
        document = list(self.nlp.pipe(x["list-sentences"]))
        matches = []
        empty_row = {column: np.nan for column in self.output_dataset_columns}
        if not bool(self.category_column):
            matches = [
                (matcher(sentence, as_spans=True), sentence) for sentence in document
            ]
            self._get_tags_row_per_keyword(document, matches, x)
        else:
            self._get_tags_row_per_keyword_category(document, x)
        if not bool(self.no_matches):
            self.output_df = self.output_df.append(empty_row, ignore_index=True)
            self.df_duplicated_lines = self.df_duplicated_lines.append(x)

    def _get_tags_row_per_document_category(self, pattern, line, line_full, sentence):
        """
        Called by _row_per_document_category
        Writes the needed informations about founded tags:
        -line is a dictionary {category:tag}
        -line_full is a dictionary containing full information about the founded tags
        """
        if self.keyword_to_tag[pattern.text] not in line_full[pattern.label_]:
            json_dictionary = {
                "occurence": 1,
                "sentences": [sentence.text],
                "keywords": [pattern.text],
            }
            line_full[pattern.label_][
                self.keyword_to_tag[pattern.text]
            ] = json_dictionary
            line[pattern.label_].append(self.keyword_to_tag[pattern.text])
        else:
            line_full[pattern.label_][self.keyword_to_tag[pattern.text]][
                "occurence"
            ] += 1
            line_full[pattern.label_][self.keyword_to_tag[pattern.text]][
                "sentences"
            ].append(sentence.text)
            line_full[pattern.label_][self.keyword_to_tag[pattern.text]][
                "keywords"
            ].append(pattern.text)
        return line, line_full

    def _get_tags_row_per_document(
        self,
        sentence,
        matcher,
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
        matches = matcher(sentence)
        for _, start, end in matches:
            if self.keyword_to_tag[sentence[start:end].text] not in list_matched_tags:
                list_matched_tags.append(self.keyword_to_tag[sentence[start:end].text])
                tag_columns_in_sentence[self.output_dataset_columns[2]].append(
                    self.keyword_to_tag[sentence[start:end].text]
                )
            string_keywords = string_keywords + " " + sentence[start:end].text
            string_sentence = string_sentence + sentence.text
        if tag_columns_in_sentence != {}:
            tag_columns_in_document[self.output_dataset_columns[2]].extend(
                tag_columns_in_sentence[self.output_dataset_columns[2]]
            )
        return tag_columns_in_document, string_sentence, string_keywords

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
                    "tag": self.keyword_to_tag[span.text],
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
                    "tag": self.keyword_to_tag[ent.text],
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
        if bool(match):
            self.output_df = self.output_df.append(vals, ignore_index=True)
            self.df_duplicated_lines = self.df_duplicated_lines.append(
                [x for i in range(len(vals))]
            )
            self.no_matches = True

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
        self.output_df = pd.concat([self.text_df, self.output_df], axis=1)
        return self._arrange_columns_order(
            self.output_df, self.output_dataset_columns + [self.text_column]
        )

    def _arrange_columns_order(self, df, columns):
        """Put columns in the right order in the Output dataframe"""
        for column in columns:
            df.set_index(column, inplace=True)
            df.reset_index(inplace=True)
        return df