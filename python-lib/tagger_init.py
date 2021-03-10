from spacy_tokenizer import MultilingualTokenizer
from spacy.pipeline import EntityRuler
from spacy.matcher import PhraseMatcher
 


class TaggerInitializer:
    def __init__(self, settings):

        self.patterns = list
        self.keyword_to_tag = {}

        self.text_df = settings["text_input"].get_dataframe()
        self.ontology_df = settings["ontology_input"].get_dataframe()

        self.text_column = settings["text_column"]
        self.language = self._get_language_current(settings)

        self.tag_column = settings["tag_column"]
        self.category_column = settings["category_column"]
        self.keyword_column = settings["keyword_column"]

        self.lemmatize = settings["lemmatization"]
        self.case_sensitive = settings["case_sensitivity"]
        self.normalize = settings["unicode_normalization"]

        self.mode = settings["output_format"]
        self.output_dataset_columns = self._get_output_dataset_columns()

    def _get_output_dataset_columns(self):
        """Returns the list of additional columns for the output dataframe"""
        if bool(self.category_column):
            if self.mode == "one_row_per_doc_json":

                return ["tag_json_full", "tag_json_categories"]

            if self.mode == "one_row_per_doc":
                return ["tag_keywords", "tag_sentences"]
            if self.mode == "one_row_per_tag":
                return ["tag_keyword", "tag_sentence", "tag_category", "tag"]
        else:
            if self.mode == "one_row_per_doc_json":
                return ["tag_json_full"]
            if self.mode == "one_row_per_doc":
                return ["tag_keywords", "tag_sentences", "tag_list"]
            if self.mode == "one_row_per_tag":
                return ["tag_keyword", "tag_sentence", "tag"]

    def _get_language(self, settings):
        """Returns language of the tagger"""
        return (
            settings.language_column
            if settings.language == "language_column"
            else settings.language
        )

    def _get_language_current(self, settings):
        """Returns language of the tagger (temporary function)"""
        if settings.language != "language_column":
            return settings.language
        else:
            raise NotImplementedError(
                "Document Dataset with multiple languages has not been implemented yet."
            )

    def get_patterns(self):
        """
        Public function called in tagger.py
        Creates the list of patterns :
        - If there aren't category -> list of the keywords (string list)
        - If there are categories  -> list of dictionaries, {"label": category, "pattern": keyword}
        """
        list_of_tags = self.ontology_df[self.tag_column].values.tolist()
        match_list = self.ontology_df[self.keyword_column].values.tolist()
        self.keyword_to_tag = dict(zip(match_list, list_of_tags))

        if self.category_column is not None:
            list_of_categories = self.ontology_df[self.category_column].values.tolist()
            self.patterns = [
                {"label": label, "pattern": pattern}
                for label, pattern in zip(list_of_categories, match_list)
            ]
        else:
            self.patterns = match_list

    def _list_sentences(self, x, nlp):
        """Auxiliary function called in nlp_pipeline
        Applies sentencizer and return list of sentences"""
        return [elt.text for elt in nlp(x).sents]

    def nlp_pipeline(self):
        """
        Public function called in tagger.py
        Uses a spacy pipeline
        -Creates tokenizer from MultingualTokenizer class
        -Split sentences by applying sentencizer on documents
        -Uses the right Matcher depending on the presence of categories
        """

        # tokenization
        tokenizer = MultilingualTokenizer()
        tokenizer._add_spacy_tokenizer(self.language)
        self.nlp = tokenizer.spacy_nlp_dict[self.language]
        # sentence splitting
        self.nlp.add_pipe("sentencizer")
        self.text_df["list-sentences"] = self.text_df[self.text_column].apply(
            self._list_sentences, args=[self.nlp]
        )
        _, _ = self.nlp.remove_pipe("sentencizer")

        # matching
        if self.category_column is not None:
            ruler = self.nlp.add_pipe("entity_ruler")
            ruler.add_patterns(self.patterns)
            return
        else:
            matcher = PhraseMatcher(self.nlp.vocab)
            self.patterns = list(self.nlp.tokenizer.pipe(self.patterns))
            matcher.add("PatternList", self.patterns)
            return matcher