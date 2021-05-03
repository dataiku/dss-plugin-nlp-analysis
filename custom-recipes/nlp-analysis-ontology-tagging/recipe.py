# -*- coding: utf-8 -*-
import dataiku
from formatter_instanciator import COLUMNS_DESCRIPTION
from dkulib_io_utils import set_column_descriptions
from dku_plugin_config_loading import DkuConfigLoadingOntologyTagging
from ontology_tagger import Tagger
import spacy
import unicodedata
import pandas as pd
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")


def _split_sentences(row: pd.Series, text_column, nlp):
    """Called if there is only one language specified.Apply sentencizer and return list of sentences

    Args:
        row (pandas.Series): row which contains text to process

    Returns:
        List : Document splitted into tokenized sentences as strings.

    """
    document = row[text_column]
    return [sentence.text for sentence in nlp(document).sents]


def _match_no_category(list_of_tags, list_of_keywords, nlp, language):
    nlp.remove_pipe("sentencizer")
    patterns = _tokenize_keywords(language, list_of_tags, list_of_keywords)
    matcher = PhraseMatcher(nlp.vocab)
    matcher.add("PatternList", patterns)
    print(patterns)
    return matcher


def get_keyword(text, normalize_case):
    """Return text in its wanted-case form"""
    return text.lower() if normalize_case else text


def _tokenize_keywords(language, tags, keywords):
    """Called when self.category_column is not None. Tokenize the keywords and fill in the dictionary _keyword_to_tag.
    The keywords are tokenized depending on the given language.

    Args:
        language (str) : Language code in ISO 639-1 format to use to tokenize the keywords.
        keywords (list): The keywords to tokenize.

    Returns:
        List: The tokenized keywords.

    """
    tokenized_keywords = list(
        nlp.tokenizer.pipe(
            [
                unicodedata.normalize("NFD", get_keyword(keyword, False))
                for keyword in keywords
            ]
        )
    )
    return tokenized_keywords


# normalize and tokenize the keywords
# list_of_keywords = ["Cl\u00e9ment"]
# list_of_keywords = [nlp(unicodedata.normalize('NFD',keyword)) for keyword in list_of_keywords]
# print("keywords : ",list_of_keywords)
tags = ["cl\u00e9ment"]
keywords = ["cl\u00e9ment"]


df = pd.DataFrame({"text": ["here is clément"]})
df["text_tokenized"] = df.apply(_split_sentences, args=["text", nlp], axis=1)
matcher = _match_no_category(tags, keywords, nlp, "en")


def _get_document_to_match(row: pd.Series, language):
    normalized_texte = [
        unicodedata.normalize("NFD", sentence) for sentence in row["text_tokenized"]
    ]
    normalized_texte = list(nlp.pipe(normalized_texte))
    return normalized_texte


def _write_row(row, language_column, matcher):
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
    document_to_match = _get_document_to_match(row, "en")
    print(document_to_match)
    for idx, sentence in enumerate(document_to_match):
        match = matcher(sentence, as_spans=True)
        for keyword in match:
            print("find: ", keyword.text)


df.apply(_write_row, args=[None, matcher], axis=1)


# normalized_texte = [unicodedata.normalize('NFD',sentence) for sentence in row[self.text_column_tokenized]
# normalized_texte = list(self.tokenizer.spacy_nlp_dict[language].pipe(normalized_texte))
# texte = "A sentence. An other with Clément."
# nlp.add_pipe("sentencizer")
# texte = nlp(texte)


# print(list(texte.sents))
# nlp.remove_pipe("sentencizer")
# texte = [nlp(unicodedata.normalize('NFD',t.text)) for t in list(texte.sents)]

# print("texte :", texte)
# for sent in normalized_texte:
#  z = matcher(sent)
#  print("Matches:")
#  for id,start,end in z:
#    print(sent[start:end].text)
dku_config = DkuConfigLoadingOntologyTagging()
settings = dku_config.load_settings()
text_dataframe = settings.text_input.get_dataframe(infer_with_pandas=False)
ontology_dataframe = settings.ontology_input.get_dataframe(
    columns=settings.ontology_columns, infer_with_pandas=False
)
languages = (
    text_dataframe[settings.language_column].dropna().unique()
    if settings.language == "language_column"
    else [settings.language]
)
dku_config._check_languages(languages)

tagger = Tagger(
    ontology_df=ontology_dataframe,
    tag_column=settings.tag_column,
    category_column=settings.category_column,
    keyword_column=settings.keyword_column,
    language=settings.language,
    lemmatization=settings.lemmatization,
    normalize_case=settings.normalize_case,
    normalization=settings.unicode_normalization,
)
output_df = tagger.tag_and_format(
    text_df=text_dataframe,
    text_column=settings.text_column,
    language_column=settings.language_column,
    output_format=settings.output_format,
    languages=languages,
)
settings.output_dataset.write_with_schema(output_df)
set_column_descriptions(
    output_dataset=settings.output_dataset, column_descriptions=COLUMNS_DESCRIPTION
)
