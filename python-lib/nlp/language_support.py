# -*- coding: utf-8 -*-
"""Module with constants defining the language support of underlying NLP libraries"""

SUPPORTED_LANGUAGES_SPACY = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "ga": "Irish",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "lb": "Luxembourgish",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "nb": "Norwegian Bokmål",
    "ne": "Nepali",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sa": "Sanskrit",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "tt": "Tatar",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "yo": "Yoruba",
    "zh": "Chinese (simplified)",
}

"""dict: Languages supported by spaCy: https://spacy.io/usage/models#languages

Dictionary with ISO 639-1 language code (key) and language name (value)
Korean is excluded for now because of system installation issues
"""

SPACY_LANGUAGE_MODELS = {
    "de": "de_core_news_sm",  # OntoNotes
    "en": "en_core_web_sm",  # OntoNotes
    "es": "es_core_news_sm",  # Wikipedia
    "fr": "fr_core_news_sm",  # Wikipedia
    "nb": "nb_core_news_sm",  # NorNE
    "pl": "pl_core_news_sm",  # NKJP
    "ru": "ru_core_news_sm",  # Nerus
    "zh": "zh_core_web_sm",  # OntoNotes
}

"""dict: Mapping between ISO 639-1 language code and spaCy model identifiers

Models with Creative Commons licenses are not included because this plugin is licensed under Apache-2
"""

SPACY_LANGUAGE_MODELS_LEMMATIZATION = ["de", "en", "es", "fr", "nb", "pl", "ru"]
"""list: Languages that have a SpaCy pre-trained model with a Lemmatizer component.
When using a pre-trained pipeline to lemmatize, you need to have in your SpaCy Language pipeline:
-either SpaCy 'morphologizer' + 'lemmatizer'
-or SpaCy 'tagger' + 'attribute ruler' + 'lemmatizer'
depending on the pre-trained pipeline built-in components"""

SPACY_LANGUAGE_MODELS_MORPHOLOGIZER = ["de", "es", "fr", "nb", "pl", "ru"]
"""list: Languages that have a SpaCy pre-trained model with a Morphologizer component."""

SPACY_LANGUAGE_LOOKUP = [
    "ca",
    "cs",
    "da",
    "de",
    "en",
    "es",
    "fr",
    "hr",
    "hu",
    "id",
    "it",
    "lb",
    "lt",
    "mk",
    "nb",
    "nl",
    "pt",
    "ro",
    "sr",
    "sv",
    "tl",
    "tr",
    "ur",
]
"""list: Languages that have available SpaCy lookup tables for lemmatization. 
The lookup tables are available at https://github.com/explosion/spacy-lookups-data/tree/master/spacy_lookups_data/data
"""

SPACY_LANGUAGE_RULES = ["bn", "el", "fa"]
"""list: Languages that have available SpaCy rule tables for lemmatization
The rule tables are available at https://github.com/explosion/spacy-lookups-data/tree/master/spacy_lookups_data/data
"""
