# Text Analysis Plugin 🚧 WORK IN PROGRESS

![Build status](https://github.com/dataiku/dss-plugin-nlp-analysis/actions/workflows/auto-make.yml/badge.svg) ![GitHub release (latest by date)](https://img.shields.io/github/v/release/dataiku/dss-plugin-nlp-analysis?logo=github)  ![Support level](https://img.shields.io/badge/support-Unsupported-orange)

This Dataiku DSS plugin provides recipes to analyze text data. With this plugin, you will be able to:

- Tag documents matching keywords by string-matching, within a corpus of text documents, for 59 languages.

Note that languages are defined as per the ISO 639-1 standard with 2-letter codes.
## How to set up
### Ontology tagging
Right after installing the plugin, you will need to build its code environment. Note that Python version 3.6 or 3.7 is required.
## How to use
### Ontology tagging
This recipe assigns tags to text documents. To assign tags to a corpus of documents, you will need two input datasets:
#### Document dataset:
Let’s assume that you have a Dataiku DSS project with a dataset containing raw text data. This text data must be stored in a dataset, inside a text column, with one row for each document. If your corpus is composed with document of different languages, you must have a language column to indicate each document language. 

#### Example 
| Text     | Language          | 
| ------------- |:-------------:| 
| Our most famous recreation center sports are football and judo.   | en | 
| Ofrecemos lecciones de guitarra    | es    | 

#### Ontology Dataset, with the following columns:
- a column such as "keywords", which are the terms that will be searched in the corpus of documents.
- a column such as "tags" which are the tags you want to assign to your documents. A tag can be linked to a set of keywords. 
- (Optional) a column such as "category", which are the categories you want to assign to your documents. A category can be linked to a set of tags.

#### Example

| Keywords      | Tags           | Category  |
| ------------- |:-------------:| -----:|
| Football     | Team sport | Sport |
| Basketball      | Team sport     |   Sport |
| Taekwondo | Martial arts   |    Sport |
| Judo | Martial arts      |    Sport |
| Guitar | String instrument     |    Music |
| Violin | String instrument      |    Music |

Let's assume you have your two datasets in a project. Now you can : 
- Navigate to the Flow
- Click on the + RECIPE button and access the Text Analysis menu. If your dataset is selected, you can directly find the plugin on the right panel.

#### Parameter settings

##### Input parameters



*   Text dataset:
    *   Text column
    *   Language: the language parameter lets you choose among 59 supported languages if your text is monolingual. Else, the Multilingual option will let you set a Language column.This Language column parameter should contain ISO 639-1 supported language codes. 
*   Ontology dataset:
    *   tag column
    *   keyword column
    *   (Optional) category column

 
##### Matching parameters

You can widen the search by activating the following matching parameters:

*   Button normalize case: To match documents by ignoring case (e.g will try to match 'guitar','Guitar', 'GUITAR')
*   Button normalize diacritics: To match documents by ignoring diacritics marks like accents, cedillas, tildes
*   Button lemmatization: To match documents by simplifying words to their lemma form (e.g. going → go, mice → mouse). This option supports 28 languages. 
###### List of the 28 supported languages for lemmatization in ISO 639-1 format:
de, es, fr, nb, pl, ru, ca, cs, da, en, hr ,hu , id, it, lb, lt, mk, nb, nl, pt, ro, sr, sv, tl, tr, ur, bn, el, fa
##### Output format

There are three available output formats. All the outputs preserve the initial datas of the dataset.
* Default output “one row per document (array columns)”: A row for each document, with additional columns:
  *   the assigned tags of the document (array-like)
  *   the keywords that match with the document (array-like)
  *   the concatenated sentences that matched the keywords (string-like)
  However, if you gave a 'category column' in the matching parameters, you won't have a unique tag column but one column per category, each one containing a list of associated tags
 
* Output “one row per match”: A row per document per match. This mode creates duplicates from the input dataset, as you can have multiple matches in a document. 
In this case, the additional columns are:
  * a tag column (string-like)
  * a keyword column (string-like)
  * a sentence column (string-like)
  * (Optional) a category column (string-like)
  
* Output “one row per document (nested JSON)”: A row for each document, with additional columns:
  * tag_json_full column : a dictionary with details about occurences of each tag, matched keywords for each tag, and the sentences where they appear. (Object-like)
  * If you gave a 'category column' in the matching parameters, you will also have a column 'tag_json_categories' (Object-like), which is a simplified version of the previous dictionary: it contains categories (keys) and associated tags (values)


## License

This plugin is distributed under the [Apache License version 2.0](LICENSE).
