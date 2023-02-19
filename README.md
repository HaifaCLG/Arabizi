# Identifying Code-switching in Arabizi

We describe a corpus of social media posts that include utterances in Arabizi, a Roman-script rendering of Arabic, mixed with other languages, notably English, French, and Arabic written in the Arabic script. We manually annotated a subset of the texts with word-level language IDs; this is a non-trivial task due to the nature of mixed-language writing, especially on social media. We developed classifiers that can accurately predict the language ID tags. Then, we extended the word-level predictions to identify sentences that include Arabizi (and code-switching), and applied the classifiers to the raw corpus, thereby harvesting a large number of additional instances. The result is a large-scale dataset of Arabizi, with precise indications of code-switching between Arabizi and English, French, and Arabic.

If you use this dataset, please cite:
@inproceedings{shehadi-wintner-2022-identifying,
    title = "Identifying Code-switching in {A}rabizi",
    author = "Shehadi, Safaa  and
      Wintner, Shuly",
    booktitle = "Proceedings of the The Seventh Arabic Natural Language Processing Workshop (WANLP)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.wanlp-1.18",
    pages = "194--204",
}
