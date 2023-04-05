import json
from typing import Dict, List

from transformers import pipeline


class LanguageDetector:
    """Language detector for 50 languages.
    The algorithm uses language detection model jb2k/bert-base-multilingual-cased-language-detection.
    Hugging Face reference: https://huggingface.co/jb2k/bert-base-multilingual-cased-language-detection
    Languages: Arabic, Basque, Breton, Catalan, Chinese_China, Chinese_Hongkong, Chinese_Taiwan, Chuvash, Czech,
               Dhivehi, Dutch, English, Esperanto, Estonian, French, Frisian, Georgian, German, Greek, Hakha_Chin,
               Indonesian, Interlingua, Italian, Japanese, Kabyle, Kinyarwanda, Kyrgyz, Latvian, Maltese, Mongolian,
               Persian, Polish, Portuguese, Romanian, Romansh_Sursilvan, Russian, Sakha, Slovenian, Spanish, Swedish,
               Tamil, Tatar, Turkish, Ukranian, Welsh
    """

    def __init__(self, model_path: str, threshold: float) -> None:
        self.classifier = pipeline("text-classification", model=model_path)
        self.threshold = threshold

        # read language list from config file
        languages = set()
        with open(f"{model_path}/config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            for _, lang in config["id2label"].items():
                languages.add(lang)
            self.languages = sorted(list(languages))

    def classify(self, text: str) -> Dict[str, float]:
        """Language detection implementation.

        :param text: text to detect language on
        :return: list of detected language names with scoring
        """
        # apply pipeline to detect language
        langs = self.classifier(text)
        result = dict()

        # filter languages by threshold value
        for lang in langs:
            if lang["score"] >= self.threshold:
                result[lang["label"]] = lang["score"]

        return result

    def get_languages(self) -> List[str]:
        """Return list of available languages

        :return: list of language names
        """
        return list(self.languages)
