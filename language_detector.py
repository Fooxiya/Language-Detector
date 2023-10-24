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

    # language label to language name mapping
    LABEL_TO_LANG: Dict[str, str] = {
        "LABEL_0": "Arabic",
        "LABEL_1": "Basque",
        "LABEL_2": "Breton",
        "LABEL_3": "Catalan",
        "LABEL_4": "Chinese_China",
        "LABEL_5": "Chinese_Hongkong",
        "LABEL_6": "Chinese_Taiwan",
        "LABEL_7": "Chuvash",
        "LABEL_8": "Czech",
        "LABEL_9": "Dhivehi",
        "LABEL_10": "Dutch",
        "LABEL_11": "English",
        "LABEL_12": "Esperanto",
        "LABEL_13": "Estonian",
        "LABEL_14": "French",
        "LABEL_15": "Frisian",
        "LABEL_16": "Georgian",
        "LABEL_17": "German",
        "LABEL_18": "Greek",
        "LABEL_19": "Hakha_Chin",
        "LABEL_20": "Indonesian",
        "LABEL_21": "Interlingua",
        "LABEL_22": "Italian",
        "LABEL_23": "Japanese",
        "LABEL_24": "Kabyle",
        "LABEL_25": "Kinyarwanda",
        "LABEL_26": "Kyrgyz",
        "LABEL_27": "Latvian",
        "LABEL_28": "Maltese",
        "LABEL_29": "Mongolian",
        "LABEL_30": "Persian",
        "LABEL_31": "Polish",
        "LABEL_32": "Portuguese",
        "LABEL_33": "Romanian",
        "LABEL_34": "Romansh_Sursilvan",
        "LABEL_35": "Russian",
        "LABEL_36": "Sakha",
        "LABEL_37": "Slovenian",
        "LABEL_38": "Spanish",
        "LABEL_39": "Swedish",
        "LABEL_40": "Tamil",
        "LABEL_41": "Tatar",
        "LABEL_42": "Turkish",
        "LABEL_43": "Ukranian",
        "LABEL_44": "Welsh",
    }

    # language names
    LANGUAGES: List[str] = [
        "Arabic",
        "Basque",
        "Breton",
        "Catalan",
        "Chinese_China",
        "Chinese_Hongkong",
        "Chinese_Taiwan",
        "Chuvash",
        "Czech",
        "Dhivehi",
        "Dutch",
        "English",
        "Esperanto",
        "Estonian",
        "French",
        "Frisian",
        "Georgian",
        "German",
        "Greek",
        "Hakha_Chin",
        "Indonesian",
        "Interlingua",
        "Italian",
        "Japanese",
        "Kabyle",
        "Kinyarwanda",
        "Kyrgyz",
        "Latvian",
        "Maltese",
        "Mongolian",
        "Persian",
        "Polish",
        "Portuguese",
        "Romanian",
        "Romansh_Sursilvan",
        "Russian",
        "Sakha",
        "Slovenian",
        "Spanish",
        "Swedish",
        "Tamil",
        "Tatar",
        "Turkish",
        "Ukranian",
        "Welsh",

    ]

    def __init__(self, model_path: str, threshold: float) -> None:
        self.classifier = pipeline("text-classification", model=model_path)
        self.threshold = threshold

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
                label = lang["label"]
                print(f"label: {label}")
                if label in self.LABEL_TO_LANG:
                    result[self.LABEL_TO_LANG[label]] = lang["score"]
                else:
                    raise ValueError(f"Label {label} is not in the list of languages")
        return result

    def get_languages(self) -> List[str]:
        """Return list of available languages

        :return: list of language names
        """
        return self.LANGUAGES
