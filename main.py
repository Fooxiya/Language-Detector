import json
from typing import Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from language_detector import LanguageDetector


class Text(BaseModel):
    text: str


class DetectionResult(BaseModel):
    result: Dict[str, float]


app = FastAPI()
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
lang_detector = LanguageDetector(
    model_path=config["model_path"], threshold=config["threshold"]
)


@app.get("/languages")
async def get_languages() -> List[str]:
    return lang_detector.get_languages()


@app.post("/detect")
def detect(text_input: Text) -> DetectionResult:
    return DetectionResult(result=lang_detector.classify(text_input.text))
