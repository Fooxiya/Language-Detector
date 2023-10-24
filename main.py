import json
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from language_detector import LanguageDetector
from time import perf_counter


class Text(BaseModel):
    text: str


class DetectionResult(BaseModel):
    result: Dict[str, float]
    time: int


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
    try:
        t1_start = perf_counter()
        result = lang_detector.classify(text_input.text)
        t1_stop = perf_counter()
        return DetectionResult(result=result, time=int((t1_stop - t1_start) * 1000))
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
