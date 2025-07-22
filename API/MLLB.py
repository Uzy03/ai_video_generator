from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import uvicorn

app = FastAPI()

# モデルとトークナイザーのロード
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# 日本語→英語に修正
translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang="jpn_Jpan", tgt_lang="eng_Latn")

class TranslationRequest(BaseModel):
    text: str

@app.post("/translate")
def translate(req: TranslationRequest):
    result = translator(req.text)
    return {"translatedText": result[0]['translation_text']}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 