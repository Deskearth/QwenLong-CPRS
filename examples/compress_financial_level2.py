import os
import glob
import json
from typing import List

from examples.infer import compress_api_call_local

DOC_DIR = "/mnt/workspace/zhiyuan/corpus-eval_/corpus_levels/financial_zh/level2"
QUESTION_FILE = "/mnt/workspace/zhiyuan/corpus-eval_/question/questions_financial_zh.json"
OUTPUT_PATH = "output/compressed_financial_level2.jsonl"

CPRS_PROMPT = (
    "You are an expert for information extraction, your task is to extract "
    "some sentences from the documents as the supporting facts of the user's question.\n"
    "## tagging rule:\n- tag the supporting facts with 'fact'"
)


def load_documents() -> str:
    files = sorted(glob.glob(os.path.join(DOC_DIR, "*.md")))
    documents: List[str] = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            documents.append(f.read())
    prompt = ""
    for i, doc in enumerate(documents, 1):
        prompt += f"## Document{i}:\n{doc}\n\n"
    return prompt


def load_questions() -> List[dict]:
    questions = []
    with open(QUESTION_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    document_prompt = load_documents()
    questions = load_questions()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_file:
        for q in questions:
            messages = [
                {"role": "system", "content": CPRS_PROMPT},
                {"role": "user", "content": q["question"]},
                {"role": "context", "content": document_prompt},
            ]
            compressed = compress_api_call_local(messages)
            result = {
                "level": 2,
                "id": q["id"],
                "compressed": compressed,
            }
            out_file.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
