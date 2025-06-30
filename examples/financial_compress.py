import argparse
import glob
import json
import os
from tqdm import tqdm

# reuse the helper from infer.py to call the compression server
from infer import compress_api_call_local


def build_document(doc_dir: str) -> str:
    """Read all md files from directory and combine them."""
    documents = []
    for path in sorted(glob.glob(os.path.join(doc_dir, '*.md'))):
        with open(path, 'r', encoding='utf-8') as f:
            documents.append(f.read())
    prompt = ''
    for i, doc in enumerate(documents, 1):
        prompt += f"## Document{i}:\n{doc}\n\n"
    return prompt


def load_questions(question_path: str):
    """Load questions from a jsonl file."""
    questions = []
    with open(question_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            questions.append(data)
    return questions


def compress_documents(document: str, questions, level: int, output_path: str, prompt: str):
    """Compress document for each question and write results."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fw:
        for q in tqdm(questions, desc='Compressing'):
            messages = [
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': q['question']},
                {'role': 'context', 'content': document}
            ]
            result = compress_api_call_local(messages)
            record = {
                'level': level,
                'id': q['id'],
                'question': q['question'],
                'compressed': result
            }
            fw.write(json.dumps(record, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compress documents for financial questions')
    parser.add_argument('--level', type=int, default=2, help='level id of documents')
    parser.add_argument('--doc_dir', type=str, default='/mnt/workspace/zhiyuan/corpus-eval_/corpus_levels/financial_zh/level2')
    parser.add_argument('--questions', type=str, default='/mnt/workspace/zhiyuan/corpus-eval_/question/questions_financial_zh.jsonl')
    parser.add_argument('--output', type=str, default='output/financial_compressed.jsonl')
    parser.add_argument('--compress_prompt', type=str, default='You are an expert for information extraction, your task is to compress the given document to answer the user question.\n## tagging rule:\n- tag the supporting facts with \"fact\"')
    args = parser.parse_args()

    document = build_document(args.doc_dir)
    questions = load_questions(args.questions)

    compress_documents(document, questions, args.level, args.output, args.compress_prompt)
