import argparse
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def compress_documents(document: str, questions, level: int, output_path: str, prompt: str, server_urls):
    """Compress document for each question and write results."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def worker(q, url):
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": q["question"]},
            {"role": "context", "content": document},
        ]
        result = compress_api_call_local(messages, url=url)
        return {
            "level": level,
            "id": q["id"],
            "question": q["question"],
            "compressed": result,
        }

    with open(output_path, "w", encoding="utf-8") as fw:
        futures = []
        with ThreadPoolExecutor(max_workers=len(server_urls)) as executor:
            for idx, q in enumerate(questions):
                url = server_urls[idx % len(server_urls)]
                futures.append(executor.submit(worker, q, url))

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Compressing"):
                record = fut.result()
                fw.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compress documents for questions')
    parser.add_argument('--corpus', type=str, default='financial_zh', 
                        help='corpus name (e.g., financial_zh, medical_en, etc.)')
    parser.add_argument('--level', type=int, default=2, 
                        help='level id of documents')
    parser.add_argument('--doc_dir', type=str, default=None,
                        help='directory containing documents (default: /mnt/workspace/zhiyuan/corpus-eval_/corpus_levels/{corpus}/level{level})')
    parser.add_argument('--questions', type=str, default=None,
                        help='path to questions file (default: /mnt/workspace/zhiyuan/corpus-eval_/question/questions_{corpus}.jsonl)')
    parser.add_argument('--output', type=str, default=None,
                        help='output path (default: output/{corpus}_compressed.jsonl)')
    parser.add_argument('--compress_prompt', type=str, 
                        default='You are an expert for information extraction, your task is to compress the given document to answer the user question.\n## tagging rule:\n- tag the supporting facts with \"fact\"',
                        help='prompt for compression')
    parser.add_argument('--server_urls', type=str,
                        default='http://0.0.0.0:8091/qwen_long_compress_server',
                        help='Comma separated compression server URLs')
    
    args = parser.parse_args()
    
    # Set default paths based on corpus if not provided
    if args.doc_dir is None:
        args.doc_dir = f'/mnt/workspace/zhiyuan/corpus-eval_/corpus_levels/{args.corpus}/level{args.level}'
    
    if args.questions is None:
        args.questions = f'/mnt/workspace/zhiyuan/corpus-eval_/question/questions_{args.corpus}.jsonl'
    
    if args.output is None:
        args.output = f'output/{args.corpus}_level{args.level}_compressed.jsonl'
    
    # Print configuration for verification
    print(f"Corpus: {args.corpus}")
    print(f"Document directory: {args.doc_dir}")
    print(f"Questions file: {args.questions}")
    print(f"Output file: {args.output}")
    
    server_urls = [u.strip() for u in args.server_urls.split(',') if u.strip()]

    document = build_document(args.doc_dir)
    questions = load_questions(args.questions)

    compress_documents(document, questions, args.level, args.output, args.compress_prompt, server_urls)
