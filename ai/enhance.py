import os
import json
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from queue import Queue
from threading import Lock
# INSERT_YOUR_CODE
import requests

import dotenv
import argparse
from tqdm import tqdm

import langchain_core.exceptions
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from structure import Structure

if os.path.exists('.env'):
    dotenv.load_dotenv()
template = open("template.txt", "r").read()
system = open("system.txt", "r").read()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    parser.add_argument("--rpm_limit", type=int, default=15, help="Rate limit: requests per minute")
    return parser.parse_args()

class RateLimiter:
    """
    简单的速率限制器，确保每分钟不超过指定数量的请求
    Simple rate limiter to ensure no more than specified number of requests per minute
    """

    def __init__(self, max_requests_per_minute: int = 15):
        self.max_requests = max_requests_per_minute
        self.requests = deque()
        self.lock = Lock()

    def wait_if_needed(self):
        """
        检查是否需要等待以遵守速率限制
        Check if need to wait to comply with rate limit
        """
        with self.lock:
            now = time.time()

            # 移除60秒前的请求记录
            # Remove request records older than 60 seconds
            while self.requests and now - self.requests[0] >= 60:
                self.requests.popleft()

            # 如果已达到限制，等待
            # If limit reached, wait
            if len(self.requests) >= self.max_requests:
                # 计算需要等待的时间，直到最早的请求超过60秒
                # Calculate how long to wait until the earliest request is over 60 seconds
                wait_time = 60 - (now - self.requests[0])
                if wait_time > 0:
                    print(f"⏱️  速率限制: 达到 {self.max_requests} RPM, 等待 {wait_time:.2f} 秒... / Rate limit: {self.max_requests} RPM reached, waiting {wait_time:.2f}s...", file=sys.stderr)
                    time.sleep(wait_time)
                    # 重新检查并清理过期的请求记录
                    # Re-check and clean up expired request records
                    now = time.time()
                    while self.requests and now - self.requests[0] >= 60:
                        self.requests.popleft()

            # 记录本次请求时间
            # Record this request time
            self.requests.append(now)

def process_single_item(chain, item: Dict, language: str) -> Dict:
    def is_sensitive(content: str) -> bool:
        """
        调用 spam.dw-dengwei.workers.dev 接口检测内容是否包含敏感词。
        返回 True 表示触发敏感词，False 表示未触发。
        """
        try:
            resp = requests.post(
                "https://spam.dw-dengwei.workers.dev",
                json={"text": content},
                timeout=5
            )
            if resp.status_code == 200:
                result = resp.json()
                # 约定接口返回 {"sensitive": true/false, ...}
                return result.get("sensitive", True)
            else:
                # 如果接口异常，默认不触发敏感词
                print(f"Sensitive check failed with status {resp.status_code}", file=sys.stderr)
                return True
        except Exception as e:
            print(f"Sensitive check error: {e}", file=sys.stderr)
            return True

    # 检查 summary 字段
    if is_sensitive(item.get("summary", "")):
        return None

    """处理单个数据项"""
    # Default structure with meaningful fallback values
    default_ai_fields = {
        "tldr": "Summary generation failed",
        "motivation": "Motivation analysis unavailable",
        "method": "Method extraction failed",
        "result": "Result analysis unavailable",
        "conclusion": "Conclusion extraction failed"
    }
    
    try:
        response: Structure = chain.invoke({
            "language": language,
            "content": item['summary']
        })
        item['AI'] = response.model_dump()
    except langchain_core.exceptions.OutputParserException as e:
        # 尝试从错误信息中提取 JSON 字符串并修复
        error_msg = str(e)
        partial_data = {}
        
        if "Function Structure arguments:" in error_msg:
            try:
                # 提取 JSON 字符串
                json_str = error_msg.split("Function Structure arguments:", 1)[1].strip().split('are not valid JSON')[0].strip()
                # 预处理 LaTeX 数学符号 - 使用四个反斜杠来确保正确转义
                json_str = json_str.replace('\\', '\\\\')
                # 尝试解析修复后的 JSON
                partial_data = json.loads(json_str)
            except Exception as json_e:
                print(f"Failed to parse JSON for {item.get('id', 'unknown')}: {json_e}", file=sys.stderr)
        
        # Merge partial data with defaults to ensure all fields exist
        item['AI'] = {**default_ai_fields, **partial_data}
        print(f"Using partial AI data for {item.get('id', 'unknown')}: {list(partial_data.keys())}", file=sys.stderr)
    except Exception as e:
        # Catch any other exceptions and provide default values
        print(f"Unexpected error for {item.get('id', 'unknown')}: {e}", file=sys.stderr)
        item['AI'] = default_ai_fields
    
    # Final validation to ensure all required fields exist
    for field in default_ai_fields.keys():
        if field not in item['AI']:
            item['AI'][field] = default_ai_fields[field]

    # 检查 AI 生成的所有字段
    for v in item.get("AI", {}).values():
        if is_sensitive(str(v)):
            return None
    return item

def process_all_items(data: List[Dict], model_name: str, language: str, max_workers: int, rate_limiter: RateLimiter) -> List[Dict]:
    """并行处理所有数据项"""
    llm = ChatOpenAI(model=model_name).with_structured_output(Structure, method="function_calling")
    print('Connect to:', model_name, file=sys.stderr)
    print(f'Rate limit: {rate_limiter.max_requests} requests per minute', file=sys.stderr)
    print(f'Max workers: {max_workers}', file=sys.stderr)

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(template=template)
    ])

    chain = prompt_template | llm

    # 使用线程池并行处理
    processed_data = [None] * len(data)  # 预分配结果列表
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(process_single_item_with_rate_limit, chain, item, language, rate_limiter): idx
            for idx, item in enumerate(data)
        }

        # 使用tqdm显示进度
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(data),
            desc="Processing items"
        ):
            idx = future_to_idx[future]
            try:
                result = future.result()
                processed_data[idx] = result
            except Exception as e:
                print(f"Item at index {idx} generated an exception: {e}", file=sys.stderr)
                # Add default AI fields to ensure consistency
                processed_data[idx] = data[idx]
                processed_data[idx]['AI'] = {
                    "tldr": "Processing failed",
                    "motivation": "Processing failed",
                    "method": "Processing failed",
                    "result": "Processing failed",
                    "conclusion": "Processing failed"
                }

    return processed_data

def process_single_item_with_rate_limit(chain, item: Dict, language: str, rate_limiter: RateLimiter) -> Dict:
    """
    带速率限制的单个数据项处理函数
    Single item processing function with rate limiting
    """
    # 在调用AI API之前检查速率限制
    # Check rate limit before calling AI API
    rate_limiter.wait_if_needed()

    # 然后调用原有的处理函数
    # Then call the original processing function
    return process_single_item(chain, item, language)

def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", 'deepseek-chat')
    language = os.environ.get("LANGUAGE", 'Chinese')

    # 创建速率限制器
    rate_limiter = RateLimiter(max_requests_per_minute=args.rpm_limit)

    # 检查并删除目标文件
    target_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f'Removed existing file: {target_file}', file=sys.stderr)

    # 读取数据
    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 去重
    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data
    print('Open:', args.data, file=sys.stderr)
    print(f'Total unique papers to process: {len(data)}', file=sys.stderr)

    # 并行处理所有数据
    processed_data = process_all_items(
        data,
        model_name,
        language,
        args.max_workers,
        rate_limiter
    )

    # 保存结果
    with open(target_file, "w") as f:
        for item in processed_data:
            if item is not None:
                f.write(json.dumps(item) + "\n")

    print(f'\n✅ 处理完成! / Processing completed!', file=sys.stderr)
    print(f'📊 处理了 {len([x for x in processed_data if x is not None])} 篇论文 / Processed {len([x for x in processed_data if x is not None])} papers', file=sys.stderr)

if __name__ == "__main__":
    main()
