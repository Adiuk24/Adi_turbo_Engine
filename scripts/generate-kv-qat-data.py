#!/usr/bin/env python3
"""Generate KV-cache QAT training data by distilling from BF16 model.

Sends diverse prompts to the BF16 model running on llama-server,
captures the full responses, and saves as chat-format JSONL for MLX QLoRA.

The goal: the model will be fine-tuned with TQ3_S KV cache simulation
so it learns to produce activations that survive 3-bit compression.
"""

import json, os, sys, time
from urllib.request import Request, urlopen

# Configurable via env or argv: generate-kv-qat-data.py [output.jsonl]
API = os.environ.get("ADITURBO_API", "http://localhost:11434/v1/chat/completions")
MODEL = os.environ.get("ADITURBO_MODEL", "gemma4")
OUTPUT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ADITURBO_QAT_OUT", "kv-qat-distill.jsonl")

# Diverse prompts covering all capabilities Eyla needs
PROMPTS = [
    # English knowledge
    "What is the capital of France?",
    "Explain photosynthesis in 3 sentences.",
    "What causes earthquakes?",
    "Who invented the telephone?",
    "What is the speed of light?",
    "Explain how a CPU works to a 10-year-old.",
    "What are the three states of matter?",
    "Name the planets in order from the sun.",
    "What is the difference between DNA and RNA?",
    "Explain supply and demand in economics.",
    "What is machine learning?",
    "How does encryption work?",
    "What is the Pythagorean theorem?",
    "Explain the water cycle.",
    "What is blockchain?",
    "How do vaccines work?",
    "What is the theory of relativity?",
    "Explain how the internet works.",
    "What is quantum computing?",
    "How does a neural network learn?",

    # Bangla knowledge
    "বাংলাদেশের রাজধানী কী?",
    "পদ্মা সেতু সম্পর্কে বলো।",
    "বাংলাদেশের মুক্তিযুদ্ধ কবে হয়েছিল?",
    "রবীন্দ্রনাথ ঠাকুর কে ছিলেন?",
    "বাংলাদেশের জাতীয় ফুল কী?",
    "সুন্দরবন সম্পর্কে তিন লাইনে বলো।",
    "বাংলা নববর্ষ কবে পালিত হয়?",
    "বঙ্গবন্ধু শেখ মুজিবুর রহমান কে ছিলেন?",
    "বাংলাদেশের প্রধান রপ্তানি পণ্য কী?",
    "একুশে ফেব্রুয়ারি কেন গুরুত্বপূর্ণ?",

    # Math
    "What is 15 * 23?",
    "Solve: 2x + 5 = 17",
    "What is the square root of 144?",
    "If a train travels 60 km/h for 2.5 hours, how far does it go?",
    "What is 30% of 250?",

    # Code
    "Write a Python function to reverse a string.",
    "Write a function to check if a number is prime in Python.",
    "What does 'git rebase' do?",
    "Explain the difference between == and === in JavaScript.",
    "Write a SQL query to find duplicate emails in a users table.",

    # Reasoning
    "If all dogs are animals and some animals can fly, can dogs fly?",
    "A farmer has 17 sheep. All but 9 run away. How many are left?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",

    # Creative
    "Write a haiku about rain.",
    "Tell me a short joke.",

    # Instruction following
    "List 5 fruits that start with the letter A.",
    "Translate 'good morning' into Spanish, French, and German.",
    "Summarize the concept of gravity in one sentence.",
]

# Tool calling examples — model learns to generate structured tool calls
TOOL_PROMPTS = [
    ("What time is it right now?", [{"type":"function","function":{"name":"get_time","description":"Get current date and time","parameters":{"type":"object","properties":{}}}}]),
    ("List files in /tmp", [{"type":"function","function":{"name":"shell_exec","description":"Execute a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]),
    ("Search the web for latest news about Bangladesh", [{"type":"function","function":{"name":"web_search","description":"Search the web","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}]),
    ("Read the file at /etc/hostname", [{"type":"function","function":{"name":"file_read","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]),
    ("Create a reminder to buy groceries", [{"type":"function","function":{"name":"todo_create","description":"Create a todo item","parameters":{"type":"object","properties":{"title":{"type":"string"},"description":{"type":"string"}},"required":["title"]}}}]),
    ("What apps are running on my computer?", [{"type":"function","function":{"name":"app_list","description":"List running applications","parameters":{"type":"object","properties":{}}}}]),
    ("Open the calculator app", [{"type":"function","function":{"name":"open_app","description":"Open an application","parameters":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}}]),
    ("Find all Python files in the current directory", [{"type":"function","function":{"name":"code_glob","description":"Find files matching a pattern","parameters":{"type":"object","properties":{"pattern":{"type":"string"}},"required":["pattern"]}}}]),
    ("Search for 'TODO' in the codebase", [{"type":"function","function":{"name":"code_grep","description":"Search file contents","parameters":{"type":"object","properties":{"pattern":{"type":"string"}},"required":["pattern"]}}}]),
    ("What is my computer's IP address?", [{"type":"function","function":{"name":"shell_exec","description":"Execute a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]),
    ("Write 'hello world' to /tmp/test.txt", [{"type":"function","function":{"name":"file_write","description":"Write to a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]),
    ("How much disk space do I have?", [{"type":"function","function":{"name":"shell_exec","description":"Execute a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]),
    ("Remember that my favorite color is blue", [{"type":"function","function":{"name":"memory_learn","description":"Store a fact in memory","parameters":{"type":"object","properties":{"fact":{"type":"string"}},"required":["fact"]}}}]),
    ("What do you remember about me?", [{"type":"function","function":{"name":"memory_recall","description":"Recall stored facts","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}]),
    ("Browse to google.com and tell me what you see", [{"type":"function","function":{"name":"browse_url","description":"Open a URL and read its content","parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}}}]),
]

def ask(prompt, max_tokens=1024, tools=None):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    if tools:
        body["tools"] = tools
    data_bytes = json.dumps(body).encode()
    req = Request(API, data=data_bytes, headers={"Content-Type": "application/json"})
    resp = urlopen(req, timeout=300)
    data = json.loads(resp.read())
    msg = data["choices"][0]["message"]
    return msg.get("content", ""), msg.get("tool_calls", [])

def main():
    total = len(PROMPTS) + len(TOOL_PROMPTS)
    print(f"Generating KV-QAT training data: {len(PROMPTS)} text + {len(TOOL_PROMPTS)} tool = {total} prompts")
    print(f"Output: {OUTPUT}")

    count = 0
    with open(OUTPUT, "w") as f:
        # Text prompts
        for i, prompt in enumerate(PROMPTS):
            try:
                response, _ = ask(prompt)
                if not response.strip():
                    print(f"  [{i+1}/{total}] SKIP (empty): {prompt[:50]}")
                    continue
                entry = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ]
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
                print(f"  [{i+1}/{total}] OK: {prompt[:50]}... → {len(response)} chars")
            except Exception as e:
                print(f"  [{i+1}/{total}] ERROR: {e}")

        # Tool calling prompts
        for i, (prompt, tools) in enumerate(TOOL_PROMPTS):
            idx = len(PROMPTS) + i + 1
            try:
                content, tool_calls = ask(prompt, tools=tools)
                if not tool_calls:
                    print(f"  [{idx}/{total}] SKIP (no tool call): {prompt[:50]}")
                    continue
                # Build assistant message with tool calls
                assistant_msg = {"role": "assistant", "content": content or ""}
                assistant_msg["tool_calls"] = [
                    {"type": "function", "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                    for tc in tool_calls
                ]
                entry = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        assistant_msg,
                    ]
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
                tool_names = [tc["function"]["name"] for tc in tool_calls]
                print(f"  [{idx}/{total}] TOOL: {prompt[:40]}... → {tool_names}")
            except Exception as e:
                print(f"  [{idx}/{total}] ERROR: {e}")

    print(f"\nDone. {count}/{total} examples saved to {OUTPUT}")

if __name__ == "__main__":
    main()
