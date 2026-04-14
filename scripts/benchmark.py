#!/usr/bin/env python3
"""Quick benchmark for AdiTurbo models via chat completions API.
Runs MMLU, GSM8K, ARC subsets and reports accuracy vs published baselines."""

import json, re, sys, time
from urllib.request import Request, urlopen

API = "http://localhost:8081/v1/chat/completions"
MODEL = "gemma4"

def ask(prompt, max_tokens=50):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = Request(API, data=body, headers={"Content-Type": "application/json"})
    resp = urlopen(req, timeout=300)
    data = json.loads(resp.read())
    return data["choices"][0]["message"].get("content", "")

def extract_answer(text):
    """Extract single letter A/B/C/D from model response."""
    text = text.strip()
    # Direct letter
    if text and text[0] in "ABCD":
        return text[0]
    # "The answer is X"
    m = re.search(r'[Aa]nswer\s*(?:is|:)\s*\(?([A-D])\)?', text)
    if m: return m.group(1)
    # Any standalone letter
    m = re.search(r'\b([A-D])\b', text)
    if m: return m.group(1)
    return None

# ── MMLU Questions (10 per category, 5 categories) ──
MMLU = [
    # Abstract Algebra
    ("Find the degree of the extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\nA. 0\nB. 4\nC. 2\nD. 6", "B"),
    ("Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\nA. 0\nB. 1\nC. 2\nD. 3", "B"),
    ("Statement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.\nA. True, True\nB. False, False\nC. True, False\nD. False, True", "C"),
    # High School Mathematics
    ("What is the sum of all positive integers less than 100 that are squares of perfect squares?\nA. 98\nB. 100\nC. 99\nD. 97", "A"),
    ("If f(x) = 2x + 3 and g(x) = x^2 - 1, what is f(g(2))?\nA. 7\nB. 9\nC. 11\nD. 5", "B"),
    # World History
    ("The longest ruling dynasty in the history of China was the\nA. Shang\nB. Zhou\nC. Han\nD. Ming", "B"),
    ("The Magna Carta was signed in what year?\nA. 1066\nB. 1215\nC. 1346\nD. 1453", "B"),
    # Computer Science
    ("Which of the following is a correct statement about the time complexity of searching in a balanced binary search tree with n elements?\nA. O(1)\nB. O(log n)\nC. O(n)\nD. O(n log n)", "B"),
    ("In the TCP/IP model, which layer is responsible for routing?\nA. Application\nB. Transport\nC. Internet\nD. Network Access", "C"),
    ("What does SQL stand for?\nA. Standard Query Language\nB. Structured Query Language\nC. Simple Query Language\nD. Sequential Query Language", "B"),
    # Biology
    ("Which of the following is NOT a function of the liver?\nA. Detoxification\nB. Bile production\nC. Insulin production\nD. Glycogen storage", "C"),
    ("The process by which mRNA is synthesized from DNA is called\nA. Translation\nB. Replication\nC. Transcription\nD. Transduction", "C"),
    ("Which organelle is known as the powerhouse of the cell?\nA. Nucleus\nB. Ribosome\nC. Mitochondria\nD. Golgi apparatus", "C"),
    # Physics
    ("A 2 kg object is thrown vertically upward with a velocity of 20 m/s. What is its kinetic energy at the highest point? (g=10 m/s²)\nA. 0 J\nB. 200 J\nC. 400 J\nD. 100 J", "A"),
    ("What is the SI unit of electric current?\nA. Volt\nB. Ohm\nC. Ampere\nD. Watt", "C"),
    # General Knowledge
    ("What is the chemical symbol for gold?\nA. Au\nB. Ag\nC. Fe\nD. Cu", "A"),
    ("Which planet is known as the Red Planet?\nA. Venus\nB. Mars\nC. Jupiter\nD. Saturn", "B"),
    ("What is the smallest prime number?\nA. 0\nB. 1\nC. 2\nD. 3", "C"),
    ("What is the largest ocean on Earth?\nA. Atlantic\nB. Indian\nC. Arctic\nD. Pacific", "D"),
    ("Who wrote Romeo and Juliet?\nA. Charles Dickens\nB. William Shakespeare\nC. Jane Austen\nD. Mark Twain", "B"),
    # Medical
    ("Which vitamin is primarily obtained from sunlight?\nA. Vitamin A\nB. Vitamin B12\nC. Vitamin C\nD. Vitamin D", "D"),
    # Law
    ("In most common law jurisdictions, the burden of proof in criminal cases rests on\nA. The defendant\nB. The prosecution\nC. The judge\nD. The jury", "B"),
    # Economics
    ("What does GDP stand for?\nA. General Domestic Product\nB. Gross Domestic Product\nC. Gross Development Product\nD. General Development Product", "B"),
    # Philosophy
    ("'I think, therefore I am' is attributed to\nA. Plato\nB. Aristotle\nC. Descartes\nD. Kant", "C"),
    # Geography
    ("What is the longest river in the world?\nA. Amazon\nB. Nile\nC. Mississippi\nD. Yangtze", "B"),
]

# ── GSM8K-style Math Questions ──
GSM8K = [
    ("Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Answer with just the number.", "18"),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take? Answer with just the number.", "3"),
    ("Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make? Answer with just the number.", "70000"),
    ("James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year? Answer with just the number.", "624"),
    ("Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. If the carry-over feed from the morning is 10 cups, how many chickens does Wendi have? Answer with just the number.", "20"),
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Answer with just the number.", "72"),
    ("Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet? Answer with just the number.", "5"),
    ("Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day? Answer with just the number.", "48"),
    ("A trader buys some bags of wheat from a farmer at a rate of $20 per bag. If the trader sells all the bags at a rate of $30 per bag and earns a total profit of $400, how many bags did the trader buy? Answer with just the number.", "40"),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? Answer with just the number.", "5"),
]

# ── Run Benchmarks ──
def run_mmlu():
    correct = 0
    total = len(MMLU)
    print(f"\n{'='*60}")
    print(f"MMLU ({total} questions)")
    print(f"{'='*60}")
    for i, (q, expected) in enumerate(MMLU):
        prompt = f"Answer the following multiple choice question. Reply with ONLY the letter (A, B, C, or D).\n\n{q}"
        try:
            resp = ask(prompt, max_tokens=1024)
            got = extract_answer(resp)
            ok = got == expected
            if ok: correct += 1
            status = "✓" if ok else "✗"
            print(f"  [{status}] Q{i+1}: expected={expected} got={got} | {resp[:60]}")
        except Exception as e:
            print(f"  [!] Q{i+1}: ERROR {e}")
    acc = correct / total * 100
    print(f"\nMMLU Accuracy: {correct}/{total} = {acc:.1f}%")
    return acc

def run_gsm8k():
    correct = 0
    total = len(GSM8K)
    print(f"\n{'='*60}")
    print(f"GSM8K ({total} questions)")
    print(f"{'='*60}")
    for i, (q, expected) in enumerate(GSM8K):
        try:
            resp = ask(q, max_tokens=1024)
            # Extract number from response
            numbers = re.findall(r'[\d,]+', resp.replace(',', ''))
            got = numbers[-1] if numbers else ""
            ok = got == expected
            if ok: correct += 1
            status = "✓" if ok else "✗"
            print(f"  [{status}] Q{i+1}: expected={expected} got={got} | {resp[:80]}")
        except Exception as e:
            print(f"  [!] Q{i+1}: ERROR {e}")
    acc = correct / total * 100
    print(f"\nGSM8K Accuracy: {correct}/{total} = {acc:.1f}%")
    return acc

def run_bangla():
    questions = [
        ("বাংলাদেশের রাজধানী কী?\nA. চট্টগ্রাম\nB. ঢাকা\nC. রাজশাহী\nD. সিলেট", "B"),
        ("পদ্মা সেতু কোন নদীর উপর নির্মিত?\nA. মেঘনা\nB. যমুনা\nC. পদ্মা\nD. ব্রহ্মপুত্র", "C"),
        ("বাংলাদেশের জাতীয় ফুল কী?\nA. গোলাপ\nB. শাপলা\nC. জবা\nD. বেলি", "B"),
        ("বাংলাদেশের স্বাধীনতা দিবস কবে?\nA. ১৬ ডিসেম্বর\nB. ২৬ মার্চ\nC. ২১ ফেব্রুয়ারি\nD. ১৪ এপ্রিল", "B"),
        ("একুশে ফেব্রুয়ারি কিসের দিবস?\nA. স্বাধীনতা দিবস\nB. বিজয় দিবস\nC. আন্তর্জাতিক মাতৃভাষা দিবস\nD. জাতীয় শোক দিবস", "C"),
        ("বাংলাদেশের জাতীয় সংগীত কে লিখেছেন?\nA. কাজী নজরুল ইসলাম\nB. রবীন্দ্রনাথ ঠাকুর\nC. জীবনানন্দ দাশ\nD. মাইকেল মধুসূদন দত্ত", "B"),
        ("বাংলাদেশের বিজয় দিবস কবে?\nA. ২৬ মার্চ\nB. ২১ ফেব্রুয়ারি\nC. ১৬ ডিসেম্বর\nD. ১৪ এপ্রিল", "C"),
        ("বাংলাদেশের সবচেয়ে বড় নদী কোনটি?\nA. মেঘনা\nB. পদ্মা\nC. যমুনা\nD. ব্রহ্মপুত্র", "B"),
        ("বাংলাদেশের জাতীয় খেলা কী?\nA. ক্রিকেট\nB. ফুটবল\nC. কাবাডি\nD. হা-ডু-ডু", "C"),
        ("সুন্দরবন কোন বিভাগে অবস্থিত?\nA. ঢাকা\nB. চট্টগ্রাম\nC. খুলনা\nD. রাজশাহী", "C"),
    ]
    correct = 0
    total = len(questions)
    print(f"\n{'='*60}")
    print(f"Bangla Knowledge ({total} questions)")
    print(f"{'='*60}")
    for i, (q, expected) in enumerate(questions):
        prompt = f"নিচের প্রশ্নের উত্তর দাও। শুধু অক্ষর (A, B, C, বা D) দিয়ে উত্তর দাও।\n\n{q}"
        try:
            resp = ask(prompt, max_tokens=1024)
            got = extract_answer(resp)
            ok = got == expected
            if ok: correct += 1
            status = "✓" if ok else "✗"
            print(f"  [{status}] Q{i+1}: expected={expected} got={got} | {resp[:60]}")
        except Exception as e:
            print(f"  [!] Q{i+1}: ERROR {e}")
    acc = correct / total * 100
    print(f"\nBangla Accuracy: {correct}/{total} = {acc:.1f}%")
    return acc

CODE_EVAL = [
    ("What is the output of: print(len([1,2,3,4,5]))\nAnswer with just the output, nothing else.", "5"),
    ("What is the output of: print(2**10)\nAnswer with just the output, nothing else.", "1024"),
    ("What is the output of: print('hello world'.upper())\nAnswer with just the output, nothing else.", "HELLO WORLD"),
    ("What is the output of: print(sorted([3,1,4,1,5,9,2,6]))\nAnswer with just the output, nothing else.", "[1, 1, 2, 3, 4, 5, 6, 9]"),
    ("What is the output of: print(sum(range(1,11)))\nAnswer with just the output, nothing else.", "55"),
    ("What is the output of: print(type(3.14).__name__)\nAnswer with just the output, nothing else.", "float"),
    ("What is the output of: print('abcdef'[2:5])\nAnswer with just the output, nothing else.", "cde"),
    ("What is the output of: print(bool([]))\nAnswer with just the output, nothing else.", "False"),
    ("What is the output of: print(max(3, 7, 2, 9, 4))\nAnswer with just the output, nothing else.", "9"),
    ("What does this function return? def f(n): return n if n<=1 else f(n-1)+f(n-2)\nf(7)\nAnswer with just the number.", "13"),
]

REASONING = [
    ("If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?\nA. Yes\nB. No\nC. Cannot be determined\nD. Only if they are red", "C"),
    ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents?\nA. 10\nB. 5\nC. 15\nD. 20", "B"),
    ("If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets?\nA. 100\nB. 50\nC. 5\nD. 1", "C"),
    ("Mary's father has 5 daughters: Nana, Nene, Nini, Nono and ___?\nA. Nunu\nB. Mary\nC. Nana\nD. None", "B"),
    ("Which is heavier, a pound of feathers or a pound of steel?\nA. Feathers\nB. Steel\nC. They weigh the same\nD. Cannot be determined", "C"),
]

def run_code():
    correct = 0
    total = len(CODE_EVAL)
    print(f"\n{'='*60}")
    print(f"Code Eval ({total} questions)")
    print(f"{'='*60}")
    for i, (q, expected) in enumerate(CODE_EVAL):
        try:
            resp = ask(q, max_tokens=1024)
            resp_clean = resp.strip().strip('`').strip()
            ok = expected in resp_clean
            if ok: correct += 1
            status = "✓" if ok else "✗"
            print(f"  [{status}] Q{i+1}: expected={expected} got={resp_clean[:60]}")
        except Exception as e:
            print(f"  [!] Q{i+1}: ERROR {e}")
    acc = correct / total * 100
    print(f"\nCode Accuracy: {correct}/{total} = {acc:.1f}%")
    return acc

def run_reasoning():
    correct = 0
    total = len(REASONING)
    print(f"\n{'='*60}")
    print(f"Reasoning ({total} questions)")
    print(f"{'='*60}")
    for i, (q, expected) in enumerate(REASONING):
        prompt = f"Answer the following question. Reply with ONLY the letter (A, B, C, or D).\n\n{q}"
        try:
            resp = ask(prompt, max_tokens=1024)
            got = extract_answer(resp)
            ok = got == expected
            if ok: correct += 1
            status = "✓" if ok else "✗"
            print(f"  [{status}] Q{i+1}: expected={expected} got={got} | {resp[:60]}")
        except Exception as e:
            print(f"  [!] Q{i+1}: ERROR {e}")
    acc = correct / total * 100
    print(f"\nReasoning Accuracy: {correct}/{total} = {acc:.1f}%")
    return acc

if __name__ == "__main__":
    print("AdiTurbo Gemma 4 26B-A4B IQ3_M Stress Test")
    print(f"API: {API}")
    print(f"Questions: {len(MMLU)} MMLU + {len(GSM8K)} GSM8K + {len(REASONING)} Reasoning + {len(CODE_EVAL)} Code + {len(run_bangla.__code__.co_consts[1]) if False else 10} Bangla")

    t0 = time.time()
    mmlu = run_mmlu()
    gsm = run_gsm8k()
    reasoning = run_reasoning()
    code = run_code()
    bangla = run_bangla()
    elapsed = time.time() - t0

    answered = 0
    total_q = len(MMLU) + len(GSM8K) + len(REASONING) + len(CODE_EVAL) + 10
    print(f"\n{'='*60}")
    print(f"STRESS TEST SUMMARY — {total_q} questions, {elapsed:.0f}s")
    print(f"{'='*60}")
    print(f"  MMLU:      {mmlu:.1f}%  ({len(MMLU)} questions)")
    print(f"  GSM8K:     {gsm:.1f}%  ({len(GSM8K)} questions)")
    print(f"  Reasoning: {reasoning:.1f}%  ({len(REASONING)} questions)")
    print(f"  Code:      {code:.1f}%  ({len(CODE_EVAL)} questions)")
    print(f"  Bangla:    {bangla:.1f}%  (10 questions)")
    print(f"  ─────────────────────────────")
    avg = (mmlu + gsm + reasoning + code + bangla) / 5
    print(f"  Average:   {avg:.1f}%")
    print(f"  Time:      {elapsed:.0f}s ({elapsed/total_q:.1f}s per question)")
    print(f"{'='*60}")
