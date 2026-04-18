import sys

with open('src/llama-context.cpp', 'r') as f:
    lines = f.readlines()

new_lines = []
skip = 0
for i, line in enumerate(lines):
    if skip > 0:
        skip -= 1
        continue
    
    # Identify the accidental patch in graph_reserve (around line 2130)
    if '// Manual input buffering' in line and i < 2700:
        # Skip this block (approx 10 lines)
        skip = 9 # Adjust based on the size of the block I inserted
        print(f"Skipping accidental patch at line {i}")
        continue
    
    new_lines.append(line)

with open('src/llama-context.cpp', 'w') as f:
    f.writelines(new_lines)
print("Removed accidental patch")
