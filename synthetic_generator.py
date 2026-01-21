# Create placeholder files: experiment_matrix.csv and synthetic_generator.py
import pandas as pd

# Create a basic experiment matrix
import itertools

L_vals = [3,5,7]
R_vals = [0,10,20,30,50]
types = ["content","function","random","contiguous"]

rows = []
for L,R,t in itertools.product(L_vals,R_vals,types):
    rows.append({"L":L,"missing_rate":R,"missing_type":t,"N":150})

df = pd.DataFrame(rows)
csv_path = "/mnt/data/experiment_matrix.csv"
df.to_csv(csv_path, index=False)

# Create a basic synthetic_generator.py script
py_content = """import json
import csv
import random
import re

def normalize_transcription(t):
    t = t.upper()
    t = re.sub(r'[^A-ZÇĞİÖŞÜ\\s]', '', t)
    t = re.sub(r'\\s+', ' ', t).strip()
    return t.split()

def apply_missingness(tokens, rate, mtype, seed=42):
    random.seed(seed)
    n = len(tokens)
    k = int(n * rate / 100)
    indices = list(range(n))
    if k == 0:
        return tokens, []

    if mtype == 'contiguous':
        start = random.randint(0, max(0, n-k))
        miss = list(range(start, start+k))
    else:
        miss = random.sample(indices, k)
    corrupted = [tok for i, tok in enumerate(tokens) if i not in miss]
    return corrupted, miss

# Placeholder main
if __name__ == "__main__":
    print("Synthetic generator placeholder. Integrate TID JSON loading manually.")
"""

py_path = "/mnt/data/synthetic_generator.py"
with open(py_path, "w") as f:
    f.write(py_content)

csv_path, py_path
