import sqlite3
import sys

db_path = "data/nsys-hero/fastvideo/trace_20260223_213127.sqlite"
conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
cursor = conn.cursor()

# Find all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cursor.fetchall()]

kernel_table = next((t for t in tables if "KERNEL" in t.upper()), None)
print(f"Kernel table found: {kernel_table}")

if kernel_table:
    cursor.execute(f"PRAGMA table_info({kernel_table})")
    cols = cursor.fetchall()
    print("Columns:")
    for c in cols:
        print(f"  {c[1]} ({c[2]})")

print("Checking for Tensor/TARGET_INFO tables...")
for t in tables:
    if "TARGET" in t.upper() or "EXEC" in t.upper() or "TENSOR" in t.upper():
        print(f"Found table: {t}")
        cursor.execute(f"PRAGMA table_info({t})")
        for c in cursor.fetchall():
            print(f"  {c[1]} ({c[2]})")

conn.close()
