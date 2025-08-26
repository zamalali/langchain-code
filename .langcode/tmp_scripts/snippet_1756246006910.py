
import os

count = 0
for root, _, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            count += 1

print(f"Number of Python files: {count}")
