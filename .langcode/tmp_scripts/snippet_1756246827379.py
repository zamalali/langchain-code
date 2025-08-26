
import os

def count_python_files(path):
    count = 0
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".py"):
                count += 1
    return count

if __name__ == "__main__":
    num_files = count_python_files(".")
    print(f"Number of Python files: {num_files}")
