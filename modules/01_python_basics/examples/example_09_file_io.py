"""
Example 9: File I/O for .NET Developers
This file demonstrates Python file operations with detailed comments and C# comparisons.
"""

import json
import os

# ============================================
# SECTION 1: OPENING FILES - BASIC SYNTAX
# ============================================

print("=== SECTION 1: OPENING FILES - BASIC SYNTAX ===\n")

# C#: var file = File.OpenText("example.txt");
# Python: file = open("example.txt", "r")

# File modes:
# "r" = Read (default) - File must exist
# "w" = Write - Creates new or overwrites
# "a" = Append - Adds to end
# "x" = Exclusive - Fails if exists
# "b" = Binary mode (rb, wb, etc.)
# "r+" = Read and write

print("File modes:")
print("  'r'  - Read (file must exist)")
print("  'w'  - Write (overwrites if exists)")
print("  'a'  - Append (adds to end)")
print("  'x'  - Exclusive create (error if exists)")
print("  'b'  - Binary mode (rb, wb)")
print("  'r+' - Read and write")

print()

# ============================================
# SECTION 2: CONTEXT MANAGERS (with STATEMENT)
# ============================================

print("=== SECTION 2: CONTEXT MANAGERS (with STATEMENT) ===\n")

# BEST PRACTICE: Use 'with' statement!
# Automatically closes file, even if error occurs

# C#:
# using (var reader = File.OpenText("example.txt"))
# {
#     var content = reader.ReadToEnd();
# }

# Python:
# with open("example.txt", "r") as file:
#     content = file.read()

print("The 'with' statement:")
print("- Automatically closes file")
print("- Works even if error occurs")
print("- No need to call file.close()")
print()

# Create a test file first
print("Creating test file 'demo.txt'...")
with open("demo.txt", "w") as file:
    file.write("Hello, World!\n")
    file.write("This is line 2.\n")
    file.write("This is line 3.\n")
print("File created!")

print()

# ============================================
# SECTION 3: READING FILES
# ============================================

print("=== SECTION 3: READING FILES ===\n")

# Method 1: Read entire file
# C#: File.ReadAllText("demo.txt")
# Python: open("demo.txt").read()

print("Method 1: Read entire file:")
with open("demo.txt", "r") as file:
    content = file.read()  # Read all content as single string
    print(content)

print()

# Method 2: Read all lines into list
# C#: File.ReadAllLines("demo.txt")
# Python: file.readlines()

print("Method 2: Read all lines (as list):")
with open("demo.txt", "r") as file:
    lines = file.readlines()  # Returns list of lines
    print(f"Lines: {lines}")
    print(f"First line: {lines[0]}")

print()

# Method 3: Read one line at a time
print("Method 3: Read one line at a time:")
with open("demo.txt", "r") as file:
    line1 = file.readline()  # Read first line
    line2 = file.readline()  # Read next line
    print(f"Line 1: {line1.strip()}")
    print(f"Line 2: {line2.strip()}")

print()

# Method 4: Iterate over lines (BEST!)
# C#: foreach (var line in File.ReadLines("demo.txt"))
# Python: for line in file:

print("Method 4: Iterate over lines (best way):")
with open("demo.txt", "r") as file:
    for line in file:
        print(f"  {line.strip()}")  # strip() removes newline

print()

# Method 5: Read specific number of characters
print("Method 5: Read specific characters:")
with open("demo.txt", "r") as file:
    chunk = file.read(10)  # Read first 10 characters
    print(f"First 10 chars: '{chunk}'")

print()

# ============================================
# SECTION 4: WRITING FILES
# ============================================

print("=== SECTION 4: WRITING FILES ===\n")

# Write mode ('w') - Overwrites file!
# C#: File.WriteAllText("output.txt", "content");
# Python: open("output.txt", "w").write("content")

print("Write mode ('w') - Creates new or overwrites:")
with open("output.txt", "w") as file:
    file.write("Hello, Python!\n")
    file.write("This is a new file.\n")
    file.write("Write mode overwrites!\n")
print("File 'output.txt' created!")

print()

# Read it back
print("Reading back 'output.txt':")
with open("output.txt", "r") as file:
    print(file.read())

# ============================================
# SECTION 5: APPENDING TO FILES
# ============================================

print("=== SECTION 5: APPENDING TO FILES ===\n")

# Append mode ('a') - Adds to end
# C#: File.AppendAllText("output.txt", "more content");
# Python: open("output.txt", "a").write("more content")

print("Append mode ('a') - Adds to end:")
with open("output.txt", "a") as file:
    file.write("This line was appended.\n")
    file.write("Another appended line.\n")
print("Lines appended!")

print()

# Read updated file
print("Reading updated 'output.txt':")
with open("output.txt", "r") as file:
    print(file.read())

# ============================================
# SECTION 6: WRITING MULTIPLE LINES
# ============================================

print("=== SECTION 6: WRITING MULTIPLE LINES ===\n")

# writelines() - Write list of strings
# C#: File.WriteAllLines("lines.txt", lines);
# Python: file.writelines(lines)

print("Using writelines():")
lines = [
    "First line\n",
    "Second line\n",
    "Third line\n"
]

with open("lines.txt", "w") as file:
    file.writelines(lines)
print("File 'lines.txt' created!")

print()

# Read it back
print("Reading 'lines.txt':")
with open("lines.txt", "r") as file:
    print(file.read())

# ============================================
# SECTION 7: READING LINE BY LINE WITH ENUMERATE
# ============================================

print("=== SECTION 7: READING WITH LINE NUMBERS ===\n")

print("Reading with line numbers:")
with open("demo.txt", "r") as file:
    for line_num, line in enumerate(file, 1):
        print(f"Line {line_num}: {line.strip()}")

print()

# ============================================
# SECTION 8: FILE PATHS
# ============================================

print("=== SECTION 8: FILE PATHS ===\n")

# Relative paths
print("Relative paths:")
print("  'file.txt' - Same directory")
print("  'data/file.txt' - Subdirectory")
print("  '../file.txt' - Parent directory")

print()

# Absolute paths
print("Absolute paths:")
print("  Windows: 'C:\\\\Users\\\\Name\\\\file.txt' or r'C:\\Users\\Name\\file.txt'")
print("  Linux/Mac: '/home/user/file.txt'")

print()

# Check if file exists
# C#: File.Exists("demo.txt")
# Python: os.path.exists("demo.txt")

print("Checking if file exists:")
if os.path.exists("demo.txt"):
    print("  'demo.txt' exists!")
else:
    print("  'demo.txt' not found!")

if os.path.exists("nonexistent.txt"):
    print("  'nonexistent.txt' exists!")
else:
    print("  'nonexistent.txt' not found!")

print()

# ============================================
# SECTION 9: WORKING WITH JSON
# ============================================

print("=== SECTION 9: WORKING WITH JSON ===\n")

# JSON = JavaScript Object Notation
# Python's dict/list maps perfectly to JSON

# Writing JSON
# C#: File.WriteAllText("data.json", JsonSerializer.Serialize(data));
# Python: json.dump(data, file)

print("Writing JSON:")
data = {
    "name": "Alice",
    "age": 30,
    "city": "NYC",
    "hobbies": ["reading", "coding", "music"]
}

with open("data.json", "w") as file:
    json.dump(data, file, indent=4)  # indent=4 for pretty printing
print("JSON file 'data.json' created!")

print()

# Read JSON file
print("Reading JSON file 'data.json':")
with open("data.json", "r") as file:
    loaded_data = json.load(file)  # Converts JSON to Python dict
    print(f"Loaded data: {loaded_data}")
    print(f"Name: {loaded_data['name']}")
    print(f"Hobbies: {loaded_data['hobbies']}")

print()

# JSON strings (not files)
print("JSON strings:")

# Convert Python to JSON string
json_string = json.dumps(data, indent=2)
print(f"Python → JSON string:\n{json_string}")

print()

# Convert JSON string to Python
json_text = '{"name": "Bob", "age": 25}'
parsed = json.loads(json_text)
print(f"JSON string → Python: {parsed}")

print()

# ============================================
# SECTION 10: PRACTICAL EXAMPLES
# ============================================

print("=== SECTION 10: PRACTICAL EXAMPLES ===\n")

# Example 1: Write list to file
print("Example 1: Write list to file:")
fruits = ["apple", "banana", "orange", "grape"]

with open("fruits.txt", "w") as file:
    for fruit in fruits:
        file.write(f"{fruit}\n")
print("Fruits written to 'fruits.txt'")

# Read back into list
with open("fruits.txt", "r") as file:
    loaded_fruits = [line.strip() for line in file]
    print(f"Loaded fruits: {loaded_fruits}")

print()

# Example 2: Count lines, words, characters
print("Example 2: File statistics:")

def file_stats(filename):
    """Calculate file statistics."""
    with open(filename, "r") as file:
        content = file.read()
        lines = content.split("\n")
        words = content.split()
        chars = len(content)

    print(f"  File: {filename}")
    print(f"  Lines: {len(lines)}")
    print(f"  Words: {len(words)}")
    print(f"  Characters: {chars}")

file_stats("demo.txt")

print()

# Example 3: Search in file
print("Example 3: Search in file:")

def search_in_file(filename, search_term):
    """Search for term in file."""
    print(f"  Searching for '{search_term}' in {filename}:")
    with open(filename, "r") as file:
        for line_num, line in enumerate(file, 1):
            if search_term in line:
                print(f"    Line {line_num}: {line.strip()}")

search_in_file("demo.txt", "line")

print()

# Example 4: Copy file
print("Example 4: Copy file:")

def copy_file(source, destination):
    """Copy file."""
    with open(source, "r") as src:
        with open(destination, "w") as dst:
            dst.write(src.read())
    print(f"  Copied '{source}' → '{destination}'")

copy_file("demo.txt", "demo_copy.txt")

print()

# Example 5: Append to log file
print("Example 5: Logging to file:")

def log_message(message):
    """Append message to log file with timestamp."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("app.log", "a") as file:
        file.write(f"[{timestamp}] {message}\n")
    print(f"  Logged: {message}")

log_message("Application started")
log_message("User logged in")
log_message("Processing data")

# Read log
print("\n  Log contents:")
with open("app.log", "r") as file:
    print(file.read())

# ============================================
# SECTION 11: ERROR HANDLING WITH FILES
# ============================================

print("=== SECTION 11: ERROR HANDLING WITH FILES ===\n")

# FileNotFoundError - File doesn't exist
print("Example 1: FileNotFoundError:")
try:
    with open("nonexistent.txt", "r") as file:
        content = file.read()
except FileNotFoundError:
    print("  Error: File not found!")

print()

# Better - check first
print("Example 2: Check before opening:")
filename = "demo.txt"
if os.path.exists(filename):
    with open(filename, "r") as file:
        print(f"  File '{filename}' opened successfully")
else:
    print(f"  File '{filename}' not found")

print()

# Safe file reading function
print("Example 3: Safe file reader:")

def safe_read_file(filename):
    """Read file with error handling."""
    try:
        with open(filename, "r") as file:
            return file.read()
    except FileNotFoundError:
        print(f"  Error: '{filename}' not found")
        return None
    except PermissionError:
        print(f"  Error: No permission to read '{filename}'")
        return None
    except Exception as e:
        print(f"  Unexpected error: {e}")
        return None

content = safe_read_file("demo.txt")
if content:
    print(f"  Read {len(content)} characters")

content = safe_read_file("nonexistent.txt")

print()

# ============================================
# SECTION 12: CLEANUP - DELETE TEST FILES
# ============================================

print("=== SECTION 12: CLEANUP ===\n")

# Clean up test files
test_files = [
    "demo.txt", "output.txt", "lines.txt",
    "data.json", "fruits.txt", "demo_copy.txt", "app.log"
]

print("Cleaning up test files:")
for filename in test_files:
    if os.path.exists(filename):
        os.remove(filename)
        print(f"  Deleted '{filename}'")

print("\nTest files cleaned up!")

print()

# ============================================
# SUMMARY
# ============================================

print("=== SUMMARY ===\n")

summary = """
File I/O for .NET Developers:

OPENING FILES:
  C#: var file = File.OpenText("file.txt");
  Python: file = open("file.txt", "r")

FILE MODES:
  "r"  - Read (default, must exist)
  "w"  - Write (creates/overwrites)
  "a"  - Append (add to end)
  "x"  - Exclusive create (error if exists)
  "b"  - Binary mode (rb, wb)
  "r+" - Read and write

CONTEXT MANAGER (BEST PRACTICE):
  C#:
    using (var reader = File.OpenText("file.txt"))
    {
        var content = reader.ReadToEnd();
    }

  Python:
    with open("file.txt", "r") as file:
        content = file.read()

  Benefits:
  - Automatically closes file
  - Works even if error occurs
  - No need for file.close()

READING FILES:
  C#: File.ReadAllText("file.txt")
  Python: file.read()  # Read all

  C#: File.ReadAllLines("file.txt")
  Python: file.readlines()  # List of lines

  C#: foreach (var line in File.ReadLines("file.txt"))
  Python: for line in file:  # Iterate (best!)

  file.readline()  # Read one line
  file.read(10)    # Read 10 characters

WRITING FILES:
  C#: File.WriteAllText("file.txt", "content");
  Python: file.write("content")

  C#: File.WriteAllLines("file.txt", lines);
  Python: file.writelines(lines)

APPEND:
  C#: File.AppendAllText("file.txt", "more");
  Python: open("file.txt", "a").write("more")

JSON:
  Write:
    C#: File.WriteAllText("data.json", JsonSerializer.Serialize(obj));
    Python: json.dump(obj, file)

  Read:
    C#: JsonSerializer.Deserialize<T>(File.ReadAllText("data.json"));
    Python: json.load(file)

  Strings:
    json.dumps(obj) - Python → JSON string
    json.loads(str) - JSON string → Python

FILE PATHS:
  Relative: "file.txt", "data/file.txt", "../file.txt"
  Absolute: "C:\\\\path\\\\file.txt" or r"C:\\path\\file.txt"

CHECK EXISTS:
  C#: File.Exists("file.txt")
  Python: os.path.exists("file.txt")

ERROR HANDLING:
  try:
      with open("file.txt", "r") as file:
          content = file.read()
  except FileNotFoundError:
      print("File not found!")

COMMON PATTERNS:
  Read all lines:
    with open("file.txt", "r") as file:
        lines = [line.strip() for line in file]

  Write list:
    with open("file.txt", "w") as file:
        for item in items:
            file.write(f"{item}\\n")

  Append to log:
    with open("app.log", "a") as file:
        file.write(f"[{timestamp}] {message}\\n")

C# → Python Quick Reference:
  File.ReadAllText()        → file.read()
  File.ReadAllLines()       → file.readlines()
  File.WriteAllText()       → file.write()
  File.WriteAllLines()      → file.writelines()
  File.AppendAllText()      → open("file", "a").write()
  File.Exists()             → os.path.exists()
  using (...)               → with ... as ...:
"""

print(summary)

print("="*60)
print("Next: example_10_error_handling.py - Learn exception handling!")
print("="*60)
