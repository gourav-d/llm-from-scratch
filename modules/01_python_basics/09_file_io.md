# Lesson 1.9: File I/O (Reading and Writing Files)

## 🎯 What You'll Learn
- Reading files
- Writing files
- File modes
- Context managers (`with` statement)
- Working with different file formats

---

## Opening Files

### Basic Syntax

```python
# Open a file
file = open("filename.txt", "r")  # r = read mode

# Read content
content = file.read()

# IMPORTANT: Always close the file!
file.close()
```

**File Modes:**
- `"r"` → Read (default) - File must exist
- `"w"` → Write - Creates new or overwrites existing
- `"a"` → Append - Adds to end of file
- `"x"` → Exclusive creation - Fails if file exists
- `"r+"` → Read and write
- `"b"` → Binary mode (e.g., `"rb"`, `"wb"`)

---

## Reading Files

### Method 1: Read Entire File

```python
file = open("example.txt", "r")
content = file.read()
print(content)
file.close()
```

### Method 2: Read Line by Line

```python
file = open("example.txt", "r")

# Read all lines into a list
lines = file.readlines()
for line in lines:
    print(line, end="")  # end="" prevents double newlines

file.close()
```

### Method 3: Read One Line at a Time

```python
file = open("example.txt", "r")

# Read first line
line1 = file.readline()
print(line1)

# Read next line
line2 = file.readline()
print(line2)

file.close()
```

### Method 4: Iterate Over Lines (Best!)

```python
file = open("example.txt", "r")

for line in file:
    print(line.strip())  # strip() removes newline

file.close()
```

---

## Writing Files

### Write Mode (`"w"`) - Overwrites

```python
# This creates a new file or OVERWRITES existing
file = open("output.txt", "w")

file.write("Hello, World!\n")
file.write("This is a new file.\n")

file.close()
```

**Warning:** `"w"` mode **deletes** existing content!

### Append Mode (`"a"`) - Adds to End

```python
# This ADDS to existing file
file = open("output.txt", "a")

file.write("This line is added.\n")
file.write("Another line.\n")

file.close()
```

### Writing Multiple Lines

```python
file = open("output.txt", "w")

lines = [
    "First line\n",
    "Second line\n",
    "Third line\n"
]

file.writelines(lines)
file.close()
```

---

## Context Managers (`with` statement)

**Best Practice:** Use `with` to automatically close files!

```python
# Traditional way
file = open("example.txt", "r")
content = file.read()
file.close()

# Better way - with statement
with open("example.txt", "r") as file:
    content = file.read()
    print(content)
# File is automatically closed here!
```

**Benefits:**
- Automatically closes file (even if error occurs)
- Cleaner code
- No need to remember `file.close()`

**C# Comparison:**
```csharp
// C# using statement
using (var file = File.OpenText("example.txt"))
{
    var content = file.ReadToEnd();
}

// Python with statement
with open("example.txt", "r") as file:
    content = file.read()
```

---

## Reading Examples

### Read and Print All Lines

```python
with open("example.txt", "r") as file:
    for line in file:
        print(line.strip())
```

### Read Specific Number of Characters

```python
with open("example.txt", "r") as file:
    # Read first 10 characters
    chunk = file.read(10)
    print(chunk)
```

### Read and Process Lines

```python
# Count lines
with open("example.txt", "r") as file:
    line_count = sum(1 for line in file)
    print(f"Total lines: {line_count}")

# Count words
with open("example.txt", "r") as file:
    word_count = sum(len(line.split()) for line in file)
    print(f"Total words: {word_count}")
```

### Read into List

```python
with open("example.txt", "r") as file:
    lines = file.readlines()
    # lines is now a list of all lines

# Or use list comprehension
with open("example.txt", "r") as file:
    lines = [line.strip() for line in file]
```

---

## Writing Examples

### Write Simple Text

```python
with open("output.txt", "w") as file:
    file.write("Hello, Python!\n")
    file.write("File I/O is easy.\n")
```

### Write from List

```python
data = ["Apple", "Banana", "Orange"]

with open("fruits.txt", "w") as file:
    for fruit in data:
        file.write(f"{fruit}\n")
```

### Write Formatted Data

```python
students = [
    {"name": "Alice", "score": 95},
    {"name": "Bob", "score": 87},
    {"name": "Charlie", "score": 92}
]

with open("scores.txt", "w") as file:
    for student in students:
        file.write(f"{student['name']}: {student['score']}\n")
```

---

## File Paths

### Relative Paths

```python
# Same directory
with open("file.txt", "r") as file:
    content = file.read()

# Subdirectory
with open("data/file.txt", "r") as file:
    content = file.read()

# Parent directory
with open("../file.txt", "r") as file:
    content = file.read()
```

### Absolute Paths

```python
# Windows
with open("C:\\Users\\Name\\Documents\\file.txt", "r") as file:
    content = file.read()

# Or use raw strings
with open(r"C:\Users\Name\Documents\file.txt", "r") as file:
    content = file.read()

# Linux/Mac
with open("/home/user/documents/file.txt", "r") as file:
    content = file.read()
```

### Using pathlib (Modern Way)

```python
from pathlib import Path

# Create path object
file_path = Path("data/file.txt")

# Read
content = file_path.read_text()

# Write
file_path.write_text("Hello, World!")

# Check if exists
if file_path.exists():
    print("File exists!")
```

---

## Working with CSV Files

### Reading CSV

```python
# Manual parsing
with open("data.csv", "r") as file:
    for line in file:
        values = line.strip().split(",")
        print(values)

# Using csv module (better!)
import csv

with open("data.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)  # Each row is a list

# Read as dictionaries
with open("data.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row)  # Each row is a dictionary
```

### Writing CSV

```python
import csv

data = [
    ["Name", "Age", "City"],
    ["Alice", "30", "NYC"],
    ["Bob", "25", "LA"]
]

with open("output.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)

# Write dictionaries
people = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25}
]

with open("output.csv", "w", newline="") as file:
    fieldnames = ["name", "age"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()  # Write header row
    writer.writerows(people)
```

---

## Working with JSON Files

```python
import json

# Reading JSON
with open("data.json", "r") as file:
    data = json.load(file)  # Converts JSON to Python dict/list
    print(data)

# Writing JSON
data = {
    "name": "Alice",
    "age": 30,
    "hobbies": ["reading", "coding"]
}

with open("output.json", "w") as file:
    json.dump(data, file, indent=4)  # indent=4 for pretty printing

# Convert Python to JSON string
json_string = json.dumps(data, indent=4)
print(json_string)

# Convert JSON string to Python
data = json.loads(json_string)
```

**Example JSON file:**
```json
{
    "users": [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25}
    ]
}
```

**Reading:**
```python
import json

with open("users.json", "r") as file:
    data = json.load(file)
    for user in data["users"]:
        print(f"{user['name']}: {user['age']}")
```

---

## Error Handling with Files

```python
# Check if file exists
try:
    with open("nonexistent.txt", "r") as file:
        content = file.read()
except FileNotFoundError:
    print("File not found!")

# Better - check first
from pathlib import Path

file_path = Path("example.txt")
if file_path.exists():
    content = file_path.read_text()
else:
    print("File doesn't exist!")
```

---

## Binary Files

### Reading Binary

```python
# Read image
with open("image.jpg", "rb") as file:  # rb = read binary
    data = file.read()
    print(f"File size: {len(data)} bytes")

# Copy file
with open("source.jpg", "rb") as source:
    with open("destination.jpg", "wb") as dest:
        dest.write(source.read())
```

---

## Practical Examples

### Example 1: Count Lines, Words, Characters

```python
def file_stats(filename):
    with open(filename, "r") as file:
        content = file.read()
        lines = content.split("\n")
        words = content.split()
        chars = len(content)

    print(f"Lines: {len(lines)}")
    print(f"Words: {len(words)}")
    print(f"Characters: {chars}")

file_stats("example.txt")
```

### Example 2: Search in File

```python
def search_in_file(filename, search_term):
    with open(filename, "r") as file:
        for line_num, line in enumerate(file, 1):
            if search_term in line:
                print(f"Line {line_num}: {line.strip()}")

search_in_file("example.txt", "Python")
```

### Example 3: Replace Text

```python
def replace_in_file(filename, old_text, new_text):
    # Read
    with open(filename, "r") as file:
        content = file.read()

    # Replace
    content = content.replace(old_text, new_text)

    # Write back
    with open(filename, "w") as file:
        file.write(content)

replace_in_file("example.txt", "old", "new")
```

### Example 4: Log File

```python
from datetime import datetime

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("log.txt", "a") as file:
        file.write(f"[{timestamp}] {message}\n")

log_message("Application started")
log_message("User logged in")
log_message("Error occurred")
```

---

## 💡 Key Takeaways

1. **Always use `with`** → Automatically closes files
2. **Read modes** → `"r"`, `"w"`, `"a"`
3. **Read entire** → `file.read()`
4. **Read lines** → `file.readlines()` or iterate
5. **Write** → `file.write(text)`
6. **CSV** → Use `csv` module
7. **JSON** → Use `json` module
8. **Paths** → Use `pathlib` for modern code
9. **Check exists** → Use `Path.exists()`

---

## ✏️ Practice Exercise

Create `file_io_practice.py`:

```python
# 1. Write to file
with open("test.txt", "w") as file:
    file.write("Hello, File I/O!\n")
    file.write("This is line 2.\n")
    file.write("This is line 3.\n")

# 2. Read from file
with open("test.txt", "r") as file:
    content = file.read()
    print(content)

# 3. Append to file
with open("test.txt", "a") as file:
    file.write("This line was added.\n")

# 4. Read line by line
with open("test.txt", "r") as file:
    for line_num, line in enumerate(file, 1):
        print(f"{line_num}: {line.strip()}")

# 5. Write list to file
fruits = ["Apple", "Banana", "Orange"]
with open("fruits.txt", "w") as file:
    for fruit in fruits:
        file.write(f"{fruit}\n")

# 6. Read into list
with open("fruits.txt", "r") as file:
    fruits = [line.strip() for line in file]
    print(fruits)

# 7. JSON example
import json

data = {
    "name": "Alice",
    "age": 30,
    "hobbies": ["reading", "coding"]
}

# Write JSON
with open("person.json", "w") as file:
    json.dump(data, file, indent=4)

# Read JSON
with open("person.json", "r") as file:
    person = json.load(file)
    print(person)
```

**Run it:** `python file_io_practice.py`

---

## 🤔 Quick Quiz

1. What's the best way to open files in Python?
   <details>
   <summary>Answer</summary>

   Use `with` statement:
   ```python
   with open("file.txt", "r") as file:
       content = file.read()
   ```
   </details>

2. What's the difference between `"w"` and `"a"` modes?
   <details>
   <summary>Answer</summary>

   - `"w"` → Overwrites file (deletes existing content)
   - `"a"` → Appends to file (adds to end)
   </details>

3. How do you read all lines into a list?
   <details>
   <summary>Answer</summary>

   ```python
   with open("file.txt", "r") as file:
       lines = file.readlines()
   # or
   lines = [line.strip() for line in file]
   ```
   </details>

4. How do you read a JSON file?
   <details>
   <summary>Answer</summary>

   ```python
   import json
   with open("file.json", "r") as file:
       data = json.load(file)
   ```
   </details>

5. Why use `with` statement?
   <details>
   <summary>Answer</summary>

   Automatically closes file, even if error occurs. No need to manually call `file.close()`
   </details>

---

**Next Lesson:** [10_error_handling.md](10_error_handling.md) - Learn to handle errors gracefully!
