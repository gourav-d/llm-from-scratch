"""
Module 01 - Python Basics
Exercise 09: File I/O (using io.StringIO for in-memory practice)

NOTE: These exercises use io.StringIO so you don't need to create
      actual files on disk. StringIO acts like a file in memory.
      Once you understand the pattern, reading/writing real files
      uses the exact same API -- just replace StringIO with open().

GLOSSARY
--------
open()        : Opens a real file. open("file.txt", "r") for reading,
                open("file.txt", "w") for writing, open("file.txt", "a") to append.
                Like C# File.Open() or File.ReadAllText().
read()        : Read ALL content of a file as one string.
                Like C# File.ReadAllText().
write()       : Write a string to a file.
                Like C# File.WriteAllText() or StreamWriter.Write().
close()       : Close the file when done. Releases the file handle.
                Python does this automatically when using 'with'.
with statement: Context manager. Automatically closes the file when done.
                Like C# 'using' keyword: using (var f = File.Open(...)) { ... }
StringIO      : A file-like object stored IN MEMORY, not on disk.
                from io import StringIO -- then use StringIO() like a file.
                Perfect for practicing file I/O without touching the filesystem.
mode 'r'      : Open for READING. Raises error if file doesn't exist.
mode 'w'      : Open for WRITING. Creates file if needed; OVERWRITES existing content.
mode 'a'      : Open for APPENDING. Creates file if needed; adds to end.
flush()       : Forces any buffered data to be written immediately.
                Like C# StreamWriter.Flush().
seek()        : Move the file cursor to a position.
                seek(0) moves back to the start (needed to read after writing).
"""

import io    # import the 'io' module -- contains StringIO for in-memory files

print("=" * 60)    # "=" repeated 60 times -- visual separator
print("Exercise 09: File I/O (using io.StringIO)")  # exercise title
print("=" * 60)    # separator
print()            # blank line


# ============================================================
#  EXERCISE 1
#  Topic: Parse lines from a multi-line string
#
#  Background:
#    When you read a text file, you often get one big string with
#    newline characters (\n) between lines. You need to:
#    1. Split by newline to get individual lines
#    2. Strip (trim) whitespace from each line
#    3. Ignore blank lines
#
#    In Python:
#      text = "line1\nline2\nline3"
#      lines = text.split("\n")           # split into a list of strings
#      stripped = [line.strip() for line in lines]   # remove whitespace
#      non_empty = [l for l in stripped if l]        # skip blank lines
#
#    In C#:
#      string[] lines = text.Split('\n');
#      var cleaned = lines.Select(l => l.Trim()).Where(l => l.Length > 0).ToList();
#
#  Your Task:
#    Given a multi-line string, return a list of non-empty lines
#    with leading/trailing whitespace removed.
#
#  C# Analogy:
#    text.Split('\n').Select(l=>l.Trim()).Where(l=>l.Length>0).ToList()
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 1: Parse Lines from a String")  # section title
print("-" * 50)    # separator
print()            # blank line


def parse_lines(text):
    """
    Split text into cleaned, non-empty lines.

    Args:
        text (str): A multi-line string.

    Returns:
        list: Non-empty lines with whitespace stripped.
    """
    # TODO: Split, strip, and filter lines
    # lines = text.split("\n")                  # split the big string into a list of lines
    # stripped = [line.strip() for line in lines]   # strip whitespace from each line
    # result = [line for line in stripped if line]  # keep only non-empty lines
    # return result
    pass   # replace with your code


sample_text = "  Hello, World!  \n\n  Python is fun  \n   \nLearning LLMs  "  # test string
result1 = parse_lines(sample_text)   # call the function
if result1 is not None:              # only print if something returned
    for i, line in enumerate(result1):    # enumerate gives (index, value) pairs
        print(f"  Line {i}: '{line}'")    # show each cleaned line
print()
print("  Expected: 'Hello, World!', 'Python is fun', 'Learning LLMs'")
print()


# ============================================================
#  EXERCISE 2
#  Topic: Count lines in a string
#
#  Background:
#    A common file task: count how many lines a file has.
#    With a real file: sum(1 for line in file)
#    With a string: count non-empty lines after splitting.
#
#    Useful Python trick: len(text.splitlines())
#    splitlines() handles different line endings (\n, \r\n, \r).
#    Like C# File.ReadAllLines().Length
#
#  Your Task:
#    Count the total number of lines AND the number of non-empty lines.
#    Return (total_lines, non_empty_lines).
#
#  C# Analogy:
#    string[] allLines = text.Split('\n');
#    int total = allLines.Length;
#    int nonEmpty = allLines.Count(l => l.Trim().Length > 0);
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 2: Count Lines")  # section title
print("-" * 50)    # separator
print()            # blank line


def count_lines(text):
    """
    Count total lines and non-empty lines in the text.

    Args:
        text (str): Multi-line string.

    Returns:
        tuple: (total_lines, non_empty_lines)
    """
    # TODO: Count total and non-empty lines
    # all_lines    = text.splitlines()              # split using all line ending styles
    # total_lines  = len(all_lines)                 # count ALL lines including blanks
    # non_empty    = len([l for l in all_lines if l.strip()])  # count only non-blank
    # return (total_lines, non_empty)
    pass   # replace with your code


file_content = "alpha\n\nbeta\n\n\ngamma\ndelta"   # test: 7 lines, 4 non-empty
result2 = count_lines(file_content)                # call the function
if result2 is not None:                            # only print if something returned
    total, non_empty = result2                     # unpack the 2 values
    print(f"  Total lines     : {total}")          # should be 7
    print(f"  Non-empty lines : {non_empty}")      # should be 4
print()
print("  Expected: total=7, non_empty=4")
print()


# ============================================================
#  EXERCISE 3
#  Topic: Find lines containing a keyword (like grep)
#
#  Background:
#    'grep' is a command-line tool for finding lines with a keyword.
#    We can do the same in Python:
#      matching = [line for line in lines if keyword in line]
#
#    Case-insensitive search:
#      if keyword.lower() in line.lower()
#      Like C# StringComparison.OrdinalIgnoreCase
#
#    In C#:
#      lines.Where(l => l.Contains(keyword, StringComparison.OrdinalIgnoreCase))
#
#  Your Task:
#    Given a multi-line string and a keyword, return all lines
#    that contain the keyword (case-insensitive).
#    Lines should be stripped of whitespace.
#
#  C# Analogy:
#    text.Split('\n').Where(l=>l.Contains(keyword, OrdinalIgnoreCase)).ToList()
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 3: Find Lines Containing Keyword (grep)")  # section title
print("-" * 50)    # separator
print()            # blank line


def grep_lines(text, keyword):
    """
    Return lines that contain the keyword (case-insensitive).

    Args:
        text (str): Multi-line string to search.
        keyword (str): The word to search for.

    Returns:
        list: Lines containing the keyword, stripped of whitespace.
    """
    # TODO: Filter lines by keyword (case-insensitive)
    # lines    = text.splitlines()              # split into individual lines
    # stripped = [line.strip() for line in lines]   # remove whitespace
    # keyword_lower = keyword.lower()           # convert keyword to lowercase once
    # matching = [line for line in stripped     # keep lines where...
    #             if keyword_lower in line.lower()]  # ...keyword appears (case-insensitive)
    # return matching
    pass   # replace with your code


log_text = (                         # multi-line string using string concatenation
    "INFO: Server started\n"
    "ERROR: Connection failed\n"
    "INFO: Retrying connection\n"
    "ERROR: Timeout after 30s\n"
    "INFO: Connection restored\n"
)
result3 = grep_lines(log_text, "error")   # find all lines with "error" (case-insensitive)
if result3 is not None:                   # only print if something returned
    print(f"  Lines with 'error': {len(result3)} found")
    for line in result3:                  # show each matching line
        print(f"    > {line}")
print()
print("  Expected: 2 lines containing 'ERROR'")
print()


# ============================================================
#  EXERCISE 4
#  Topic: Write lines to StringIO and read them back
#
#  Background:
#    StringIO is an in-memory file. You write to it, then move the
#    cursor back to the start (seek(0)), then read from it.
#
#    This is the SAME API as real file I/O:
#      # Real file:    f = open("myfile.txt", "w")
#      # In-memory:    f = io.StringIO()
#      f.write("line1\n")   # write to file
#      f.seek(0)            # move cursor back to start
#      content = f.read()   # read all content
#
#    In C#:
#      var sb = new StringBuilder();
#      sb.AppendLine("line1");
#      string content = sb.ToString();
#
#  Your Task:
#    1. Create a StringIO object
#    2. Write 3 lines: "Python\n", "is\n", "great\n"
#    3. Seek to position 0 (start)
#    4. Read all content back as a string
#    5. Return (the_string, list_of_lines)
#
#  C# Analogy:
#    var sw = new StringWriter();
#    sw.WriteLine("Python"); sw.WriteLine("is"); sw.WriteLine("great");
#    string content = sw.ToString();
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 4: Write to StringIO and Read Back")  # section title
print("-" * 50)    # separator
print()            # blank line


def write_and_read():
    """
    Write lines to StringIO and read them back.

    Returns:
        tuple: (full_string, list_of_lines)
    """
    # TODO: Use io.StringIO to write and read
    # buffer = io.StringIO()        # create an in-memory file
    # buffer.write("Python\n")      # write first line (include \n for newline)
    # buffer.write("is\n")          # write second line
    # buffer.write("great\n")       # write third line
    # buffer.seek(0)                # move cursor to the START (position 0)
    # content = buffer.read()       # read everything back as one string
    # lines = content.splitlines()  # split into a list of lines
    # return (content, lines)
    pass   # replace with your code


result4 = write_and_read()        # call the function
if result4 is not None:           # only print if something returned
    content, lines = result4      # unpack 2 values
    print(f"  Full content  : {repr(content)}")   # repr() shows \n characters
    print(f"  Lines         : {lines}")
print()
print("  Expected: content='Python\\nis\\ngreat\\n', lines=['Python','is','great']")
print()


# ============================================================
#  EXERCISE 5
#  Topic: CSV parsing -- split comma-separated values
#
#  Background:
#    CSV (Comma-Separated Values) is a simple file format.
#    Each line is a row; each value is separated by a comma.
#    No external library needed for simple CSVs.
#
#    Example CSV:
#      name,age,score
#      Alice,20,95
#      Bob,22,78
#
#    Parsing steps:
#      1. Split text into lines
#      2. First line = headers
#      3. Each remaining line = a data row; split by comma
#      4. Zip headers with row values to make a dict per row
#
#    In C#:
#      Like manually parsing string.Split(',') or using CsvHelper library.
#
#  Your Task:
#    Parse the CSV string below into a list of dicts.
#    Each dict maps column names to values.
#    Return: [{"name":"Alice","age":"20","score":"95"}, ...]
#
#  C# Analogy:
#    var rows = csvText.Split('\n').Skip(1)
#               .Select(l => { var parts = l.Split(','); return new{...}; });
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 5: CSV Parsing")  # section title
print("-" * 50)    # separator
print()            # blank line


def parse_csv(csv_text):
    """
    Parse a simple CSV string into a list of dictionaries.

    Args:
        csv_text (str): CSV formatted string with header row.

    Returns:
        list: List of dicts, one per data row.
    """
    # TODO: Parse the CSV into a list of dicts
    # lines   = csv_text.strip().splitlines()      # split into lines, remove extra whitespace
    # headers = lines[0].split(",")                # first line contains column names
    # result  = []                                 # list to collect row dicts
    # for line in lines[1:]:                       # lines[1:] skips the header row
    #     values = line.split(",")                 # split each row by comma
    #     row    = {}                              # empty dict for this row
    #     for i, header in enumerate(headers):     # pair each header with its value
    #         row[header] = values[i]              # map header -> value
    #     result.append(row)                       # add this row dict to the list
    # return result
    pass   # replace with your code


csv_data = "name,age,score\nAlice,20,95\nBob,22,78\nCarol,21,88"   # sample CSV
result5 = parse_csv(csv_data)          # call the function
if result5 is not None:                # only print if something returned
    for row in result5:                # loop over each parsed row
        print(f"  {row}")             # show the dict
print()
print("  Expected:")
print("    {'name': 'Alice', 'age': '20', 'score': '95'}")
print("    {'name': 'Bob',   'age': '22', 'score': '78'}")
print("    {'name': 'Carol', 'age': '21', 'score': '88'}")
print()

print("=" * 60)                       # closing separator
print("All exercises complete!")      # completion message
print("=" * 60)                       # closing separator
