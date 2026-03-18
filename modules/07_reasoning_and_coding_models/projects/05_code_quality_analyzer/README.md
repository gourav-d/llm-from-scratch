# Project 5: Code Quality Analyzer

**Measure and track code quality automatically**

---

## What You'll Build

A comprehensive code quality system that:
- Measures code quality metrics
- Tracks quality trends over time
- Identifies technical debt
- Integrates with CI/CD pipelines
- Provides actionable recommendations

**Like SonarQube, but AI-powered and free!**

---

## Why This Is Useful

### Problem: How Good Is Your Code?

Without metrics, you can't answer:
- Is this code maintainable?
- Where is our technical debt?
- Is quality improving or degrading?
- Which modules need refactoring?

### This Tool Answers All That:

```bash
quality-analyzer analyze .

# OUTPUT:
# ==========================================
# CODE QUALITY REPORT
# ==========================================
# Overall Score: 7.8/10 (Good)
#
# ✅ Strengths:
#   - Low complexity (avg: 5.2)
#   - Good test coverage (82%)
#   - No security issues
#
# ⚠️  Areas for Improvement:
#   - Code duplication (15%)
#   - Large functions (3 functions >100 lines)
#   - Missing docstrings (23% of functions)
#
# 🎯 Top 3 Actions:
#   1. Refactor UserService.process() (complexity: 25)
#   2. Extract duplicate validation code
#   3. Add docstrings to API handlers
```

**Frequency of Use:** Daily in CI/CD, weekly for team reviews

---

## Key Features

### 1. Quality Score
```python
from code_quality import QualityAnalyzer

analyzer = QualityAnalyzer()
score = analyzer.analyze_directory("./src")

print(f"Quality Score: {score.overall}/10")
print(f"  Complexity:      {score.complexity}/10")
print(f"  Maintainability: {score.maintainability}/10")
print(f"  Duplication:     {score.duplication}/10")
print(f"  Test Coverage:   {score.coverage}%")
print(f"  Security:        {score.security}/10")
```

### 2. Complexity Analysis
```python
# Find complex code
complex_functions = analyzer.find_complex_code(threshold=10)

# OUTPUT:
# Complex Functions (cyclomatic complexity > 10):
# 1. src/payment/processor.py:145 - process_payment()
#    Complexity: 25 (VERY HIGH)
#    Recommendation: Split into smaller functions
#
# 2. src/auth/permissions.py:78 - check_access()
#    Complexity: 15 (HIGH)
#    Recommendation: Simplify conditional logic
```

### 3. Duplicate Detection
```python
# Find copy-pasted code
duplicates = analyzer.find_duplicates(min_lines=6)

# OUTPUT:
# Found 8 duplicate blocks (157 total lines)
#
# Duplicate 1: (23 lines)
#   src/api/users.py:45-67
#   src/api/admins.py:89-111
#   Recommendation: Extract to shared utility
#
# Potential savings: 134 lines of code
```

### 4. Trend Analysis
```python
# Track quality over time
trends = analyzer.analyze_trends(since="2026-01-01")

# OUTPUT:
# Quality Trends (Last 3 Months)
# ==============================
#
# Overall:     7.2 → 7.8 (+0.6) ✅ IMPROVING
# Complexity:  8.1 → 9.2 (+1.1) ✅ IMPROVING
# Duplication: 6.8 → 5.5 (-1.3) ⚠️  DEGRADING
# Coverage:    75% → 82% (+7%)  ✅ IMPROVING
#
# 📈 You're making progress! Keep it up!
# ⚠️  Watch the duplication - it's increasing
```

### 5. Quality Gates
```python
# Set quality gates for CI/CD
gate = analyzer.quality_gate(
    min_score=7.5,
    min_coverage=80,
    max_complexity=15,
    max_duplication=10
)

if not gate.passed:
    print("❌ Quality gate FAILED:")
    for violation in gate.violations:
        print(f"  - {violation}")
    sys.exit(1)  # Fail the build
```

---

## Metrics Tracked

### 1. Complexity Metrics

**Cyclomatic Complexity:**
```python
# Measures number of decision points
def example():
    if condition1:        # +1
        if condition2:    # +1
            pass
    elif condition3:      # +1
        pass
    for item in items:    # +1
        pass
# Complexity = 4
```

**Cognitive Complexity:**
```python
# Measures mental effort to understand code
# Nested conditions increase complexity more
```

**Recommendations:**
- **1-5:** Simple (Good!)
- **6-10:** Moderate (OK)
- **11-20:** Complex (Refactor soon)
- **20+:** Very Complex (Refactor now!)

### 2. Maintainability Index

Calculated from:
- Lines of code
- Cyclomatic complexity
- Number of parameters
- Comment density

**Score:**
- **85-100:** Highly maintainable ✅
- **65-84:** Moderately maintainable 🟡
- **0-64:** Hard to maintain ⚠️

### 3. Code Duplication

```python
# Detects:
# - Exact duplicates (copy-paste)
# - Near duplicates (minor changes)
# - Structural duplicates (same logic, different vars)

# Acceptable levels:
# - <5%: Excellent
# - 5-10%: Good
# - 10-20%: Needs improvement
# - >20%: Major refactoring needed
```

### 4. Test Coverage

```python
# Measures:
# - Line coverage (% of lines executed)
# - Branch coverage (% of branches tested)
# - Function coverage (% of functions tested)

# Industry standards:
# - 80%+: Good
# - 90%+: Excellent
# - <60%: Risky
```

### 5. Security Issues

```python
# Scans for:
# - SQL injection vulnerabilities
# - XSS vulnerabilities
# - Hardcoded secrets
# - Insecure dependencies
# - Known CVEs

# Severity levels:
# - Critical: Fix immediately
# - High: Fix this sprint
# - Medium: Fix soon
# - Low: Nice to fix
```

---

## Example Usage

### Analyze Project
```bash
# Full analysis
quality-analyzer analyze ./src

# OUTPUT:
# Analyzing 247 files...
#
# ==========================================
# CODE QUALITY REPORT
# ==========================================
#
# 📊 Overall Score: 8.2/10 (Very Good)
#
# Detailed Breakdown:
# ------------------------------------------
# Complexity:      9.1/10  ✅ Excellent
#   Avg complexity: 4.8
#   Functions >10:  5 (2%)
#   Functions >20:  0
#
# Maintainability: 8.5/10  ✅ Very Good
#   Avg MI score:   78
#   Hard to maintain: 12 files (5%)
#
# Duplication:     7.2/10  🟡 Good
#   Duplicate lines: 8.5%
#   Duplicate blocks: 23
#   Potential savings: 450 lines
#
# Test Coverage:   82%     ✅ Good
#   Line coverage:   82%
#   Branch coverage: 75%
#   Untested files:  8
#
# Security:        10/10   ✅ Excellent
#   Critical issues: 0
#   High issues:     0
#   Medium issues:   0
#
# 🎯 Top Recommendations:
#   1. Reduce duplication in auth module
#   2. Increase coverage to 85%
#   3. Simplify UserService.process()
```

### Track Trends
```bash
# Compare with last week
quality-analyzer compare --since "1 week ago"

# OUTPUT:
# Quality Changes (Last 7 Days)
# ==============================
#
# Overall:     8.0 → 8.2 (+0.2) ✅
# Complexity:  9.0 → 9.1 (+0.1) ✅
# Coverage:    80% → 82% (+2%)  ✅
# Duplication: 7.5 → 7.2 (-0.3) ⚠️
#
# 📈 3 improvements, 1 regression
#
# Recent Changes:
#   ✅ Refactored payment module (-5 complexity)
#   ✅ Added 47 new tests (+2% coverage)
#   ⚠️  Copy-pasted code in validators (+23 duplicates)
```

### CI/CD Integration
```yaml
# .github/workflows/quality.yml
name: Code Quality Check
on: [pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Install Quality Analyzer
        run: pip install -r requirements.txt

      - name: Run Quality Check
        run: |
          python -m quality_analyzer gate \
            --min-score 7.5 \
            --min-coverage 80 \
            --max-complexity 15

      - name: Comment on PR
        run: |
          python -m quality_analyzer report \
            --format markdown \
            | gh pr comment --body-file -
```

---

## Real-World Use Cases

### 1. Code Reviews
```bash
# Before reviewing PR
quality-analyzer diff origin/main

# Shows quality changes in PR:
# ✅ Overall quality: +0.3
# ⚠️  Added 2 complex functions
# ✅ Coverage increased by 5%
```

### 2. Refactoring Priorities
```bash
# Find worst code first
quality-analyzer worst --top 10

# OUTPUT:
# Top 10 Files Needing Refactoring:
# 1. src/legacy/processor.py (Score: 3.2/10)
# 2. src/utils/helpers.py (Score: 4.1/10)
# ...
```

### 3. Team Health Metrics
```bash
# Generate team report
quality-analyzer team-report --format pdf

# Creates PDF with:
# - Quality trends
# - Team contribution to quality
# - Technical debt estimate
# - Improvement recommendations
```

### 4. Sprint Planning
```bash
# Estimate tech debt
quality-analyzer debt

# OUTPUT:
# Technical Debt Estimate:
# ========================
# Total debt: 23 days of work
#
# By Category:
#   Complexity:   8 days
#   Duplication:  6 days
#   Missing tests: 5 days
#   Security:      2 days
#   Documentation: 2 days
#
# Recommendation: Allocate 20% of sprint to tech debt
```

---

## Project Structure

```
05_code_quality_analyzer/
├── README.md
├── requirements.txt
│
├── quality_analyzer.py      # Main analyzer
├── metrics/
│   ├── complexity.py        # Complexity metrics
│   ├── maintainability.py   # MI calculation
│   ├── duplication.py       # Duplicate detection
│   ├── coverage.py          # Test coverage
│   └── security.py          # Security scanning
│
├── analyzers/
│   ├── file_analyzer.py     # Analyze files
│   ├── project_analyzer.py  # Analyze projects
│   └── trend_analyzer.py    # Track trends
│
├── reporters/
│   ├── console_reporter.py  # Terminal output
│   ├── html_reporter.py     # HTML reports
│   ├── json_reporter.py     # JSON API
│   └── markdown_reporter.py # GitHub comments
│
└── examples/
    ├── example_01_analyze.py
    ├── example_02_trends.py
    ├── example_03_gate.py
    └── example_04_ci_cd.py
```

---

## Getting Started

```bash
cd projects/05_code_quality_analyzer
pip install -r requirements.txt

# Analyze your project
python -m quality_analyzer analyze ./src

# Set up quality gate
python -m quality_analyzer gate --min-score 7.0

# Generate HTML report
python -m quality_analyzer report --format html > quality.html
```

---

## Difficulty: ⭐⭐⭐ Advanced

**Time Estimate:** 8-10 hours

**Prerequisites:**
- Module 7 Lessons 6-10 (Code analysis)
- Understanding of code metrics
- Testing knowledge (coverage)

---

## Commercial Tool Comparison

| Feature | SonarQube | CodeClimate | This Tool |
|---------|-----------|-------------|-----------|
| Complexity | ✅ | ✅ | ✅ |
| Duplication | ✅ | ✅ | ✅ |
| Coverage | ✅ | ✅ | ✅ |
| Security | ✅ | ✅ | ✅ |
| Trends | ✅ | ✅ | ✅ |
| AI Analysis | ❌ | ❌ | ✅ |
| **Cost** | $150/dev | $99/dev | **FREE** |

---

## Success Criteria

- [x] Measures all key metrics
- [x] Provides actionable recommendations
- [x] Works in CI/CD
- [x] Tracks trends over time
- [x] Team actually uses it

---

**Build quality into your development process!** 📊

Track it, improve it, celebrate it!
