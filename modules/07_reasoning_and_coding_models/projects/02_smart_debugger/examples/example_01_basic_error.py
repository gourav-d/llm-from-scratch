"""
Example 1: Basic Error Analysis

This example shows how to use the Smart Debugger to analyze common errors.

Run: python examples/example_01_basic_error.py
"""

import sys
sys.path.append('..')  # Add parent directory to path

from smart_debugger import SmartDebugger

def main():
    print("="*80)
    print("SMART DEBUGGER - Example 1: Basic Error Analysis")
    print("="*80)
    print()

    debugger = SmartDebugger()

    # Example 1: IndexError
    print("Example 1: IndexError - Off by one")
    print("-"*80)

    code1 = """
def get_last_item(items):
    index = len(items)  # Bug: Should be len(items) - 1
    return items[index]
"""
    error1 = "IndexError: list index out of range"

    analysis1 = debugger.analyze_error(error1, code1)
    print(analysis1)

    print("\n" + "="*80)
    print()

    # Example 2: ZeroDivisionError
    print("Example 2: ZeroDivisionError - No validation")
    print("-"*80)

    code2 = """
def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)  # Bug: Crashes if numbers is empty
"""
    error2 = "ZeroDivisionError: division by zero"

    analysis2 = debugger.analyze_error(error2, code2)
    print(analysis2)

    print("\n" + "="*80)
    print()

    # Example 3: AttributeError
    print("Example 3: AttributeError - None check missing")
    print("-"*80)

    code3 = """
def get_username(user):
    return user.name  # Bug: user might be None
"""
    error3 = "AttributeError: 'NoneType' object has no attribute 'name'"

    analysis3 = debugger.analyze_error(error3, code3)
    print(analysis3)

    print("\n" + "="*80)
    print()

    # Example 4: KeyError
    print("Example 4: KeyError - Missing key check")
    print("-"*80)

    code4 = """
def get_config_value(config, key):
    return config[key]  # Bug: Key might not exist
"""
    error4 = "KeyError: 'database_url'"

    analysis4 = debugger.analyze_error(error4, code4)
    print(analysis4)


if __name__ == "__main__":
    main()
