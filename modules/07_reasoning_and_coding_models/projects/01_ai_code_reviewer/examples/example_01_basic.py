"""
Example 1: Basic Code Review

This example shows how to use the AI Code Reviewer on simple code.

Run: python examples/example_01_basic.py
"""

import sys
sys.path.append('..')  # Add parent directory to path

from ai_code_reviewer import CodeReviewer

def main():
    print("="*80)
    print("AI CODE REVIEWER - Example 1: Basic Usage")
    print("="*80)
    print()

    # Example 1: Code with security issue
    print("Example 1: Reviewing code with SQL injection vulnerability")
    print("-"*80)

    vulnerable_code = """
def get_user_by_email(email):
    query = "SELECT * FROM users WHERE email = '" + email + "'"
    result = database.execute(query)
    return result
"""

    reviewer = CodeReviewer()
    issues = reviewer.review_code(vulnerable_code, filename="user_service.py")

    print(f"\nCode to review:")
    print(vulnerable_code)

    print(f"\nFound {len(issues)} issue(s)")
    print(reviewer.format_report())

    print("\n" + "="*80)
    print()

    # Example 2: Code with bug
    print("Example 2: Reviewing code with potential bug")
    print("-"*80)

    buggy_code = """
def get_last_element(items):
    try:
        return items[len(items)]
    except:
        pass
"""

    reviewer2 = CodeReviewer()
    issues2 = reviewer2.review_code(buggy_code, filename="utils.py")

    print(f"\nCode to review:")
    print(buggy_code)

    print(f"\nFound {len(issues2)} issue(s)")
    print(reviewer2.format_report())

    print("\n" + "="*80)
    print()

    # Example 3: Good code
    print("Example 3: Reviewing good code")
    print("-"*80)

    good_code = """
def get_user_by_email(email: str) -> Optional[User]:
    '''Safely retrieve user by email'''
    if not email:
        raise ValueError("Email cannot be empty")

    # Use parameterized query
    query = "SELECT * FROM users WHERE email = ?"
    result = database.execute(query, [email])

    return result.first() if result else None
"""

    reviewer3 = CodeReviewer()
    issues3 = reviewer3.review_code(good_code, filename="user_service.py")

    print(f"\nCode to review:")
    print(good_code)

    print(f"\nFound {len(issues3)} issue(s)")
    print(reviewer3.format_report())


if __name__ == "__main__":
    main()
