"""
Module 01 - Python Basics
Exercise 08: Classes and Object-Oriented Programming (OOP)

GLOSSARY
--------
class         : A blueprint/template for creating objects. Like a C# class.
                Defines what data (attributes) and behavior (methods) an object has.
__init__      : The constructor method. Called automatically when you create an object.
                Like C# constructor: public MyClass(int x) { this.x = x; }
                In Python: def __init__(self, x):  self.x = x
self          : Reference to the current instance. Like C# 'this'.
                Must be the FIRST parameter in every instance method.
                Python requires it explicitly; C# has it implicitly.
method        : A function defined inside a class. Called on an instance: obj.method().
                Like C# instance method.
instance      : A specific object created from a class.
                dog = Dog("Rex") -- 'dog' is an instance of Dog.
attribute     : A variable that belongs to an instance. Accessed via self.attribute_name.
                Like C# properties/fields: this.name, this.age.
inheritance   : A child class that extends a parent class.
                class Dog(Animal):  -- Dog inherits from Animal.
                Like C#: class Dog : Animal
super()       : Calls the parent class method. Like C# base.Method().
                super().__init__(...)  -- call parent's constructor.
__str__       : Special method that defines string representation.
                Like C# public override string ToString() { ... }
encapsulation : Keeping data inside a class. Access via methods, not directly.
"""

print("=" * 60)    # "=" repeated 60 times -- visual separator
print("Exercise 08: Classes and OOP")  # exercise title
print("=" * 60)    # separator
print()            # blank line


# ============================================================
#  EXERCISE 1
#  Topic: Simple class with __init__ and a method
#
#  Background:
#    In C# you write:
#      public class Circle {
#          public double Radius { get; }
#          public Circle(double radius) { Radius = radius; }
#          public double Area() { return Math.PI * Radius * Radius; }
#      }
#
#    In Python:
#      class Circle:                        # no 'public', no braces
#          def __init__(self, radius):      # constructor; 'self' = 'this'
#              self.radius = radius         # store radius as instance attribute
#          def area(self):                  # method; 'self' required
#              return 3.14159 * self.radius ** 2
#
#    Creating an instance:
#      C#:     var c = new Circle(5.0);
#      Python: c = Circle(5.0)     (no 'new' keyword in Python!)
#
#  Your Task:
#    Complete the Circle class:
#    1. __init__ stores self.radius
#    2. area() returns 3.14159 * radius**2
#    3. circumference() returns 2 * 3.14159 * radius
#
#  C# Analogy:
#    public class Circle {
#        public double Radius;
#        public Circle(double r) { Radius = r; }
#        public double Area() { return 3.14159 * Radius * Radius; }
#        public double Circumference() { return 2 * 3.14159 * Radius; }
#    }
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 1: Simple Class (Circle)")  # section title
print("-" * 50)    # separator
print()            # blank line


class Circle:
    """A class representing a circle with a given radius."""

    def __init__(self, radius):
        """
        Initialize Circle with a radius.

        Args:
            radius (float): The circle's radius.
        """
        # TODO: Store radius as an instance attribute
        # self.radius = radius   # 'self.radius' creates an attribute on this instance
        pass   # replace with your code

    def area(self):
        """
        Compute and return the area of the circle.

        Returns:
            float: Area = 3.14159 * radius**2
        """
        # TODO: Return the area
        # return 3.14159 * self.radius ** 2   # use self.radius to access the stored value
        pass   # replace with your code

    def circumference(self):
        """
        Compute and return the circumference of the circle.

        Returns:
            float: Circumference = 2 * 3.14159 * radius
        """
        # TODO: Return the circumference
        # return 2 * 3.14159 * self.radius   # 2 * pi * r
        pass   # replace with your code


c = Circle(5)                        # create a Circle instance with radius=5
print(f"  Circle radius       : {c.radius if hasattr(c, 'radius') else 'not set'}")
a = c.area()                         # call area method
circ = c.circumference()             # call circumference method
if a is not None:
    print(f"  Area                : {a:.2f}")         # :.2f means 2 decimal places
if circ is not None:
    print(f"  Circumference       : {circ:.2f}")
print()
print("  Expected: area=78.54, circumference=31.42")
print()


# ============================================================
#  EXERCISE 2
#  Topic: Class with multiple attributes and a method
#
#  Background:
#    A class can store multiple pieces of data (attributes).
#    Each attribute is set via self.attribute_name in __init__.
#
#    In C#:
#      public class Student {
#          public string Name;
#          public int Age;
#          public double GPA;
#          public Student(string name, int age, double gpa) {
#              Name=name; Age=age; GPA=gpa;
#          }
#          public string Summary() { return $"{Name} age {Age}, GPA {GPA}"; }
#      }
#
#  Your Task:
#    Complete the Student class:
#    1. __init__ takes name (str), age (int), gpa (float)
#    2. Store all three as self.name, self.age, self.gpa
#    3. is_honors() returns True if gpa >= 3.5, else False
#    4. summary() returns f"{name}, age {age}, GPA: {gpa:.1f}"
#
#  C# Analogy:
#    public class Student { ... } (see above)
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 2: Class with Multiple Attributes (Student)")  # section title
print("-" * 50)    # separator
print()            # blank line


class Student:
    """A class representing a student with name, age, and GPA."""

    def __init__(self, name, age, gpa):
        """
        Initialize Student with name, age, and GPA.

        Args:
            name (str): Student's name.
            age (int): Student's age.
            gpa (float): Student's GPA (0.0 to 4.0).
        """
        # TODO: Store all three attributes
        # self.name = name   # store the name
        # self.age  = age    # store the age
        # self.gpa  = gpa    # store the GPA
        pass   # replace with your code

    def is_honors(self):
        """
        Check if student qualifies for honors (GPA >= 3.5).

        Returns:
            bool: True if honors, False otherwise.
        """
        # TODO: Return True if self.gpa >= 3.5
        # return self.gpa >= 3.5   # comparison returns a bool
        pass   # replace with your code

    def summary(self):
        """
        Return a formatted string summary of the student.

        Returns:
            str: "Name, age X, GPA: Y.Y"
        """
        # TODO: Return a formatted f-string summary
        # return f"{self.name}, age {self.age}, GPA: {self.gpa:.1f}"
        # :.1f means format float to 1 decimal place
        pass   # replace with your code


s1 = Student("Alice", 20, 3.8)      # create a Student instance
s2 = Student("Bob", 22, 2.9)        # create another Student instance
for s in [s1, s2]:                  # loop over both students
    if hasattr(s, 'name'):          # only print if __init__ stored the attributes
        honors = s.is_honors()      # check honors status
        summ = s.summary()          # get the summary string
        print(f"  {summ}  |  Honors: {honors}")   # display all info
print()
print("  Expected:")
print("    Alice, age 20, GPA: 3.8  |  Honors: True")
print("    Bob, age 22, GPA: 2.9    |  Honors: False")
print()


# ============================================================
#  EXERCISE 3
#  Topic: Method that modifies self (mutating state)
#
#  Background:
#    Methods can MODIFY the object's own attributes via self.
#    In C#: void Deposit(double amount) { Balance += amount; }
#    Python: def deposit(self, amount):  self.balance += amount
#
#    This is like a C# void method that changes a field.
#    The method doesn't need to return anything -- it changes the object.
#
#  Your Task:
#    Complete the BankAccount class:
#    1. __init__ takes owner (str) and initial_balance (float, default 0)
#    2. deposit(amount) adds amount to self.balance
#    3. withdraw(amount) subtracts amount from self.balance
#       BUT if amount > balance, print "Insufficient funds" and do nothing
#    4. get_balance() returns the current balance
#
#  C# Analogy:
#    public class BankAccount {
#        public string Owner;
#        public double Balance;
#        public void Deposit(double a) { Balance += a; }
#        public void Withdraw(double a) {
#            if (a > Balance) Console.WriteLine("Insufficient funds");
#            else Balance -= a;
#        }
#    }
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 3: Mutating State (BankAccount)")  # section title
print("-" * 50)    # separator
print()            # blank line


class BankAccount:
    """A simple bank account that supports deposit and withdraw."""

    def __init__(self, owner, initial_balance=0):
        """
        Initialize BankAccount.

        Args:
            owner (str): Account holder's name.
            initial_balance (float): Starting balance. Default 0.
        """
        # TODO: Store owner and balance as attributes
        # self.owner   = owner             # store the owner's name
        # self.balance = initial_balance   # store the starting balance
        pass   # replace with your code

    def deposit(self, amount):
        """Add amount to balance."""
        # TODO: Add amount to self.balance
        # self.balance += amount   # balance = balance + amount
        pass   # replace with your code

    def withdraw(self, amount):
        """Subtract amount from balance if sufficient funds exist."""
        # TODO: Check funds, then subtract if OK
        # if amount > self.balance:          # not enough money
        #     print(f"  Insufficient funds (balance: {self.balance})")
        # else:
        #     self.balance -= amount         # subtract from balance
        pass   # replace with your code

    def get_balance(self):
        """Return the current balance."""
        # TODO: Return self.balance
        # return self.balance
        pass   # replace with your code


acct = BankAccount("Alice", 100)      # create account with 100 starting balance
if hasattr(acct, 'balance'):          # only test if __init__ worked
    print(f"  Initial balance  : {acct.get_balance()}")   # 100
    acct.deposit(50)                  # add 50
    print(f"  After deposit 50 : {acct.get_balance()}")   # 150
    acct.withdraw(30)                 # subtract 30
    print(f"  After withdraw 30: {acct.get_balance()}")   # 120
    acct.withdraw(200)                # try to overdraw
    print(f"  After bad withdraw: {acct.get_balance()}")  # still 120
print()
print("  Expected: 100, 150, 120, [Insufficient funds message], 120")
print()


# ============================================================
#  EXERCISE 4
#  Topic: Inheritance -- Child class extends Parent
#
#  Background:
#    Inheritance lets a child class reuse and extend a parent class.
#    In C#:   class Dog : Animal { ... }
#    Python:  class Dog(Animal):  -- put parent name in parentheses
#
#    super().__init__() calls the parent's constructor.
#    Like C#: base(args) in constructor.
#
#    Overriding a method: define a method with the SAME name in the child.
#    Python automatically uses the child version (polymorphism).
#
#  Your Task:
#    Given Animal class (provided below), create a Dog class that:
#    1. Inherits from Animal
#    2. __init__ calls super().__init__(name) and stores breed
#    3. speak() returns "{name} says: Woof!"  (override Animal.speak)
#    4. info() returns "{name} ({breed})"
#
#  C# Analogy:
#    class Dog : Animal {
#        string Breed;
#        Dog(string name, string breed) : base(name) { Breed = breed; }
#        public override string Speak() { return $"{Name} says: Woof!"; }
#        public string Info() { return $"{Name} ({Breed})"; }
#    }
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 4: Inheritance (Dog extends Animal)")  # section title
print("-" * 50)    # separator
print()            # blank line


class Animal:
    """Base class for all animals. PROVIDED -- do not change this."""

    def __init__(self, name):
        """Store the animal's name."""
        self.name = name      # all animals have a name

    def speak(self):
        """Return a generic sound."""
        return f"{self.name} says: ..."   # default -- child classes override this


class Dog(Animal):   # Dog INHERITS from Animal (put parent name in parentheses)
    """A dog that inherits from Animal and can woof."""

    def __init__(self, name, breed):
        """
        Initialize Dog with name and breed.

        Args:
            name (str): Dog's name.
            breed (str): Dog's breed.
        """
        # TODO: Call parent __init__ and store breed
        # super().__init__(name)   # call Animal.__init__ to set self.name
        # self.breed = breed       # additionally store the breed
        pass   # replace with your code

    def speak(self):
        """Override Animal.speak to return a dog-specific sound."""
        # TODO: Return "{name} says: Woof!"
        # return f"{self.name} says: Woof!"   # self.name comes from the parent class
        pass   # replace with your code

    def info(self):
        """Return a formatted string with name and breed."""
        # TODO: Return "{name} ({breed})"
        # return f"{self.name} ({self.breed})"
        pass   # replace with your code


generic = Animal("Cat")          # create a base Animal
dog = Dog("Rex", "Labrador")     # create a Dog (uses our new class)
print(f"  Animal.speak()  : {generic.speak()}")    # Cat says: ...
if hasattr(dog, 'breed'):        # only test Dog methods if __init__ worked
    print(f"  Dog.speak()     : {dog.speak()}")    # Rex says: Woof!
    print(f"  Dog.info()      : {dog.info()}")     # Rex (Labrador)
print()
print("  Expected:")
print("    Animal: 'Cat says: ...'")
print("    Dog speak: 'Rex says: Woof!'")
print("    Dog info:  'Rex (Labrador)'")
print()


# ============================================================
#  EXERCISE 5
#  Topic: __str__ method (like C# ToString())
#
#  Background:
#    __str__ is a "magic method" (dunder method) that Python calls
#    when you convert an object to a string (e.g., print(obj) or str(obj)).
#    Like C#: public override string ToString() { return ...; }
#
#    Without __str__:  print(obj)  ->  <__main__.Book object at 0x...>
#    With __str__:     print(obj)  ->  your custom string
#
#    Pattern:
#      def __str__(self):
#          return f"Book: {self.title} by {self.author}"
#
#  Your Task:
#    Complete the Book class:
#    1. __init__ takes title (str), author (str), pages (int)
#    2. __str__ returns "'{title}' by {author} ({pages} pages)"
#    3. is_long() returns True if pages > 300
#
#  C# Analogy:
#    public class Book {
#        ...
#        public override string ToString() {
#            return $"'{Title}' by {Author} ({Pages} pages)";
#        }
#    }
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 5: __str__ Method (ToString Equivalent)")  # section title
print("-" * 50)    # separator
print()            # blank line


class Book:
    """A class representing a book, with a custom string representation."""

    def __init__(self, title, author, pages):
        """
        Initialize Book with title, author, and page count.

        Args:
            title (str): Book title.
            author (str): Author name.
            pages (int): Number of pages.
        """
        # TODO: Store all three attributes
        # self.title  = title    # store the title
        # self.author = author   # store the author
        # self.pages  = pages    # store the page count
        pass   # replace with your code

    def __str__(self):
        """
        Return a human-readable string representation of the book.

        Returns:
            str: "'{title}' by {author} ({pages} pages)"
        """
        # TODO: Return a formatted f-string
        # return f"'{self.title}' by {self.author} ({self.pages} pages)"
        pass   # replace with your code

    def is_long(self):
        """Return True if this book has more than 300 pages."""
        # TODO: Return True if pages > 300
        # return self.pages > 300   # comparison returns bool directly
        pass   # replace with your code


b1 = Book("Clean Code", "Robert Martin", 431)     # create a long book
b2 = Book("The Go-Giver", "Bob Burg", 127)        # create a short book
for b in [b1, b2]:                                # loop over both books
    if hasattr(b, 'title'):                       # only test if __init__ worked
        print(f"  str(book)  : {str(b)}")         # calls __str__ via str()
        print(f"  is_long()  : {b.is_long()}")    # check length
        print()
print("  Expected:")
print("    'Clean Code' by Robert Martin (431 pages) | is_long: True")
print("    'The Go-Giver' by Bob Burg (127 pages)    | is_long: False")
print()

print("=" * 60)                       # closing separator
print("All exercises complete!")      # completion message
print("=" * 60)                       # closing separator
