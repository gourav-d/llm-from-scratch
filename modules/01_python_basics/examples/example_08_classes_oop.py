"""
Example 8: Classes and Object-Oriented Programming for .NET Developers
This file demonstrates Python OOP with detailed comments and C# comparisons.
"""

# ============================================
# SECTION 1: BASIC CLASS DEFINITION
# ============================================

print("=== SECTION 1: BASIC CLASS DEFINITION ===\n")

# C#:
# public class Dog {
#     private string name;
#     private int age;
#
#     public Dog(string name, int age) {
#         this.name = name;
#         this.age = age;
#     }
#
#     public void Bark() {
#         Console.WriteLine($"{name} says: Woof!");
#     }
# }

# Python:
class Dog:
    """A simple Dog class demonstrating basic OOP."""

    # Constructor (like C# constructor but named __init__)
    # self = this (in C#)
    def __init__(self, name, age):
        """Initialize a new Dog instance."""
        self.name = name  # Instance variable
        self.age = age    # Instance variable

    # Instance method
    def bark(self):
        """Make the dog bark."""
        print(f"{self.name} says: Woof!")

    def get_info(self):
        """Get dog information."""
        return f"{self.name} is {self.age} years old"

# Create instances (no 'new' keyword!)
# C#: var dog1 = new Dog("Buddy", 3);
# Python: dog1 = Dog("Buddy", 3)
dog1 = Dog("Buddy", 3)
dog2 = Dog("Max", 5)

# Call methods
dog1.bark()
dog2.bark()

# Access attributes (public by default!)
print(f"Dog 1 name: {dog1.name}")
print(f"Dog 2 age: {dog2.age}")

# Call info method
print(dog1.get_info())
print(dog2.get_info())

print()

# ============================================
# SECTION 2: __init__ METHOD (CONSTRUCTOR)
# ============================================

print("=== SECTION 2: __init__ METHOD (CONSTRUCTOR) ===\n")

# __init__ is the constructor
# - Always takes 'self' as first parameter
# - Called automatically when creating instance
# - Can have default parameters

class Person:
    """A Person class with default parameters."""

    def __init__(self, name, age=18, city="Unknown"):
        """Initialize a Person."""
        self.name = name
        self.age = age
        self.city = city

# Create with all parameters
person1 = Person("Alice", 30, "NYC")
print(f"{person1.name}, {person1.age}, {person1.city}")

# Create with defaults
person2 = Person("Bob")
print(f"{person2.name}, {person2.age}, {person2.city}")

# Create with some defaults
person3 = Person("Charlie", 25)
print(f"{person3.name}, {person3.age}, {person3.city}")

print()

# ============================================
# SECTION 3: INSTANCE VARIABLES AND METHODS
# ============================================

print("=== SECTION 3: INSTANCE VARIABLES AND METHODS ===\n")

# Instance variables: Unique to each instance
# Instance methods: Work with instance data

class BankAccount:
    """A simple bank account class."""

    def __init__(self, owner, balance=0):
        """Initialize a bank account."""
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        """Deposit money into account."""
        self.balance += amount
        print(f"Deposited ${amount}. New balance: ${self.balance}")

    def withdraw(self, amount):
        """Withdraw money from account."""
        if amount > self.balance:
            print("Insufficient funds!")
        else:
            self.balance -= amount
            print(f"Withdrew ${amount}. New balance: ${self.balance}")

    def get_balance(self):
        """Get current balance."""
        return self.balance

# Create account
account = BankAccount("Alice", 1000)
print(f"Owner: {account.owner}")
print(f"Initial balance: ${account.get_balance()}")

# Perform operations
account.deposit(500)
account.withdraw(200)
account.withdraw(2000)  # Insufficient funds

print()

# ============================================
# SECTION 4: CLASS VARIABLES
# ============================================

print("=== SECTION 4: CLASS VARIABLES ===\n")

# Class variables: Shared by ALL instances
# C#: public static string Species = "Canis familiaris";
# Python: species = "Canis familiaris"  (outside __init__)

class Cat:
    """A Cat class with class variables."""

    # Class variable (shared by all instances)
    species = "Felis catus"
    count = 0  # Track number of cats created

    def __init__(self, name, breed):
        # Instance variables (unique to each instance)
        self.name = name
        self.breed = breed
        # Increment class variable
        Cat.count += 1

# Create cats
cat1 = Cat("Whiskers", "Persian")
cat2 = Cat("Shadow", "Siamese")
cat3 = Cat("Luna", "Maine Coon")

# Access class variable (same for all)
print(f"cat1.species: {cat1.species}")
print(f"cat2.species: {cat2.species}")
print(f"Cat.species: {Cat.species}")

# Access instance variables (unique)
print(f"cat1.name: {cat1.name}")
print(f"cat2.name: {cat2.name}")

# Check count
print(f"Total cats created: {Cat.count}")

print()

# ============================================
# SECTION 5: CLASS METHODS AND STATIC METHODS
# ============================================

print("=== SECTION 5: CLASS METHODS AND STATIC METHODS ===\n")

class MathHelper:
    """A class with class and static methods."""

    pi = 3.14159

    @classmethod
    def circle_area(cls, radius):
        """Calculate circle area (class method)."""
        # cls refers to the class (like self for instance)
        return cls.pi * radius ** 2

    @staticmethod
    def add(a, b):
        """Add two numbers (static method)."""
        # No automatic first parameter (no self, no cls)
        return a + b

    @staticmethod
    def is_even(n):
        """Check if number is even."""
        return n % 2 == 0

# Call class method (can access class variables)
# C#: MathHelper.CircleArea(5)
# Python: MathHelper.circle_area(5)
area = MathHelper.circle_area(5)
print(f"Circle area (r=5): {area:.2f}")

# Call static method (no class access needed)
result = MathHelper.add(10, 20)
print(f"10 + 20 = {result}")

print(f"Is 8 even? {MathHelper.is_even(8)}")
print(f"Is 7 even? {MathHelper.is_even(7)}")

print()

# ============================================
# SECTION 6: __str__ AND __repr__ METHODS
# ============================================

print("=== SECTION 6: __str__ AND __repr__ METHODS ===\n")

# __str__: Human-readable string (like C# ToString())
# __repr__: Developer-friendly representation

class Book:
    """A Book class with string representation."""

    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages

    def __str__(self):
        """String representation for users."""
        # C#: public override string ToString()
        return f'"{self.title}" by {self.author} ({self.pages} pages)'

    def __repr__(self):
        """String representation for developers."""
        return f"Book('{self.title}', '{self.author}', {self.pages})"

# Create book
book = Book("1984", "George Orwell", 328)

# print() uses __str__
print(f"Using str(): {str(book)}")
print(f"Using print: {book}")

# repr() uses __repr__
print(f"Using repr(): {repr(book)}")

print()

# ============================================
# SECTION 7: PROPERTIES (@property DECORATOR)
# ============================================

print("=== SECTION 7: PROPERTIES (@property DECORATOR) ===\n")

# @property: Create getters and setters
# C#: public string Name { get; set; }
# Python: @property

class Employee:
    """An Employee class with properties."""

    def __init__(self, name, salary):
        self._name = name      # Convention: _ for "protected"
        self._salary = salary

    @property
    def name(self):
        """Getter for name."""
        return self._name

    @property
    def salary(self):
        """Getter for salary."""
        return self._salary

    @salary.setter
    def salary(self, value):
        """Setter for salary with validation."""
        if value < 0:
            raise ValueError("Salary cannot be negative")
        self._salary = value

    @property
    def annual_salary(self):
        """Calculated property (no setter)."""
        return self._salary * 12

# Create employee
emp = Employee("Alice", 5000)

# Use like attributes (no parentheses!)
print(f"Name: {emp.name}")
print(f"Monthly salary: ${emp.salary}")
print(f"Annual salary: ${emp.annual_salary}")

# Set salary (calls setter)
emp.salary = 5500
print(f"New monthly salary: ${emp.salary}")
print(f"New annual salary: ${emp.annual_salary}")

# Try invalid value
try:
    emp.salary = -1000  # Raises ValueError
except ValueError as e:
    print(f"Error: {e}")

print()

# ============================================
# SECTION 8: INHERITANCE
# ============================================

print("=== SECTION 8: INHERITANCE ===\n")

# C#: public class Dog : Animal
# Python: class Dog(Animal):

# Parent class (base class)
class Animal:
    """Base Animal class."""

    def __init__(self, name, species):
        self.name = name
        self.species = species

    def make_sound(self):
        """Make a sound."""
        print(f"{self.name} makes a sound")

    def info(self):
        """Get animal info."""
        print(f"{self.name} is a {self.species}")

# Child class (derived class)
class DogInherit(Animal):
    """Dog class inheriting from Animal."""

    def make_sound(self):
        """Override make_sound method."""
        print(f"{self.name} says: Woof!")

class CatInherit(Animal):
    """Cat class inheriting from Animal."""

    def make_sound(self):
        """Override make_sound method."""
        print(f"{self.name} says: Meow!")

# Create instances
dog = DogInherit("Buddy", "Dog")
cat = CatInherit("Whiskers", "Cat")

# Call methods
dog.make_sound()  # Uses overridden method
cat.make_sound()  # Uses overridden method

dog.info()  # Uses inherited method
cat.info()  # Uses inherited method

print()

# ============================================
# SECTION 9: USING super()
# ============================================

print("=== SECTION 9: USING super() ===\n")

# super(): Call parent class methods
# C#: base.Method()
# Python: super().method()

class Vehicle:
    """Base Vehicle class."""

    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def info(self):
        """Display vehicle info."""
        print(f"{self.brand} {self.model}")

class Car(Vehicle):
    """Car class inheriting from Vehicle."""

    def __init__(self, brand, model, doors):
        # Call parent constructor
        # C#: base(brand, model)
        # Python: super().__init__(brand, model)
        super().__init__(brand, model)
        self.doors = doors

    def info(self):
        """Display car info (extended)."""
        # Call parent method
        super().info()
        print(f"Doors: {self.doors}")

# Create car
car = Car("Toyota", "Camry", 4)
car.info()

print()

# ============================================
# SECTION 10: ENCAPSULATION (PRIVATE VARIABLES)
# ============================================

print("=== SECTION 10: ENCAPSULATION ===\n")

# Python doesn't have true private variables
# Conventions:
# - _variable: Protected (don't access from outside)
# - __variable: Private (name mangling)

class BankAccountSecure:
    """Bank account with encapsulation."""

    def __init__(self, owner, balance):
        self._owner = owner          # Protected
        self.__balance = balance      # Private (name mangling)

    def deposit(self, amount):
        """Deposit money."""
        self.__balance += amount

    def withdraw(self, amount):
        """Withdraw money."""
        if amount <= self.__balance:
            self.__balance -= amount
        else:
            print("Insufficient funds!")

    def get_balance(self):
        """Get balance (public accessor)."""
        return self.__balance

account = BankAccountSecure("Alice", 1000)

# Can still access _owner (but shouldn't!)
print(f"Owner (protected): {account._owner}")

# Cannot access __balance directly
try:
    print(account.__balance)  # AttributeError
except AttributeError:
    print("Cannot access __balance directly")

# Access through method
print(f"Balance (via method): ${account.get_balance()}")

# Name mangling: can still access (but very discouraged!)
print(f"Balance (name mangling): ${account._BankAccountSecure__balance}")

print()

# ============================================
# SECTION 11: SPECIAL METHODS (MAGIC METHODS)
# ============================================

print("=== SECTION 11: SPECIAL METHODS (MAGIC METHODS) ===\n")

class Point:
    """A 2D point with magic methods."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        """String representation."""
        return f"Point({self.x}, {self.y})"

    def __add__(self, other):
        """Add two points."""
        return Point(self.x + other.x, self.y + other.y)

    def __eq__(self, other):
        """Check equality."""
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        """Less than comparison (by distance from origin)."""
        return (self.x**2 + self.y**2) < (other.x**2 + other.y**2)

# Create points
p1 = Point(3, 4)
p2 = Point(1, 2)
p3 = Point(3, 4)

print(f"p1: {p1}")
print(f"p2: {p2}")

# Use + operator (calls __add__)
p4 = p1 + p2
print(f"p1 + p2: {p4}")

# Use == operator (calls __eq__)
print(f"p1 == p2: {p1 == p2}")
print(f"p1 == p3: {p1 == p3}")

# Use < operator (calls __lt__)
print(f"p2 < p1: {p2 < p1}")

print()

# ============================================
# SECTION 12: PRACTICAL EXAMPLE - COMPLETE CLASS
# ============================================

print("=== SECTION 12: PRACTICAL EXAMPLE ===\n")

class Rectangle:
    """A complete Rectangle class with all features."""

    # Class variable
    count = 0

    def __init__(self, width, height):
        """Initialize rectangle."""
        self.width = width
        self.height = height
        Rectangle.count += 1

    @property
    def area(self):
        """Calculate area (computed property)."""
        return self.width * self.height

    @property
    def perimeter(self):
        """Calculate perimeter (computed property)."""
        return 2 * (self.width + self.height)

    def scale(self, factor):
        """Scale the rectangle."""
        self.width *= factor
        self.height *= factor

    def __str__(self):
        """String representation."""
        return f"Rectangle({self.width}x{self.height})"

    def __eq__(self, other):
        """Check equality (by area)."""
        return self.area == other.area

    @classmethod
    def square(cls, side):
        """Create a square (class method)."""
        return cls(side, side)

# Create rectangles
rect1 = Rectangle(10, 5)
rect2 = Rectangle(5, 10)
square = Rectangle.square(7)

print(f"Rectangle 1: {rect1}")
print(f"  Area: {rect1.area}")
print(f"  Perimeter: {rect1.perimeter}")

print(f"\nRectangle 2: {rect2}")
print(f"  Area: {rect2.area}")

print(f"\nSquare: {square}")
print(f"  Area: {square.area}")

# Scale
rect1.scale(2)
print(f"\nAfter scaling rect1 by 2: {rect1}")
print(f"  New area: {rect1.area}")

# Compare
print(f"\nrect1 == rect2 (by area): {rect1 == rect2}")

# Check count
print(f"\nTotal rectangles created: {Rectangle.count}")

print()

# ============================================
# SECTION 13: MULTIPLE INHERITANCE
# ============================================

print("=== SECTION 13: MULTIPLE INHERITANCE ===\n")

# Python supports multiple inheritance (C# uses interfaces)

class Flyable:
    """Mixin for flying ability."""

    def fly(self):
        print(f"{self.name} is flying!")

class Swimmable:
    """Mixin for swimming ability."""

    def swim(self):
        print(f"{self.name} is swimming!")

class Duck(Animal, Flyable, Swimmable):
    """Duck can fly and swim."""

    def __init__(self, name):
        super().__init__(name, "Duck")

# Create duck
duck = Duck("Donald")
duck.info()      # From Animal
duck.fly()       # From Flyable
duck.swim()      # From Swimmable

print()

# ============================================
# SUMMARY
# ============================================

print("=== SUMMARY ===\n")

summary = """
Classes and OOP for .NET Developers:

CLASS DEFINITION:
  C#:
    public class Dog {
        private string name;
        public Dog(string name) { this.name = name; }
    }

  Python:
    class Dog:
        def __init__(self, name):
            self.name = name

KEY DIFFERENCES:
  - No access modifiers (public, private, protected)
  - Use 'self' instead of 'this'
  - Use '__init__' instead of constructor name
  - No type declarations
  - Methods use snake_case

CONSTRUCTOR:
  C#: public Dog(string name)
  Python: def __init__(self, name)
  - Always takes 'self' as first parameter
  - Can have default parameters

CREATING INSTANCES:
  C#: var dog = new Dog("Buddy");
  Python: dog = Dog("Buddy")  # No 'new' keyword!

INSTANCE VARIABLES:
  self.name = name  # Unique to each instance

CLASS VARIABLES:
  species = "Canis familiaris"  # Shared by all instances
  C# equivalent: public static string Species

METHODS:
  Instance method: def bark(self):
  Class method: @classmethod def method(cls):
  Static method: @staticmethod def method():

PROPERTIES:
  C#: public string Name { get; set; }
  Python:
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

INHERITANCE:
  C#: public class Dog : Animal
  Python: class Dog(Animal):

  Call parent: super().__init__() or super().method()

SPECIAL METHODS:
  __init__: Constructor
  __str__: ToString() equivalent
  __repr__: Developer representation
  __len__: len() support
  __eq__: == operator
  __lt__: < operator
  __add__: + operator

ENCAPSULATION:
  _variable: Protected (convention, not enforced)
  __variable: Private (name mangling)
  No true private in Python!

C# → Python Quick Reference:
  public class Dog              → class Dog:
  public Dog(...)               → def __init__(self, ...):
  this                          → self
  new Dog()                     → Dog()
  public static                 → @classmethod or @staticmethod
  override ToString()           → def __str__(self):
  class Dog : Animal            → class Dog(Animal):
  base.Method()                 → super().method()
  public string Name { get; }   → @property
"""

print(summary)

print("="*60)
print("Next: example_09_file_io.py - Learn file operations!")
print("="*60)
