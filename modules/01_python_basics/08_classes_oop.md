# Lesson 1.8: Classes and Object-Oriented Programming

## 🎯 What You'll Learn
- Creating classes in Python
- Constructors (__init__)
- Instance variables and methods
- Class variables and methods
- Inheritance
- Python OOP vs C# OOP

---

## Basic Class Definition

### C# vs Python

**C#:**
```csharp
public class Dog
{
    private string name;
    private int age;

    public Dog(string name, int age)
    {
        this.name = name;
        this.age = age;
    }

    public void Bark()
    {
        Console.WriteLine($"{name} says: Woof!");
    }
}
```

**Python:**
```python
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def bark(self):
        print(f"{self.name} says: Woof!")
```

**Key Differences:**
- No access modifiers (public, private) needed
- Use `def __init__` instead of constructor name
- Use `self` instead of `this`
- No type declarations
- Methods use snake_case (not PascalCase)

---

## Creating a Class

```python
class Person:
    # Constructor (called when creating instance)
    def __init__(self, name, age):
        self.name = name  # Instance variable
        self.age = age    # Instance variable

    # Instance method
    def introduce(self):
        print(f"Hi, I'm {self.name} and I'm {self.age} years old")

# Create instances (objects)
person1 = Person("Alice", 30)
person2 = Person("Bob", 25)

# Call methods
person1.introduce()  # Hi, I'm Alice and I'm 30 years old
person2.introduce()  # Hi, I'm Bob and I'm 25 years old

# Access attributes
print(person1.name)  # Alice
print(person2.age)   # 25
```

**Line-by-line explanation:**
- `class Person:` → Define class (use PascalCase for class names)
- `def __init__(self, ...)` → Constructor (special method)
- `self` → Reference to current instance (like C#'s `this`)
- `self.name = name` → Create instance variable
- `person1 = Person("Alice", 30)` → Create instance (no `new` keyword!)

---

## The `__init__` Method (Constructor)

```python
class Car:
    def __init__(self, brand, model, year):
        self.brand = brand
        self.model = model
        self.year = year
        self.mileage = 0  # Default value

car = Car("Toyota", "Camry", 2020)
print(car.brand)    # Toyota
print(car.mileage)  # 0
```

**Explanation:**
- `__init__` is the constructor (double underscores!)
- `self` is always the first parameter
- You can set default values

**With default parameters:**

```python
class Person:
    def __init__(self, name, age=18, city="Unknown"):
        self.name = name
        self.age = age
        self.city = city

p1 = Person("Alice")
print(p1.age)   # 18 (default)

p2 = Person("Bob", 25, "NYC")
print(p2.city)  # NYC
```

---

## Instance Methods

Methods that work with instance data

```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        print(f"Deposited ${amount}. New balance: ${self.balance}")

    def withdraw(self, amount):
        if amount > self.balance:
            print("Insufficient funds!")
        else:
            self.balance -= amount
            print(f"Withdrew ${amount}. New balance: ${self.balance}")

    def get_balance(self):
        return self.balance

# Use the class
account = BankAccount("Alice", 1000)
account.deposit(500)    # Deposited $500. New balance: $1500
account.withdraw(200)   # Withdrew $200. New balance: $1300
print(account.get_balance())  # 1300
```

**Explanation:**
- All instance methods have `self` as first parameter
- Use `self.variable` to access instance variables
- Methods can modify instance state

---

## Class Variables vs Instance Variables

```python
class Dog:
    # Class variable (shared by all instances)
    species = "Canis familiaris"
    count = 0

    def __init__(self, name, breed):
        # Instance variables (unique to each instance)
        self.name = name
        self.breed = breed
        Dog.count += 1  # Increment class variable

# Create instances
dog1 = Dog("Buddy", "Golden Retriever")
dog2 = Dog("Max", "German Shepherd")

# Access class variable
print(dog1.species)  # Canis familiaris
print(dog2.species)  # Canis familiaris
print(Dog.species)   # Canis familiaris

# Access instance variables
print(dog1.name)     # Buddy
print(dog2.name)     # Max

# Check count
print(Dog.count)     # 2 (two dogs created)
```

**Explanation:**
- **Class variables** → Defined inside class but outside methods
- **Instance variables** → Defined with `self.` in `__init__`
- Class variables are shared, instance variables are unique

**C# Comparison:**
```csharp
// C#
public class Dog
{
    public static string Species = "Canis familiaris";  // Class variable (static)
    public string Name { get; set; }                     // Instance variable

    public Dog(string name)
    {
        Name = name;
    }
}

// Python
class Dog:
    species = "Canis familiaris"  # Class variable

    def __init__(self, name):
        self.name = name  # Instance variable
```

---

## Class Methods and Static Methods

```python
class Math:
    pi = 3.14159

    @classmethod
    def circle_area(cls, radius):
        # cls refers to the class (like self for instance)
        return cls.pi * radius ** 2

    @staticmethod
    def add(a, b):
        # Static method - doesn't need class or instance
        return a + b

# Call class method
area = Math.circle_area(5)
print(area)  # 78.53975

# Call static method
result = Math.add(10, 20)
print(result)  # 30
```

**Explanation:**
- `@classmethod` → Receives class as first parameter (`cls`)
- `@staticmethod` → No automatic first parameter
- Can be called on class without creating instance

**C# Comparison:**
```csharp
// C#
public class Math
{
    public static double Pi = 3.14159;

    public static double CircleArea(double radius)  // Static method
    {
        return Pi * radius * radius;
    }
}

// Python
class Math:
    pi = 3.14159

    @staticmethod
    def circle_area(radius):
        return Math.pi * radius ** 2
```

---

## Special Methods (Magic Methods)

Methods with double underscores (`__method__`)

### `__str__` - String Representation

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"Person(name={self.name}, age={self.age})"

person = Person("Alice", 30)
print(person)  # Person(name=Alice, age=30)
```

**C# Comparison:** Like `ToString()` override

### `__repr__` - Developer Representation

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __repr__(self):
        return f"Person('{self.name}', {self.age})"

person = Person("Alice", 30)
print(repr(person))  # Person('Alice', 30)
```

### `__len__` - Length

```python
class Playlist:
    def __init__(self):
        self.songs = []

    def add_song(self, song):
        self.songs.append(song)

    def __len__(self):
        return len(self.songs)

playlist = Playlist()
playlist.add_song("Song 1")
playlist.add_song("Song 2")
print(len(playlist))  # 2
```

### Comparison Methods

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __eq__(self, other):
        return self.age == other.age

    def __lt__(self, other):
        return self.age < other.age

p1 = Person("Alice", 30)
p2 = Person("Bob", 25)

print(p1 == p2)  # False
print(p1 > p2)   # True (30 > 25)
```

---

## Inheritance

```python
# Parent class
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        print(f"{self.name} makes a sound")

# Child class
class Dog(Animal):
    def speak(self):
        print(f"{self.name} says: Woof!")

class Cat(Animal):
    def speak(self):
        print(f"{self.name} says: Meow!")

# Create instances
dog = Dog("Buddy")
cat = Cat("Whiskers")

dog.speak()  # Buddy says: Woof!
cat.speak()  # Whiskers says: Meow!
```

**Explanation:**
- `class Dog(Animal):` → Dog inherits from Animal
- Child class can override parent methods
- No `override` keyword needed

### Using `super()`

```python
class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species

    def info(self):
        print(f"{self.name} is a {self.species}")

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Dog")  # Call parent constructor
        self.breed = breed

    def info(self):
        super().info()  # Call parent method
        print(f"Breed: {self.breed}")

dog = Dog("Buddy", "Golden Retriever")
dog.info()
# Output:
# Buddy is a Dog
# Breed: Golden Retriever
```

**C# Comparison:**
```csharp
// C#
public class Dog : Animal
{
    public Dog(string name, string breed) : base(name, "Dog")
    {
        Breed = breed;
    }
}

// Python
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Dog")
        self.breed = breed
```

---

## Properties (Getters and Setters)

### Using @property Decorator

```python
class Person:
    def __init__(self, name, age):
        self._name = name  # Convention: _ for "private"
        self._age = age

    @property
    def name(self):
        """Getter for name"""
        return self._name

    @property
    def age(self):
        """Getter for age"""
        return self._age

    @age.setter
    def age(self, value):
        """Setter for age with validation"""
        if value < 0:
            raise ValueError("Age cannot be negative")
        self._age = value

# Use like attributes
person = Person("Alice", 30)
print(person.name)  # Alice (calls getter)
print(person.age)   # 30 (calls getter)

person.age = 31     # Calls setter
print(person.age)   # 31

person.age = -5     # Error! (validation in setter)
```

**Explanation:**
- `@property` → Creates getter
- `@age.setter` → Creates setter
- Use like attributes (no parentheses!)
- Allows validation and logic

---

## Encapsulation (Private Variables)

Python doesn't have true private variables, but conventions:

```python
class BankAccount:
    def __init__(self, balance):
        self._balance = balance      # Protected (convention)
        self.__secret = "123"         # Name mangling (harder to access)

    def get_balance(self):
        return self._balance

account = BankAccount(1000)

# Can still access _balance (but shouldn't!)
print(account._balance)  # 1000 (not recommended)

# Harder to access __secret
# print(account.__secret)  # Error!
print(account._BankAccount__secret)  # 123 (name mangling)
```

**Convention:**
- `_variable` → Protected (don't access from outside)
- `__variable` → Private (name mangling, harder to access)
- No true private in Python!

---

## Practical Example: Complete Class

```python
class Rectangle:
    """A class representing a rectangle"""

    count = 0  # Class variable

    def __init__(self, width, height):
        self.width = width
        self.height = height
        Rectangle.count += 1

    @property
    def area(self):
        """Calculate and return area"""
        return self.width * self.height

    @property
    def perimeter(self):
        """Calculate and return perimeter"""
        return 2 * (self.width + self.height)

    def scale(self, factor):
        """Scale the rectangle by a factor"""
        self.width *= factor
        self.height *= factor

    def __str__(self):
        return f"Rectangle({self.width}x{self.height})"

    def __eq__(self, other):
        return self.area == other.area

# Use the class
rect1 = Rectangle(10, 5)
rect2 = Rectangle(5, 10)

print(rect1)              # Rectangle(10x5)
print(f"Area: {rect1.area}")         # Area: 50
print(f"Perimeter: {rect1.perimeter}")  # Perimeter: 30

rect1.scale(2)
print(rect1)              # Rectangle(20x10)

print(rect1 == rect2)     # True (same area)
print(Rectangle.count)    # 2 (two rectangles created)
```

---

## 💡 Key Takeaways

1. **Class definition** → `class ClassName:`
2. **Constructor** → `def __init__(self, ...)`
3. **self** → Like C#'s `this`
4. **No `new` keyword** → `obj = ClassName()`
5. **Inheritance** → `class Child(Parent):`
6. **super()** → Call parent method
7. **@property** → Create getters/setters
8. **Magic methods** → `__str__`, `__len__`, `__eq__`, etc.
9. **No true private** → Use `_var` convention

---

## ✏️ Practice Exercise

Create `classes_practice.py`:

```python
# Create a simple class
class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages

    def __str__(self):
        return f'"{self.title}" by {self.author} ({self.pages} pages)'

book = Book("1984", "George Orwell", 328)
print(book)

# Create a class with methods
class Circle:
    pi = 3.14159

    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return Circle.pi * self.radius ** 2

    def circumference(self):
        return 2 * Circle.pi * self.radius

circle = Circle(5)
print(f"Area: {circle.area():.2f}")
print(f"Circumference: {circle.circumference():.2f}")

# Inheritance
class Vehicle:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def info(self):
        print(f"{self.brand} {self.model}")

class Car(Vehicle):
    def __init__(self, brand, model, doors):
        super().__init__(brand, model)
        self.doors = doors

    def info(self):
        super().info()
        print(f"Doors: {self.doors}")

car = Car("Toyota", "Camry", 4)
car.info()
```

**Run it:** `python classes_practice.py`

---

## 🤔 Quick Quiz

1. What's the Python equivalent of C#'s `this`?
   <details>
   <summary>Answer</summary>

   `self`
   </details>

2. What method is the constructor in Python?
   <details>
   <summary>Answer</summary>

   `__init__`
   </details>

3. How do you inherit from a parent class?
   <details>
   <summary>Answer</summary>

   `class Child(Parent):`
   </details>

4. What decorator makes a method a property?
   <details>
   <summary>Answer</summary>

   `@property`
   </details>

5. What's the naming convention for "private" variables?
   <details>
   <summary>Answer</summary>

   Start with underscore: `_variable` or `__variable`
   </details>

---

**Next Lesson:** [09_file_io.md](09_file_io.md) - Learn to read and write files!
