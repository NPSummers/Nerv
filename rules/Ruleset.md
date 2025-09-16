# Programming Language Ruleset

## Comments
- Single-line comments: `# This is a comment`
- Multi-line comments: `# { This is a multi-line comment }`

## Variable Declaration
- Variables are declared using `plug` (equivalent to `var` or `let`):
  ``` 
  plug x: type = 10
  ```

## Data Types
- **Integer (`int`)**: Whole numbers
  ```
  plug x: int = 42
  ```
- **Float (`float`)**: Floating-point numbers
  ```
  plug y: float = 3.14
  ```
- **Boolean (`bool`)**: True or false values
  ```
  plug is_active: bool = true
  ```
- **String (`string`)**: Sequence of characters
  ```
  plug name: string = "Hello"
  ```
- **Character (`char`)**: Single character
  ```
  plug letter: char = 'A'
  ```
- **List (`list`)**: Ordered, mutable collection
  ```
  plug numbers: list = [1, 2, 3]
  ```
- **Dictionary (`dict`)**: Key-value pairs, immutable by default
  ```
  plug info: dict = {"key": "value", "id": 1}
  ```
  - To modify a dictionary, explicitly mark it as mutable:
    ```
    plug mutable info: dict = {"key": "value"}
    info["key"] = "new_value" # Allowed only if mutable
    ```

## Imports
- Import modules using `import` and `from`:
  ```
  import math
  from utils import helper_function
  ```

## Classes
- Defined using `class` keyword followed by the class name and curly braces:
  ```
  class Person {
      plug name: string
      plug age: int
      
      plug Person(name: string, age: int) {
          this.name = name
          this.age = age
      }
  }
  ```

## Functions
- Defined using `plug` followed by the function name, parameters, and curly braces:
  ```
  plug add(a: int, b: int): int {
      return a + b
  }
  ```

## Notes
- Dictionaries are immutable by default to ensure data integrity. Use `mutable` keyword for mutable dictionaries.
- Type annotations are optional but recommended for clarity.
- Functions and classes use curly braces `{}` to define their scope.
- The language supports standard control structures (if, for, while) but they are not detailed here.