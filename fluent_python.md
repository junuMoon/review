# Fluent Python - Luciano Ramhalo

## Part 3. Classes and Protocols

### Goose Typing

- So, here’s my valediction: whenever you’re implementing a class embodying any of the concepts represented in the ABCs in numbers, collections.abc, or other framework you may be using, be sure (if needed) to subclass it from, or register it into, the corresponding ABC.
    - At the start of your programs using some library or framework defining classes which have omitted to do that, perform the registrations yourself; then, when you must check for (most typically) an argument being, e.g, “a sequence,” check whether: `isinstance(the_arg,collections.abc.Sequence)`
    - And, don’t define custom ABCs (or metaclasses) in production code. If you feel the urge to do so, I’d bet it’s likely to be a case of the “all problems look like a nail” syndrome for somebody who just got a shiny new hammer you(and future maintainers of your code) will be much happier sticking with straightforward and simple code, eschewing such depths. Valē!
- This is done by calling a register class method on the ABC. The registered class then becomes a virtual subclass of the ABC, and will be recognized as such by issub class, but it does not inherit any methods or attributes from the ABC.
    - 이게 왜 필요한거지
    - an object's suitability for a role is determined more by its methods and attributes rather than its actual class. register allows you to formally acknowledge that a class conforms to an ABC's interface, even if it doesn't directly inherit from it.

```python
from typing import TypeVar, Protocol
T = TypeVar('T')

class Repeatable(Protocol):
    def __mul__(self: T, repeat_count: int) -> T: ...

RT = TypeVar('RT', bound=Repeatable)
def double(x: RT) -> RT: returnx*2
```

- This example shows why PEP 544 is titled “Protocols: Structural subtyping (static duck typing).” The nominal type of the actual argument x given to double is irrelevant as long as it quacks—that is, as long as it implements `__mul__`.
- Typing 종류
    - Nominal Typing: 오리라는 이름을 가져야 오리
    - Structural Typing: 오리의 구조를 가지면 오리
    - Static Typing: 컴파일 할 때 오리인지 체크
    - Duck(Dynamic) Typing: 런타임에 오리처럼 행동하면 오리임
    - Static Duck Typing(Protocols): 컴파일 시 오리처럼 행동하는지 확인한다면, 런타임에 실제 타입은 신경 쓰지 않는다.
        - Python can enforce type constraints at the static analysis level (like when using tools such as mypy) while still embracing a flexible, duck-typing-like approach.

<img width="586" alt="image" src="https://github.com/junuMoon/review/assets/52732827/eb366189-4b5f-4c16-b33b-2f6a7eac4f28">

### Inheritanc: For Better or for Worse

- The MRO takes into account not only the inheritance graph but also the order in which superclasses are listed in a subclass declaration.
    - subclass에서 method call은 Method Resolution Order를 따름. MRO는 inheritance graph뿐만 아니라 순서도 따름
- A mixin class is designed to be subclassed together with at least one other class in a multiple inheritance arrangement
    -Mixins allow for the modification or enhancement of class behaviors without using inheritance from the base class directly.
    - Minin을 사용하면 상위 class의 메소드를 수정하지 않고서도 행동을 제어할 수 있음
    - a mixin does not define a new type; it merely bundles methods for reuse. 
- Understand Why Inheritance Is Used in Each Case:
    - Inheritance of interface creates a subtype, implying an “is-a” relationship. This is best done with ABCs
    - Inheritance of implementation avoids code duplication by reuse. Mixins can help with this
- Chosun-gpt에 적용해보면 ABC <- Generator <- ArticleGenerator 이렇게 하고, Styler는 아무 것도 상속받지 말고 Generator를 내부에서 사용하도록 갔어야 한다. 

