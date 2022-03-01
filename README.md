# mindshit
An imperative programming language that compiles to brainfuck.

## Language Reference

### Literals

#### Integer

An integer is any positive whole number between 0 and 255. It is written as a number without any decimal point.

    int index = 5

#### Boolean

A boolean is `true` or `false` and is stored as a 1 or a 0. It is mainly used by conditional blocks suck as `if` or `while`.

    bool is_enabled = true

#### Character

A character is an integer value represented via ASCII or Unicode. It is written as a single character or an escaped character surrounded by either single (`''`) or double (`""`) quotes.

    char letter = 'a'

#### String

A string is an array/collection of characters. It is written as multiple characters surrounded by either single (`''`) or double (`""`) quotes.

    str message = 'Hello world!'

#### Long integer

A long integer is an array/collection of integers in order to represent a single large integer. It is written as an integer greater than 255 (underscores may be used instead of commas).

    long clicks = 13_370_000