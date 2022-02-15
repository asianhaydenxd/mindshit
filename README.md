# mindshit
A compiled language written in python that compiles to brainfuck.

## Features

### Literals
    10 -> 10
    'H' -> 72
    true -> 1

### Assignment
`<cell> = <literal>`
*Go to cell, set it to zero, and add the literal*

    &0 = 1
    &1 = 2

### Operator Assignment
`<cell> +=/-= <literal>`
*Go to cell and add or subtract it with the literal*

    &0 += 4
    &0 -= 2

### Output
`out <cell>`
*Output the value of a cell*

    &0 = 1
    out &0

### Output Statement
`out <statement>`
*Carry out statement and output its return value*

    out &0 = 1

### Aliasing
`out <statement>`
*Attach identifiers to certain cells*

    a: &0
    b: &1 = 5
    a = 1

### Relocating
`<cell> -> <cell>`
*Set target cell to current cell and clear the current cell*

    &0 = 1
    &0 -> &1

## Todo
- Create a function to make new compiler operator implementations more efficient as opposed to comparing every input for every operator
- Implement `<->` operator for swapping two cells
- Allow assigning to other cells `&1 = &0`
- Implement `in` keyword for input (returns input ASCII integer)
- Implement arrays (`z: &0 = 0, 1, 2` for defining, `z#0` or `&0#0` for accessing)