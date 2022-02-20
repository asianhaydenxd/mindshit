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

### Input
`in <cell>`
*Write input to a cell*

    a: in &0
    out a

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

### Swapping
`<cell> -> <cell>`
*Swap the values of two cells*

    &0 = 1
    &1 = 2
    &0 <-> &1

### Assignment to cells
`<cell> = <cell>`
*Set the cell value to the value of the target cell*

    &0 = 1
    &1 = &0

### While loops
`while <cell> ... end`
*Execute a block of code until the specified condition cell is false*

    i: &0 = 5
    while i
        i -= 1
    end

### If statements
`if <cell> ... end`
*Execute a block of code if the specified condition cell is true*

    i: &0 = 1
    if i
        i += 1
    end

### If-else statements
`if <cell> ... else ... end`
*Execute one block of code if the specified condition cell is true and another if false*

    i: &0 = 1
    if i
        i += 1
    else
        i += 2
    end