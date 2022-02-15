# mindshit
A compiled language written in python that compiles to brainfuck.

## Literals
    10 -> 10
    'H' -> 72
    true -> 1

## Assignment
`<cell> = <literal>`
*Go to cell, set it to zero, and add the literal*

    &0 = 1
    &1 = 2

## Operator Assignment
`<cell> +=/-= <literal>`
*Go to cell and add or subtract it with the literal*

    &0 += 4
    &0 -= 2

## Output
`out <cell>`
*Output the value of a cell*

    &0 = 1
    out &0

## Output Statement
`out <statement>`
*Carry out statement and output its return value*

    out &0 = 1