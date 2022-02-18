#!/usr/bin/python
#
# Brainfuck Interpreter
# Copyright 2011 Sebastian Kaspari
#
# Usage: ./brainfuck.py [FILE]

# From http://code.activestate.com/recipes/134892/
# "getch()-like unbuffered character reading from stdin
#  on both Windows and Unix (Python recipe)"

import sys


def execute(filename):
    f = open(filename, "r")
    evaluate(f.read())
    f.close()


def evaluate(code, returning: bool = False, default_input: str = ''):
    out = ''
    code = cleanup(list(code))
    bracemap = buildbracemap(code)

    cells, codeptr, cellptr, inputchars = [0], 0, 0, list(default_input).reverse()

    while codeptr < len(code):
        command = code[codeptr]

        if command == ">":
            cellptr += 1
            if cellptr == len(cells):
                cells.append(0)
        if command == "<":
            cellptr = 0 if cellptr <= 0 else cellptr - 1

        if command == "+": cells[cellptr] = cells[cellptr] + 1 if cells[cellptr] < 255 else 0
        if command == "-": cells[cellptr] = cells[cellptr] - 1 if cells[cellptr] > 0 else 255

        if command == "[" and cells[cellptr] == 0: codeptr = bracemap[codeptr]
        if command == "]" and cells[cellptr] != 0: codeptr = bracemap[codeptr]

        if command == ".":
            if returning:
                out += chr(cells[cellptr])
            else:
                sys.stdout.write(chr(cells[cellptr]))

        if command == ",":
            if returning:
                if len(inputchars) == 0:
                    inputchars.append(0)
                cells[cellptr] = inputchars[-1]
                inputchars.pop()

            else:
                if len(inputchars) == 0:
                    read = input()
                    if read:
                        for letter in reversed(read):
                            inputchars.append(ord(letter))
                    else:
                        inputchars.append(0)
                cells[cellptr] = inputchars[-1]
                inputchars.pop()

        codeptr += 1

    return out


def cleanup(code):
    return ''.join(filter(lambda x: x in ['.', ',', '[', ']', '<', '>', '+', '-'], code))


def buildbracemap(code):
    temp_bracestack, bracemap = [], {}

    for position, command in enumerate(code):
        if command == "[":
            temp_bracestack.append(position)
        if command == "]":
            start = temp_bracestack.pop()
            bracemap[start] = position
            bracemap[position] = start
    return bracemap

def main():
    if len(sys.argv) == 2:
        execute(sys.argv[1])
    else:
        print("Usage:", sys.argv[0], "filename")


if __name__ == "__main__":
    main()
