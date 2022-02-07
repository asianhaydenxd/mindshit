'''
MINDSHIT

Data types
    int     Stores a plain integer 
    char    Stores a character
    bool    Stores 1 for true, 0 for false
    float   Stores a floating-point number

Special arrays
    int[]   Stores an array of digits for longer numbers # ! Not to be implemented
    char[]  Stores an array of chars to form a string

Variable declaration
    <type> <cell>;                  Reserve <cell> for type <type>)
    <type> <id>;                    Search for available cell, call it <id>, and reserve it for type <type>
    <type> <cell> <id>;             Take <cell>, call it <id>, and reserve it for type <type>
    <type> <cell> := <value>;         Reserve <cell> for type <type> with value <value>
    <type> <id> := <value>;           Search for available cell, call it <id>, and reserve it for type <type> with value <value>
    <type> <cell> <id> := <value>;    Take <cell>, call it <id>, and reserve it for type <type> with value <value>

Primitives
    <int>[@n]           Integer value
    '<char>'[@n]        Char value
    <true|false>[@n]    Boolean value
    <float>[@n]         Float value
    "<string>"[@n]      String value # ! Strings take up multiple cells
Float is distinguished from int with a dot placed anywhere within the number.
The optional @n after each template is necessary if you want to choose where the data will be stored yourself.

Data reassignment
    <cell|id> := <value>;
    
    <cell|id> += <value>;
    <cell|id> -= <value>;
    <cell|id> *= <value>;
    <cell|id> /= <value>;
    <cell|id> %= <value>;

While loop
    while (true) {
        /* code here */
    }

For loop
    for (int i := 0; i < 5; i:+1) {
        /* code here */
    }

For-in loop
    for (i in "Hello world!") {
        /* code here */
    }

If-elif-else statement
    if (true) {
        /* code here */
    } elif (true) {
        /* some more */
    } else {
        /* even more */
    }

Function (inline)
    fn add(int a, int b) -> int {
        return a + b;
    }
During compilation, all references to functions would be replaced by their contents.
# ! Recursive functions are not possible with inline functions.

Print
    print '<char>';
    print "<string>";

Read
    char <cell> <id>: read; Store input without modification
    int <cell> <id>: read;  Convert ASCII input to digit

Delete identifier
    del <id>;

'''


# Imports
from types import CellType
from typing import TypeVar

from matplotlib.pyplot import cla
Self = TypeVar('Self')


# Constants
DIGITS = '0123456789'
LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZŠŒŽšœžŸÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþ'
WHITESPACE = ' \t\n'


# Error
class Error:
    def __init__(self, name, info, start, end=None):
        self.name = name
        self.info = info
        self.start = start
        self.end = end
        if not end:
            self.end = start.copy().next()
    
    def __str__(self):
        result = f'{self.name}: {self.info}\n'
        result += f'File "{self.start.file_name}", line {self.start.line + 1}\n'
        result += self.point()
        return result

    def point(self):
        result = ''

        # Calculate indices
        start_index = max(self.start.file_text.rfind('\n', 0, self.start.index), 0)
        end_index = self.start.file_text.find('\n', start_index + 1)
        
        if end_index < 0:
            end_index = len(self.start.file_text)
        
        # Generate each line
        line_count = self.end.line - self.start.line + 1
        for i in range(line_count):
            # Calculate line columns
            line = self.start.file_text[start_index:end_index]
            column_start = self.start.column if i == 0 else 0
            column_end = self.end.column if i == line_count - 1 else len(line) - 1

            # Append to result
            result += line + '\n'
            result += ' ' * column_start + '^' * (column_end - column_start)

            # Re-calculate indices
            start_index = end_index
            end_index = self.start.file_text.find('\n', start_index + 1)
            
            if end_index < 0:
                end_index = len(self.start.file_text)

        return result.replace('\t', '')
    

class IllegalCharError(Error):
    def __init__(self, info, start, end=None):
        super().__init__('Illegal Character', info, start, end)
    

class InvalidSyntaxError(Error):
    def __init__(self, info, start, end=None):
        super().__init__('Invalid Syntax', info, start, end)



###################################################
# LEXER
###################################################

# Position
class Position:
    def __init__(self, index: int, line: int, column: int, file_name: str, file_text: str) -> None:
        self.index = index
        self.line = line
        self.column = column
        self.file_name = file_name
        self.file_text = file_text
        
    def next(self, char: str = None) -> Self:
        self.index += 1
        self.column += 1
        
        if char == '\n':
            self.line += 1
            self.column = 0
        
        return self
    
    def copy(self) -> Self:
        return Position(self.index, self.line, self.column, self.file_name, self.file_text)


# Tokens
class Tk:
    KW = 'kw'
    ID = 'id'
    OP = 'op'

    INT = 'int'
    FLOAT = 'float'
    CHAR = 'char'
    STR = 'string'

    EOF = 'eof'

    KEYWORDS = [
        'print',
        'read',
        
        'if',
        'elif',
        'else',
        'while',
        'for',
        'in', # Used for looping through arrays
        'fn', # Function declaration
        
        # Boolean values
        'true',
        'false',
        
        # Variable declarations
        'int', # Integer from 0-255
        'float', # Floating point; 
        'char', # Character from 0-255 in extended ASCII
        'bool', # Boolean value set to either 255 (true) or 0 (false)
    ]


class Token:
    def __init__(self, type_: str, value: str = None, start: Position = None, end: Position = None) -> None:
        self.type = type_
        self.value = value
        
        self.full = (self.type)
        if self.value:
            self.full = (self.type, self.value)
        
        if start:
            self.start = start.copy()
            self.end = start.copy().next()
        
        if end:
            self.end = end.copy()
    
    def __str__(self) -> str:
        if self.value != None:
            if type(self.value) == str:
                return f'[{self.type}: \'{self.value}\']'

            return f'[{self.type}: {self.value}]'

        return f'[{self.type}]'

    def matches(self, type_: str, value: any) -> str:
        return self.type == type_ and self.value == value


# Lexer
class Lexer:
    def __init__(self, file_name: str, text: str) -> None:
        self.file_name = file_name
        self.text = text
        self.pos = Position(-1, 0, -1, file_name, text)
        self.char = None
        self.next()
    
    def next(self, count=1) -> Self:
        for _ in range(count):
            self.pos.next(self.char)
        
        self.char = self.text[self.pos.index] if self.pos.index < len(self.text) else None
        
        return self
    
    def lex(self) -> list[Token]:
        tokens = []
        
        while self.char != None:
            if self.char in WHITESPACE:
                self.next()
                
            elif self.char in DIGITS:
                tokens.append(self.make_number())

            elif self.char in LETTERS:
                tokens.append(self.make_text())
                
            elif self.char == '@':
                tokens.append(Token(Tk.OP, '@', self.pos))
                self.next()

            elif self.char == ';':
                tokens.append(Token(Tk.OP, ';', self.pos))
                self.next()
            
            elif self.char == '(':
                tokens.append(Token(Tk.OP, '(', self.pos))
                self.next()
            
            elif self.char == ')':
                tokens.append(Token(Tk.OP, ')', self.pos))
                self.next()
            
            elif self.char == '[':
                tokens.append(Token(Tk.OP, '[', self.pos))
                self.next()
            
            elif self.char == ']':
                tokens.append(Token(Tk.OP, ']', self.pos))
                self.next()
            
            elif self.char == '{':
                tokens.append(Token(Tk.OP, '{', self.pos))
                self.next()
            
            elif self.char == '}':
                tokens.append(Token(Tk.OP, '}', self.pos))
                self.next()
            
            elif self.char == "'":
                token, error = self.make_char()
                if error: return [], error
                tokens.append(token)
            
            elif self.char == '"':
                token, error = self.make_string()
                if error: return [], error
                tokens.append(token)
            
            elif self.chars(2) == ':=':
                token, error = self.make_long_assign(':=')
                if error: return [], error
                tokens.append(token)
                
            elif self.chars(2) == '+=':
                token, error = self.make_long_assign('+=')
                if error: return [], error
                tokens.append(token)
            
            elif self.chars(2) == '-=':
                token, error = self.make_long_assign('-=')
                if error: return [], error
                tokens.append(token)
            
            elif self.chars(2) == '*=':
                token, error = self.make_long_assign('*=')
                if error: return [], error
                tokens.append(token)
            
            elif self.chars(2) == '/=':
                token, error = self.make_long_assign('/=')
                if error: return [], error
                tokens.append(token)
            
            elif self.chars(2) == '%=':
                token, error = self.make_long_assign('%=')
                if error: return [], error
                tokens.append(token)
            
            elif self.chars(2) == '//':
                self.skip_oneline_comment()
                
            elif self.chars(2) == '/*':
                self.skip_multiline_comment()

            else:
                return [], IllegalCharError(f"'{self.char}'", self.pos)
        
        tokens.append(Token(Tk.EOF, start=self.pos))
        
        return tokens, None
    
    
    def chars(self, length: int) -> str:
        chars = ''
        pointer = self.pos.copy()
        
        for _ in range(length):
            if pointer.index >= len(self.text):
                raise IndexError('char length is too long')
            
            chars += self.text[pointer.index]
            pointer.next(self.text[pointer.index])
        
        return chars
    
    def make_number(self) -> Token:
        num_str = ''
        dot_count = 0
        start_pos = self.pos.copy()
        
        while self.char != None and self.char in DIGITS + '.':
            if self.char == '.':
                if dot_count == 1:
                    break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.char
            
            self.next()
        
        if dot_count == 0:
            return Token(Tk.INT, int(num_str), start_pos, self.pos)
        else:
            return Token(Tk.FLOAT, float(num_str), start_pos, self.pos)
    
    def make_text(self) -> Token:
        text_str = ''
        start_pos = self.pos.copy()
        
        while self.char != None and self.char in LETTERS + DIGITS + '_':
            text_str += self.char
            self.next()
        
        token_type = Tk.KW if text_str in Tk.KEYWORDS else Tk.ID
        
        return Token(token_type, text_str, start_pos, self.pos)

    def make_char(self) -> Token:
        self.next()
        
        token = Token(Tk.CHAR, self.char, self.pos)
        
        self.next()
        
        if self.char == None or self.char != "'":
            return [], InvalidSyntaxError('expected "\'"', self.pos)
        
        self.next()
        
        return token, None
    
    def make_string(self) -> Token:
        text_str = ''
        start_pos = self.pos.copy()
        
        self.next()
        
        while self.char != None and self.char != '"':
            text_str += self.char
            self.next()
        
        self.next()
        
        return Token(Tk.STR, text_str, start_pos, self.pos), None
    
    def make_long_assign(self, value: str) -> Token:
        self.next()
        
        while self.char != None and self.char == '=':
            self.next()
        
        return Token(Tk.OP, value, self.pos), None
    
    def skip_oneline_comment(self) -> None:
        while self.char != None and self.char != '\n':
            self.next()
    
    def skip_multiline_comment(self) -> None:
        while self.char != None and self.chars(2) != '*/':
            self.next()
        
        self.next(2)


def run(file_name: str, text: str) -> None:
    lexer = Lexer(file_name, text)
    tokens, error = lexer.lex()
        
    if error:
        return None, error
    
    print(", ".join(map(lambda x: x.__str__(), tokens)))
    
    return None, None


with open("main.ms") as f:
    result, error = run("main.ms", f.read())

if error:
    print(error)
elif result:
    print(result)