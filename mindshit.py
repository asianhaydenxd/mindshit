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
        return: a + b;
    }
During compilation, all references to functions would be replaced by their contents.
# ! Recursive functions are not possible with inline functions.

The function above can be called via the following syntax:
    add: <a>, <b>;
...or can apply its return value with the following:
    add(<a>, <b>)

Print
    print: '<char>';
    print: "<string>";

Read
    char <cell> <id>: read; Store input without modification
    int <cell> <id>: read;  Convert ASCII input to digit

Delete identifier
    del <id>;

'''


# Imports
from typing import TypeVar, Union, List, Tuple
import json

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
    BOOL = 'bool'
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
        
        'and',
        'or',
        'not',
        
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
    
    def __repr__(self) -> str:
        if self.value != None:
            if type(self.value) == str:
                return '{' + f"'{self.type}': '{self.value}'" + '}'

            return '{' + f"'{self.type}': {self.value}" + '}'

        return '{' + f"'{self.type}'" + '}'

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
    
    def lex(self) -> List[Token]:
        tokens = []
        
        while self.char != None:
            if self.char in WHITESPACE:
                self.next()
                
            elif self.char in DIGITS:
                tokens.append(self.make_number())

            elif self.char in LETTERS:
                tokens.append(self.make_text())
            
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
                
            elif self.chars(2) == '!=':
                tokens.append(Token(Tk.OP, '!=', self.pos))
                self.next(2)
                
            elif self.chars(2) == '>=':
                tokens.append(Token(Tk.OP, '>=', self.pos))
                self.next(2)
                
            elif self.chars(2) == '<=':
                tokens.append(Token(Tk.OP, '<=', self.pos))
                self.next(2)
                
            elif self.char == '=':
                tokens.append(Token(Tk.OP, '=', self.pos))
                self.next()
            
            elif self.char == '>':
                tokens.append(Token(Tk.OP, '>', self.pos))
                self.next()
            
            elif self.char == '<':
                tokens.append(Token(Tk.OP, '<', self.pos))
                self.next()
                
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
            
            elif self.char == '+':
                tokens.append(Token(Tk.OP, '+', self.pos))
                self.next()
            
            elif self.char == '-':
                tokens.append(Token(Tk.OP, '-', self.pos))
                self.next()
            
            elif self.char == '*':
                tokens.append(Token(Tk.OP, '*', self.pos))
                self.next()
            
            elif self.char == '/':
                tokens.append(Token(Tk.OP, '/', self.pos))
                self.next()
            
            elif self.char == '%':
                tokens.append(Token(Tk.OP, '%', self.pos))
                self.next()
            
            elif self.char == "'":
                token, error = self.make_char()
                if error: return [], error
                tokens.append(token)
            
            elif self.char == '"':
                token, error = self.make_string()
                if error: return [], error
                tokens.append(token)

            else:
                return [], IllegalCharError(f"'{self.char}'", self.pos)
        
        tokens.append(Token(Tk.EOF, start=self.pos))
        
        return tokens, None
    
    
    def chars(self, length: int) -> str:
        chars = ''
        pointer = self.pos.copy()
        
        for _ in range(length):
            if pointer.index >= len(self.text):
                return None
            
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
        


###################################################
# PARSER
###################################################

# Nodes
class Node:
    def __repr__(self) -> str:
        return str(vars(self))

class CellNode(Node):
    def __init__(self, address) -> None:
        self.address = address

class TypeNode(Node):
    def __init__(self, type: str, length: int = 1, datatype: Self = None) -> None:
        self.type = type
        self.length = length
        
        if type == 'array':
            self.datatype = datatype

class ValueNode(Node):
    def __init__(self, type: TypeNode, value: Union[int, List[Self]]) -> None:
        self.type = type
        self.value = value

class BinaryOpNode(Node):
    def __init__(self, left: ValueNode, token: Token, right: ValueNode) -> None:
        self.left = left
        self.token = token
        self.right = right
        
class UnaryOpNode(Node):
    def __init__(self, token: Token, value: ValueNode) -> None:
        self.token = token
        self.value = value
        
class FnCallNode(Node):
    def __init__(self, name: str) -> None:
        self.name = name
    
class DeclarationNode(Node):
    def __init__(self, parent: any, datatype: TypeNode, cell: CellNode = None, alias: str = None, value: ValueNode = None) -> None:
        self.parent = parent
        self.datatype = datatype
        self.cell = cell
        self.alias = alias
        self.value = value

class ReassignNode(Node):
    def __init__(self, parent: any, cell: CellNode = None, alias: str = None, value: ValueNode = None) -> None:
        self.parent = parent
        self.cell = cell
        self.alias = alias
        self.value = value

class IfNode(Node):
    def __init__(self, parent: any, body: List[any], else_node: List[any] = None, condition: ValueNode = None) -> None:
        self.parent = parent
        self.condition = condition
        self.body = body
        self.else_node = else_node
    
    def add_child(self, node) -> any:
        self.body.append(node)
        return node
    
class ElseNode(Node):
    def __init__(self, parent: any, body: List[any]) -> None:
        self.parent = parent
        self.body = body
    
    def add_child(self, node) -> any:
        self.body.append(node)
        return node

class MainNode(Node):
    def __init__(self, body: List[any]) -> None:
        self.body = body
        
    def add_child(self, node) -> any:
        self.body.append(node)
        return node


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.index = -1
        self.scope = MainNode([])
        self.pointers = {}
        self.next()
    
    def next(self, index: int = 1):
        self.index += index
        self.token = self.tokens[self.index] if self.index < len(self.tokens) else None
        return self.token
    
    def parse(self) -> Tuple[MainNode, Error]:
        self.scope = MainNode([])
        while self.token != None:
            if self.token.full in [(Tk.KW, 'int'), (Tk.KW, 'char'), (Tk.KW, 'bool')]:
                self.declaration()
            
            elif self.token.full == (Tk.KW, 'if'):
                self.if_statement()
            
            elif self.token.full == (Tk.OP, '}'):
                block_node = self.scope # Get the block node
                self.scope = self.scope.parent
                self.next()

                # If the block node that was just terminated is an if node, check if 'else' follows
                if self.token.full == (Tk.KW, 'else') and type(block_node) == IfNode:
                    self.scope = block_node
                    self.else_statement()
                    
                if type(block_node) == ElseNode:
                    self.scope = self.scope.parent
                
            else:
                self.next()
        
        return self.scope, None
    
    def declaration(self):
        decl_node = DeclarationNode(self.scope, TypeNode(self.token.value))
        self.scope.add_child(decl_node)
        
        type_ = self.token.value
        
        self.next()
        
        if self.token.full == (Tk.OP, '['):
            self.next()
            if self.token.type != Tk.INT: raise
            decl_node.datatype = TypeNode('array', self.token.value, type_)
                
            self.next()
            if self.token.full != (Tk.OP, ']'): raise
            
            self.next()
        
        if self.token.full != (Tk.OP, '@'): raise
        
        self.next()
        
        if self.token.type != Tk.INT: raise

        decl_node.cell = CellNode(self.token.value)
        
        self.next()
            
        if self.token.type != Tk.ID: raise

        decl_node.alias = self.token.value
        
        self.next()

        if self.token.full != (Tk.OP, ':='): raise
        
        self.next()
        decl_node.value = self.expr()
        self.pointers[decl_node.alias] = decl_node.cell.address

        if self.token.full != (Tk.OP, ';'): raise

        self.next()
        
    def if_statement(self):
        if_node = IfNode(self.scope, [])
        self.scope.add_child(if_node)
        
        self.next()
        if self.token.full != (Tk.OP, '('): raise
        
        self.next()
        if_node.condition = self.expr()
        self.scope = if_node

        self.next()
        if self.token.full != (Tk.OP, '{'): raise

        self.next()

    def else_statement(self):
        if type(self.scope) != IfNode: raise
        
        else_node = ElseNode(self.scope, [])
        self.scope.else_node = else_node
        self.scope = else_node

        self.next()
        if self.token.full != (Tk.OP, '{'): raise

        self.next()
        
        
    # Algebra parser
        
    def factor(self):
        token = self.token
        
        if self.token.type in [Tk.INT, Tk.FLOAT, Tk.CHAR]:
            self.next()
            return ValueNode(token.type, token.value)
        
        elif self.token.full in [(Tk.KW, 'true'), (Tk.KW, 'false')]:
            self.next()
            return ValueNode('bool', token.value)
        
        elif self.token.type == Tk.ID and self.token.value in self.pointers:
            self.next()
            return CellNode(self.pointers[token.value])
        
        elif self.token.full == (Tk.OP, '('):
            self.next()
            expr = self.expr()
            if self.token.full == (Tk.OP, ')'):
                self.next()
                return expr
        
    def term(self):
        return self.binary_op(self.factor, ((Tk.OP, '*'), (Tk.OP, '/'), (Tk.OP, '%')))
    
    def arith_expr(self):
        return self.binary_op(self.term, ((Tk.OP, '+'), (Tk.OP, '-')))
    
    def comp_expr(self):
        return self.binary_op(self.arith_expr, ((Tk.OP, '='), (Tk.OP, '>'), (Tk.OP, '<'), (Tk.OP, '>='), (Tk.OP, '<='), (Tk.OP, '!=')))
    
    def not_expr(self):
        return self.unary_op(self.comp_expr, [(Tk.KW, 'not')])
    
    def expr(self):
        return self.binary_op(self.not_expr, ((Tk.KW, 'and'), (Tk.KW, 'or')))
    
    def binary_op(self, function, ops):
        left = function()
        
        while self.token.full in ops:
            op_token = self.token
            self.next()
            right = function()
            left = BinaryOpNode(left, op_token, right)
            
        return left
    
    def unary_op(self, function, ops):
        value = function()
        while self.token.full in ops:
            op_token = self.token
            self.next()
            value = UnaryOpNode(op_token, value)
        
        return value
        

def run(file_name: str, text: str) -> None:
    lexer = Lexer(file_name, text)
    tokens, error = lexer.lex()
        
    if error:
        return None, error
    
    token_json = json.loads(str(list(map(lambda x: x.__str__(), tokens))))
    with open('debug/tokens.json', 'w') as token_json_file:
        json.dump(token_json, token_json_file, indent=4)
    
    parser = Parser(tokens)
    ast, parse_error = parser.parse()
    
    # Load syntax tree into ast.json
    ast_json = json.loads(str(ast).replace("'", '"').replace('...', '').replace('None', 'null'))
    with open('debug/ast.json', 'w') as ast_json_file:
        json.dump(ast_json, ast_json_file, indent=4)
    
    if parse_error:
        return None, parse_error
    elif ast: # Remove this elif once compiler is implemented
        return ast, None


def main():
    with open("main.ms") as f:
        result, error = run("main.ms", f.read())

    if error:
        print(error)

if __name__ == '__main__':
    main()