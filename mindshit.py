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
    add(<a>, <b>);
    

Print
    print('<char>');
    print("<string>");

Read
    char <cell> <id>: read; Store input without modification
    int <cell> <id>: read;  Convert ASCII input to digit

Delete identifier
    del <id>;

'''


# Imports
from typing import Callable, TypeVar, Union, List, Tuple
import json

Self = TypeVar('Self')


# Constants
DIGITS = '0123456789'
LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZŠŒŽšœžŸÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþ'
WHITESPACE = ' \t\n'
CELLS = 25


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


# Error
class Error:
    def __init__(self, name: str, info: str, start: Position, end: Position = None):
        self.name = name
        self.info = info
        self.start = start
        self.end = end
        if not end:
            self.end = start.copy().next()
    
    def __str__(self) -> str:
        result = f'{self.name}: {self.info}\n'
        result += f'File "{self.start.file_name}", line {self.start.line + 1}\n'
        result += self.point()
        return result

    def point(self) -> str:
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
    def __init__(self, info: str, start: Position, end: Position = None) -> None:
        super().__init__('Illegal Character', info, start, end)
    

class InvalidSyntaxError(Error):
    def __init__(self, info: str, start: Position, end: Position = None) -> None:
        super().__init__('Invalid Syntax', info, start, end)



###################################################
# LEXER
###################################################

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
        'include',
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
        
        # Brainfuck Instructions
        'move',
        'right',
        'left',
        'set',
        'add',
        'sub',
        'output',
        'input',
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
    
    def next(self, count: int = 1) -> Self:
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
            
            elif self.char == ',':
                tokens.append(Token(Tk.OP, ',', self.pos))
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
    
    def make_number(self) -> Tuple[Token, Error]:
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
    
    def make_text(self) -> Tuple[Token, Error]:
        text_str = ''
        start_pos = self.pos.copy()
        
        while self.char != None and self.char in LETTERS + DIGITS + '_':
            text_str += self.char
            self.next()
        
        token_type = Tk.KW if text_str in Tk.KEYWORDS else Tk.ID
        
        return Token(token_type, text_str, start_pos, self.pos)

    def make_char(self) -> Tuple[Token, Error]:
        self.next()
        
        token = Token(Tk.CHAR, self.char, self.pos)
        
        self.next()
        
        if self.char == None or self.char != "'":
            return [], InvalidSyntaxError('expected "\'"', self.pos)
        
        self.next()
        
        return token, None
    
    def make_string(self) -> Tuple[Token, Error]:
        text_str = ''
        start_pos = self.pos.copy()
        
        self.next()
        
        while self.char != None and self.char != '"':
            text_str += self.char
            self.next()
        
        self.next()
        
        return Token(Tk.STR, text_str, start_pos, self.pos), None
    
    def make_long_assign(self, value: str) -> Tuple[Token, Error]:
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
    def __init__(self, address: int) -> None:
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
    
class DeclarationNode(Node):
    def __init__(self, datatype: TypeNode, cell: CellNode = None, alias: str = None, value: ValueNode = None) -> None:
        self.datatype = datatype
        self.cell = cell
        self.alias = alias
        self.value = value

class ReassignNode(Node):
    def __init__(self, cell: CellNode = None, alias: str = None, value: ValueNode = None) -> None:
        self.cell = cell
        self.alias = alias
        self.value = value

class ParamNode(Node):
    def __init__(self, alias: str) -> None:
        self.alias = alias

class ParamCallNode(Node):
    def __init__(self, param_num: int) -> None:
        self.param_num = param_num

class InstructionNode(Node):
    def __init__(self, command: str, argument: Union[CellNode, int] = None) -> None:
        self.command = command
        self.argument = argument
        if argument == None and command in ['right', 'left', 'add', 'sub']:
            self.argument = 1
        
class FnNode(Node):
    def __init__(self, name: str, params: List[ParamNode] = None, body: List[any] = None) -> None:
        self.name = name
        self.params = params
        self.body = body

class FnCallNode(Node):
    def __init__(self, name: str, fn: List[any], params: List[ParamNode]) -> None:
        self.name = name
        self.fn = fn
        self.params = params

class IfNode(Node):
    def __init__(self, body: List[any], else_node: List[any] = None, condition: ValueNode = None) -> None:
        self.condition = condition
        self.body = body
        self.else_node = else_node
    
    def add_child(self, node) -> any:
        self.body.append(node)
        return node
    
class ElseNode(Node):
    def __init__(self, body: List[any]) -> None:
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
    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.index = -1
        
        self.scope = MainNode([])
        self.scope_address = [self.scope]
        self.pointers = {}
        self.functions = []
        
        self.next()
    
    def next(self, index: int = 1) -> Token:
        self.index += index
        self.token = self.tokens[self.index] if self.index < len(self.tokens) else None
        return self.token
    
    def parse(self) -> Tuple[MainNode, Error]:
        self.preprocess()
        
        while self.token != None:
            if self.token.full in [(Tk.KW, 'int'), (Tk.KW, 'char'), (Tk.KW, 'bool')]:
                self.declaration()
            
            elif self.token.full in [(Tk.KW, 'move'), (Tk.KW, 'right'), (Tk.KW, 'left'), (Tk.KW, 'set'), (Tk.KW, 'add'), (Tk.KW, 'sub'), (Tk.KW, 'output'), (Tk.KW, 'input')]:
                instruction = InstructionNode(self.token.value)
                self.next()
                instruction.argument = self.expr()
                 
                self.scope.add_child(instruction)
            
            elif self.token.full == (Tk.KW, 'if'):
                self.if_statement()
            
            elif self.token.type == Tk.ID and self.token.value in map(lambda x: x.name, self.functions):
                self.call_fn(self.token.value)
            
            elif self.token.full == (Tk.OP, '}'):
                block_node = self.scope # Get the block node
                self.rise_scope_address()
                self.next()

                # If the block node that was just terminated is an if node, check if 'else' follows
                if self.token.full == (Tk.KW, 'else') and type(block_node) == IfNode:
                    self.lower_scope_address(block_node)
                    self.else_statement()
                    
                if type(block_node) == ElseNode:
                    self.rise_scope_address()
                
            else:
                self.next()
        
        return self.scope, None
    
    def rise_scope_address(self) -> None:
        self.scope_address.pop()
        self.scope = self.scope_address[-1]
    
    def lower_scope_address(self, node: any) -> None:
        self.scope_address.append(node)
        self.scope = node
    
    def preprocess(self) -> None:
        while self.token != None:
            if self.token.full == (Tk.KW, 'include'):
                self.next()
                if self.token.full == (Tk.ID, 'io'):
                    # Standard library
                    print_fn = FnNode('print', 
                        [ParamNode('output')], 
                        [
                            InstructionNode('output', ParamCallNode(0))
                        ]
                    )
                    self.functions.append(print_fn)
                    
                    read_fn = FnNode('read', 
                        [], 
                        [
                            InstructionNode('input', ParamCallNode(0))
                        ]
                    )
                    self.functions.append(read_fn)
                
            self.next()
            
        self.index = -1
        self.next()
    
    def call_fn(self, fn_name: str) -> None:
        fn_node = None
        for fn in self.functions:
            if fn.name == fn_name:
                fn_node = fn
                break
        
        fn_call_node = FnCallNode(fn_name, fn_node.body, [])
        
        self.next()
        
        if self.token.full != (Tk.OP, '('): raise
        
        self.next()
        
        while self.token.full != (Tk.OP, ')'):
            fn_call_node.params.append(self.expr())
            if self.token.full == (Tk.OP, ','):
                self.next()
        
        self.scope.add_child(fn_call_node)
        
        self.next()
    
    def declaration(self) -> None:
        decl_node = DeclarationNode(TypeNode(self.token.value))
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
        
        if self.token.full == (Tk.OP, '@'):
            self.next()
            
            if self.token.type != Tk.INT: raise

            decl_node.cell = CellNode(self.token.value)
            
            self.next()
        else:
            decl_node.cell = CellNode((list(self.pointers.values())[-1] + 2) if list(self.pointers.values()) else 0)
        
        if self.token.type != Tk.ID: raise

        decl_node.alias = self.token.value
        
        self.next()

        if self.token.full != (Tk.OP, ':='): raise
        
        self.next()
        decl_node.value = self.expr()
        self.pointers[decl_node.alias] = decl_node.cell.address
        
    def if_statement(self) -> None:
        if_node = IfNode([])
        self.scope.add_child(if_node)
        
        self.next()
        if self.token.full != (Tk.OP, '('): raise
        
        self.next()
        if_node.condition = self.expr()
        self.lower_scope_address(if_node)

        self.next()
        if self.token.full != (Tk.OP, '{'): raise

        self.next()

    def else_statement(self) -> None:
        if type(self.scope) != IfNode: raise
        
        else_node = ElseNode([])
        self.scope.else_node = else_node
        self.lower_scope_address(else_node)

        self.next()
        if self.token.full != (Tk.OP, '{'): raise

        self.next()
        
        
    # Algebra parser
        
    def factor(self) -> any:
        token = self.token
        
        if self.token.type in [Tk.INT, Tk.FLOAT, Tk.CHAR, Tk.STR]:
            self.next()
            return ValueNode(token.type, token.value)
        
        elif self.token.full in [(Tk.KW, 'true'), (Tk.KW, 'false')]:
            self.next()
            return ValueNode('bool', token.value)
        
        elif self.token.type == Tk.ID and self.token.value in self.pointers:
            self.next()
            return CellNode(self.pointers[token.value])

        elif self.token.full == (Tk.OP, '@'):
            self.next()
            token = self.token
            self.next()
            return CellNode(token.value)
        
        elif self.token.full == (Tk.OP, '('):
            self.next()
            expr = self.expr()
            if self.token.full == (Tk.OP, ')'):
                self.next()
                return expr
        
    def term(self) -> BinaryOpNode:
        return self.binary_op(self.factor, ((Tk.OP, '*'), (Tk.OP, '/'), (Tk.OP, '%')))
    
    def arith_expr(self) -> BinaryOpNode:
        return self.binary_op(self.term, ((Tk.OP, '+'), (Tk.OP, '-')))
    
    def comp_expr(self) -> BinaryOpNode:
        return self.binary_op(self.arith_expr, ((Tk.OP, '='), (Tk.OP, '>'), (Tk.OP, '<'), (Tk.OP, '>='), (Tk.OP, '<='), (Tk.OP, '!=')))
    
    def not_expr(self) -> UnaryOpNode:
        return self.unary_op(self.comp_expr, [(Tk.KW, 'not')])
    
    def expr(self) -> BinaryOpNode:
        return self.binary_op(self.not_expr, ((Tk.KW, 'and'), (Tk.KW, 'or')))
    
    def binary_op(self, function: Callable, ops: List[Tuple[str]]) -> BinaryOpNode:
        left = function()
        
        while self.token.full in ops:
            op_token = self.token
            self.next()
            right = function()
            left = BinaryOpNode(left, op_token, right)
            
        return left
    
    def unary_op(self, function: Callable, ops: List[Tuple[str]]) -> UnaryOpNode:
        value = function()
        while self.token.full in ops:
            op_token = self.token
            self.next()
            value = UnaryOpNode(op_token, value)
        
        return value



###################################################
# COMPILER
###################################################

class Compiler: # Go through AST and return string in Brainfuck
    def __init__(self, mainnode: MainNode) -> None:
        self.mainnode = mainnode
        
    def compile(self) -> None:
        self.result = ''
        self.pointer = 0
        self.cells = [0 for _ in range(CELLS)]
        self.visit(self.mainnode)
        print(self.cells)
        return self.result, None
    
    def visit(self, node: any) -> str:
        for child in node.body:
            if type(child) == InstructionNode:
                self.result += self.visit_instruction(child)
                
            elif type(child) == DeclarationNode:
                self.result += self.visit_declaration(child)
                
            elif type(child) == FnCallNode:
                self.result += self.visit_fncall(child)
    
    def visit_instruction(self, node: InstructionNode) -> str:
        if node.command == 'move':
            return self.move(node.argument)
        
        elif node.command == 'set':
            return self.set(node.argument)

        elif node.command == 'right':
            return self.right(node.argument)

        elif node.command == 'left':
            return self.left(node.argument)

        elif node.command == 'add':
            return self.add(node.argument)

        elif node.command == 'sub':
            return self.sub(node.argument)

        elif node.command == 'output':
            return self.output(node.argument)
        
        elif node.command == 'input':
            return self.input(node.argument)

        return ''

    def visit_declaration(self, node: DeclarationNode):
        return self.move(node.cell) + self.set(node.value)
    
    def visit_fncall(self, node: FnCallNode):
        result = ''
        
        for cmd in node.fn:
            if type(cmd.argument) == ParamCallNode:
                cmd.argument = node.params[cmd.argument.param_num]
            result += self.visit_instruction(cmd)

        return result
        
    # Turn other value types into integers from 0-255

    def correct(self, cell: ValueNode):
        new_value = 0
        if cell.type == 'int':
            new_value = cell.value
        elif cell.type == 'bool':
            if cell.value == 'true': new_value = 1
        elif cell.type == 'char':
            new_value = ord(cell.value)
        elif cell.type == 'array':
            return [self.correct(i) for i in cell.value]
        
        if new_value < 0 or new_value > 255: raise
        return new_value

    # Instructions
    
    def move(self, cell_target: CellNode) -> str:
        pointer = self.pointer
        self.pointer = cell_target.address
        
        if cell_target.address > pointer:
            return '>' * (cell_target.address - pointer)
        return '<' * (pointer - cell_target.address)
    
    def set(self, value_target: ValueNode) -> str:
        new_value = self.correct(value_target)
        cell = self.cells[self.pointer]
        self.cells[self.pointer] = new_value
        
        if new_value > cell:
            return '+' * (new_value - cell)
        return '-' * (cell - new_value)

    def right(self, cell_increase: ValueNode) -> str:
        new_value = self.correct(cell_increase)
        self.pointer += new_value if new_value != None else 1
        return '>' * new_value

    def left(self, cell_decrease: ValueNode) -> str:
        new_value = self.correct(cell_decrease)
        self.pointer -= new_value if new_value != None else 1
        return '<' * new_value

    def add(self, value_increase: ValueNode) -> str:
        new_value = self.correct(value_increase)
        self.cells[self.pointer] += new_value if new_value != None else 1
        return '+' * new_value

    def sub(self, value_decrease: ValueNode) -> str:
        new_value = self.correct(value_decrease)
        self.cells[self.pointer] -= new_value if new_value != None else 1
        return '-' * new_value

    def output(self, output_cell: CellNode = None) -> str:
        return self.move_append(output_cell, '.')

    def input(self, input_cell: CellNode = None) -> str:
        return self.move_append(input_cell, ',')

    def move_append(self, cell: CellNode, symbol: str) -> str:
        if cell != None:
            pointer = self.pointer
            self.pointer = cell
        
            if cell.address > pointer:
                return '>' * (cell.address - pointer) + symbol
            if cell.address < pointer:
                return '<' * (pointer - cell.address) + symbol
        return symbol



###################################################
# RUN
###################################################

def run(file_name: str, text: str) -> None:
    lexer = Lexer(file_name, text)
    tokens, error = lexer.lex()
        
    if error:
        return None, error
    
    # Load tokens to tokens.json
    token_json = json.loads(str(list(map(lambda x: x.__str__(), tokens))))
    with open('debug/tokens.json', 'w') as token_json_file:
        json.dump(token_json, token_json_file, indent=4)
    
    parser = Parser(tokens)
    ast, parse_error = parser.parse()
    
    # Load syntax tree into ast.json
    ast_json = json.loads(str(ast).replace("'", '"').replace('None', 'null'))
    with open('debug/ast.json', 'w') as ast_json_file:
        json.dump(ast_json, ast_json_file, indent=4)
    
    if parse_error:
        return None, parse_error
    
    compiler = Compiler(ast)
    bf, compiler_error = compiler.compile()

    # Load brainfuck output into target.bf
    with open('debug/target.bf', 'w') as target_bf_file:
        target_bf_file.write(bf)
    
    if compiler_error:
        return None, compiler_error
    
    return bf, None


def main() -> None:
    with open("main.ms") as f:
        result, error = run("main.ms", f.read())

    if error:
        print(error)

if __name__ == '__main__':
    main()