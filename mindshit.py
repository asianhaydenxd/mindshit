#!/usr/bin/env python3

# Imports
from typing import Callable, Iterable, TypeVar, Union, List, Tuple
import json, sys

Self = TypeVar('Self')
Node = TypeVar('Node')

# Constants
DIGITS = '0123456789'
LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZŠŒŽšœžŸÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþ'
WHITESPACE = ' \t\n'
OCTDIGITS = '01234567'
HEXDIGITS = '0123456789ABCDEF'

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
    def __init__(self, name: str, info: str, start: Position, end: Position = None) -> None:
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

# Tokens
class Tk:
    KW = 'kw'
    OP = 'op'
    ID = 'id'

    INT = 'int'
    CHAR = 'char'
    STR = 'string'

    EOF = 'eof'

    KEYWORDS = [
        # Loop blocks
        'while',
        
        # Conditional blocks
        'if',
        'elif',
        'else',
        
        # Block tokens
        'do',
        'end',
        
        # Type declarations
        'int',
        'char',
        'bool',
        'void',
        
        # Boolean literals
        'true',
        'false',
        
        # Logical operators
        'and',
        'or',
        'not',
        
        # IO
        'print',
        'input',
    ]
    
    OPERATORS = [
        # Data relocation
        '<->', '<-',
        
        # Arithmetic assignment
        '+=', '-=', '*=', '/=',
        
        # Comparison
        '==', '!=', '<=', '>=',
        '<', '>',
        
        # Misc
        '&', '=', ':', ',',
        
        # Brackets
        '(', ')', '[', ']',
        
        # Arithmetic
        '+', '-', '*', '/', '%',
    ]

class Type:
    INT = 'int' # Integer (whole number)
    CHAR = 'char' # Character (ASCII/Unicode)
    BOOL = 'bool' # Boolean (true/false)
    VOID = 'void' # Void (programmer-inaccessible)

    @staticmethod
    def str_to_type(typestring):
        if typestring == 'int':
            return Type.INT
        if typestring == 'char':
            return Type.CHAR
        if typestring == 'bool':
            return Type.BOOL

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
            return f'[{self.type}: {self.value}]'
        return f'[{self.type}]'
    
    def reprJSON(self) -> dict:
        if self.value != None:
            return dict(type=self.type, value=self.value)
        return dict(type=self.type)

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
    
    def lex(self) -> Tuple[List[Token], Error]:
        tokens = []
        
        while self.char != None:
            if self.char in WHITESPACE:
                self.next()
                
            elif self.char in DIGITS:
                tokens.append(self.make_number())
            
            elif self.char in LETTERS + '_':
                tokens.append(self.make_text())
            
            elif self.char in ["'", '"']:
                token, error = self.make_char()
                if error: return [], error
                tokens.append(token)
            
            elif self.chars(2) == '//':
                self.skip_oneline_comment()
                
            elif self.chars(2) == '/*':
                self.skip_multiline_comment()
            
            else:
                for op in Tk.OPERATORS:
                    if self.chars(len(op)) == op:
                        tokens.append(Token(Tk.OP, op, self.pos))
                        self.next(len(op))
                        break
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
        start_pos = self.pos.copy()
        
        while self.char != None and self.char in DIGITS:
            num_str += self.char
            self.next()
        
        return Token(Tk.INT, int(num_str), start_pos, self.pos)
    
    def make_text(self) -> Tuple[Token, Error]:
        text_str = ''
        start_pos = self.pos.copy()
        
        while self.char != None and self.char in LETTERS + DIGITS + '_':
            text_str += self.char
            self.next()
        
        token_type = Tk.KW if text_str in Tk.KEYWORDS else Tk.ID
        
        return Token(token_type, text_str, start_pos, self.pos)

    def make_char(self) -> Tuple[Token, Error]:
        text_str = ''
        start_pos = self.pos.copy()
        quote_type = self.char
        
        self.next()
        
        while self.char != None and self.char != quote_type:
            if self.char == '\\':
                self.next()
                if   self.char == '\\': text_str += '\\'
                elif self.char == "'": text_str += '\''
                elif self.char == '"': text_str += '\"'
                elif self.char == 'a': text_str += '\a'
                elif self.char == 'b': text_str += '\b'
                elif self.char == 'f': text_str += '\f'
                elif self.char == 'n': text_str += '\n'
                elif self.char == 'r': text_str += '\r'
                elif self.char == 's': text_str += ' '
                elif self.char == 't': text_str += '\t'
                elif self.char == 'v': text_str += '\v'
                elif self.char == 'x':
                    self.next()
                    num = ''
                    while self.char in HEXDIGITS:
                        num += self.char
                        self.next()
                    text_str += chr(int(num, 16))
                    continue
                elif self.char == 'd':
                    self.next()
                    num = ''
                    while self.char in DIGITS:
                        num += self.char
                        self.next()
                    text_str += chr(int(num, 10))
                    continue
                elif self.char == 'B':
                    self.next()
                    num = ''
                    while self.char in ['0', '1']:
                        num += self.char
                        self.next()
                    text_str += chr(int(num, 2))
                    continue
                elif self.char in OCTDIGITS:
                    num = ''
                    while self.char in OCTDIGITS:
                        num += self.char
                        self.next()
                    text_str += chr(int(num, 8))
                    continue
                else: text_str += self.char
                self.next()
                continue
            
            text_str += self.char
            self.next()
        
        self.next()
        
        return Token(Tk.CHAR if len(text_str) == 1 else Tk.STR, text_str, start_pos, self.pos), None
    
    def skip_oneline_comment(self) -> None:
        while self.char != None and self.char != '\n':
            self.next()
    
    def skip_multiline_comment(self) -> None:
        while self.char != None and self.chars(2) != '*/':
            self.next()
        self.next(2)

class LiteralNode:
    def __init__(self, value: int, type_: Type) -> None:
        self.value = value
        self.type = type_
    
    def reprJSON(self) -> str:
        return dict(value=self.value)

class AddressNode:
    def __init__(self, address: int) -> None:
        self.address = address
    
    def reprJSON(self) -> str:
        return dict(address=self.address)
    
class ArrayNode:
    def __init__(self, array: List[AddressNode]):
        self.array = array
        self.index = 0
        
    def reprJSON(self) -> str:
        return dict(array=self.array)

class ArrayAccessNode:
    def __init__(self, array_title: str, index: Node) -> None:
        self.array_title = array_title
        self.index = index

    def reprJSON(self) -> str:
        return dict(array_name=self.array_title, index=self.index)

class IdentifierNode:
    def __init__(self, title: str) -> None:
        self.title = title
    
    def reprJSON(self) -> str:
        return dict(title=self.title)

class BinaryOpNode:
    def __init__(self, left: Node, token: Token, right: Node) -> None:
        self.left = left
        self.token = token
        self.right = right
    
    def reprJSON(self) -> str:
        return dict(token=self.token, left=self.left, right=self.right)

class UnaryOpNode:
    def __init__(self, token: Token, right: Node) -> None:
        self.token = token
        self.right = right
    
    def reprJSON(self) -> str:
        return dict(token=self.token, right=self.right)
    
class FunctionOpNode:
    def __init__(self, token: Token, args: List[Node]) -> None:
        self.token = token
        self.args = args
    
    def reprJSON(self) -> str:
        return dict(token=self.token, args=self.args)

class ConditionalNode:
    def __init__(self, token: Token, condition: Node, body: List[Node], elsebody: List[Node] = []) -> None:
        self.token = token
        self.condition = condition
        self.body = body
        self.elsebody = elsebody
    
    def reprJSON(self) -> str:
        return dict(token=self.token, condition=self.condition, body=self.body, elsebody=self.elsebody)

class DoNode:
    def __init__(self, name: str, body: List[Node]) -> None:
        self.name = name
        self.body = body
    
    def reprJSON(self) -> str:
        return dict(name=self.name, body=self.body)

class Parser:
    def __init__(self, file_name: str, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.index = -1
        
        self.location = [DoNode(file_name, [])]
        
        self.next()
    
    def next(self, index: int = 1) -> Token:
        self.index += index
        self.token = self.tokens[self.index] if self.index < len(self.tokens) else None
        return self.token
           
    # TODO: create parsing errors
    def parse(self) -> Tuple[DoNode, Error]:
        # TODO: implement "include" for including tokens of another file
        while self.token.full != Tk.EOF:
            expr, error = self.expr()
            if error:
                return None, error
            self.location[-1].body.append(expr)
        
        return self.location[-1], None

    def unary_op(self, ops: List[Tuple[str]], function: Callable) -> UnaryOpNode:
        while self.token.full in ops:
            op_token = self.token
            self.next()
            right, error = function()
            return UnaryOpNode(op_token, right), error
        return function()
    
    def binary_op(self, ops: List[Tuple[str]], function: Callable) -> Union[BinaryOpNode, UnaryOpNode]:
        left, error = function()
        while self.token.full in ops:
            op_token = self.token
            self.next()
            right, error = function()
            left = BinaryOpNode(left, op_token, right)
        return left, error
    
    def function_op(self, ops: List[Tuple[str]], argc: int, function: Callable) -> UnaryOpNode:
        while self.token.full in ops:
            op_token = self.token
            self.next()
            args = []
            for _ in range(argc):
                right, error = function()
                args.append(right)
            return FunctionOpNode(op_token, args), error
        return function()

    def conditional_op(self, ops: List[Tuple[str]], function: Callable) -> Union[ConditionalNode, BinaryOpNode, UnaryOpNode]:
        if self.token.full in ops:
            op_token = self.token
            self.next()
            condition, error = function()
            blocknode = ConditionalNode(op_token, condition, [])
            
            while not self.token.full in [(Tk.KW, 'end'), (Tk.KW, 'else'), (Tk.KW, 'elif')]:
                instruction, error = self.expr()
                blocknode.body.append(instruction)
                
            childblocknode = blocknode
                
            while self.token.full == (Tk.KW, 'elif'):
                op_token = self.token
                self.next()
                condition, error = function()
                newblocknode = ConditionalNode(op_token, condition, [])
                
                while not self.token.full in [(Tk.KW, 'end'), (Tk.KW, 'else'), (Tk.KW, 'elif')]:
                    instruction, error = self.expr()
                    newblocknode.body.append(instruction)
                
                childblocknode.elsebody = [newblocknode]
                childblocknode = newblocknode
                
            if self.token.full == (Tk.KW, 'else'):
                self.next()
                while self.token.full != (Tk.KW, 'end'):
                    instruction, error = self.expr()
                    childblocknode.elsebody = []
                    childblocknode.elsebody.append(instruction)
                
            if self.token.full == (Tk.KW, 'end'):
                self.next()
            
            return blocknode, error
        return function()
    
    # TODO: implement switch/match block
    
    # TODO: implement inline functions and function calls
    
    # TODO: implement data structures
    
    def expr(self) -> Union[ConditionalNode, BinaryOpNode, UnaryOpNode]:
        return  self.conditional_op([(Tk.KW, 'while'), (Tk.KW, 'if')],
        lambda: self.function_op([(Tk.KW, 'int'), (Tk.KW, 'char'), (Tk.KW, 'bool')], 1,
        lambda: self.binary_op([(Tk.OP, '='), (Tk.OP, '+='), (Tk.OP, '-='), (Tk.OP, '*='), (Tk.OP, '/='), (Tk.OP, '<-'), (Tk.OP, '<->')], 
        lambda: self.binary_op([(Tk.KW, 'or')],
        lambda: self.binary_op([(Tk.KW, 'and')],
        lambda: self.unary_op([(Tk.KW, 'not')],
        lambda: self.binary_op([(Tk.OP, '=='), (Tk.OP, '!='), (Tk.OP, '>'), (Tk.OP, '<'), (Tk.OP, '>='), (Tk.OP, '<=')],
        lambda: self.binary_op([(Tk.OP, '+'), (Tk.OP, '-')],
        lambda: self.binary_op([(Tk.OP, '*'), (Tk.OP, '/'), (Tk.OP, '%')],
        lambda: self.binary_op([(Tk.OP, ':')],
        lambda: self.function_op([(Tk.KW, 'input')], 1,
        lambda: self.function_op([(Tk.KW, 'print')], 1, 
                self.factor
        ))))))))))))
            
    def factor(self) -> Node:
        token = self.token
        self.next()
        
        if token.type == Tk.INT:
            return LiteralNode(token.value, Type.INT), None
        
        if token.type == Tk.CHAR:
            return LiteralNode(ord(token.value), Type.CHAR), None
        
        if token.type == Tk.STR:
            return ArrayNode([LiteralNode(ord(char), Type.CHAR) for char in token.value]), None

        if token.full == (Tk.KW, 'true'):
            return LiteralNode(1, Type.BOOL), None
        
        if token.full == (Tk.KW, 'false'):
            return LiteralNode(0, Type.BOOL), None
        
        if token.full == (Tk.OP, '&'):
            address_token = self.token
            self.next()
            return AddressNode(address_token.value), None
        
        if token.full == (Tk.KW, 'do'):
            donode = DoNode('do', [])
            while self.token.full not in [(Tk.EOF), (Tk.KW, 'end')]:
                expr, error = self.expr()
                if error:
                    return None, error
                donode.body.append(expr)
            self.next()
            return donode, None
        
        if token.full == (Tk.OP, '['):
            array = ArrayNode([])
            while self.token.full not in [(Tk.EOF), (Tk.OP, ']')]:
                if self.token.full == (Tk.OP, ','): self.next()
                element, error = self.expr()
                array.array.append(element)
            self.next()
            return array, None

        if token.full == (Tk.OP, '('):
            expr = self.expr()
            if self.token.full == (Tk.OP, ')'):
                self.next()
                return expr

        if token.type == Tk.ID:
            if self.token.full == (Tk.OP, '['):
                self.next()
                index = self.expr()
                if self.token.full == (Tk.OP, ']'):
                    self.next()
                    return ArrayAccessNode(token.value, index[0]), None
                raise
            return IdentifierNode(token.value), None
        
        return None, Error('Exception Raised', 'invalid factor', token.start, token.end)

# Infinity-simulating list
class InfiniteList:
    def __init__(self, object):
        self.object = object
        self.list = []
        
    def __getitem__(self, index):
        try:
            return self.list[index]
        except IndexError:
            self.list += [self.object for _ in range(index+1 - len(self.list))]
            return self.list[index]
    
    def __setitem__(self, index, value):
        try:
            self.list[index] = value
        except IndexError:
            self.list += [self.object for _ in range(index+1 - len(self.list))]
            self.list[index] = value
            
    def __iter__(self) -> Iterable:
        return iter(self.list)
    
    def __repr__(self) -> str:
        return str(self.list)

# Infinity-like list of boolean values indicating whether each slot is used along with their type
class MemoryUsageList(InfiniteList):
    def __init__(self) -> None:
        super().__init__(None)
    
    def use(self, index: int, type_: Type) -> None:
        self[index] = type_
    
    def rmv(self, *indices: int) -> None:
        for index in indices:
            self[index] = False
    
    def get_cell(self) -> int:
        for index, value in enumerate(self.list):
            if not value:
                return index
        self[len(self.list)] = None
        return len(self.list) - 1
    
    def get_array(self, size: int) -> int:
        total_size = 0
        start_index = None
        for index, used in enumerate(self.list):
            if not used:
                total_size += 1
            else:
                total_size = 0
                
            if total_size == 1:
                start_index = index
                
            if total_size == size:
                return start_index
        self[len(self.list)-1+size] = None
        return len(self.list)-size
    
    def allocate(self, *types) -> int:
        if len(types) == 1:
            cell_found = self.get_cell()
            self.use(cell_found, types[0])
            return cell_found
            
        cells = []
        for type_ in types:
            cell_found = self.get_cell()
            self.use(cell_found, type_)
            cells.append(cell_found)
        return tuple(cells)
    
    def allocate_block(self, size: int, type_: Type) -> int:
        block_found = self.get_array(size)
        for i in range(size):
            self.use(block_found + i, type_)
        return block_found

class Compiler:
    def __init__(self, mainnode) -> None:
        self.mainnode = mainnode
    
    def compile(self) -> str:
        self.aliases = {}
        self.arrays = {}
        self.literals = {}
        self.memory = MemoryUsageList()
        self.pointer = 0
        return self.visit(self.mainnode)
    
    def visit(self, node) -> None:
        result = ''

        # TODO: extract node compilers into separate functions (with left and right params)
        if type(node) == DoNode:
            return self.visit_do(node)
        
        if type(node) == ConditionalNode:
            if node.token.full == (Tk.KW, 'while'):
                return self.visit_while(node.condition, node.body)
            
            if node.token.full in [(Tk.KW, 'if'), (Tk.KW, 'elif')]:
                return self.visit_if(node.condition, node.body, node.elsebody)
        
        if type(node) == BinaryOpNode:
            if node.token.full == (Tk.OP, '='):
                # TODO: raise error when types do not match
                if type(node.right) == ArrayNode:
                    return self.visit_array_assign(node.left, node.right)
                
                return self.visit_assign(node.left, node.right)
                    
            if node.token.full == (Tk.OP, '+='):
                return self.visit_add_assign(node.left, node.right)
                    
            if node.token.full == (Tk.OP, '-='):
                return self.visit_subtract_assign(node.left, node.right)
            
            if node.token.full == (Tk.OP, '*='):
                return self.visit_multiply_assign(node.left, node.right)
            
            if node.token.full == (Tk.OP, '/='):
                return self.visit_divide_assign(node.left, node.right)

            if node.token.full == (Tk.OP, '<-'):
                return self.visit_relocate(node.left, node.right)
            
            if node.token.full == (Tk.OP, '<->'):
                return self.visit_swap(node.left, node.right)
            
            if node.token.full == (Tk.OP, '+'):
                return self.visit_op_add(node.left, node.right)
            
            if node.token.full == (Tk.OP, '-'):
                return self.visit_op_subtract(node.left, node.right)
            
            if node.token.full == (Tk.OP, '*'):
                return self.visit_op_multiply(node.left, node.right)
            
            if node.token.full == (Tk.OP, '/'):
                return self.visit_op_divide(node.left, node.right)
            
            if node.token.full == (Tk.OP, '%'):
                return self.visit_op_modulus(node.left, node.right)
            
            if node.token.full == (Tk.OP, '=='):
                return self.visit_op_equal(node.left, node.right)
            
            if node.token.full == (Tk.OP, '!='):
                return self.visit_op_notequal(node.left, node.right)

            if node.token.full == (Tk.OP, '<'):
                return self.visit_op_less(node.left, node.right)
            
            if node.token.full == (Tk.OP, '<='):
                return self.visit_op_lessequal(node.left, node.right)
            
            if node.token.full == (Tk.OP, '>'):
                return self.visit_op_greater(node.left, node.right)
            
            if node.token.full == (Tk.OP, '>='):
                return self.visit_op_greaterthan(node.left, node.right)
                
            if node.token.full == (Tk.OP, ':'):
                if type(node.left) == IdentifierNode:
                    try:
                        self.aliases[node.left.title] = node.right.address
                    except AttributeError:
                        self.aliases[node.left.title] = node.right.right.address
                    return self.visit(node.right)
                        
            raise RuntimeError('binary operator not defined in compiler')
        
        if type(node) == UnaryOpNode:            
            if node.token.full == (Tk.KW, 'not'):
                return self.visit_op_not(node.right)
        
        if type(node) == FunctionOpNode:
            if node.token.full == (Tk.KW, 'print'):
                return self.visit_print(node.args[0])

                raise TypeError(f'unsupported type {self.memory[self.pointer]}')
            
            if node.token.full == (Tk.KW, 'input'):
                # TODO: take multi-character input
                return self.visit(node.args[0]) + self.input()
            
            # FIXME: detect types automatically (including arrays) and rather replace everything with keyword let
            if node.token.type == Tk.KW and node.token.value in ['int', 'char', 'bool']:
                return self.visit_var_declaration(node.token.value, node.args[0])
            
            raise RuntimeError('unary operator not defined in compiler')
        
        if type(node) == AddressNode:
            return self.move(node.address)

        if type(node) == int:
            return self.move(node)
        
        if type(node) == ArrayAccessNode:
            if node.array_title in self.arrays:
                return self.visit_array_access(right)

            raise RuntimeError('specified array has not been declared')
        
        if type(node) == IdentifierNode:
            if node.title in self.aliases:
                return self.move(self.aliases[node.title])
            if node.title in self.arrays:
                return self.move(self.arrays[node.title]['position'])
            raise RuntimeError('specified identifier has not been declared')
        
        if type(node) == LiteralNode:
            cell_found = self.memory.allocate(node.type)
            return self.move(cell_found) + self.assign(node.value)
        
        if type(node) == ArrayNode:
            return self.visit_array_literal(node.array)
    
    def visit_do(self, node) -> None:
        result = ''
        for child in node.body:
            result += self.visit(child)
        return result

    def visit_while(self, condition, body) -> None:
        result = ''

        temp0, temp1 = self.memory.allocate(Type.INT, Type.INT)
        
        result += self.bf_parse('x[b0x]r_b0',
            t0 = temp0,
            x  = lambda: self.visit_assign(temp1, condition),
            b0 = body,
        )
        
        self.memory.rmv(temp0, temp1)
        return result
    
    def visit_if(self, condition, body, elsebody) -> None:
        result = ''

        returned, temp0, temp1, temp2 = self.memory.allocate(Type.BOOL, Type.INT, Type.INT, Type.INT)
        
        result += self.visit(condition)
        condition = self.pointer
        
        result += self.bf_parse('t0[-]+t1[-]x[b0t2[-]rv[-]r_b0[rv+t2+r_b0-]t2[r_b0+t2-]t0-x[t1+x-]]t1[x+t1-]t0[b1t2[-]rv[-]r_b1[rv+t2+r_b1-]t2[r_b1+t2-]t0-]rv',
            #                                   \=====================================/                          \=====================================/
            #                                       Set return value to if's return                                 Set return value to else's return
            t0 = temp0,
            t1 = temp1,
            t2 = temp2,
            x  = condition,
            b0 = body,
            b1 = elsebody,
            rv = returned,
        )
        
        self.memory.rmv(temp0, temp1, temp2)
        return result

    def visit_assign(self, left, right) -> None:
        result = ''

        temp0 = self.memory.allocate(Type.INT)
        
        result += self.visit(left)
        left = self.pointer
        
        result += self.visit(right)
        right = self.pointer

        result += self.bf_parse('t0[-]x[-]y[x+t0+y-]t0[y+t0-]x',
            t0 = temp0,
            x  = left,
            y  = right,
        )
        
        self.memory.rmv(temp0)
        return result
    
    def visit_array_assign(self, left, right) -> None:
        result = ''

        for i, subnode in enumerate(right.array):
            if type(subnode) == LiteralNode:
                result += self.move(self.arrays[left.array_title]['position'] + i)
                result += self.assign(subnode.value)
                continue
            result += self.visit_assign(self.arrays[left.array_title]['position'] + i, subnode)
        result += self.move(self.arrays[left.array_title]['position'])
        return result

    def visit_add_assign(self, left, right) -> None:
        result = ''

        temp0 = self.memory.allocate(Type.INT)
        
        result += self.visit(left)
        left = self.pointer
        
        result += self.visit(right)
        right = self.pointer
        
        result += self.bf_parse('t0[-]y[x+t0+y-]t0[y+t0-]x',
            t0 = temp0,
            x  = left,
            y  = right,
        )
        
        self.memory.rmv(temp0)
        return result
    
    def visit_subtract_assign(self, left, right) -> None:
        result = ''

        temp0 = self.memory.allocate(Type.INT)
        
        result += self.visit(left)
        left = self.pointer
        
        result += self.visit(right)
        right = self.pointer
        
        result += self.bf_parse('t0[-]y[x-t0+y-]t0[y+t0-]x',
            t0 = temp0,
            x  = left,
            y  = right,
        )
        
        self.memory.rmv(temp0)
        return result

    def visit_multiply_assign(self, left, right) -> None:
        result = ''

        temp0, temp1 = self.memory.allocate(Type.INT, Type.INT)
        
        result += self.visit(left)
        left = self.pointer
        
        result += self.visit(right)
        right = self.pointer
        
        result += self.bf_parse('t0[-]t1[-]x[t1+x-]t1[y[x+t0+y-]t0[y+t0-]t1-]x',
            t0 = temp0,
            t1 = temp1,
            x  = left,
            y  = right,
        )
        
        self.memory.rmv(temp0, temp1)
        return result

    def visit_divide_assign(self, left, right) -> None:
        result = ''

        temp0, temp1, temp2, temp3 = self.memory.allocate(Type.INT, Type.INT, Type.INT, Type.INT)
        
        result += self.visit(left)
        left = self.pointer
        
        result += self.visit(right)
        right = self.pointer
        
        result += self.bf_parse('t0[-]t1[-]t2[-]t3[-]x[t0+x-]t0[y[t1+t2+y-]t2[y+t2-]t1[t2+t0-[t2[-]t3+t0-]t3[t0+t3-]t2[t1-[x-t1[-]]+t2-]t1-]x+t0]x',
            t0 = temp0,
            t1 = temp1,
            t2 = temp2,
            t3 = temp3,
            x  = left,
            y  = right,
        )
        
        self.memory.rmv(temp0, temp1, temp2, temp3)
        return result
    
    def visit_relocate(self, left, right) -> None:
        result = ''

        result += self.visit(left)
        left = self.pointer
        
        result += self.visit(right)
        right = self.pointer
        
        result += self.bf_parse('x[-]y[x+y-]x',
            x = left,
            y = right,
        )
        
        return result
    
    def visit_swap(self, left, right) -> None:
        result = ''

        temp0 = self.memory.allocate(Type.INT)
        
        result += self.visit(left)
        left = self.pointer
        
        result += self.visit(right)
        right = self.pointer
        
        result += self.bf_parse('t0[-]x[t0+x-]y[x+y-]t0[y+t0-]x',
            t0 = temp0,
            x = left,
            y = right,
        )
        
        self.memory.rmv(temp0)
        return result

    def visit_op_add(self, left, right) -> None:
        result = ''

        returned, temp0 = self.memory.allocate(Type.INT, Type.INT)
        
        result += self.visit(left)
        left = self.pointer
        
        result += self.visit(right)
        right = self.pointer
        
        result += self.bf_parse('t0[-]r[-]x[r+t0+x-]t0[x+t0-]t0[-]y[r+t0+y-]t0[y+t0-]r',
            t0 = temp0,
            r  = returned,
            x  = left,
            y  = right,
        )
        
        self.memory.rmv(temp0)
        return result
    
    def visit_op_subtract(self, left, right) -> None:
        result = ''

        returned, temp0 = self.memory.allocate(Type.INT, Type.INT)
        
        result = self.visit(left)
        left = self.pointer
        
        result += self.visit(right)
        right = self.pointer
        
        result += self.bf_parse('t0[-]r[-]x[r+t0+x-]t0[x+t0-]t0[-]y[r-t0+y-]t0[y+t0-]r',
            t0 = temp0,
            r  = returned,
            x  = left,
            y  = right,
        )
        
        self.memory.rmv(temp0)
        return result

    def visit_op_multiply(self, left, right) -> None:
        result = ''

        returned, temp0, temp1 = self.memory.allocate(Type.INT, Type.INT, Type.INT)
        
        result = self.visit(left)
        left = self.pointer
        
        result += self.visit(right)
        right = self.pointer
        
        result += self.bf_parse('t0[-]r[-]x[r+t0+x-]t0[x+t0-]t0[-]t1[-]r[t1+r-]t1[y[r+t0+y-]t0[y+t0-]t1-]r',
            t0 = temp0,
            t1 = temp1,
            r  = returned,
            x  = left,
            y  = right,
        )
        
        self.memory.rmv(temp0, temp1)
        return result

    def visit_op_divide(self, left, right) -> None:
        result = ''

        returned, temp0, temp1, temp2, temp3 = self.memory.allocate(Type.INT, Type.INT, Type.INT, Type.INT, Type.INT)
        
        result += self.visit(left)
        left = self.pointer
        
        result += self.visit(right)
        right = self.pointer
        
        result += self.bf_parse('t0[-]r[-]x[r+t0+x-]t0[x+t0-]t0[-]t1[-]t2[-]t3[-]r[t0+r-]t0[y[t1+t2+y-]t2[y+t2-]t1[t2+t0-[t2[-]t3+t0-]t3[t0+t3-]t2[t1-[r-t1[-]]+t2-]t1-]r+t0]r',
            t0 = temp0,
            t1 = temp1,
            t2 = temp2,
            t3 = temp3,
            r = returned,
            x  = left,
            y  = right,
        )
        
        self.memory.rmv(temp0, temp1, temp2, temp3)
        return result
    
    def visit_op_modulus(self, left, right) -> None:
        result = ''

        returned = self.memory.allocate(Type.INT)
        temp_block = self.memory.allocate_block(6, Type.INT)

        result += self.visit(left)
        left = self.pointer
        
        result += self.visit(right)
        right = self.pointer

        result += self.bf_parse('x[-t+>+<x]t[-x+t] y[-t+>>+<<y]t[-y+t]>[>->+<[>]>[<+>-]<<[<]>-]>[-]>>[-<<<r+t>>>]r',
            t = temp_block,
            r = returned,
            x = left,
            y = right,
        )

        self.memory.rmv(*(temp_block + i for i in range(6)))
        return result

    def visit_op_equal(self, left, right) -> None:
        result = ''

        returned, temp0, temp1 = self.memory.allocate(Type.BOOL, Type.INT, Type.INT)
        
        result += self.visit(left)
        left = self.pointer
        
        result += self.visit(right)
        right = self.pointer
        
        result += self.bf_parse('t0[-]r[-]x[r+t0+x-]t0[x+t0-]t1[-]r[t1+r-]+y[t1-t0+y-]t0[y+t0-]t1[r-t1[-]]r',
            t0 = temp0,
            t1 = temp1,
            r  = returned,
            x  = left,
            y  = right,
        )
        
        self.memory.rmv(temp0, temp1)
        return result

    def visit_op_notequal(self, left, right) -> None:
        result = ''

        returned, temp0, temp1 = self.memory.allocate(Type.BOOL, Type.INT, Type.INT)
        
        result += self.visit(left)
        left = self.pointer
        
        result += self.visit(right)
        right = self.pointer
        
        result += self.bf_parse('t0[-]r[-]x[r+t0+x-]t0[x+t0-]t1[-]r[t1+r-]y[t1-t0+y-]t0[y+t0-]t1[r+t1[-]]r',
            t0 = temp0,
            t1 = temp1,
            r  = returned,
            x  = left,
            y  = right,
        )
        
        self.memory.rmv(temp0, temp1)
        return result

    def visit_op_less(self, left, right) -> None:
        result = ''

        temp_block = self.memory.allocate_block(5, Type.VOID)
        returned = self.memory.allocate(Type.BOOL)

        result += self.visit(left)
        left = self.pointer
        
        result += self.visit(right)
        right = self.pointer

        result += self.bf_parse('r[-]t[-]x[-r+t+x]t[-x+t]t>[-]>[-]+>[-]<<<y[t+>+<y-]t[y+t-]r[t+r-]+t>[>-]>[<r-t[-]>>->]<+<<[>-[>-]>[<<r-t[-]+>>->]<+<<-]r',
            t = temp_block,
            r = returned,
            x = left,
            y = right,
        )

        self.memory.rmv(*(temp_block + i for i in range(5)))
        return result

    def visit_op_lessequal(self, left, right) -> None:
        result = ''
        
        temp_block = self.memory.allocate_block(5, Type.VOID)
        returned = self.memory.allocate(Type.BOOL)

        result += self.visit(left)
        left = self.pointer
        
        result += self.visit(right)
        right = self.pointer

        result += self.bf_parse('r[-]t[-]x[-r+t+x]t[-x+t]t>[-]>[-]+>[-]<<<y[t+>+<y-]t>[<y+t>-]<r[t>+<r-]t>[>-]>[<r+t[-]>>->]<+<<[>-[>-]>[<<r+t[-]+>>->]<+<<-]r',
            t = temp_block,
            r = returned,
            x = left,
            y = right,
        )

        self.memory.rmv(*(temp_block + i for i in range(5)))
        return result

    def visit_op_greater(self, left, right) -> None:
        result = ''

        temp_block = self.memory.allocate_block(5, Type.VOID)
        returned = self.memory.allocate(Type.BOOL)

        result += self.visit(left)
        left = self.pointer
        
        result += self.visit(right)
        right = self.pointer

        result += self.bf_parse('r[-]t[-]y[-r+t+y]t[-y+t]t>[-]>[-]+>[-]<<<x[t+>+<x-]t[x+t-]r[t+r-]+t>[>-]>[<r-t[-]>>->]<+<<[>-[>-]>[<<r-t[-]+>>->]<+<<-]r',
            t = temp_block,
            r = returned,
            x = left,
            y = right,
        )

        self.memory.rmv(*(temp_block + i for i in range(5)))
        return result

    def visit_op_greaterthan(self, left, right) -> None:
        result = ''

        temp_block = self.memory.allocate_block(5, Type.VOID)
        returned = self.memory.allocate(Type.BOOL)

        result += self.visit(left)
        left = self.pointer
        
        result += self.visit(right)
        right = self.pointer

        result += self.bf_parse('r[-]t[-]y[-r+t+y]t[-y+t]t>[-]>[-]+>[-]<<<x[t+>+<x-]t>[<x+t>-]<r[t>+<r-]t>[>-]>[<r+t[-]>>->]<+<<[>-[>-]>[<<r+t[-]+>>->]<+<<-]r',
            t = temp_block,
            r = returned,
            x = left,
            y = right,
        )

        self.memory.rmv(*(temp_block + i for i in range(5)))
        return result

    def visit_op_not(self, right) -> None:
        result = ''

        temp0, returned = self.memory.allocate(Type.INT, Type.BOOL)
        
        result += self.visit(node.right)
        right = self.pointer
        
        result += self.bf_parse('t0[-]r[-]x[r+t0+x-]t0[x+t0-]r-[t0-r-]t0[r+t0-]r',
            t0 = temp0,
            r  = returned,
            x  = right,
        )
        
        self.memory.rmv(temp0)
        return result
    
    def visit_print(self, right) -> None:
        result = self.visit(right)

        if type(right) == ArrayNode:
            result += self.visit_print_array(right)

        elif self.memory[self.pointer] == Type.CHAR:
            result += self.visit_print_char(right)

        elif self.memory[self.pointer] == Type.BOOL:
            result += self.visit_print_bool(right)

        else:
            result += self.visit_print_integer(right)
        
        return result

    def visit_print_array(self, right) -> None:
        result = ''

        for value in right.array:
            result += self.visit_print(value)
        return result

    def visit_print_char(self, right) -> None:
        result = self.visit(right)
        result += self.output()
        return result

    def visit_print_integer(self, right) -> None:
        result = ''

        temp_block = self.memory.allocate_block(8, Type.VOID)
        temp = self.memory.allocate(Type.INT)
        result += self.bf_parse('t0[-]tb[-]x[-t0+tb+x]t0[-x+t0]tb>[-]>[-]+>[-]+<[>[-<-<<[->+>+<<]>[-<+>]>>]++++++++++>[-]+>[-]>[-]>[-]<<<<<[->-[>+>>]>[[-<+>]+>+>>]<<<<<]>>-[-<<+>>]<[-]++++++++[-<++++++>]>>[-<<+>>]<<]<[.[-]<]<x',
            tb = temp_block,
            t0 = temp,
            x = self.pointer
        )

        self.memory.rmv(*(temp_block + i for i in range(8)))
        return result

    def visit_print_bool(self, right) -> None:
        result = ''

        temp_block = self.memory.allocate_block(4, Type.CHAR)

        result += self.bf_parse('t[-]+>[-]<x[t>>>+++++++++++[<++++++++++>-]<++++++.--.+++.>++++[<---->-]<.[-]<<-x[t>+<x-]]t>[<x+t>-]<[>>>++++++++++[<++++++++++>-]<++.-----.+++++++++++.+++++++.>+++++[<--->-]<+.[-]<<-]',
            t = temp_block,
            x = self.pointer
        )

        self.memory.rmv(*(temp_block + i for i in range(4)))
        return result
    
    def visit_var_declaration(self, type_, right) -> None:
        result = ''

        new_variable = right.left if type(right) == BinaryOpNode else right

        if type(new_variable) == ArrayAccessNode:
            self.arrays[new_variable.array_title] = {
                'position': self.memory.allocate_block(new_variable.index.value, Type.str_to_type(type_)),
                'size': new_variable.index.value,
                'type': Type.str_to_type(type_)
            }
            result += self.move(self.arrays[new_variable.array_title]['position'])
        else:
            self.aliases[new_variable.title] = self.memory.allocate(Type.str_to_type(type_))
            result += self.move(self.aliases[new_variable.title])
            
        result += self.visit(right)
        return result
    
    def visit_array_access(self, left, right) -> None:
        # FIXME: arrays have been reformatted to concise blocks of values.
        #        in order to access variable indices, the former method
        #        of storing arrays must be recreated as a temporary block
        #        which will just return its own value, followed by deletion
        #        of such block.
        #           FORMER STORAGE: [0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0]
        #           NEW STORAGE: [1, 2, 3, 4]
        result = ''

        temp0 = self.memory.allocate(Type.INT)

        result += self.visit(node.index) or ''
        index = self.pointer

        result += self.bf_parse('t0[-]s>>[-]<<i[s>>+<<t0+i-]t0[i+t0-]s>>+[-[+>>]+[<<]>>-]>>[[-]+>>]<[-<[<<]<<+>+>>>[>>]<]<[<<]<<[->>>>[>>]<+<[<<]<<]>>>>[>>]<<[[-]<<]',
            t0 = temp0,
            s = self.arrays[node.array_title]['position'],
            i = index
        )
        result += self.left()

        self.memory.rmv(temp0)
        return result
    
    def visit_array_literal(self, array) -> None:
        # TODO: specifiy array types
        result = ''
        
        array_address = self.memory.allocate_block(len(array), Type.str_to_type(array[0].type))
        for i, array_node in enumerate(array):
            if type(array_node) == LiteralNode:
                result += self.move(array_address + i)
                result += self.assign(array_node.value)
                continue
            result += self.visit_assign(array_address + i, array_node)
        result += self.move(array_address)
        return result
    
    # Brainfuck Parsing
    
    def bf_parse(self, bf: str, **mapping: Union[Callable, List[Callable]]):
        def repl(id_str):
            if type(mapping[id_str]) == int:
                return self.move(mapping[id_str])
            if callable(mapping[id_str]):
                return mapping[id_str]()
            total = ''
            returned = 0
            for child in mapping[id_str]:
                total += self.visit(child)
                returned = self.pointer
            mapping['r_'+id_str] = returned
            return total

        i = 0
        while i < len(bf):
            character = bf[i]
            if character in LETTERS + DIGITS + '_':
                id_str = ''
                while id_str not in mapping.keys() and i < len(bf):
                    character = bf[i]
                    id_str += character
                    i += 1
                repl_str = repl(id_str)
                bf = bf.replace(id_str, repl_str, 1)
                i += len(repl_str) - len(id_str)
            else:
                i += 1
        return bf
                
    # Instructions
    
    def move(self, address_target: int) -> str:
        pointer = self.pointer
        self.pointer = address_target
        
        if address_target > pointer:
            return '>' * (address_target - pointer)
        return '<' * (pointer - address_target)
    
    def assign(self, value_target: int) -> str:
        return '[-]' + '+' * value_target

    def right(self, address_increment: int = 1) -> str:
        self.pointer += address_increment if address_increment != None else 1
        return '>' * address_increment

    def left(self, address_decrement: int = 1) -> str:
        self.pointer -= address_decrement if address_decrement != None else 1
        return '<' * address_decrement

    def add(self, value_increment: int = 1) -> str:
        return '+' * value_increment if value_increment != None else 1

    def sub(self, value_decrement: int = 1) -> str:
        return '-' * value_decrement if value_decrement != None else 1

    def output(self, output_address: int = None) -> str:
        return self.move_append(output_address, '.')

    def input(self, input_address: int = None) -> str:
        return self.move_append(input_address, ',')

    def move_append(self, address: int, symbol: str) -> str:
        if address != None:
            pointer = self.pointer
            self.pointer = address
        
            if address > pointer:
                return '>' * (address - pointer) + symbol
            if address < pointer:
                return '<' * (pointer - address) + symbol
        return symbol

def ComplexEncoder(object: Node) -> Union[dict, str]:
    if hasattr(object, 'reprJSON'):
        return object.reprJSON()
    else:
        return repr(object)

def run(filename: str, filetext: str, debug: bool = False):
    lexer = Lexer(filename, filetext)
    tokens, error = lexer.lex()
    
    if error: return None, error
    
    if debug:
        with open('debug/tokens.txt', 'w') as tokens_txt:
            tokens_txt.write('\n'.join(repr(token) for token in tokens))
        
    parser = Parser(filename, tokens)
    parsetree, error = parser.parse()
    
    if error: return None, error
    
    if debug:
        with open('debug/parsetree.json', 'w') as parsetree_json:
            json.dump(parsetree, parsetree_json, default=ComplexEncoder, indent=4)

    compiler = Compiler(parsetree)
    bf = compiler.compile()
    
    if debug:
        with open('debug/compiled.bf', 'w') as compiled_bf:
            compiled_bf.write('\n'.join(bf[i:i+64] for i in range(0, len(bf), 64)))

    return bf, None

def main():
    if len(sys.argv) > 1: file_name = sys.argv[1]
    else: file_name = 'debug/main.ms'
    
    with open(file_name, 'r') as file:
        bf, error = run(file_name, file.read(), debug = True)
    if error:
        print(error)
    else:
        import brainfuck
        brainfuck.evaluate(bf)

if __name__ == '__main__': main()
