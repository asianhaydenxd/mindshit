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
    
    def allocate_array(self, size: int, type_: Type) -> int:
        array_found = self.get_array(size)
        self.use(array_found, Type.VOID)
        self.use(array_found + 1, Type.VOID)
        self.use(array_found + 2, Type.VOID)
        for i in range(size):
            self.use(array_found + i*2 + 3, type_)
            self.use(array_found + i*2 + 4, Type.VOID)
        return array_found

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
        
        if type(node) == DoNode:
            for child in node.body:
                result += self.visit(child)
            return result
        
        if type(node) == ConditionalNode:
            if node.token.full == (Tk.KW, 'while'):
                temp0, temp1 = self.memory.allocate(Type.INT, Type.INT)
                
                result += self.bf_parse('x[b0x]r_b0',
                    t0 = temp0,
                    x  = [BinaryOpNode(AddressNode(temp1), Token(Tk.OP, '='), node.condition)],
                    b0 = node.body,
                )
                
                self.memory.rmv(temp0, temp1)
                return result
            
            if node.token.full in [(Tk.KW, 'if'), (Tk.KW, 'elif')]:
                returned, temp0, temp1, temp2 = self.memory.allocate(Type.BOOL, Type.INT, Type.INT, Type.INT)
                
                result += self.visit(node.condition)
                condition = self.pointer
                
                result += self.bf_parse('t0[-]+t1[-]x[b0t2[-]rv[-]r_b0[rv+t2+r_b0-]t2[r_b0+t2-]t0-x[t1+x-]]t1[x+t1-]t0[b1t2[-]rv[-]r_b1[rv+t2+r_b1-]t2[r_b1+t2-]t0-]rv',
                    #                                   \=====================================/                          \=====================================/
                    #                                       Set return value to if's return                                 Set return value to else's return
                    t0 = temp0,
                    t1 = temp1,
                    t2 = temp2,
                    x  = condition,
                    b0 = node.body,
                    b1 = node.elsebody,
                    rv = returned,
                )
                
                self.memory.rmv(temp0, temp1, temp2)
                return result
        
        if type(node) == BinaryOpNode:
            if node.token.full == (Tk.OP, '='):
                # TODO: raise error when types do not match
                if type(node.right) == ArrayNode:
                    for i, subnode in enumerate(node.right.array):
                        if type(subnode) == LiteralNode:
                            result += self.move(self.arrays[node.left.array_title]['position'] + i*2 + 3)
                            result += self.assign(subnode.value)
                            continue
                        result += self.visit(BinaryOpNode(AddressNode(self.arrays[node.left.array_title]['position'] + i*2 + 3), Token(Tk.OP, '='), subnode))
                    result += self.move(self.arrays[node.left.array_title]['position'])
                    return result
                
                temp0 = self.memory.allocate(Type.INT)
                
                result += self.visit(node.left)
                left = self.pointer
                
                result += self.visit(node.right)
                right = self.pointer

                result += self.bf_parse('t0[-]x[-]y[x+t0+y-]t0[y+t0-]x',
                    t0 = temp0,
                    x  = left,
                    y  = right,
                )
                
                self.memory.rmv(temp0)
                return result
                    
            if node.token.full == (Tk.OP, '+='):
                temp0 = self.memory.allocate(Type.INT)
                
                result += self.visit(node.left)
                left = self.pointer
                
                result += self.visit(node.right)
                right = self.pointer
                
                result += self.bf_parse('t0[-]y[x+t0+y-]t0[y+t0-]x',
                    t0 = temp0,
                    x  = left,
                    y  = right,
                )
                
                self.memory.rmv(temp0)
                return result
                    
            if node.token.full == (Tk.OP, '-='):
                temp0 = self.memory.allocate(Type.INT)
                
                result += self.visit(node.left)
                left = self.pointer
                
                result += self.visit(node.right)
                right = self.pointer
                
                result += self.bf_parse('t0[-]y[x-t0+y-]t0[y+t0-]x',
                    t0 = temp0,
                    x  = left,
                    y  = right,
                )
                
                self.memory.rmv(temp0)
                return result
            
            if node.token.full == (Tk.OP, '*='):
                temp0, temp1 = self.memory.allocate(Type.INT, Type.INT)
                
                result += self.visit(node.left)
                left = self.pointer
                
                result += self.visit(node.right)
                right = self.pointer
                
                result += self.bf_parse('t0[-]t1[-]x[t1+x-]t1[y[x+t0+y-]t0[y+t0-]t1-]x',
                    t0 = temp0,
                    t1 = temp1,
                    x  = left,
                    y  = right,
                )
                
                self.memory.rmv(temp0, temp1)
                return result
            
            if node.token.full == (Tk.OP, '/='):
                temp0, temp1, temp2, temp3 = self.memory.allocate(Type.INT, Type.INT, Type.INT, Type.INT)
                
                result += self.visit(node.left)
                left = self.pointer
                
                result += self.visit(node.right)
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

            if node.token.full == (Tk.OP, '<-'):
                result += self.visit(node.left)
                left = self.pointer
                
                result += self.visit(node.right)
                right = self.pointer
                
                result += self.bf_parse('x[-]y[x+y-]x',
                    x = left,
                    y = right,
                )
                
                return result
            
            if node.token.full == (Tk.OP, '<->'):
                temp0 = self.memory.allocate(Type.INT)
                
                result += self.visit(node.left)
                left = self.pointer
                
                result += self.visit(node.right)
                right = self.pointer
                
                result += self.bf_parse('t0[-]x[t0+x-]y[x+y-]t0[y+t0-]x',
                    t0 = temp0,
                    x = left,
                    y = right,
                )
                
                self.memory.rmv(temp0)
                return result
            
            if node.token.full == (Tk.OP, '+'):
                returned, temp0 = self.memory.allocate(Type.INT, Type.INT)
                
                result += self.visit(node.left)
                left = self.pointer
                
                result += self.visit(node.right)
                right = self.pointer
                
                result += self.bf_parse('t0[-]r[-]x[r+t0+x-]t0[x+t0-]t0[-]y[r+t0+y-]t0[y+t0-]r',
                    t0 = temp0,
                    r  = returned,
                    x  = left,
                    y  = right,
                )
                
                self.memory.rmv(temp0)
                return result
            
            if node.token.full == (Tk.OP, '-'):
                returned, temp0 = self.memory.allocate(Type.INT, Type.INT)
                
                result = self.visit(node.left)
                left = self.pointer
                
                result += self.visit(node.right)
                right = self.pointer
                
                result += self.bf_parse('t0[-]r[-]x[r+t0+x-]t0[x+t0-]t0[-]y[r-t0+y-]t0[y+t0-]r',
                    t0 = temp0,
                    r  = returned,
                    x  = left,
                    y  = right,
                )
                
                self.memory.rmv(temp0)
                return result
            
            if node.token.full == (Tk.OP, '*'):
                returned, temp0, temp1 = self.memory.allocate(Type.INT, Type.INT, Type.INT)
                
                result = self.visit(node.left)
                left = self.pointer
                
                result += self.visit(node.right)
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
            
            if node.token.full == (Tk.OP, '/'):
                returned, temp0, temp1, temp2, temp3 = self.memory.allocate(Type.INT, Type.INT, Type.INT, Type.INT, Type.INT)
                
                result += self.visit(node.left)
                left = self.pointer
                
                result += self.visit(node.right)
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
            
            # TODO: implement modulus
            
            if node.token.full == (Tk.OP, '=='):
                returned, temp0, temp1 = self.memory.allocate(Type.BOOL, Type.INT, Type.INT)
                
                result += self.visit(node.left)
                left = self.pointer
                
                result += self.visit(node.right)
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
            
            if node.token.full == (Tk.OP, '!='):
                returned, temp0, temp1 = self.memory.allocate(Type.BOOL, Type.INT, Type.INT)
                
                result += self.visit(node.left)
                left = self.pointer
                
                result += self.visit(node.right)
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
            
            # TODO: implement more comparison operators
                
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
        
        if type(node) == FunctionOpNode:
            if node.token.full == (Tk.KW, 'print'):
                # FIXME: weird formatting, needs extensive testing
                if type(node.args[0]) == ArrayNode:
                    result += self.visit(node.args[0])
                    for _ in range(len(node.args[0].array) + 1):
                        result += self.output()
                        result += self.args[0](1)
                    return result
                return self.visit(node.args[0]) + self.output()
            
            if node.token.full == (Tk.KW, 'input'):
                return self.visit(node.args[0]) + self.input()
            
            if node.token.type == Tk.KW and node.token.value in ['int', 'char', 'bool']:
                str_to_type = {'int': Type.INT, 'char': Type.CHAR, 'bool': Type.BOOL}
                new_variable = node.args[0].left if type(node.args[0]) == BinaryOpNode else node.args[0]

                if type(new_variable) == ArrayAccessNode:
                    self.arrays[new_variable.array_title] = {
                        'position': self.memory.allocate_array(new_variable.index.value, str_to_type[node.token.value]),
                        'size': new_variable.index.value,
                        'type': str_to_type[node.token.value]
                    }
                    result += self.move(self.arrays[new_variable.array_title]['position'])
                else:
                    self.aliases[new_variable.title] = cell_found = self.memory.allocate(str_to_type[node.token.value])
                    result += self.move(self.aliases[new_variable.title])
                    
                result += self.visit(node.args[0])
                return result
            
            raise RuntimeError('unary operator not defined in compiler')
        
        if type(node) == AddressNode:
            return self.move(node.address)
        
        if type(node) == ArrayAccessNode:
            if node.array_title in self.arrays:
                temp0 = self.memory.allocate(Type.INT)

                result += self.visit(node.index) or ''
                index = self.pointer

                result += self.bf_parse('t0[-]s>>[-]<<i[s>>+<<t0+i-]t0[i+t0-]s>>+[-[+>>]+[<<]>>-]>>[[-]+>>]<[-<[<<]<<+>+>>>[>>]<]<[<<]<<[->>>>[>>]<+<[<<]<<]>>>>[>>]<<[[-]<<]',
                    t0 = temp0,
                    s = self.arrays[node.array_title]['position'],
                    i = index
                )
                result += self.left(1)

                self.memory.rmv(temp0)
                return result

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
            # TODO: specifiy array types
            array_address = self.memory.allocate_array(len(node.array), Type.INT)
            for i, subnode in enumerate(node.array):
                if type(subnode) == LiteralNode:
                    result += self.move(array_address + i*2 + 3)
                    result += self.assign(subnode.value)
                    continue
                result += self.visit(BinaryOpNode(AddressNode(array_address + i*2 + 3), Token(Tk.OP, '='), subnode))
            result += self.move(array_address)
            return result
    
    def bf_parse(self, bf: str, **mapping: Union[Callable, List[Callable]]):
        def repl(id_str):
            if type(mapping[id_str]) == int:
                return self.move(mapping[id_str])
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

    def right(self, address_increment: int) -> str:
        self.pointer += address_increment if address_increment != None else 1
        return '>' * address_increment

    def left(self, address_decrement: int) -> str:
        self.pointer -= address_decrement if address_decrement != None else 1
        return '<' * address_decrement

    def add(self, value_increment: int) -> str:
        return '+' * value_increment if value_increment != None else 1

    def sub(self, value_decrement: int) -> str:
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
