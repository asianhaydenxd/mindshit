# Imports
from typing import Callable, TypeVar, Union, List, Tuple

Self = TypeVar('Self')

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
        'end',
        
        # Conditional blocks
        'if',
        'elif',
        'else',
        
        # Boolean literals
        'true',
        'false',
        
        # IO
        'out',
        'in',
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
            return '{' + f'{self.type}: {self.value}' + '}'

        return '{' + self.type + '}'

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
            
            elif self.char in LETTERS:
                tokens.append(self.make_text())
            
            elif self.char in ["'", '"']:
                token, error = self.make_char()
                if error: return [], error
                tokens.append(token)

            elif self.chars(3) == '<->':
                tokens.append(Token(Tk.OP, '<->', self.pos))
                self.next(3)

            elif self.chars(2) == '//':
                self.skip_oneline_comment()
                
            elif self.chars(2) == '/*':
                self.skip_multiline_comment()

            elif self.chars(2) == '+=':
                tokens.append(Token(Tk.OP, '+=', self.pos))
                self.next(2)

            elif self.chars(2) == '-=':
                tokens.append(Token(Tk.OP, '-=', self.pos))
                self.next(2)

            elif self.chars(2) == '->':
                tokens.append(Token(Tk.OP, '->', self.pos))
                self.next(2)
                
            elif self.char == '&':
                tokens.append(Token(Tk.OP, '&', self.pos))
                self.next()
            
            elif self.char == '=':
                tokens.append(Token(Tk.OP, '=', self.pos))
                self.next()
            
            elif self.char == ':':
                tokens.append(Token(Tk.OP, ':', self.pos))
                self.next()
            
            elif self.char == '(':
                tokens.append(Token(Tk.OP, '(', self.pos))
                self.next()
                
            elif self.char == ')':
                tokens.append(Token(Tk.OP, ')', self.pos))
                self.next()
            
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
    def __init__(self, value: int) -> None:
        self.value = value
    
    def __repr__(self) -> str:
        return str(self.__dict__)

class AddressNode:
    def __init__(self, address: int) -> None:
        self.address = address
    
    def __repr__(self) -> str:
        return str(self.__dict__)

ValueNode = Union[LiteralNode, AddressNode]

class IdentifierNode:
    def __init__(self, title) -> None:
        self.title = title
    
    def __repr__(self) -> str:
        return str(self.__dict__)

class BinaryOpNode:
    def __init__(self, left, token, right) -> None:
        self.left = left
        self.token = token
        self.right = right
    
    def __repr__(self) -> str:
        return str(vars(self))

class UnaryOpNode:
    def __init__(self, token, right) -> None:
        self.token = token
        self.right = right
    
    def __repr__(self) -> str:
        return str(vars(self))

class ConditionalNode:
    def __init__(self, token, condition, body, elsebody = []) -> None:
        self.token = token
        self.condition = condition
        self.body = body
        self.elsebody = elsebody
    
    def __repr__(self) -> str:
        return str(self.__dict__)

class MainNode:
    def __init__(self, body) -> None:
        self.body = body
    
    def __repr__(self) -> str:
        return str(self.__dict__)

class Parser:
    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.index = -1
        
        self.location = [MainNode([])]
        
        self.next()
    
    def next(self, index: int = 1) -> Token:
        self.index += index
        self.token = self.tokens[self.index] if self.index < len(self.tokens) else None
        return self.token
           
    def parse(self) -> Tuple[MainNode, Error]:
        while self.token.full != Tk.EOF:
            expr, error = self.expr()
            if error:
                return None, error
            self.location[-1].body.append(expr)
        
        return self.location[-1], None
    
    def binary_op(self, ops: List[Tuple[str]], function: Callable):
        left, error = function()
        if self.token.full in ops:
            op_token = self.token
            self.next()
            right, error = function()
            left = BinaryOpNode(left, op_token, right)
        return left, error

    def unary_op(self, ops: List[Tuple[str]], function: Callable):
        if self.token.full in ops:
            op_token = self.token
            self.next()
            right, error = function()
            return UnaryOpNode(op_token, right), error
        return function()

    def conditional_op(self, ops: List[Tuple[str]], function: Callable):
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
            
    def expr(self):
        return self.conditional_op([(Tk.KW, 'while'), (Tk.KW, 'if')],
            lambda: self.unary_op([(Tk.KW, 'out')], 
                lambda: self.binary_op([(Tk.OP, '='), (Tk.OP, '+='), (Tk.OP, '-='), (Tk.OP, '->'), (Tk.OP, '<->')], 
                    lambda: self.binary_op([(Tk.OP, ':')],
                        lambda: self.unary_op([(Tk.KW, 'in')],
                            self.factor
                        )
                    )
                )
            )
        )
            
    def factor(self) -> ValueNode:
        token = self.token
        self.next()
        
        if token.type == Tk.INT:
            return LiteralNode(token.value), None
        
        if token.type == Tk.CHAR:
            return LiteralNode(ord(token.value)), None

        if token.full == (Tk.KW, 'true'):
            return LiteralNode(1), None
        
        if token.full == (Tk.KW, 'false'):
            return LiteralNode(0), None
        
        if token.full == (Tk.OP, '&'):
            address_token = self.token
            self.next()
            return AddressNode(address_token.value), None

        if token.full == (Tk.OP, '('):
            expr = self.expr()
            if self.token.full == (Tk.OP, ')'):
                self.next()
                return expr

        if token.type == Tk.ID:
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
            
    def __iter__(self):
        return iter(self.list)
    
    def __repr__(self):
        return str(self.list)

# Infinity-like list of boolean values indicating whether each slot is used
class MemoryUsageList(InfiniteList):
    def __init__(self):
        super().__init__(False)
    
    def use(self, index):
        self[index] = True
    
    def rmv(self, index):
        self[index] = False
    
    def find_cell(self):
        for index, value in enumerate(self.list):
            if not value:
                return index
        self[len(self.list)] = False
        return len(self.list) - 1
    
    def find_array(self, size):
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
        self[len(self.list)-1+size] = False
        return len(self.list)-size
    
    def find_use_cell(self):
        cell_found = self.find_cell()
        self.use(cell_found)
        return cell_found
    
    def find_use_array(self):
        array_found = self.find_array()
        self.use(array_found)
        return array_found

class Compiler:
    def __init__(self, mainnode) -> None:
        self.mainnode = mainnode
    
    def compile(self) -> str:
        self.result = ''
        self.aliases = {}
        self.literals = {}
        self.memoryusage = MemoryUsageList()
        self.pointer = 0
        self.visit(self.mainnode)
        return self.result
    
    def visit(self, node) -> None:
        if type(node) == MainNode:
            for child in node.body:
                self.result += self.visit(child)
        
        if type(node) == ConditionalNode:
            if node.token.full == (Tk.KW, 'while'):
                result = self.visit(node.condition) + '['
                for child in node.body:
                    result += self.visit(child)
                result += self.visit(node.condition) + ']'
                return result
            
            if node.token.full in [(Tk.KW, 'if'), (Tk.KW, 'elif')]:
                temp0 = self.memoryusage.find_use_cell()
                temp1 = self.memoryusage.find_use_cell()
                result = self.cmd_move(temp0) + '[-]+'
                result += self.cmd_move(temp1) + '[-]'
                result += self.visit(node.condition) + '['
                for child in node.body:
                    result += self.visit(child)
                result += self.cmd_move(temp0) + '-'
                result += self.visit(node.condition) + '['
                result += self.cmd_move(temp1) + '+'
                result += self.visit(node.condition) + '-]]'
                result += self.cmd_move(temp1) + '['
                result += self.visit(node.condition) + '+'
                result += self.cmd_move(temp1) + '-]'
                result += self.cmd_move(temp0) + '['
                for child in node.elsebody:
                    result += self.visit(child)
                result += self.cmd_move(temp0) + '-]'
                result += self.visit(node.condition)
                self.memoryusage.rmv(temp0)
                self.memoryusage.rmv(temp1)
                return result
        
        if type(node) == BinaryOpNode:
            if node.token.full == (Tk.OP, '='):
                temp0 = self.memoryusage.find_use_cell()
                if type(node.right) == LiteralNode:
                    return self.visit(node.left) + self.cmd_set(node.right.value)
                result = self.cmd_move(temp0) + '[-]'
                result += self.visit(node.left) + '[-]'
                result += self.visit(node.right) + '['
                result += self.visit(node.left) + '+'
                result += self.cmd_move(temp0) + '+'
                result += self.visit(node.right) + '-]'
                result += self.cmd_move(temp0) + '['
                result += self.visit(node.right) + '+'
                result += self.cmd_move(temp0) + '-]'
                result += self.visit(node.left)
                self.memoryusage.rmv(temp0)
                return result
                    
            if node.token.full == (Tk.OP, '+='):
                temp0 = self.memoryusage.find_use_cell()
                if type(node.right) == LiteralNode:
                    return self.visit(node.left) + self.cmd_add(node.right.value)
                result = self.cmd_move(temp0) + '[-]'
                result += self.visit(node.right) + '['
                result += self.visit(node.left) + '+'
                result += self.cmd_move(temp0) + '+'
                result += self.visit(node.right) + '-]'
                result += self.cmd_move(temp0) + '['
                result += self.visit(node.right) + '+'
                result += self.cmd_move(temp0) + '-]'
                self.memoryusage.rmv(temp0)
                return result
                    
            if node.token.full == (Tk.OP, '-='):
                temp0 = self.memoryusage.find_use_cell()
                if type(node.right) == LiteralNode:
                    return self.visit(node.left) + self.cmd_sub(node.right.value)
                result = self.cmd_move(temp0) + '[-]'
                result += self.visit(node.right) + '['
                result += self.visit(node.left) + '-'
                result += self.cmd_move(temp0) + '+'
                result += self.visit(node.right) + '-]'
                result += self.cmd_move(temp0) + '['
                result += self.visit(node.right) + '+'
                result += self.cmd_move(temp0) + '-]'
                self.memoryusage.rmv(temp0)
                return result

            if node.token.full == (Tk.OP, '->'):
                result = self.visit(node.right) + '[-]'
                result += self.visit(node.left) + '['
                result += self.visit(node.right) + '+'
                result += self.visit(node.left) + '-]'
                result += self.visit(node.right)
                return result
            
            if node.token.full == (Tk.OP, '<->'):
                temp0 = self.memoryusage.find_use_cell()
                result = self.cmd_move(temp0) + '[-]'
                result += self.visit(node.left) + '['
                result += self.cmd_move(temp0) + '+'
                result += self.visit(node.left) + '-]'
                result += self.visit(node.right) + '['
                result += self.visit(node.left) + '+'
                result += self.visit(node.right) + '-]'
                result += self.cmd_move(temp0) + '['
                result += self.visit(node.right) + '+'
                result += self.cmd_move(temp0) + '-]'
                self.memoryusage.rmv(temp0)
                return result
                
            if node.token.full == (Tk.OP, ':'):
                if type(node.left) == IdentifierNode:
                    try:
                        self.aliases[node.left.title] = node.right.address
                    except AttributeError:
                        self.aliases[node.left.title] = node.right.right.address
                    return self.visit(node.right)
                        
            raise RuntimeError('binary operator not defined in compiler')
        
        if type(node) == UnaryOpNode:
            if node.token.full == (Tk.KW, 'out'):
                return self.visit(node.right) + self.cmd_output()
            
            if node.token.full == (Tk.KW, 'in'):
                return self.visit(node.right) + self.cmd_input()
            
            raise RuntimeError('unary operator not defined in compiler')
        
        if type(node) == AddressNode:
            return self.cmd_move(node.address)
        
        if type(node) == IdentifierNode:
            if node.title in self.aliases:
                return self.cmd_move(self.aliases[node.title])
            cell_found = self.memoryusage.find_use_cell()
            self.aliases[node.title] = cell_found
            return self.cmd_move(self.aliases[node.title])
                
    # Instructions
    
    def cmd_move(self, address_target: int) -> str:
        pointer = self.pointer
        self.pointer = address_target
        
        if address_target > pointer:
            return '>' * (address_target - pointer)
        return '<' * (pointer - address_target)
    
    def cmd_set(self, value_target: int) -> str:
        return '[-]' + '+' * value_target

    def cmd_right(self, address_increment: int) -> str:
        self.pointer += address_increment if address_increment != None else 1
        return '>' * address_increment

    def cmd_left(self, address_decrement: int) -> str:
        self.pointer -= address_decrement if address_decrement != None else 1
        return '<' * address_decrement

    def cmd_add(self, value_increment: int) -> str:
        return '+' * value_increment if value_increment != None else 1

    def cmd_sub(self, value_decrement: int) -> str:
        return '-' * value_decrement if value_decrement != None else 1

    def cmd_output(self, output_address: int = None) -> str:
        return self.move_append(output_address, '.')

    def cmd_input(self, input_address: int = None) -> str:
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

def run(filename: str, filetext: str, debug: bool = False):
    lexer = Lexer(filename, filetext)
    tokens, error = lexer.lex()
    
    if error: return None, error

    if debug: print(tokens)
        
    parser = Parser(tokens)
    ast, error = parser.parse()
    
    if error: return None, error

    compiler = Compiler(ast)
    bf = compiler.compile()
    
    if debug: print(bf)

    return bf, None

if __name__ == '__main__':
    file_name = 'main.ms'
    with open(file_name, 'r') as file:
        bf, error = run(file_name, file.read(), debug = False)
    if error:
        print(error)
    else:
        import brainfuck
        brainfuck.evaluate(bf)