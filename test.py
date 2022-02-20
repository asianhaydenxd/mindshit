import mindshit as ms
import brainfuck as bf

class Test:
    count = 0
    def __init__(self, file_name, output):
        Test.count += 1
        self.count = Test.count
        self.file_name = file_name
        with open('tests/' + self.file_name, 'r') as file:
            self.code = file.read()
        self.output = output
        self.test()
        
    def test(self):
        result = bf.evaluate(ms.run(self.file_name, self.code)[0], returning = True)
        try:
            assert result == self.output
        except AssertionError:
            print(f'✗ Test {self.count}: {self.file_name} failed, got \'{result}\' instead of \'{self.output}\'')
            return
        print(f'✓ Test {self.count}: {self.file_name} passed')

def main():
    # TODO: split tests to be more accurate
    Test('assignment.ms', 'aac')
    Test('opassignment.ms', '053')
    Test('relocate.ms', 'aa')
    Test('swap.ms', 'ba')
    Test('ifstatement.ms', 'abbc')
    Test('whilestatement.ms', '98765432109')
    
    print('Testing complete')

if __name__ == '__main__': main()