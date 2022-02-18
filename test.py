import mindshit as ms
import brainfuck as bf
import sys

def test(test_num: int, purpose: str, text: str, expected_bf: str):
    details = len(sys.argv) > 1 and sys.argv[1] == 'details'
    result = ms.run('<test>', text)[0]
    try:
        assert result == '>' * ms.RAM_SIZE + expected_bf
    except AssertionError:
        print(f'✗ Test {test_num} failed' +  (f': For {purpose}, {text} got {result[ms.RAM_SIZE:]}, expected {expected_bf}' if details else ''))
        return
    print(f'✓ Test {test_num} passed' + (f': For {purpose}, {text} got {result[ms.RAM_SIZE:]}' if details else ''))

class Test:
    count = 0
    def __init__(self, label, code, output):
        Test.count += 1
        self.count = Test.count
        self.label = label
        self.code = code
        self.output = output
        
    def test(self):
        result = bf.evaluate(ms.run('test.ms', self.code)[0], returning = True)
        try:
            assert result == self.output
        except AssertionError:
            print(f'✗ Test {self.count}: {self.label} failed, got \'{result}\' instead of \'{self.output}\'')
            return
        print(f'✓ Test {self.count}: {self.label} passed')

def main():
    Test('Assignment', 'char1: &0 = "a" out char1', 'b').test()
    
    # test(1, 'Assignment', '&0 = 1', '[-]+')
    # test(2, 'Multiple assignments', '&0 = 1 &1 = 2', '[-]+>[-]++')
    # test(3, 'Operator assignments', '&0 += 1', '+')
    # test(4, 'Output cells', '&0 = 1 out &0', '[-]+.')
    # test(5, 'Output statements', 'out &0 = 1', '[-]+.')
    # test(6, 'Aliasing', 'a: &0 a = 1', '[-]+')
    # test(7, 'Output alias', 'a: &0 out a', '.')
    # test(8, 'Alias statement', 'a: &0 = 1 a += 1', '[-]++')
    # test(9, 'Output alias definition', 'out a: &0', '.')
    # test(10, 'Output alias statement', 'out a: &0 = 1', '[-]+.')
    # test(11, 'Relocating', '&0 = 1 &0 -> &1', '[-]+>[-]<[>+<-]>')
    # test(12, 'Relocating alias', 'a: &0 = 1 a -> &1', '[-]+>[-]<[>+<-]>')
    # test(13, 'Relocating alias to alias', 'a: &0 = 1 b: &1 a -> b', '[-]+>[-]<[>+<-]>')

if __name__ == '__main__':
    main()
    print('Testing complete')