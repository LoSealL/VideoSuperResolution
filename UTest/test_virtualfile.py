"""
Unit test for DataLoader.VirtualFile
"""

from VSR.DataLoader.VirtualFile import *

if __name__ == '__main__':
    """ Test File """
    file = File('.', False)
    try:
        _count = 0
        while True:
            print(str(file.read(100), 'utf-8'))
            _count += 1
            assert file.tell() == _count * 100
    except EOFError:
        print('EOF!')
        assert len(file) == file.tell()
    file = File('.', True)
    print(str(file.read(len(file)), 'utf-8'))
    print(str(file.read(len(file)), 'utf-8'))
    print(str(file.read(), 'utf-8'))
    """ Test RawFile """

    """ Test ImageFile """
