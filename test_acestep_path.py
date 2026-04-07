import sys
print('Python path:')
for p in sys.path:
    print(f'  {p}')
print('\nTrying to import acestep...')
import acestep
print(f'acestep location: {acestep.__file__}')
