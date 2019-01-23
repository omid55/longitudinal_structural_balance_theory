import preprocessing as prep
import cProfile
import pstats
import re

# or you can simply call:  python3 -m cProfile -o my.txt preprocessing.py

if __name__ == '__main__':
    # cProfile.run('re.compile("prep.main()")', 'restats')
    cProfile.run('prep.main()', 'restats')
    p = pstats.Stats('restats')
    p.strip_dirs().sort_stats(-1).print_stats()
