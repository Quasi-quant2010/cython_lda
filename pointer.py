# -*- coding: utf-8 -*-

import memoryview1 as m1
import pointer1 as p1
import pointer2 as p2
import numpy as np
import time
import pstats, cProfile

if __name__ == "__main__":

    print 'memoryview'
    cProfile.runctx("m1.main()", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()
    
    print 'pointer1'
    cProfile.runctx("p1.main()", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()

    print 'pointer2'
    cProfile.runctx("p2.main()", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()
