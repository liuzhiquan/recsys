#!/usr/bin/env python
#encoding=utf8

import subprocess
import os
import MySQLdb
from math import sqrt

from recommender import *

if __name__ == '__main__':
    r = Recommender()
    categories = r.get_categories()
    a = []
    for category in categories:
        t = subprocess.Popen(args='python recommender.py 0 %s' % category, shell=True)
        a.append(t)
    for s in a:
        s.terminate()
