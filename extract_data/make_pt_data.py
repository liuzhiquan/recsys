#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This program extracts data from database and write to disk.
"""

import MySQLdb
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import chardet


def write_torrents(cursor, category):
    filename = 'u%i.item' % category
    f = open(filename, 'w')
    f = sys.stdout

    cursor.execute('select id, filename, descr from torrents where category=%i' % category)
    for row in cursor.fetchall():
        torrentid, torrentname, descr = str(row[0]), row[1], row[2]
        try:
            torrentname = row[1].decode('utf8')
            descr = row[2].decode('utf8')
        except UnicodeDecodeError:
            encoding = chardet.detect(torrentname)['encoding']
            torrentname = row[1].decode(encoding)
            descr = row[2].decode(encoding)
        print >> f, '|'.join((torrentid, torrentname, descr))
        #f.write('%i|%s|%s' % (row[0], row[1].decode('utf8'), row[2].decode('utf8')))
    f.close()

def write_users(cursor):
    cursor.execute('select id,username from users')
    filename = 'u.user'
    f = open(filename, 'w')
    for row in cursor.fetchall():
        print >> f, '|'.join(str(x) for x in row)
    f.close()

def write_record(cursor):
    pass

if __name__ == '__main__':
    conn = MySQLdb.connect(user='root', passwd='', db='putao')
    cursor = conn.cursor()
    write_torrents(cursor, 402)
