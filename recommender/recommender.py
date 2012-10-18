
import MySQLdb

conn = MySQLdb.connect(user='root', passwd='', db='putao')
cursor = conn.cursor()
cursor.execute('select count(*) from torrents where category=%i' % 402);
print cursor.fetchone()[0]

class Recommender():
    def __init__(self, category):
        cursor.execute('select id from users order by id asc')
        self.users = [row[0] for row in cursor.fetchall()]

        cursor.execute('select id from torrents where category=%s \
                        order by id asc' % category)
        self.items = [row[0] for row in cursor.fetchall()]

        ratings = dict.fromkeys(self.items)
        for item in self.items:
            cursor.execute('select userid from snatched where \
                torrentid=%i' % item)
            users = [row[0] for row in cursor.fetchall()]
            ratings[item] = dict.fromkeys(users, 1)

        print len(ratings)




if __name__ == '__main__':
    r = Recommender(402)
    
