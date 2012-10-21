
import os
import MySQLdb
from math import sqrt


class Recommender():
    def __init__(self, category):
        self.category = category
        self.users = []
        self.items = []
        self.ratings = {}
        self.get_sim = self.cosine_sim
        self.sims = {}
    
        self.conn = MySQLdb.connect(user='root', passwd='', db='putao')
        self.cursor = self.conn.cursor()

    def load_data_from_db(self):
        # load data from database
        cursor = self.cursor
        cursor.execute('select id from users order by id asc')
        self.users = [row[0] for row in cursor.fetchall()]

        cursor.execute('select id from torrents where category=%s \
                        order by id asc' % self.category)
        self.items = [row[0] for row in cursor.fetchall()]

        ratings = dict.fromkeys(self.items)
        for item in self.items:
            cursor.execute('select userid from snatched where \
                torrentid=%i' % item)
            users = [row[0] for row in cursor.fetchall()]
            ratings[item] = dict.fromkeys(users, 1)
        self.ratings = ratings
    
    def cal_sim(self, items=None, ratings=None, N=20):
        """calculate similarities for each item in items"""
        if items is None: items = self.items
        if ratings is None: ratings = self.ratings
        
        sims = {}
        if os.path.exists('sims'):
            for line in open('sims').readlines():
                fields = line.strip().split('\t')
                item = int(fields[0])
                sim_list = []
                for sim_item_pair in fields[1:]:
                    sim, item2 = sim_item_pair.split(':')
                    sim = float(sim)
                    item2 = int(item2)
                    sim_list.append((sim, item2))
                sims[item] = sim_list
            self.sims = sims
            return

        for item1 in items:
            sim_list = []
            for item2 in ratings:
                if item1 != item2:
                    sim = self.get_sim(ratings[item1], ratings[item2])
                    sim_list.append((sim, item2))
            sim_list.sort(reverse=True)
            sims[item1] = sim_list[:N]
        self.sims = sims

        f = open('sims', 'w')
        for key in sorted(sims.keys()):
            f.write('%i\t' % key)
            sim_list = sorted(sims[key], reverse=True)[:N]
            f.write('\t'.join('%f:%i' % (sim, item) for (sim, item) in sim_list))
            f.write('\n')      
        f.close()

    def recommend(self, items, sims):
        cursor = self.cursor
        for item in items:
            #cursor.execute('select name from torrents where id=%i' % item)
            #print cursor.fetchone()[0]
            print item
            print
            sim_list = sims.get(item, [])
            for sim, recommended_item in sim_list[:20]:
                #cursor.execute('select name from torrents where id=%i' % recommended_item)
                #name = cursor.fetchone()[0]
                #print name, sim
                print sim, recommended_item
            print
            print

    def cosine_sim(self, dict1, dict2):
        numerator = 0.0
        numerator = len(set(dict1.keys()) & set(dict2.keys()))
        denominator = sqrt(sum(dict1.values())) * sqrt(sum(dict2.values()))
        if denominator == 0.0:
            return 0.0
        else:
            return numerator / denominator


if __name__ == '__main__':
    r = Recommender(402)
    r.load_data_from_db()
    r.cal_sim(r.items, r.ratings, N=20)
    r.recommend(r.items, r.sims)
    
