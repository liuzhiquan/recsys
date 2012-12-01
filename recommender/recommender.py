#!/usr/bin/env python
#encoding=utf8

import os
import MySQLdb
from math import sqrt

DBUSER = 'root'
DBPASSWD = ''
DB = 'putao'

class Recommender():
    def __init__(self, category=0):
        self.category = category
        self.users = []
        self.items = []
        self.ratings = {}
        self.get_sim = self.cosine_sim
        self.sims = {}
        self.sim_tmpfile = 'sims-%s.tmp' % self.category
    
        self.conn = MySQLdb.connect(user=DBUSER, passwd=DBPASSWD, db=DB)
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

        self.popular_items = [item for item,ratings in sorted(ratings.items(), key=lambda x:len(ratings), reverse=True)]
    
    def cal_sim(self, items=None, ratings=None, N=20):
        """calculate similarities for each item in items"""
        if items is None: items = self.items
        if ratings is None: ratings = self.ratings
        
        sims = {}

        '''
        # if similarity matrix file exists, load the similarity from disk
        if os.path.exists(self.sim_tmpfile):
            for line in open(self.sim_tmpfile).readlines():
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
        '''

        for item1 in items:
            sim_list = []
            for item2 in ratings:
                if item1 != item2:
                    sim = self.get_sim(ratings[item1], ratings[item2])
                    sim_list.append((sim, item2))
            sim_list.sort(reverse=True)
            sims[item1] = sim_list[:N]
        self.sims = sims
        
        '''
        # store simalarity matrix into disk
        f = open(self.sim_tmpfile, 'w')
        for key in sorted(sims.keys()):
            f.write('%i\t' % key)
            sim_list = sorted(sims[key], reverse=True)[:N]
            f.write('\t'.join('%f:%i' % (sim, item) for (sim, item) in sim_list))
            f.write('\n')      
        f.close()
        '''

    def recommend(self, items, sims=None):
        """print recommendations to disk
        """
        if sims is None: 
            sims = self.sims
        cursor = self.cursor
        for item in items:
            cursor.execute('select name from torrents where id=%i' % item)
            print cursor.fetchone()[0]
            print item
            print
            sim_list = sims.get(item, [])
            for sim, recommended_item in sim_list[:20]:
                cursor.execute('select name from torrents where id=%i' % recommended_item)
                name = cursor.fetchone()[0]
                print name, recommended_item, sim
                #print sim, recommended_item
            print
            print
    
    def get_recommendations(self, user, item, n=20):
        """return n recommendations for user and item
        """
        sims = self.sims
        item_sim_list = sims.get(item, [])
        if item_sim_list:
            self.cursor.execute('select torrentid from snatched where userid = %s' % user)
            downloaded = set(x[0] for x in self.cursor.fetchall())
            recommendations = []
            for score,item in item_sim_list:
                if not item in downloaded:
                    recommendations.append(item)
            return recommendations[:n]
        else:
            return r.most_popular(n)

    def most_popular(self, n):
        return self.popular_items[:n]

    def cosine_sim(self, dict1, dict2):
        numerator = 0.0
        numerator = len(set(dict1.keys()) & set(dict2.keys()))
        denominator = sqrt(sum(dict1.values())) * sqrt(sum(dict2.values()))
        if denominator == 0.0:
            return 0.0
        else:
            return numerator / denominator

    def get_categories(self):
        cursor = self.cursor
        cursor.execute('select distinct category from torrents')
        categories = [row[0] for row in cursor.fetchall()]
        return categories

    def store_rec_to_db(self, item, recommendations):
        cursor = self.cursor
        cursor.execute('select id from rec_torrents where torrentid = %s' % item)
        result = cursor.fetchone()
        rec_str = '|'.join(str(x) for x in recommendations)
        if result:
            cursor.execute("insert into `rec_torrents` (torrentid, torrents) values (%s, '%s')" 
                % (item, rec_str))
        else:
            cursor.execute("update rec_torrents set torrents = '%s' where torrentid = %s"
                    % (rec_str, item))
    

def run_all():
    r = Recommender()
    categories = r.get_categories()
    for category in categories[:1]:
        r = Recommender(category)
        r.load_data_from_db()
        r.cal_sim(r.items, r.ratings, N=20)
        for item in r.items:
            recommendations = r.get_recommendations(0, item, 20)
            r.store_rec_to_db(item, recommendations)

def run():
    r = Recommender(402)
    r.cursor.execute('select * from rec_torrents where torrentid=14')
    result = r.cursor.fetchone()
    print result

if __name__ == '__main__':
    run_all()
