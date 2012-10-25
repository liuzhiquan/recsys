#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from math import sqrt


class Recommender:

    def __init__(self, metric='cosine'):
        """ initialize recommender
        """
        self.userdata = {}
        self.itemdata = {}
        #self.data = {}
        self.username2id = {}
        self.userid2name = {}
        self.productid2name = {}

        # average rating
        self.avg_rating = 0.0

        # for some reason I want to save the name of the metric
        self.metric = metric
        if self.metric == 'pearson':
            self.fn = self.pearson_sim
        elif self.metric == 'cosine':
            self.fn = self.cosine_sim
        elif self.metric == 'cosine_binary':
            self.fn = self.cosine_binary
        elif self.metric == 'jaccard':
            self.fn = self.jaccard_sim

    def loadMovieLens(self, path='', ratingfile='u.data',
            userfile='u.user', itemfile='u.item'):
        self.userdata = {}
        self.itemdata = {}
        #
        # first load movie ratings
        #
        # First load movie ratings into self.userdata and self.itemdata
        #
        f = open(os.path.join(path, ratingfile))
        num_ratings = 0
        for line in f:
            num_ratings += 1
            fields = line.split('\t')
            user = fields[0]
            movie = fields[1]

            #binary rating or numeric rating
            if self.metric == 'jaccard' or self.metric == 'cosine_binary':
                rating = 1
            else:
                rating = int(fields[2].strip().strip('"'))
            
            # load ratings into self.userdata
            if user in self.userdata:
                currentRatings = self.userdata[user]
            else:
                currentRatings = {}
            currentRatings[movie] = rating
            self.userdata[user] = currentRatings

            # load ratings into self.itemdata
            if movie in self.itemdata:
                currentRatings = self.itemdata[movie]
            else:
                currentRatings = {}
            currentRatings[user] = rating
            self.itemdata[movie] = currentRatings
        f.close()
        self.num_ratings = num_ratings
    
        # calculate average rating
        total_rating = 0.0
        for user, ratings in self.userdata.iteritems():
            total_rating += sum(ratings.values())
        self.avg_rating = total_rating / self.num_ratings
        
        #
        
        
        # substract average rating
        # It turns out to give worse results.
        '''
        for user, ratings in self.userdata.iteritems():
            for item, rating in ratings.iteritems():
                rating -= self.avg_rating
                self.userdata[user][item] = rating
        '''

    def pearson_sim(self, rating1, rating2):
        sum_xy = 0
        sum_x = 0
        sum_y = 0
        sum_x2 = 0
        sum_y2 = 0
        n = 0
        for key in rating1:
            if key in rating2:
                n += 1
                x = rating1[key]
                y = rating2[key]
                sum_xy += x * y
                sum_x += x
                sum_y += y
                sum_x2 += pow(x, 2)
                sum_y2 += pow(y, 2)
        if n == 0:
            return 0
        # now compute denominator
        denominator = sqrt(sum_x2 - pow(sum_x, 2) / n) * sqrt(sum_y2 - pow(sum_y, 2) / n)
        if denominator == 0:
            return 0
        else:
            return (sum_xy - (sum_x * sum_y) / n) / denominator

    def cosine_sim(self, rating1, rating2):
        denominator = sqrt(sum(value * value for value in rating1.values()))
        denominator *= sqrt(sum(value * value for value in rating2.values()))

        numerator = 0
        for key in rating1:
            if key in rating2:
                numerator += rating1[key] * rating2[key]
        return numerator / denominator
    
    def cosine_binary(self, rating1, rating2):
        denominator = sqrt(len(rating1.values()))
        denominator *= sqrt(len(rating2.values()))

        numerator = 0
        for key in rating1:
            if key in rating2:
                numerator += 1
        return numerator / denominator

    def jaccard_sim(self, rating1, rating2):
        set1 = set(rating1.keys())
        set2 = set(rating2.keys())
        n = len(set1 | set2)
        if n == 0:
            return 0.0
        else:
            return 1.0 * len(set1 & set2) / n


    def popular_based_topNitems(self, username, prefs=None, n=20):
        """returns top n popular items. It is not a personalized algorithm, 
        so the parameter username will not be used.
        """
        if prefs is None:
            prefs = self.itemdata
        return [item for (item, users) in
                sorted(prefs.items(), key=lambda x:len(x[1]), reverse=True)[:n]]

    def item_based_topNitems(self, username, prefs=None, n=20):
        """returns top n items for the user
        """
        if prefs is None:
            prefs = self.itemdata

        topNitems = {}
        totalSim = {}
        for item, rating in self.userdata[username].items():
            for (otheritem, similarity) in self.computeNearestNeighbors(prefs, item):
                if otheritem not in topNitems:
                    topNitems[otheritem] = 0
                topNitems[otheritem] += similarity * rating

                if otheritem not in totalSim:
                    totalSim[otheritem] = 0
                totalSim[otheritem] += similarity
        
        rankings = [(item, score/totalSim[item]) for item, score in topNitems.items()
                     if totalSim[item] != 0]
        rankings.sort(key=lambda x:x[1], reverse=True)
        return [item for (item, rating) in rankings]


    def computeNearestNeighbors(self, prefs, instance):
        """creates a sorted list of instances(users or items) based on their distance to instance"""
        sims = []
        for other in prefs:
            if other != instance:
                sim = self.fn(prefs[other], prefs[instance])
                sims.append((other, sim))
        # sort based on similarites -- more similar first
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims

    def user_based_topNitems3(self, username, prefs, n=20):
        """returns top n items for username with weighted sum using normalization
        这个算法结果很差很差
        """
        top_n_items = {}
        totalSim = {}
        for user, similarity in self.computeNearestNeighbors(prefs, username)[:n]:
            if similarity == 0.0:
                continue
            for item in prefs[user]:
                if item not in top_n_items:
                    top_n_items[item] = 0
                currentRating = prefs[user][item]
                top_n_items[item] += similarity * currentRating
                totalSim[item] = totalSim.get(item, 0) + similarity

        for item, rating in top_n_items.items():
            if totalSim[item] != 0:
                top_n_items[item] = rating / totalSim[item]
            else:
                top_n_items[item] = 0
        return [item for (item, rating) in 
                sorted(top_n_items.items(), key=lambda x:x[1], reverse=True)]


    def user_based_topNitems1(self, username, prefs, n=20):
        """returns top n items for username with weighted sum, no normalization
        """
        top_n_items = {}
        for user, similarity in self.computeNearestNeighbors(prefs, username)[:n]:
            for item in prefs[user]:
                if item not in top_n_items:
                    top_n_items[item] = 0
                currentRating = prefs[user][item]
                top_n_items[item] += similarity * currentRating

        #return [item for (item, rating) in 
        #        sorted(top_n_items.items(), key=lambda x:x[1], reverse=True)]
        x = sorted(top_n_items.items(), key=lambda x:x[1], reverse=True)
        #print x
        return [item for (item, rating) in x]


    def user_based_topNitems(self, username, prefs=None, n=20):
        if prefs is None:
            prefs = self.userdata
        return self.user_based_topNitems1(username, prefs, n)


def eval_cv(movielens_path, cfmethod, metric, n):

    train_ratingfiles = ['u%i.base' % i for i in range(1, 6)]
    test_ratingfiles = ['u%i.test' % i for i in range(1, 6)]
    

    results = []
    for train_ratingfile,test_ratingfile in zip(train_ratingfiles, test_ratingfiles):
        result = eval_func(movielens_path, train_ratingfile, test_ratingfile, metric, cfmethod, n)
        results.append(result)
    
    N = len(results)
    print '======final result======'
    print 'cfmethod: %s, metric: %s, n:%i' % (cfmethod, metric, n)
    print 'prec@5 = %f' % (sum(x[0] for x in results) / N)
    print 'prec@10 = %f' % (sum(x[1] for x in results) / N)
    print 'prec@20 = %f' % (sum(x[2] for x in results) / N)

    print 'recall@5 = %f' % (sum(x[3] for x in results) / N)
    print 'recall@10 = %f' % (sum(x[4] for x in results) / N)
    print 'recall@20 = %f' % (sum(x[5] for x in results) / N)
    
    print 'f1@5 = %f' % (sum(x[6] for x in results) / N)
    print 'f1@10 = %f' % (sum(x[7] for x in results) / N)
    print 'f1@20 = %f' % (sum(x[8] for x in results) / N)
    print


def eval_func(movielens_path, train_ratingfile, test_ratingfile, metric, cfmethod, n, dump=False):
    r_train = Recommender(metric)
    r_train.loadMovieLens(path=movielens_path, ratingfile=train_ratingfile)
    r_test = Recommender(metric)
    r_test.loadMovieLens(path=movielens_path, ratingfile=test_ratingfile)

    prec_at5, prec_at10, prec_at20 = 0, 0, 0
    recall_at5, recall_at10, recall_at20 = 0, 0, 0

    if cfmethod == 'user_based':
        getTopNitems = getattr(r_train, 'user_based_topNitems')
    elif cfmethod == 'item_based':
        getTopNitems = getattr(r_train, 'item_based_topNitems')
    elif cfmethod == 'popular_based':
        getTopNitems = getattr(r_train, 'popular_based_topNitems')

    for user in r_test.userdata.keys():
        hits_at5, hits_at10, hits_at20 = 0, 0, 0
        for i, item in enumerate(getTopNitems(user, n=n)[:20]):
            if item in r_test.userdata[user]:
                if i < 20:
                    hits_at20 += 1
                if i < 10:
                    hits_at10 += 1
                if i < 5:
                    hits_at5 += 1
        prec_at5 += 1.0 * hits_at5 / 5
        prec_at10 += 1.0 * hits_at10 / 10
        prec_at20 +=  1.0 * hits_at20 / 20

        N = len(r_test.userdata[user])
        recall_at5 += 1.0 * hits_at5 / N
        recall_at10 += 1.0 * hits_at10 / N
        recall_at20 += 1.0 * hits_at20 / N

    N = len(r_test.userdata)
    prec_at5 /= N
    prec_at10 /= N
    prec_at20 /= N

    recall_at5 /= N
    recall_at10 /= N
    recall_at20 /= N

    f1_at5 = 2 * (prec_at5 * recall_at5) / (prec_at5 + recall_at5)
    f1_at10 = 2 * (prec_at10 * recall_at10) / (prec_at10 + recall_at10)
    f1_at20 = 2 * (prec_at20 * recall_at20) / (prec_at20 + recall_at20)
    
    if dump:
        print movielens_path, train_ratingfile, test_ratingfile
        print 'cfmethod: %s, metric: %s, n:%i' % (cfmethod, metric, n)
        print 'prec@5 = %f' % prec_at5
        print 'prec@10 = %f' % prec_at10
        print 'prec@20 = %f' % prec_at20
    
        print 'recall@5 = %f' % recall_at5
        print 'recall@10 = %f' % recall_at10
        print 'recall@20 = %f' % recall_at20
        
        print 'f1@5 = %f' % f1_at5
        print 'f1@10 = %f' % f1_at10
        print 'f1@20 = %f' % f1_at20
        print
    
    return (prec_at5, prec_at10, prec_at20, 
            recall_at5, recall_at10, recall_at20, 
            f1_at5, f1_at10, f1_at20)


def test():
    movielens_path = '../../movielens/ml-100k/'
    cfmethods = ['user_based']
    metrics = ['cosine', 'jaccard', 'pearson', 'cosine_binary']
    for cfmethod in cfmethods:
        for metric in metrics:
            for n in [20, 100, 200, -1]: # -1 means use all the information except the last one, python trick
                #outf = open('results.100k.%s.%s.%i' % (cfmethod, metric, n), 'w')
                #sys.stdout = outf
                eval_cv(movielens_path, cfmethod, metric, n)
                #outf.close()
 
    
if __name__ == '__main__':
    eval_func('../../movielens/ml-100k', 'u1.base', 'u1.test', 'pearson', 'user_based', n=100, dump=True)

