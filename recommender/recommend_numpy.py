#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from math import sqrt


class Recommender:

    def __init__(self, metric='cosine'):
        """ initialize recommender
        """
        #self.data = {}
        self.username2id = {}
        self.userid2name = {}
        self.productid2name = {}

        # average rating for each user
        self.avgRatings = {}


    def loadMovieLens(self, path='', ratingfile='u.data',
            userfile='u.user', itemfile='u.item'):
        self.userdata = {}
        self.itemdata = {}
        
        # load user ids
        userids = []
        f = open(os.path.join(path, userfile))
        for line in f.readlines():
            userid = line.split('|')[0]
            userid = int(userid)
            userids.append(userid)
        f.close()
        self.num_users = len(userids)
        
        # load item ids
        itemids = []
        f = open(os.path.join(path, itemfile))
        for line in f.readlines():
            itemid = line.split('|')[0]
            itemid = int(itemid)
            itemids.append(itemid)
        f.close()
        self.num_items = len(itemids)

        # load movie ratings into an array
        f = open(os.path.join(path, ratingfile))
        rating_lines = f.readlines()
        self.num_ratings = len(rating_lines)
        self.ratings = np.zeros((self.num_ratings, 3))
        for i,line in enumerate(rating_lines):
            fields = line.split('\t')
            user = int(fields[0])
            item = int(fields[1])
            rating = int(fields[2])
            self.ratings[i, :] = np.array([user, item, rating])
        f.close()

        # calculate average rating, column 3 is rating
        self.average = self.ratings[:,2].sum() / self.num_ratings
        print self.average

    def pmf(self, max_epoch, epsilon, lambda_x):
        """ max_epoch is iterating times. epsilon is learning rate
        %  lambda_x is regularization parameter 
        """
        num_features = 10
        user_factor = 0.1 * np.random.random((self.num_users, num_features)) # user feature vectors
        item_factor = 0.1 * np.random.random((self.num_items, num_features)) # moive feature vectors
        for epoch in range(max_epoch):
            print 'epoch %i' % epoch
            N = self.num_ratings # number of training instances
        print user_factor
        print item_factor
        

if __name__ == '__main__':
    movielens_path = '../movielens/ml-100k/'
    r = Recommender()
    r.loadMovieLens(movielens_path)
    r.pmf(50, 50, 0.1)
    
