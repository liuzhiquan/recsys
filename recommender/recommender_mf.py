#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from math import sqrt


class Recommender:

    def __init__(self):
        """ initialize recommender
        """
        pass

    def loadMovieLens(self, path, trainfile, testfile,
            userfile='u.user', itemfile='u.item'):

        if path.find('ml-100k') >= 0:
            delimiter = '|'
        elif path.find('ml-1m') >= 0:
            delimiter = '::'
            userfile = 'users.dat'
            itemfile = 'movies.dat'

        self.userdata = {}
        self.itemdata = {}
        
        # load user ids
        userids = {}
        f = open(os.path.join(path, userfile))
        for i, line in enumerate(f.readlines()):
            userid = line.split(delimiter)[0]
            userid = int(userid)
            userids[userid] = i
        f.close()
        self.num_users = len(userids)
        
        # load item ids
        itemids = {}
        f = open(os.path.join(path, itemfile))
        for i, line in enumerate(f.readlines()):
            itemid = line.split(delimiter)[0]
            itemid = int(itemid)
            itemids[itemid] = i
        f.close()
        self.num_items = len(itemids)

        # load movie ratings for training
        f = open(os.path.join(path, trainfile))
        rating_lines = f.readlines()
        self.num_ratings = len(rating_lines)
        self.ratings = np.zeros((self.num_users, self.num_items))
        for i,line in enumerate(rating_lines):
            fields = line.split('\t')
            user = int(fields[0])
            item = int(fields[1])
            rating = int(fields[2])

            user_index = userids[user] # turn user id into index
            item_index = itemids[item]
            self.ratings[user_index, item_index] = rating
        f.close()
        self.haverated = (self.ratings != 0)

        # load movie ratings for test
        f = open(os.path.join(path, testfile))
        rating_lines = f.readlines()
        self.num_test_ratings = len(rating_lines)
        self.test_ratings = np.zeros((self.num_users, self.num_items))
        for i,line in enumerate(rating_lines):
            fields = line.split('\t')
            user = int(fields[0])
            item = int(fields[1])
            rating = int(fields[2])

            user_index = userids[user] # turn user id into index
            item_index = itemids[item]
            self.test_ratings[user_index, item_index] = rating
        f.close()
        self.test_haverated = (self.test_ratings != 0)

        # calculate average ratings
        self.mean_rating = self.ratings.sum() / self.num_ratings
        print 'mean_rating = %f' % self.mean_rating

        self.mean_user_ratings = np.zeros(self.num_users)
        for i,user in enumerate(self.ratings):
            num = sum(user != 0)
            if num != 0:
                self.mean_user_ratings[i] = user.sum() / num
            else:
                self.mean_user_ratings[i] = self.mean_rating

        self.mean_item_ratings = np.zeros(self.num_items)
        for i,item in enumerate(self.ratings.T):
            num = sum(item != 0.0)
            if num != 0:
                self.mean_item_ratings[i] = item.sum() / num
            else:
                self.mean_item_ratings[i] = self.mean_rating

    def mean1(self):
        return np.tile(self.mean_rating, self.ratings.shape)

    def mean2(self):
        pred_ratings = (np.tile(self.mean_user_ratings, (self.num_items, 1)).T
                     + np.tile(self.mean_item_ratings, (self.num_users, 1))
                     - np.tile(self.mean_rating, self.ratings.shape))
        return pred_ratings

    def pmf(self, max_epoch, epsilon, lambda1, verbose=True):
        """ max_epoch is iterating times. epsilon is learning rate
        lambda1 is regularization parameter 
        """
        num_features = 10
        user_factors = 0.1 * np.random.random((self.num_users, num_features)) # user feature vectors
        item_factors = 0.1 * np.random.random((self.num_items, num_features)) # moive feature vectors
        
        if verbose is True:
            print 'max_epoch%i-epsilon%f-lambda%f.tmp' % (max_epoch, epsilon, lambda1)

        mean = self.mean2()
        #ratings = self.ratings - self.mean_rating
        ratings = self.ratings - mean

        for epoch in range(max_epoch):
            pred_ratings = np.dot(user_factors, item_factors.T)
            error = self.haverated * (pred_ratings - ratings)

            user_gradient = np.dot(error, item_factors) + lambda1 * user_factors
            item_gradient = np.dot(error.T, user_factors) + lambda1 * item_factors
            
            user_factors -= epsilon * user_gradient
            item_factors -= epsilon * item_gradient
            
            if verbose is True:
                print 'epoch %i' % epoch
                train_rmse = sqrt((error * error).sum() / self.num_ratings)
                print 'train RMSE %f' % train_rmse

                pred_ratings = pred_ratings + mean
                pred_ratings[np.nonzero(pred_ratings > 5)] = 5
                pred_ratings[np.nonzero(pred_ratings < 1)] = 1
                test_error = self.test_haverated * (pred_ratings - self.test_ratings)
                test_rmse = sqrt((test_error * test_error).sum() / self.num_test_ratings)
                print 'test RMSE %f' % test_rmse

        pred_ratings = np.dot(user_factors, item_factors.T) + mean
        return pred_ratings
    
    def get_train_rmse(self, pred_ratings):
        errors = pred_ratings - self.ratings
        return sqrt((self.haverated * errors * errors).sum() / self.num_ratings)

    def get_test_rmse(self, pred_ratings):
        errors = pred_ratings - self.test_ratings
        return sqrt((self.test_haverated * errors * errors).sum() / self.num_test_ratings)

    def get_prec_recall(self, pred_ratings):
        """原来svd本来就不适合用来做prec和recall的预测
        """
        pred_ratings *= self.test_haverated
        pred_indices = (pred_ratings[:, ::-1]).argsort(axis=1)
        prec_at_5 = np.zeros(self.num_users)
        prec_at_10 = np.zeros(self.num_users)
        prec_at_20 = np.zeros(self.num_users)
        for i,pred_items in enumerate(pred_indices):
            true_items = self.test_ratings[i, :].nonzero()[0]
            prec_at_5[i] = len(set(pred_items[:5]) & set(true_items))
            prec_at_10[i] = len(set(pred_items[:10]) & set(true_items))
            prec_at_20[i] = len(set(pred_items[:20]) & set(true_items))
        prec_at_5 /= 5.0
        prec_at_10 /= 10.0
        prec_at_20 /= 20.0
        return prec_at_5.mean(), prec_at_10.mean(), prec_at_20.mean()

if __name__ == '__main__':
    movielens_path = '../../movielens/ml-100k/'
    #movielens_path = '../../movielens/ml-1m/'
    r = Recommender()
    r.loadMovieLens(movielens_path, 'u1.base', 'u1.test')

    print 'baseline1: mean rating u train RMSE = %f' % r.get_train_rmse(r.mean1())
    print 'baseline1: mean rating u test  RMSE = %f' % r.get_test_rmse(r.mean1())
    print 'baseline2: mean rating u+bu+bi train RMSE = %f' % r.get_train_rmse(r.mean2())
    print 'baseline2: mean rating u+bu+bi test  RMSE = %f' % r.get_test_rmse(r.mean2())

    pred_ratings = r.pmf(200, 0.001, 0.1, verbose=False)
    print 'final train RMSE %f' % r.get_train_rmse(pred_ratings)
    print 'final test RMSE %f' % r.get_test_rmse(pred_ratings)
    
    prec_at_5, prec_at_10, prec_at_20 = r.get_prec_recall(pred_ratings)
    print 'prec@5 = %f' % prec_at_5
    print 'prec@10 = %f' % prec_at_10
    print 'prec@20 = %f' % prec_at_20

