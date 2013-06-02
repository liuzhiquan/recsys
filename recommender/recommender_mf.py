#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
from math import sqrt


class Recommender:

    def __init__(self):
        """ initialize recommender
        """
        pass

    def loadRecsys13(self, user_train = 'yelp_training_set/yelp_training_set_user.json',
                           business_train = 'yelp_training_set/yelp_training_set_business.json',
                           review_train = 'yelp_training_set/yelp_training_set_review.json',
                           user_test = 'yelp_test_set/yelp_test_set_user.json',
                           business_test = 'yelp_test_set/yelp_test_set_business.json',
                           review_test = 'yelp_test_set/yelp_test_set_review.json'):

        rating_u = {}
        rating_b = {}
    
        uid_list = []
        for line in open(user_train).readlines():
            user_json = json.loads(line)
            user_id = user_json['user_id']
            uid_list.append(user_id)
            rating_u[user_id] = {}
        uid_dict = dict((uid,i) for i,uid in enumerate(uid_list))
    
        bid_list = []
        b_star_list = []
        b_star_dict = {}
        for line in open(business_train).readlines():
            business_json = json.loads(line)
            business_id = business_json['business_id']
            rating_b[business_id] = {}
            bid_list.append(business_id)
            b_star = business_json['stars']
            b_star_list.append(b_star)
            b_star_dict[business_id] = float(b_star)
        bid_dict = dict((bid,i) for i,bid in enumerate(bid_list))
        
        for line in open(review_train).readlines():
            review_json = json.loads(line)
            user_id = review_json['user_id']
            business_id = review_json['business_id']
            stars = review_json['stars']
            if not user_id in uid_dict:
                uid_list.append(user_id)
                uid_dict[user_id] = len(uid_list)-1
                rating_u[user_id] = {}
            user_index = uid_dict[user_id]
            if not business_id in bid_dict:
                bid_list.append(business_id)
                bid_dict[business_id] = len(bid_list)-1
                rating_b[business_id] = {}
            business_index = bid_dict[business_id]
    
            rating_u[user_id][business_id] = float(stars)
            rating_b[business_id][user_id] = float(stars)

        self.rating_u = rating_u
        self.rating_b = rating_b

        self.num_users = 0
        self.uid_dict = {}
        for user in rating_u:
            if len(rating_u[user]) >=5:
                self.uid_dict[user] = self.num_users
                self.num_users += 1
        self.num_items = 0
        self.bid_dict = {}
        for item in rating_b:
            if len(rating_b[item]) >= 5:
                self.bid_dict[item] = self.num_items
                self.num_items += 1
        self.ratings = np.zeros((self.num_users, self.num_items))
        for user_id in self.uid_dict:
            user_index = self.uid_dict[user_id]
            for item_id in rating_u[user_id]:
                if not item_id in self.bid_dict:
                    continue
                item_index = self.bid_dict[item_id]
                self.ratings[user_index, item_index] = rating_u[user_id][item_id]
        self.haverated = (self.ratings != 0)
        self.num_ratings = self.haverated.sum()

        #load test dataset
        self.test_ratings = self.ratings
        self.test_haverated = (self.test_ratings != 0)
        self.num_test_ratings = self.test_haverated.sum()

        print self.num_ratings, self.num_users, self.num_items

        # calculate average ratings
        self.calculate_avg()


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
        self.calculate_avg()

    def calculate_avg(self):
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

    def obj_func(self, user_factors, item_factors, lambda1, lambda2, mean):
        pred_ratings = np.dot(user_factors, item_factors.T) + mean
        e = self.ratings - pred_ratings
        e = np.multiply(e, self.haverated)
        obj = np.multiply(e, e).sum()
        obj += lambda1 * np.multiply(user_factors, user_factors).sum()
        obj += lambda2 * np.multiply(item_factors, item_factors).sum()
        return obj


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
                print 'test  RMSE %f' % test_rmse

            print self.obj_func(user_factors, item_factors, lambda1, lambda1, mean)
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


    

def run_movielens():
    movielens_path = '../../movielens/ml-100k/'
    #movielens_path = '../../movielens/ml-1m/'
    r = Recommender()
    r.loadMovieLens(movielens_path, 'u2.base', 'u2.test')
    print 'train review size', r.haverated.sum()
    print 'test  review size', r.test_haverated.sum()
    print

    print 'baseline1: mean rating u train RMSE = %f' % r.get_train_rmse(r.mean1())
    print 'baseline1: mean rating u test  RMSE = %f' % r.get_test_rmse(r.mean1())
    print 'baseline2: mean rating u+bu+bi train RMSE = %f' % r.get_train_rmse(r.mean2())
    print 'baseline2: mean rating u+bu+bi test  RMSE = %f' % r.get_test_rmse(r.mean2())

    pred_ratings = r.pmf(200, 0.001, 0.1, verbose=False)
    print 'final train RMSE %f' % r.get_train_rmse(pred_ratings)
    print 'final test RMSE %f' % r.get_test_rmse(pred_ratings)
    print pred_ratings.shape
    print (pred_ratings == 0).sum()
    
    prec_at_5, prec_at_10, prec_at_20 = r.get_prec_recall(pred_ratings)
    print 'prec@5 = %f' % prec_at_5
    print 'prec@10 = %f' % prec_at_10
    print 'prec@20 = %f' % prec_at_20


def run_recsys13():
    r = Recommender()
    r.loadRecsys13()
    print 'train review size', r.haverated.sum()
    print 'test  review size', r.test_haverated.sum()
    print

    print 'baseline1: mean rating u train RMSE = %f' % r.get_train_rmse(r.mean1())
    print 'baseline1: mean rating u test  RMSE = %f' % r.get_test_rmse(r.mean1())
    print 'baseline2: mean rating u+bu+bi train RMSE = %f' % r.get_train_rmse(r.mean2())
    print 'baseline2: mean rating u+bu+bi test  RMSE = %f' % r.get_test_rmse(r.mean2())

    pred_ratings = r.pmf(100, 0.001, 0.1, verbose=True)
    print 'final train RMSE %f' % r.get_train_rmse(pred_ratings)
    print 'final test  RMSE %f' % r.get_test_rmse(pred_ratings)
    print pred_ratings.shape
    print (pred_ratings == 0).sum()
    
    prec_at_5, prec_at_10, prec_at_20 = r.get_prec_recall(pred_ratings)
    print 'prec@5 = %f' % prec_at_5
    print 'prec@10 = %f' % prec_at_10
    print 'prec@20 = %f' % prec_at_20


    # predict ratings for test set
    global_avg_rating = r.mean_rating
    review_test = 'yelp_test_set/yelp_test_set_review.json'
    f_test = open('submission_file100.csv', 'w')
    print >> f_test, '%s,%s,%s' % ('user_id', 'business_id', 'stars')
    for line in open(review_test).readlines():
        review_json = json.loads(line)
        user_id = review_json['user_id']
        business_id = review_json['business_id']
        rating = 0
        if user_id in r.rating_u and len(r.rating_u[user_id]) >= 5:
            rating += sum(r.rating_u[user_id].values()) / len(r.rating_u[user_id])
        else:
            rating += global_avg_rating

        if business_id in r.rating_b and len(r.rating_b[business_id]) >= 5:
            rating += sum(r.rating_b[business_id].values()) / len(r.rating_b[business_id])
        else:
            rating += global_avg_rating
        rating -= global_avg_rating

        # if (user, business) pair in the traing set
        if user_id in r.rating_u:
            if business_id in r.rating_u[user_id]:
                rating = r.rating_u[user_id][business_id]

        # if (user, business) pair in the predicted ratings
        if user_id in r.uid_dict and business_id in r.bid_dict:
            user_index = r.uid_dict[user_id]
            business_index = r.bid_dict[business_id]
            rating = pred_ratings[user_index, business_index]
        print >> f_test, '%s,%s,%g' % (user_id, business_id, rating)
    f_test.close()


def main():
    #run_movielens()
    run_recsys13()

if __name__ == '__main__':
    main()
