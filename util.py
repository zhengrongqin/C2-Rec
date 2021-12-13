import sys
import copy
import random
import numpy as np
from collections import defaultdict


def data_partition(base_path, fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(base_path + '/data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, args, sess, batch_size = 1000):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG1 = 0.0
    HT1 = 0.0
    NDCG5 = 0.0
    HT5 = 0.0
    NDCG10 = 0.0
    HT10 = 0.0
    NDCG20 = 0.0
    HT20 = 0.0
    valid_user = 0.0

    if usernum > 30000:
        users = random.sample(range(1, usernum + 1), 30000)
    else:
        users = range(1, usernum + 1)

    input_batch_u=[]
    input_batch_seq = []
    input_batch_item_idx = []
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(args.neg_test):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        input_batch_u.append([u])
        input_batch_seq.append(seq)
        input_batch_item_idx.append(item_idx)
        if len(input_batch_u) >= batch_size:
            predictions = -model.predict(sess, np.array(input_batch_u), np.array(input_batch_seq), np.array(input_batch_item_idx))
            rank_array = predictions.argsort(axis = -1).argsort(axis = -1)[:, 0]
            for rank in rank_array:
                valid_user += 1
                if rank < 1:
                    NDCG1 += 1 / np.log2(rank + 2)
                    HT1 += 1

                if rank < 5:
                    NDCG5 += 1 / np.log2(rank + 2)
                    HT5 += 1

                if rank < 10:
                    NDCG10 += 1 / np.log2(rank + 2)
                    HT10 += 1

                if rank < 20:
                    NDCG20 += 1 / np.log2(rank + 2)
                    HT20 += 1

                if valid_user % 100 == 0:
                    print('.', end=''),
                    sys.stdout.flush()
            # reset data buffer
            input_batch_u = []
            input_batch_seq = []
            input_batch_item_idx = []

    if len(input_batch_u) != 0:
        predictions = -model.predict(sess, np.array(input_batch_u), np.array(input_batch_seq), np.array(input_batch_item_idx))
        rank_array = predictions.argsort(axis=-1).argsort(axis=-1)[:, 0]
        for rank in rank_array:
            valid_user += 1
            if rank < 1:
                NDCG1 += 1 / np.log2(rank + 2)
                HT1 += 1

            if rank < 5:
                NDCG5 += 1 / np.log2(rank + 2)
                HT5 += 1

            if rank < 10:
                NDCG10 += 1 / np.log2(rank + 2)
                HT10 += 1

            if rank < 20:
                NDCG20 += 1 / np.log2(rank + 2)
                HT20 += 1

            if valid_user % 100 == 0:
                print('.', end=''),
                sys.stdout.flush()

    print('')
    return NDCG1 / valid_user, HT1 / valid_user, \
           NDCG5 / valid_user, HT5 / valid_user, \
           NDCG10 / valid_user, HT10 / valid_user, \
           NDCG20 / valid_user, HT20 / valid_user


def evaluate_valid(model, dataset, args, sess, batch_size = 1000):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG1 = 0.0
    HT1 = 0.0
    NDCG5 = 0.0
    HT5 = 0.0
    NDCG10 = 0.0
    HT10 = 0.0
    NDCG20 = 0.0
    HT20 = 0.0
    valid_user = 0.0
    input_batch_u = []
    input_batch_seq = []
    input_batch_item_idx = []

    if usernum > 5000:
        users = random.sample(range(1, usernum + 1), 5000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(args.neg_test):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        input_batch_u.append([u])
        input_batch_seq.append(seq)
        input_batch_item_idx.append(item_idx)

        if len(input_batch_u) >= batch_size:
            predictions = -model.predict(sess, np.array(input_batch_u), np.array(input_batch_seq), np.array(input_batch_item_idx))
            rank_array = predictions.argsort(axis = -1).argsort(axis = -1)[:, 0]
            for rank in rank_array:
                valid_user += 1
                if rank < 1:
                    NDCG1 += 1 / np.log2(rank + 2)
                    HT1 += 1

                if rank < 5:
                    NDCG5 += 1 / np.log2(rank + 2)
                    HT5 += 1

                if rank < 10:
                    NDCG10 += 1 / np.log2(rank + 2)
                    HT10 += 1

                if rank < 20:
                    NDCG20 += 1 / np.log2(rank + 2)
                    HT20 += 1

                if valid_user % 100 == 0:
                    print('.', end=''),
                    sys.stdout.flush()

            # reset data buffer
            input_batch_u = []
            input_batch_seq = []
            input_batch_item_idx = []

    if len(input_batch_u) != 0:
        predictions = -model.predict(sess, np.array(input_batch_u), np.array(input_batch_seq),
                                     np.array(input_batch_item_idx))
        rank_array = predictions.argsort(axis=-1).argsort(axis=-1)[:, 0]
        for rank in rank_array:
            valid_user += 1
            if rank < 1:
                NDCG1 += 1 / np.log2(rank + 2)
                HT1 += 1

            if rank < 5:
                NDCG5 += 1 / np.log2(rank + 2)
                HT5 += 1

            if rank < 10:
                NDCG10 += 1 / np.log2(rank + 2)
                HT10 += 1

            if rank < 20:
                NDCG20 += 1 / np.log2(rank + 2)
                HT20 += 1

            if valid_user % 100 == 0:
                print('.', end=''),
                sys.stdout.flush()

    print('')
    return NDCG1 / valid_user, HT1 / valid_user, \
           NDCG5 / valid_user, HT5 / valid_user, \
           NDCG10 / valid_user, HT10 / valid_user, \
           NDCG20 / valid_user, HT20 / valid_user
