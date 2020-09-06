import os
import random
import json
import gc, csv
import pickle
import gensim
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from collections import defaultdict


def load_click_features(log_file, feat_keys):
    # user_id__size_agg, user_id_ad_id_unique_agg, user_id_creative_id_unique_agg,
    # user_id_industry_unique_agg, user_id_product_id_unique_agg, user_id_time_unique_agg
    # user_id_click_times_sum_agg, user_id_click_times_mean_agg, user_id_click_times_std_agg
    inf = open(log_file)
    reader = csv.reader(inf)
    head = next(reader)
    header = {h: idx for idx, h in enumerate(head)}
    print('head', head, header)
    features = {}
    cnt = 0
    for line in reader:
        user_id = line[header['user_id']]
        if user_id not in features:
            tmp = []
            for k in feat_keys:
                tmp.append([])
            features[user_id] = tmp
            tmp.append(0)
        features[user_id][-1] += 1
        for idx, k in enumerate(feat_keys):
            features[user_id][idx].append(int(line[header[k]]))
        cnt += 1
        if cnt % 1000000 == 0:
            print('lines readed', cnt)
    return features


def agg_features(features, feat_keys):
    aggs = {'user_id': [], 'user_id__size': []}
    for idx, k in enumerate(feat_keys[:-2]):
        key = 'user_id_' + k + '_unique'
        aggs[key] = []
    aggs['user_id_click_times_sum'] = []
    aggs['user_id_click_times_mean'] = []
    aggs['user_id_click_times_std'] = []
    idx_ct = feat_keys.index('click_times')

    for user_id, feats in features.items():
        aggs['user_id'].append(user_id)
        aggs['user_id__size'].append(feats[-1])
        for idx, k in enumerate(feat_keys[:-2]):
            key = 'user_id_' + k + '_unique'
            aggs[key].append(len(set(feats[idx + 1])))
        aggs['user_id_click_times_sum'].append(np.sum(feats[idx_ct]))
        aggs['user_id_click_times_mean'].append(np.mean(feats[idx_ct]))
        aggs['user_id_click_times_std'].append(np.std(feats[idx_ct]))
    gc.collect()
    return aggs


def sequence_click_footprint(features, feat_keys):
    f1 = 'user_id'
    seqs = {'user_id': []}
    for idx, f2 in enumerate(feat_keys):
        f_name = 'sequence_text_' + f1 + '_' + f2
        seqs[f_name] = []
    for user_id, feats in features.items():
        seqs['user_id'].append(user_id)
        for idx, f2 in enumerate(feat_keys):
            f_name = 'sequence_text_' + f1 + '_' + f2
            tmp = ' '.join([str(i) for i in feats[idx]])
            seqs[f_name].append(tmp)
    return seqs


def gen_click_aggs():
    print("Extracting aggregate feature...")
    feat_keys = [
        'ad_id', 'creative_id', 'advertiser_id', "industry", "product_id",
        'time', 'click_times', 'product_category'
    ]
    features = load_click_features('data/click.csv', feat_keys)

    # statistic ad_id, create_id ...sum, times mean, std, sum
    aggs = agg_features(features, feat_keys)
    #aggs.to_csv('data/user_id_agg_features.csv')
    #print("List aggregate feature names:")
    #print(aggs.head)
    #del aggs
    print("Extracting aggregate feature done!")

    # sequence click ids to train w2v
    print("Extracting sequence feature...")
    seqs = sequence_click_footprint(features, feat_keys)
    seqs = pd.DataFrame(seqs).fillna(-1)

    aggs = pd.DataFrame(aggs).fillna(-1)
    dense_features=['user_id__size', 'user_id_ad_id_unique', 'user_id_creative_id_unique', 'user_id_advertiser_id_unique', 'user_id_industry_unique', 'user_id_product_id_unique', 'user_id_time_unique', 'user_id_click_times_sum', 'user_id_click_times_mean', 'user_id_click_times_std']
    ss = StandardScaler()
    ss.fit(aggs[dense_features])
    aggs[dense_features] = ss.transform(aggs[dense_features])

    seqs = seqs.merge(aggs, on='user_id', how='left')
    seqs.to_csv('data/user_id_agg_features.csv')
    del seqs
    gc.collect()


def gen_kfolder():
    print("Extracting Kflod feature...")
    age_gender = ['age_{}'.format(i) for i in range(10) ] + ['gender_{}'.format(i) for i in range(2)]
    print(age_gender)

    train_df = pd.read_csv('data/train_user.csv')
    test_df = pd.read_csv('data/test_user.csv')
    train_df['fold'] = train_df.index % 5
    test_df['fold'] = 5

    user_df = train_df.append(test_df)
    fold_feat_keys = [ 'creative_id', 'ad_id', 'product_id', 'advertiser_id', 'industry' ]
    features = load_click_features('data/click.csv', fold_feat_keys)
    print("loaded  features", len(features))
    cnt = 0
    for k,v in features.items():
        print(k,v, type(k))
        cnt += 1
        if cnt == 10:
            break

    # gen k folder ages, and write to files
    # fold 0-5, static other folder means,
    # such as flod_0 = mean[fold_1, fold_2, fold_3, fold_4, fold_5]
    folders = []
    for i in range(6):
        folders.append({k: {} for k in fold_feat_keys})

    # fold_0['ad_id']['ad67890'] = [0, 0, ....]
    # fold_0['create_id']['cr1234'] = [0, 1, ....]
    cnt = 0
    for index, row in user_df.iterrows():
        if row['fold'] == 5: continue
        uid = str(row['user_id'])
        if uid not in features: continue

        for pivot in fold_feat_keys:
            for fid in range(6):
                # add each user's age, gender together which in other folders.
                if row['fold'] == fid: continue
                clicks = features[uid][fold_feat_keys.index(pivot)]
                for cid in clicks:
                    if int(cid) == -1: continue
                    if cid not in folders[fid][pivot]:
                        folders[fid][pivot][cid] = [0 for _ in range(13)]
                    folders[fid][pivot][cid][row['age']] += 1
                    folders[fid][pivot][cid][row['gender'] + 10] += 1
                    folders[fid][pivot][cid][-1] += 1
                    cnt += 1
                    if cnt % 1000000 == 0:
                        print('%s fold %d cnt %d'%( pivot, fid, cnt))

    print('calc folder means')
    # get each feature mean age, gender
    for idx, fold in enumerate(folders):
        for pivot, feat_dict in fold.items():
            cnt = 0
            for cid, value_list in feat_dict.items():
                for i in range(12):
                    value_list[i] /= value_list[-1]
                if cnt < 3:
                    print('folder', idx, pivot,  cid, value_list)
                    cnt += 1
    # each user
    print('gen each user click behavior statistic')
    user_feats = {'user_id': []}
    for pivot in fold_feat_keys:
        for f in age_gender:
            k = f + '_' + pivot + '_mean'
            user_feats[k] = []

    total = 0
    for index, row in user_df.iterrows():
        total += 1
        if total % 1000000 == 0:
            print('handled user pivot mean ', total)

        uid = str(row['user_id'])
        if uid not in features:
            continue
        user_feats['user_id'].append(uid)
        fold = row['fold']
        for pivot in fold_feat_keys:
            tmp = [0 for i in range(12)]
            clicks = features[uid][fold_feat_keys.index(pivot)]
            cnt = 0
            for cid in clicks:
                if int(cid) == -1: continue
                if cid not in folders[fold][pivot]:
                    print( 'cid not in fold', cid, type(cid), fold, pivot )
                    continue
                for i in range(12):
                    tmp[i] += folders[fold][pivot][cid][i]
                cnt += 1
            for i in range(10):
                if cnt:
                    tmp[i] /= cnt
                k = 'age_%d_%s_mean' % (i, pivot)
                user_feats[k].append(tmp[i])
            for i in range(2):
                if cnt:
                    tmp[i + 10] /= cnt
                k = 'gender_%d_%s_mean' % (i, pivot)
                user_feats[k].append(tmp[i + 10])
    df = pd.DataFrame(user_feats).fillna(-1)
    ss = StandardScaler()
    dense_features=[]
    for l in age_gender:
        for f in fold_feat_keys:
            dense_features.append(l+'_'+f+'_mean')
    ss.fit(df[dense_features])
    df[dense_features] = ss.transform(df[dense_features])
    df.to_csv('data/user_id_fold_age_gender_means.csv')
    del df
    gc.collect()

    # write to w2v format
    print('write to w2v')
    for pivot in fold_feat_keys:
        fname = 'data/sequence_text_user_id_' + pivot + '_fold.12d'
        print(fname)
        std_df = {"cid":[]}
        for kf in age_gender:
            std_df[kf] = []
        for fid in range(6):
            tmp = folders[fid][pivot]
            for item, vlist in tmp.items():
                cid_fold = int(item) * 10 + fid
                std_df['cid'].append(cid_fold)
                for i in range(10):
                    std_df['age_%d'%i].append(vlist[i])
                for i in range(2):
                    std_df['gender_%d'%i].append(vlist[i+10])
        std_df = pd.DataFrame(std_df)
        print('len std_df', pivot, len(std_df))
        ss = StandardScaler()
        ss.fit(std_df[age_gender])
        std_df[age_gender] = ss.transform(std_df[age_gender])
        print('len std_df', pivot, len(std_df))

        f = open(fname+'.txt', 'w')
        f.write(str(len(std_df)) + ' ' + '12' + '\n')
        for item in std_df[['cid'] + age_gender].values:
            f.write(' '.join([str(int(item[0]))] + [str(x) for x in item[1:]]) + '\n')
        f.close()
        tmp = gensim.models.KeyedVectors.load_word2vec_format(fname+'.txt', binary=False)
        pickle.dump(tmp, open(fname, 'wb'))


def merge_all():
    train_df = pd.read_csv('data/train_user.csv')
    test_df = pd.read_csv('data/test_user.csv')
    train_users = set([str(i) for i in train_df['user_id'].values])
    test_users = set([str(i) for i in test_df['user_id'].values])
    del train_df
    del test_df

    reader1 = open('data/user_id_agg_features.csv')
    reader2 = open('data/user_id_fold_age_gender_means.csv')
    header1 = reader1.readline().rstrip().split(',')
    header2 = reader2.readline().rstrip().split(',')
    header = header1 + header2[2:]

    writer1 = csv.writer(open('data/train.csv.tmp','w'))
    writer2 = csv.writer(open('data/test.csv.tmp','w'))
    writer1.writerow(header)
    writer2.writerow(header)
    cnt1, cnt2 = 0, 0
    feats1 = {}
    feats2 = {}
    f1_done = False
    f2_done = False
    while not ( f1_done and f2_done):
        # read N recodes
        N = 1000
        c = 0
        while c < N and not f1_done:
            line = reader1.readline().rstrip().split(',')
            if not line:
                f1_done = True
                break
            if len(line) < 3:
                continue
            for i,v in enumerate(line):
                try:
                    if '.' in v and ' ' not in v:
                        line[i] = float(v)
                except:
                    pass
            feats1[line[1]] = line[1:]
            c += 1
        c = 0
        while c < N and not f2_done:
            line = reader2.readline().rstrip().split(',')
            if not line:
                f2_done = True
                break
            if len(line) < 3:
                continue
            for i,v in enumerate(line):
                try:
                    if '.' in v and ' ' not in v:
                        line[i] = float(v)
                except:
                    pass
            feats2[line[1]] = line[2:]
            c += 1
        to_rm = []
        for f, v in feats1.items():
            if f not in feats2:
                continue
            to_rm.append(f)
            if f in train_users:
                writer1.writerow([cnt1] + v + feats2[f])
                cnt1 += 1
            elif f in test_users:
                writer2.writerow([cnt2] + v + feats2[f])
                cnt2 += 1
        for f in to_rm:
            feats1.pop(f)
            feats2.pop(f)

if __name__ == "__main__":
    gen_click_aggs()
    gen_kfolder()
    merge_all()
