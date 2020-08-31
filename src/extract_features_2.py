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
        if cnt % 100000 == 0:
            print('handled', cnt)
        if cnt % 20000000 == 0:
            print('features', len(features))
    return features


def agg_features(features, feat_keys):
    aggs = {'user_id': [], 'user_id__size_agg': []}
    for idx, k in enumerate(feat_keys[:-2]):
        key = 'user_id_' + k + '_unique_agg'
        aggs[key] = []
    aggs['user_id_click_times_sum_agg'] = []
    aggs['user_id_click_times_mean_agg'] = []
    aggs['user_id_click_times_std_agg'] = []
    idx_ct = feat_keys.index('click_times')

    for user_id, feats in features.items():
        aggs['user_id'].append(user_id)
        aggs['user_id__size_agg'].append(feats[-1])
        for idx, k in enumerate(feat_keys[:-2]):
            key = 'user_id_' + k + '_unique_agg'
            aggs[key].append(len(set(feats[idx + 1])))
        aggs['user_id_click_times_sum_agg'].append(np.sum(feats[idx_ct]))
        aggs['user_id_click_times_mean_agg'].append(np.mean(feats[idx_ct]))
        aggs['user_id_click_times_std_agg'].append(np.std(feats[idx_ct]))
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
    seqs = seqs.merge(aggs, on='user_id', how='left')
    seqs.to_csv('data/user_id_agg_features.csv')
    del seqs
    gc.collect()


def gen_kfolder():
    print("Extracting Kflod feature...")
    kfold_features = ['age_{}'.format(i) for i in range(10)
                      ] + ['gender_{}'.format(i) for i in range(2)]
    print(kfold_features)

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
    for index, row in user_df.iterrows():
        if row['fold'] == 5:
            continue
        uid = str(row['user_id'])
        if uid not in features:
            continue
        for pivot in fold_feat_keys:
            for fid in range(6):
                if row['fold'] == fid:
                    continue
                clicks = features[uid][fold_feat_keys.index(pivot)]
                # add each user's age, gender together who are in other folders.
                for cid in clicks:
                    if cid not in folders[fid][pivot]:
                        folders[fid][pivot][cid] = [0 for _ in range(13)]
                    folders[fid][pivot][cid][row['age']] += 1
                    folders[fid][pivot][cid][row['gender'] + 10] += 1
                    folders[fid][pivot][cid][-1] += 1
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
        for f in kfold_features:
            k = f + '_' + pivot + '_mean'
            user_feats[k] = []
    for index, row in user_df.iterrows():
        uid = row['user_id']
        if uid not in features:
            continue
        user_feats['user_id'].append(uid)
        fold = row['fold']
        for pivot in kfold_features:
            tmp = [0 for i in range(12)]
            clicks = features[uid][fold_feat_keys.index(pivot)]
            cnt = 0
            for cid in clicks:
                for i in range(12):
                    tmp[i] += folders[fold][pivot][cid][i]
                cnt += 1
            for i in range(10):
                if cnt:
                    tmp[i] /= cnt
                k = 'gender_%d_%s_mean' % (i, pivot)
                user_feats[k].append(tmp[i])
            for i in range(2):
                if cnt:
                    tmp[i + 10] /= cnt
                k = 'gender_%d_%s_mean' % (i, pivot)
                user_feats[k].append(tmp[i + 10])
    df = pd.DataFrame(user_feats).fillna(-1)
    df.to_csv('data/user_id_fold_age_gender_means.csv')
    del df
    gc.collect()

    # write to w2v format
    print('write to w2v')
    for pivot in fold_feat_keys:
        fname = 'data/sequence_text_user_id_' + pivot + '_fold.12d'
        print(fname)
        std_df = {"cid":[]}
        for kf in kfold_features:
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
        ss.fit(std_df[kfold_features])
        std_df[kfold_features] = ss.transform(std_df[kfold_features])
        print('len std_df', pivot, len(std_df))

        f = open(fname+'.txt', 'w')
        f.write(str(len(std_df)) + ' ' + '12' + '\n')
        for item in std_df[['cid'] + kfold_features].values:
            f.write(' '.join([str(int(item[0]))] + [str(x) for x in item[1:]]) + '\n')
        f.close()
        tmp = gensim.models.KeyedVectors.load_word2vec_format(fname+'.txt', binary=False)
        pickle.dump(tmp, open(fname, 'wb'))


def merge_all():
    pass


if __name__ == "__main__":
    #gen_click_aggs()
    gen_kfolder()
    merge_all()
