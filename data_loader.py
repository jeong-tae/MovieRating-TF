import numpy as np
import pickle

class Data_loader(object):
    def __init__(self, root = './dataset/movielens-100k/', pkl_path = './data/movierating.pkl'):
        self.root = root

        self.occ2vec = dict()
        self.genre2vec = dict()
        self.user2vec = dict()
        self.item2vec = dict()

        self.read_occupation()
        self.read_genre()
        self.read_user()
        self.read_item()

        self.train_triple = self.data_ready('ua.base')
        self.test_triple = self.data_ready('ua.test')

    def read_occupation(self):
        occupations = open(self.root + 'u.occupation', 'r').readlines()
        for idx, occ in enumerate(occupations):
            one_hot = np.zeros((len(occupations)), dtype = np.float32)
            one_hot[idx] = 1
            self.occ2vec[occ.strip()] = one_hot
            
    def read_genre(self):
        genres = open(self.root + 'u.genre', 'r').readlines()
        for idx, genre in enumerate(genres):
            one_hot = np.zeros((len(genres)), dtype = np.float32)
            one_hot[idx] = 1.
            gen = genre.split('|')[0]
            self.genre2vec[gen] = one_hot
    
    def read_user(self):
        users = open(self.root + 'u.user', 'r').readlines()
        for user in users:
            user_info = user.strip().split('|')
            _id = int(user_info[0])
            _age = int(user_info[1])
            _sex = user_info[2].strip()
            _occupation = user_info[3]
            _age_vector = np.zeros((7), dtype = np.float32)
            _sex_vector = np.zeros((2), dtype = np.float32)
            if _age < 18:
                _age_vector[0] = 1.
            elif 18 <= _age and _age < 25:
                _age_vector[1] = 1.
            elif 25 <= _age and _age < 34:
                _age_vector[2] = 1.
            elif 34 <= _age and _age < 44:
                _age_vector[3] = 1.
            elif 44 <= _age and _age < 49:
                _age_vector[4] = 1.
            elif 49 <= _age and _age < 55:
                _age_vector[5] = 1.
            else:
                _age_vector[6] = 1.

            if _sex == 'M':
                _sex_vector[0] = 1.
            else:
                _sex_vector[1] = 1.
            _occ_vector = self.occ2vec[_occupation]
            user_feature = np.concatenate((_age_vector, _sex_vector, _occ_vector), axis = 0)
            self.user2vec[_id] = user_feature

    def read_item(self):
        items = open(self.root + 'u.item', 'r').readlines()
        for item in items:
            item_info = item.strip().split('|')
            _id = int(item_info[0])
            _item_vector = np.array(item_info[6:], dtype = np.float32)
            self.item2vec[_id] = _item_vector

    def data_ready(self, path):
        data_triple = []
        data = open(self.root + path, 'r').readlines()
        for datum in data:
            u_id, i_id, rating, stamp = datum.strip().split('\t')
            data_triple.append([int(u_id), int(i_id), float(rating)])
        
        return np.array(data_triple)
