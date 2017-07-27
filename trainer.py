from models import bilinear
from data_loader import Data_loader
import tensorflow as tf
import numpy as np
import os

root = './dataset/movielens-100k/'

class Trainer(object):
    def __init__(self, batch_size, lr, epoch):
        
        self.user = tf.placeholder(tf.float32, [None, 30])
        self.item = tf.placeholder(tf.float32, [None, 18])
        self.rating = tf.placeholder(tf.float32, [None, 1])

        self.batch_size = batch_size
        self.max_epoch = epoch
        self.g_step = tf.Variable(0, name = 'global_step')
        self.lr = tf.train.exponential_decay(lr, self.g_step, 50000, 0.98)

        self._d = Data_loader(root)
        self.train_data = self._d.train_triple
        self.test_data = self._d.test_triple
        self.train_len = len(self.train_data)
        self.test_len = len(self.test_data)

    def build_model(self):

        self.pred_rating = bilinear(self.user, self.item)
        self.loss = tf.reduce_mean(tf.sqrt(tf.square(self.rating - self.pred_rating)))

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss, global_step = self.g_step)

        self.sess = tf.Session()

    def train(self):
        dgen = self.data_iteration(desc = 'train')

        prev_loss = 999.
        counting = 0.
        for i in range(self.max_epoch):
            batch_len = int(self.train_len / self.batch_size)
            for step in range(batch_len):
                user, item, rating = dgen.next()
                _, r_preds, loss, g_step = self.sess.run([self.train_op, self.pred_rating, self.loss, self.g_step],
                        feed_dict = {
                            self.user: user,
                            self.item: item,
                            self.rating: rating
                        })
                print("step: %d, rme: %.2f"%(g_step, loss))

                if g_step % 100 == 0:
                    test_loss = self.test(g_step)
                    if test_loss < prev_loss:
                        self.saver.save(self.sess, os.path.join("./data/", "movierating.ckpt"), global_step = g_step)
                        prev_loss = test_loss
                        counting = 0
                    else:
                        counting += 1

                if counting > 50:
                    print(" [*] Early stopping")
                    break
        print(" [*] train end")

    def test(self, i):
        dgen = self.data_iteration(desc = 'test')

        test_len = int(self.test_len / self.batch_size)
        test_loss = []
        for step in range(test_len):
            user, item, rating = dgen.next()
            loss = self.sess.run([self.loss],
                    feed_dict = {
                        self.user: user,
                        self.item: item,
                        self.rating: rating
                    })
            test_loss.append(loss)
        test_loss = np.concatenate(test_loss, axis = 0)
        print("g_step: %d, rmse: %.2f"%(i, sum(test_loss)/float(test_len)))
        return sum(test_loss) / float(test_len)

    def data_iteration(self, desc = 'Train'):
        step = 0 # step for test data

        while True:
            user = np.zeros((self.batch_size, 30), np.float32)
            item = np.zeros((self.batch_size, 18), np.float32)
            rating = np.zeros((self.batch_size, 1), np.float32)

            batch_idxs = np.zeros((self.batch_size), np.int32)
            if desc.lower() == 'train':
                batch_idxs = np.arange(self.train_len)[np.random.randint(self.train_len, size = self.batch_size)]
                data_triple = self.train_data
            else:
                batch_offset = (step * self.batch_size) % self.test_len
                if batch_offset < self.batch_size and batch_offset != 0:
                    batch_idxs[:self.batch_size - batch_offset] = np.arange((step-1) * self.batch_size, self.test_len)
                    batch_idxs[self.batch_size - batch_offset:] = np.arange(batch_offset)
                else:
                    batch_idxs = np.arange(batch_offset, batch_offset + self.batch_size)
                data_triple = self.test_data

            for idx, b in enumerate(batch_idxs):
                u_id, i_id, r = data_triple[b]
                user[idx, :] = self._d.user2vec[u_id]
                item[idx, :] = self._d.item2vec[i_id]
                rating[idx, 0] = r

            yield user, item, rating



