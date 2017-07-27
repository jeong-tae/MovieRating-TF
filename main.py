from trainer import Trainer
import tensorflow as tf
import os

batch_size = 50
lr = 0.01
epoch = 10
checkpoint_dir = './data/'

def main(_):

    train_module = Trainer(batch_size, lr, epoch)
    train_module.build_model()

    train_module.saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if ckpt:
        train_module.saver.restore(train_module.sess, ckpt)
        print(" [*] Parameter restored from %s"%ckpt)
    else:
        print(" [!] Not found checkpoint")

    train_module.sess.run(tf.global_variables_initializer())

    train_module.train()

if __name__ == '__main__':
    tf.app.run()
