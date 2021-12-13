import os
import time
import argparse
import tensorflow as tf
import shutil
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import *


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', default=2021, type=int)
parser.add_argument('--base_path', default='./', type=str)
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--summary_dir', default='', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--neg_test', default=500, type=int)
parser.add_argument('--temperature', default=1.0, type=float)
# consistency learning
parser.add_argument('--con_alpha', default=0.0, type=float)
parser.add_argument('--rd_alpha', default=0.0, type=float)
parser.add_argument('--rd_reduce', default='mean', type=str)
parser.add_argument('--neg_sample_n', default=50, type=int)
parser.add_argument('--user_reg_type', default='kl', type=str)

args = parser.parse_args()

np.random.seed(args.random_seed)
tf.random.set_random_seed(args.random_seed)

train_dir = os.path.join(args.base_path, args.dataset, args.train_dir)
if not os.path.isdir(train_dir):
    os.makedirs(train_dir)
with open(os.path.join(train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join(["--{}={} \\".format(str(k), str(v)) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

dataset = data_partition(args.base_path, args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
num_batch = int(len(user_train) / args.batch_size)
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('average sequence length: %.2f' % (cc / len(user_train)))

f = open(os.path.join(train_dir, 'log.txt'), 'w')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, neg_sample_n=args.neg_sample_n, maxlen=args.maxlen, n_workers=20)
model = Model(usernum, itemnum, args)
sess.run(tf.global_variables_initializer())

model_ckp_point = os.path.join(train_dir, 'check_path')
if os.path.exists(model_ckp_point):
    shutil.rmtree(model_ckp_point)
os.makedirs(model_ckp_point)
model_saver = tf.train.Saver(max_to_keep=25)

vars_train = tf.trainable_variables()
for v in vars_train:
    print('{}, shape: {}'.format(v.name, v.shape))

if os.path.exists(os.path.join(train_dir, 'summary')):
    shutil.rmtree(os.path.join(train_dir, 'summary'))
train_writer = tf.summary.FileWriter(os.path.join(train_dir, 'summary'), sess.graph)

T = 0.0
t0 = time.time()
global_step_val = 0

try:
    f.write(','.join(['epoch', 'time'
                      , 'val_NDCG@1', 'val_HR@1', 'val_NDCG@5', 'val_HR@5', 'val_NDCG@10', 'val_HR@10', 'val_NDCG@20', 'val_HR@20'
                      , 'test_NDCG@1', 'test_HR@1', 'test_NDCG@5', 'test_HR@5', 'test_NDCG@10', 'test_HR@10', 'test_NDCG@20', 'test_HR@20']) + '\n')
    for epoch in range(1, args.num_epochs + 1):

        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()
            auc, loss, _, merged, g_step = sess.run([model.auc, model.loss, model.train_op, model.merged, model.global_step],
                                    {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                     model.is_training: True})
            global_step_val = g_step
            if g_step % 50 == 0 or g_step <= 1:
                train_writer.add_summary(merged, global_step=g_step)

        model_saver.save(sess=sess, save_path=model_ckp_point + "/epoch-{}-ckp".format(epoch), global_step=global_step_val)

        if epoch % 5 == 0 or epoch <= 1:
            t1 = time.time() - t0
            T += t1
            print('Evaluating'),
            t_test = evaluate(model, dataset, args, sess)
            t_valid = evaluate_valid(model, dataset, args, sess)
            print('epoch:%d, time:%.1f s, '
                  'valid NDCG@1: %.4f, HR@1: %.4f, NDCG@5: %.4f, HR@5: %.4f, NDCG@10: %.4f, HR@10: %.4f, NDCG@20: %.4f, HR@20: %.4f | '
                  'test NDCG@1: %.4f, HR@1: %.4f, NDCG@5: %.4f, HR@5: %.4f, NDCG@10: %.4f, HR@10: %.4f, NDCG@20: %.4f, HR@20: %.4f ' % (
            epoch, T, t_valid[0], t_valid[1], t_valid[2], t_valid[3], t_valid[4], t_valid[5], t_valid[6], t_valid[7],
                        t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5], t_test[6], t_test[7]))

            f.write(','.join(['%.4f' % x if i >= 2 else '%.1f' % x for i, x in enumerate(
                [epoch, T,
                 t_valid[0], t_valid[1], t_valid[2], t_valid[3], t_valid[4], t_valid[5], t_valid[6], t_valid[7],
                 t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5], t_test[6], t_test[7]])
             ]) + '\n')

            f.flush()
            t0 = time.time()

except Exception as e:
    sampler.close()
    f.close()
    print(e)
    exit(1)

f.close()
sampler.close()
print("Done")