from modules import *


class Model():
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen, args.neg_sample_n))
        pos = self.pos
        neg = self.neg
        mask_bool = tf.not_equal(self.input_seq, 0)
        mask = tf.expand_dims(tf.to_float(mask_bool), -1)
        batch_size = tf.shape(self.input_seq)[0]

        with tf.variable_scope("SASRec", reuse=tf.AUTO_REUSE):
            # sequence embedding, item embedding table
            seq_input, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=False,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )

            # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [batch_size, 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )
            seq_input  = seq_input + t

            seq_encoder = self.user_encoder(input_seq = seq_input,
                                         dropout_rate = args.dropout_rate,
                                         mask = mask,
                                         num_blocks = args.num_blocks,
                                         hidden_units = args.hidden_units,
                                         num_heads = args.num_heads)

            seq_encoder_second = self.user_encoder(input_seq = seq_input,
                                         dropout_rate = args.dropout_rate,
                                         mask = mask,
                                         num_blocks = args.num_blocks,
                                         hidden_units = args.hidden_units,
                                         num_heads = args.num_heads)

            self.seq = seq_encoder
            self.seq_second = seq_encoder_second
            self.seq_l2 = tf.nn.l2_normalize(seq_encoder, axis=-1)
            self.seq_second_l2 = tf.nn.l2_normalize(seq_encoder_second, axis=-1)

        pos = tf.reshape(pos, [batch_size * args.maxlen])
        neg = tf.reshape(neg, [batch_size * args.maxlen, args.neg_sample_n])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
        seq_emb = tf.reshape(self.seq, [batch_size * args.maxlen, args.hidden_units])
        seq_emb_second = tf.reshape(self.seq_second, [batch_size * args.maxlen, args.hidden_units])
        seq_emb_l2 = tf.reshape(self.seq_l2, [batch_size * args.maxlen, args.hidden_units])
        seq_emb_second_l2 = tf.reshape(self.seq_second_l2, [batch_size * args.maxlen, args.hidden_units])

        # test -----------------------------------------------------------
        self.test_item = tf.placeholder(tf.int32, shape=(None, args.neg_test + 1))
        test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        test_seq_emb = tf.expand_dims(self.seq[:, -1, :], 1)
        self.test_logits = tf.matmul(test_seq_emb, test_item_emb, transpose_b=True)
        self.test_logits = tf.reshape(self.test_logits, [batch_size, args.neg_test + 1])
        # =====================================loss==================================================
        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [batch_size * args.maxlen])

        with tf.variable_scope("basic_loss"):
            # prediction layer
            self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1, keepdims=True)
            self.neg_logits = tf.reshape(tf.matmul(neg_emb, tf.expand_dims(seq_emb, -1)), [batch_size * args.maxlen, args.neg_sample_n])
            logits_first = tf.concat([self.pos_logits, self.neg_logits], axis=-1)
            logits_first = tf.clip_by_value(logits_first, clip_value_min=-80, clip_value_max=80)
            prob_first = tf.nn.softmax(logits_first)
            self.loss_first = self.get_softmax_loss(prob=prob_first, is_target= istarget)

        with tf.variable_scope("basic_loss2"):
            self.pos_logits_second = tf.reduce_sum(pos_emb * seq_emb_second, -1, keepdims=True)
            self.neg_logits_second = tf.reshape(tf.matmul(neg_emb, tf.expand_dims(seq_emb_second, -1)), [batch_size * args.maxlen, args.neg_sample_n])
            logits_second = tf.concat([self.pos_logits_second, self.neg_logits_second], axis=-1)
            logits_second = tf.clip_by_value(logits_second, clip_value_min=-80, clip_value_max=80)
            prob_second = tf.nn.softmax(logits_second)
            self.loss_second = self.get_softmax_loss(prob=prob_second, is_target=istarget)

        if args.rd_alpha != 0.0:
            self.loss = (self.loss_first + self.loss_second) / 2.0
        else:
            self.loss = self.loss_first

        # seq padding bool
        seq_bool = tf.reshape(mask_bool, [-1])
        seq_emb_l2_not_pad = tf.boolean_mask(seq_emb_l2, seq_bool)
        seq_emb_second_l2_not_pad = tf.boolean_mask(seq_emb_second_l2, seq_bool)
        with tf.variable_scope("ur_loss"):
            if args.con_alpha <= 0:
                self.ur_loss = 0.0
            elif args.user_reg_type == 'cosine':
                cosine_sim = (tf.reduce_sum(seq_emb_l2_not_pad * seq_emb_second_l2_not_pad, axis=-1) + 1) / 2.0
                cosine_sim = tf.clip_by_value(cosine_sim, 0, 2.0 - 1e-10)
                self.ur_loss = - tf.reduce_mean(tf.log(cosine_sim + 1e-10))
            elif args.user_reg_type == 'l2':
                self.ur_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(seq_emb_l2_not_pad - seq_emb_second_l2_not_pad), axis=-1)))
            elif args.user_reg_type == 'kl':
                tmp_size, _ = get_shape_list(seq_emb_l2_not_pad)
                user_inner_match1 = tf.matmul(seq_emb_l2_not_pad, seq_emb_l2_not_pad, transpose_b=True) * 5 - 10000 * tf.eye(tmp_size)
                user_inner_match2 = tf.matmul(seq_emb_second_l2_not_pad, seq_emb_second_l2_not_pad, transpose_b=True) * 5 - 10000 * tf.eye(tmp_size)
                user_inner_match_prob1 = tf.nn.softmax(user_inner_match1)
                user_inner_match_prob2 = tf.nn.softmax(user_inner_match2)
                user_inner_match_prob1 = tf.clip_by_value(user_inner_match_prob1, 1e-10, 1e10)
                user_inner_match_prob2 = tf.clip_by_value(user_inner_match_prob2, 1e-10, 1e10)
                self.ur_loss = self.get_r_dropout_loss(user_inner_match_prob1, user_inner_match_prob2, 'mean')
            elif args.user_reg_type == 'cl':
                seq_emb_union = tf.concat([seq_emb_l2_not_pad, seq_emb_second_l2_not_pad], axis=0)
                con_mask = tf.eye(tf.shape(seq_emb_union)[0])
                con_sim = tf.matmul(seq_emb_union, seq_emb_union, transpose_b=True)
                self.ur_loss, _ = self.weight_info_nce(sim=con_sim, temperature=args.temperature, mask=con_mask)
            else:
                self.ur_loss = 0.0

        self.loss += self.ur_loss * args.con_alpha

        # r-dropout loss----------------------------------------------
        with tf.variable_scope("rd_loss"):
            if args.rd_alpha > 0:
                self.rd_loss = self.get_r_dropout_loss(prob1=prob_first, prob2=prob_second, reduce=args.rd_reduce, w = istarget)
            else:
                self.rd_loss = 0.0
        self.loss += self.rd_loss * args.rd_alpha

        tf.summary.scalar('basic_loss', self.loss_first, family='loss')
        tf.summary.scalar('loss', self.loss, family='loss')
        tf.summary.scalar('ur_loss', self.ur_loss, family='loss')
        tf.summary.scalar('rd_loss', self.rd_loss, family='loss')
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits[:, :1]) + 1) / 2) * tf.reshape(istarget, [-1, 1])
        ) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            tvars = tf.trainable_variables()
            gvs = self.optimizer.compute_gradients(self.loss, tvars)
            capped_gvs = [(tf.clip_by_norm(grad, 8), var) for grad, var in gvs]
            self.train_op = self.optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq, item_idx):
        return sess.run(self.test_logits,
                        {self.u: u, self.input_seq: seq, self.test_item: item_idx, self.is_training: False})\

    def user_encoder(self, input_seq, dropout_rate, mask, num_blocks, hidden_units, num_heads, reuse=None):
        with tf.variable_scope("user_encoder", reuse=reuse):
            seq = tf.layers.dropout(input_seq, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))
            seq *= mask
            seq = normalize(seq, scope='input_ln')
            # Build blocks
            for i in range(num_blocks):
                with tf.variable_scope("num_blocks_%d" % i, reuse=reuse):
                    # Self-attention
                    seq = multihead_attention(queries=seq,
                                                   keys=seq,
                                                   num_units=hidden_units,
                                                   num_heads=num_heads,
                                                   dropout_rate=dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")
                    seq = normalize(seq, scope='attention_out_ln')
                    # Feed forward
                    seq = feedforward(seq, num_units=[hidden_units, hidden_units],
                                           dropout_rate=dropout_rate, is_training = self.is_training)
                    seq *= mask
                    seq = normalize(seq, scope='feedforward_out_ln')
            return seq

    def weight_info_nce(self, sim, temperature=1.0, weight=None, mask=None, name='weight_info_nce'):
        with tf.variable_scope(name_or_scope=name):
            batch_size, col_size = get_shape_list(sim)
            tn = batch_size // 2
            sim_t = sim / temperature
            idx =tf.range(tn)
            idx = tf.reshape(tf.concat([idx + tn, idx], axis=-1), [-1, 1])
            if mask is not None:
                sim_t += -100000 * mask
            prob_sim = tf.nn.softmax(sim_t, axis=-1)
            diag_part_sim = tf.batch_gather(prob_sim, idx)
            if weight is None:
                loss = tf.reduce_mean(-tf.log(diag_part_sim))
            else:
                loss = tf.reduce_sum(-tf.log(diag_part_sim) * weight) / tf.reduce_sum(weight)
            return loss, prob_sim

    def get_softmax_loss(self, prob, is_target):
        prob_t = tf.reshape(prob[:, :1], [-1])
        return tf.reduce_sum( - tf.log(prob_t + 1e-10) * is_target) / (tf.reduce_sum(is_target) + 1e-10)

    def kl_divergence(self, p1, p2):
        return tf.reduce_sum(p1 * (tf.log(p1 + 1e-10) - tf.log(p2 + 1e-10)), axis=-1)

    def get_r_dropout_loss(self, prob1, prob2, reduce='', w=None):
        if w is None:
            w = tf.ones_like(prob1)
        if reduce == 'sum':
            kl_loss = tf.reduce_sum(self.kl_divergence(prob1, prob2) * w)
            kl_loss += tf.reduce_sum(self.kl_divergence(prob2, prob1) * w)
        else:
            kl_loss = tf.reduce_sum(self.kl_divergence(prob1, prob2) * w) / tf.reduce_sum(w)
            kl_loss += tf.reduce_sum(self.kl_divergence(prob2, prob1) * w) / tf.reduce_sum(w)
        return kl_loss / 2.0