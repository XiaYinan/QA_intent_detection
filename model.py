import tensorflow as tf

class Seq2seq(object):
    
    def build_inputs(self, config):
        self.seq_inputs = tf.placeholder(shape=(None, config.source_max_len), dtype=tf.int32, name='seq_inputs')
        self.seq_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='seq_inputs_length')
        self.seq_targets = tf.placeholder(shape=(None, config.source_max_len), dtype=tf.int32, name='seq_targets')
        self.seq_targets_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='seq_targets_length')
        self.seq_intent_targets = tf.placeholder(shape=(None,), dtype=tf.int32, name='seq_targets')
    
    def init_decoder(self, config):
        with tf.name_scope('decoder'):

            cell_fw = tf.nn.rnn_cell.GRUCell(config.hidden_dim, name='decoder/forward', activation=tf.nn.tanh)
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=0.9, output_keep_prob=0.9)
            decoder_outputs, _ = tf.nn.dynamic_rnn(cell=cell_fw, inputs=self.encoder_outputs, dtype=tf.float32, sequence_length=self.seq_inputs_length)
            decoder_concat = tf.concat([decoder_outputs, self.encoder_outputs], axis=2)
            self.decoder = tf.layers.dense(
                decoder_concat, config.target_vocab_size
            )
            self.decoder_softmax = tf.nn.softmax(self.decoder, axis=2)
            self.out = tf.arg_max(self.decoder_softmax, 2)

            labels = tf.reshape(
                tf.contrib.layers.one_hot_encoding(
                    tf.reshape(self.seq_targets, [-1]), num_classes=config.target_vocab_size),
                shape=[-1, config.source_max_len, config.target_vocab_size])
            cross_entropy = -tf.reduce_sum(labels * tf.log(self.decoder_softmax), axis=2)

            sequence_mask = tf.sign(tf.reduce_max(tf.abs(labels), axis=2))

            # sequence_mask = tf.sequence_mask(self.seq_targets_length, dtype=tf.float32)
            cross_entropy_masked = tf.reduce_sum(
                cross_entropy*sequence_mask, axis=1) / tf.cast(self.seq_targets_length, tf.float32)
            self.ner_loss = tf.reduce_mean(cross_entropy_masked)


    def __init__(self, config, w2i_target, useTeacherForcing=True, useAttention=True, useBeamSearch=1):

        self.build_inputs(config)

        with tf.variable_scope("encoder"):

            encoder_embedding = tf.Variable(tf.random_uniform([config.source_vocab_size, config.embedding_dim]), dtype=tf.float32, name='encoder_embedding')
            encoder_inputs_embedded = tf.nn.embedding_lookup(encoder_embedding, self.seq_inputs)
            
            ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.GRUCell(config.hidden_dim), 
                cell_bw=tf.nn.rnn_cell.GRUCell(config.hidden_dim), 
                inputs=encoder_inputs_embedded, 
                sequence_length=self.seq_inputs_length, 
                dtype=tf.float32, 
                time_major=False
            )
            # encoder_state = tf.concat([encoder_fw_final_state, encoder_bw_final_state], axis=-1)
            # self.encoder_outputs = tf.concat([encoder_fw_outputs, encoder_bw_outputs], axis=-1)
            encoder_state = tf.add(encoder_fw_final_state, encoder_bw_final_state)
            self.encoder_outputs = tf.add(encoder_fw_outputs, encoder_bw_outputs)

        with tf.variable_scope("intent"):
            hidden_size = int(self.encoder_outputs.shape[-1])
            self.outputs = tf.reshape(self.encoder_outputs, [-1, hidden_size], name='outputs')
            self.softmax_w = tf.get_variable('softmax_w', [hidden_size, config.intent_target_vocab_size])
            self.softmax_b = tf.get_variable('softmax_b', [config.intent_target_vocab_size])
            self.seq_intent_logits = tf.reshape(
                tf.matmul(self.outputs, self.softmax_w) + self.softmax_b,
                shape=[-1, config.source_max_len, config.intent_target_vocab_size], name='intent_logits')


            intent_targets = tf.tile(tf.expand_dims(self.seq_intent_targets, -1), [1, config.source_max_len])
            
            # intent_logits = tf.layers.dense(self.encoder_outputs, units=config.intent_target_vocab_size)


            labels = tf.reshape(
                tf.contrib.layers.one_hot_encoding(
                    tf.reshape(intent_targets, [-1]), num_classes=config.intent_target_vocab_size),
                shape=[-1, config.source_max_len, config.intent_target_vocab_size])
            intent_logits = tf.nn.softmax(self.seq_intent_logits, dim=-1)
            cross_entropy = -tf.reduce_sum(labels * tf.log(intent_logits), axis=2)
            sequence_mask = tf.sign(tf.reduce_max(tf.abs(labels), axis=2))
            # sequence_mask = tf.sequence_mask(self.seq_targets_length, dtype=tf.float32)
            cross_entropy_masked = tf.reduce_sum(
                cross_entropy*sequence_mask, axis=1) / tf.cast(self.seq_targets_length, tf.float32)
            self.intent_loss = tf.reduce_mean(cross_entropy_masked)
        self.init_decoder(config)
        self.loss = self.ner_loss + self.intent_loss   
        self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)

if __name__ == "__main__":
    
    import pandas as pd
    config = pd.Series({
        "source_vocab_size": 999,
        "source_max_len": 30,
        "embedding_dim": 100,
        "hidden_dim": 20,
        "target_vocab_size": 3,
        "target_max_len": 6,
        "batch_size": 10,
        "intent_nums": 9,
        "intent_target_vocab_size": 3
    })

    w2i_target = {
        "_GO": 1
    }
    model = Seq2seq(config, w2i_target)
    print('got it')