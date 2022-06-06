# %%

import re
import os
import io
import time
import jieba
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

file_path = 'rnn_train_data/cmn.txt'


def pre_process_english(word: str):
    word = word.lower().strip()
    word = re.sub(r"([?.!,])", r' \1 ', word)
    word = re.sub(r'[" "]+', " ", word)
    word = re.sub(r"[^a-zA-Z?.!,]+", " ", word)
    word = word.rstrip().strip()
    word = '<start> ' + word + ' <end>'
    return word


def pre_process_chinese(word: str):
    word = word.lower().strip()
    word = jieba.cut(word, cut_all=False, HMM=True)
    word = " ".join(list(word))
    word = '<start> ' + word + ' <end>'
    return word


test_en_sentence = 'May I borrow this book?'
test_chn_sentence = '我可以借这本书吗？'
print(pre_process_english(test_en_sentence))
print(pre_process_chinese(test_chn_sentence))


def create_dataset(path, num_examples=None):
    lines = open(path, encoding='utf-8').read().strip().split('\n')
    word_pairs = [[w for w in l.split('\t')] for l in lines[:num_examples]]
    word_pairs = [[pre_process_english(w[0]), pre_process_chinese(w[1])] for w in word_pairs]

    return word_pairs


word_pairs = create_dataset(file_path)
word_pairs[:20]

en, chn = zip(*create_dataset(file_path))
print(en[-1])
print(chn[-1])

print(len(en), len(chn))


def get_max_length(tensor):
    return max(len(t) for t in tensor)


# %%

def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="")
    lang_tokenizer.fit_on_texts(lang)
    text_ids = lang_tokenizer.texts_to_sequences(lang)
    padded_text_ids = tf.keras.preprocessing.sequence.pad_sequences(
        text_ids,
        padding='post'
    )
    return padded_text_ids, lang_tokenizer


# %%

def load_dataset(path, num_examples=None):
    targ_lang, inp_lang = zip(*create_dataset(path, num_examples))
    input_data, inp_lang_tokenizer = tokenize(inp_lang)
    target_data, targ_lang_tokenizer = tokenize(targ_lang)

    return input_data, target_data, inp_lang_tokenizer, targ_lang_tokenizer


# %%

num_examples = None
input_data, target_data, inp_lang, targ_lang = load_dataset(
    file_path, num_examples
)

max_length_targ, max_length_inp = get_max_length(target_data), get_max_length(input_data)
input_train, input_val, target_train, target_val = train_test_split(input_data, target_data, test_size=0.05)

print(len(input_train), len(target_train), len(input_val), len(target_val))


# %%

def convert(lang, data):
    for t in data:
        if t != 0:
            print(f"{t} -----> {lang.index_word[t]}")


print("输入: 源语言: 中文,  词和id映射关系")
convert(inp_lang, input_train[0])
print()
print("输出: 目标语言: 英文,  词和id映射关系")
convert(targ_lang, target_train[0])

# %%

BUFFER_SIZE = len(input_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_data) // BATCH_SIZE
embedding_dim = 256
units = 1024

vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices(
    (input_train, target_train)
).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape


# %%

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))


# %%

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print(f'Encode 输出的维度:(batch size, sequence length, units){sample_output.shape}')
print(sample_output[-1, -1, :] == sample_hidden[-1, :])


# %%

class AttentionLayer(tf.keras.Model):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)
        ))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


# %%

attention_layer = AttentionLayer(10)
attention_result, attention_weight = attention_layer(sample_hidden, sample_output)

print(f"attention输出的维度:(batch size, units){attention_result.shape}")
print(f"attention权值参数的维度:(batch size, sequence length, 1){attention_weight.shape}")


# %%

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.fc = tf.keras.layers.Dense(vocab_size)

        self.attention = AttentionLayer(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)

        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)

        return x, state, attention_weights


decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
sample_decoder_output, _, _ = decoder(tf.random.uniform((64, 1)), sample_hidden, sample_output)
print(f'Decoder输出的维度:(batch size, vocab size){sample_decoder_output.shape}')

# %%

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction='none'
)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


# %%

checkpoint_dir = 'checkpoint/chinese-eng'

checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(
    optimizer=optimizer,
    encoder=encoder,
    decoder=decoder
)


# %%

def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    sentence = pre_process_chinese(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs],
        maxlen=max_length_inp,
        padding='post'
    )

    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]

    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(
            dec_input,
            dec_hidden,
            enc_out

        )

        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result += targ_lang.index_word[predicted_id] + ' '
        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一 替换sans-serif字体
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二 解决坐标轴负数的负号显示问题
    plt.show()


def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print(f'Input:{sentence}')
    print(f'Predicted translation:{result}')

    attention_plot = attention_plot[:len(
        result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))


if __name__ == '__main__':
    print(tf.train.latest_checkpoint(checkpoint_dir))

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    translate('我不知道我在说什么')
