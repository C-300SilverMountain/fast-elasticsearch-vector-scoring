
import numpy as np
import fasttext
import re
import torch

model_path = '../../../../../../doc/embedded/fasttext/model/model.bin'

def test05():
    model = fasttext.load_model(model_path)
    # 如果句子以空格切分好，则句子向量为每个 (子词/norm) 相加取均值
    print('句子的向量:', model.get_sentence_vector('中置 三通 集成').tolist())
    vec = np.zeros(100)
    for word in '中置三通集成':
        embed = model.get_word_vector(word)
        embed = embed / np.sqrt(np.sum(embed ** 2))
        vec += embed
    print(vec / len('中置三通集成'))


if __name__ == '__main__':
    # a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    # print(a)
    # test05()
    # vector = np.array([1, 2])
    # vv = vector / 2
    # print(vv)
    # v = 5
    # v = v ** 2
    # print(v)

    # vector = np.array([3, 2])
    # vector = vector ** 2
    # print(vector)
    v = np.sqrt(9)
    print(v)

    vector = np.array([3, 2])
    mm = np.sum(vector)
    print(mm)
