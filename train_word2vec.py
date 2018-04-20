#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
modified and improved from https://github.com/zake7749/word2vec-tutorial
1. Download the Chinese or English Wikipedia corpus
$ wget https://dumps.wikimedia.org/zhwiki/
2. Extract the articles from wiki xml file
$ python3 wiki_to_txt.py zhwiki-20160820-pages-articles.xml.bz2
3. Using OpenCC for transforming the text from Traditional Chinese to Simplified Chinese
$ opencc -i wiki_texts.txt -o wiki_zh_tw.txt -c s2tw.json
4. Using jieba for segmenting the texts and removing the stop words
$ python3 segment.py
5. Using gensim's word2vec model for training
$ python3 train.py
6. Testing the trained model
$ python3 demo.py
"""

import logging
import os
import sys
import multiprocessing

import pandas as pd
import jieba
import jieba.posseg as pseg
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.corpora import WikiCorpus

logger = logging.getLogger("Word2Vec")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

base_path = os.getcwd() + "/data/embeddings/news/"
config = {
    "wiki_raw": "zhwiki-20160820-pages-articles.xml.bz2",
    "input_raw": base_path+"raw_text.txt",
    "input_seg": base_path+"seg_text.txt",
    "model_file": base_path+"w2v.model",
    "word_vector": base_path+"w2v.bin",
}


def to_text():
    # wiki_corpus = WikiCorpus(config['wiki_raw'], dictionary={})
    # texts_num = 0
    # with open(config['input_raw'], 'w', encoding='utf-8') as output:
    #     for text in wiki_corpus.get_texts():
    #         output.write(' '.join(text) + '\n')
    #         texts_num += 1
    #         if texts_num % 10000 == 0:
    #             logging.info("Parsed %d th articles" % texts_num)

    df = pd.read_csv(os.getcwd() + '/data/financenews/news.csv')
    title = list(df['Title'].values)
    content = list(df['NewsContent'].values)
    raw_text = title + content

    texts_num = 0
    with open(config['input_raw'], 'w', encoding='utf-8') as output:
        for text in raw_text:
            text = str(text)
            output.write(text.strip() + '\n')
            texts_num += 1
            if texts_num % 10000 == 0:
                logging.info("Parsed %d th articles" % texts_num)


def segment():
    # jieba custom setting.
    DATA_DIR = os.getcwd() + '/data/user_dict'
    jieba.load_userdict(os.path.join(DATA_DIR, 'Company.txt'))
    jieba.load_userdict(os.path.join(DATA_DIR, 'Concept.txt'))
    jieba.load_userdict(os.path.join(DATA_DIR, 'Consumer.txt'))
    jieba.load_userdict(os.path.join(DATA_DIR, 'Holder.txt'))
    jieba.load_userdict(os.path.join(DATA_DIR, 'HoldingCompany.txt'))
    jieba.load_userdict(os.path.join(DATA_DIR, 'MainComposition.txt'))
    jieba.load_userdict(os.path.join(DATA_DIR, 'Manager.txt'))
    jieba.load_userdict(os.path.join(DATA_DIR, 'Material.txt'))
    jieba.load_userdict(os.path.join(DATA_DIR, 'OtherCompetitor.txt'))
    jieba.load_userdict(os.path.join(DATA_DIR, 'Supplier.txt'))
    jieba.load_userdict(os.path.join(DATA_DIR, 'Finance.txt'))

    # load stopwords set
    stopword_set = set()
    with open(os.getcwd()+'/data/user_dict/stopWord.txt', 'r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))

    output = open(config['input_seg'], 'w', encoding='utf-8')
    with open(config['input_raw'], 'r', encoding='utf-8') as content :
        for texts_num, line in enumerate(content):
            line = line.strip('\n')
            words = jieba.cut(line, cut_all=False)
            for word in words:
                if word not in stopword_set:
                    output.write(word + ' ')
            output.write('\n')

            if (texts_num + 1) % 10000 == 0:
                logging.info("Segmented %d th articles" % (texts_num + 1))
    output.close()


def train():
    sentences = LineSentence(config['input_seg'])
    model = Word2Vec(sentences, size=300, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())

    model.save(config['model_file'])
    model.wv.save_word2vec_format(config['word_vector'], binary=True)
    logging.info("Training process done")


def demo():
    model = Word2Vec.load(config['model_file'])

    print("Provide three testing modes\n")
    print("Input a word, return 10 most similar words")
    print("Input two words, return their cosine similarity")
    print("Input three words, return the inference word")

    while True:
        try:
            query = input()
            q_list = query.split()

            if len(q_list) == 1:
                print("The 10 most similar words:")
                res = model.most_similar(q_list[0],topn = 10)
                for item in res:
                    print(item[0]+","+str(item[1]))

            elif len(q_list) == 2:
                print("Cosine similarity:")
                res = model.similarity(q_list[0],q_list[1])
                print(res)
            
            else:
                print("%s to %s, is like %s to " % (q_list[0],q_list[2],q_list[1]))
                res = model.most_similar([q_list[0],q_list[1]], [q_list[2]], topn= 10)
                for item in res:
                    print(item[0]+","+str(item[1]))
            print("----------------------------")
        except Exception as e:
            print(repr(e))



if __name__ == '__main__':
    to_text()
    segment()
    train()
    demo()
