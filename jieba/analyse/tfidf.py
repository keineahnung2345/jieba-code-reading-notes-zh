# encoding=utf-8
from __future__ import absolute_import
import os
import jieba
import jieba.posseg
from operator import itemgetter

#代碼與_compat.py裡的get_module_res類似
#但get_module_res是回傳一個開啟的檔案
#_get_module_path則是回傳檔案的路徑
_get_module_path = lambda path: os.path.normpath(os.path.join(os.getcwd(),
                                                 os.path.dirname(__file__), path))
_get_abs_path = jieba._get_abs_path

DEFAULT_IDF = _get_module_path("idf.txt")


class KeywordExtractor(object):
  
    #停用詞
    STOP_WORDS = set((
        "the", "of", "is", "and", "to", "in", "that", "we", "for", "an", "are",
        "by", "be", "as", "on", "with", "can", "if", "from", "which", "you", "it",
        "this", "then", "at", "have", "all", "not", "one", "has", "or", "that"
    ))

    #自定義停用詞，將stop_words_path裡的資料更新至self.stop_words
    def set_stop_words(self, stop_words_path):
        abs_path = _get_abs_path(stop_words_path)
        if not os.path.isfile(abs_path):
            raise Exception("jieba: file does not exist: " + abs_path)
        content = open(abs_path, 'rb').read().decode('utf-8')
        for line in content.splitlines():
            self.stop_words.add(line)

    def extract_tags(self, *args, **kwargs):
        raise NotImplementedError


class IDFLoader(object):

    def __init__(self, idf_path=None):
        self.path = ""
        #idf_freq是一個字典，記錄各詞的頻率
        self.idf_freq = {}
        #各詞詞頻的中位數
        self.median_idf = 0.0
        if idf_path:
            #讀取idf_path，更新self.idf_freq及self.median_idf
            self.set_new_path(idf_path)

    def set_new_path(self, new_idf_path):
        if self.path != new_idf_path:
            self.path = new_idf_path
            content = open(new_idf_path, 'rb').read().decode('utf-8')
            self.idf_freq = {}
            for line in content.splitlines():
                word, freq = line.strip().split(' ')
                self.idf_freq[word] = float(freq)
            #取list的中位數
            self.median_idf = sorted(
                self.idf_freq.values())[len(self.idf_freq) // 2]

    def get_idf(self):
        return self.idf_freq, self.median_idf

"""
參考維基百科中的tf-idf頁面：
TF代表的是term frequency，即文檔中各詞彙出現的頻率。
IDF代表的是inverse document frequency，代表詞彙在各文檔出現頻率倒數的對數值(以10為底)。
而TF-IDF值則是上述兩項的乘積。
TF-IDF值是在各詞彙及各文檔間計算的。如果詞彙i在文檔j中的TF-IDF值越大，則代表詞彙i在文檔j中越重要。
"""
class TFIDF(KeywordExtractor):

    def __init__(self, idf_path=None):
        #定義兩種tokenizer，分別在兩種模式下使用
        self.tokenizer = jieba.dt
        self.postokenizer = jieba.posseg.dt
        #self.STOP_WORDS繼承自KeywordExtractor類別
        self.stop_words = self.STOP_WORDS.copy()
        #DEFAULT_IDF為全局變數
        #如果有傳參數到IDFLoader建構子內，
        #那麼它就會自動呼叫set_new_path函數，
        #來將idf_freq, median_idf這兩個屬性設定好
        self.idf_loader = IDFLoader(idf_path or DEFAULT_IDF)
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()
    
    """
    如果使用者想要換一個新的idf檔案，可以直接使用set_idf_path函數。
    它會調用IDFLoader類別的set_new_path函數，讀取idf.txt這個文檔，
    並設定TFIDF物件的idf_freq及median_idf這兩個屬性。
    """
    def set_idf_path(self, idf_path):
        new_abs_path = _get_abs_path(idf_path)
        if not os.path.isfile(new_abs_path):
            raise Exception("jieba: file does not exist: " + new_abs_path)
        self.idf_loader.set_new_path(new_abs_path)
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()

    """
    extract_tags函數會先用tokenizer或postokenizer分詞後，再計算各詞的詞頻。
    得到詞頻後，再與idf_freq這個字典中相對應的詞做運算，得到每個詞的TF-IDF值。
    最後依據withWeight及topK這兩個參數來對結果做後處理再回傳。
    """
    def extract_tags(self, sentence, topK=20, withWeight=False, allowPOS=(), withFlag=False):
        """
        Extract keywords from sentence using TF-IDF algorithm.
        Parameter:
            - topK: return how many top keywords. `None` for all possible words.
            - withWeight: if True, return a list of (word, weight);
                          if False, return a list of words.
            - allowPOS: the allowed POS list eg. ['ns', 'n', 'vn', 'v','nr'].
                        if the POS of w is not in this list,it will be filtered.
            - withFlag: only work with allowPOS is not empty.
                        if True, return a list of pair(word, weight) like posseg.cut
                        if False, return a list of words
        """
        if allowPOS:
            # 參考[Python frozenset()](https://www.programiz.com/python-programming/methods/built-in/frozenset)
            # The frozenset() method returns an immutable frozenset object 
            #  initialized with elements from the given iterable.
            allowPOS = frozenset(allowPOS)
            # words為generator of pair(pair類別定義於jieba/posseg/__init__.py檔)
            # 其中pair類別的物件具有word及flag(即詞性)兩個屬性
            words = self.postokenizer.cut(sentence)
        else:
            # words為generator of str
            words = self.tokenizer.cut(sentence)
        # 計算詞頻(即TF，term frequency)
        freq = {}
        for w in words:
            if allowPOS:
                # 僅選取詞性存在於allowPOS中的詞
                if w.flag not in allowPOS:
                    continue
                # 僅回傳詞彙本身
                elif not withFlag:
                    w = w.word
            # 在allowPOS及withFlag皆為True的情況下，從w中取出詞彙本身，設為wc
            # 如果不符上述情況，則直接將wc設為w
            wc = w.word if allowPOS and withFlag else w
            if len(wc.strip()) < 2 or wc.lower() in self.stop_words:
                #略過長度小於等於1的詞及停用詞?
                continue
            freq[w] = freq.get(w, 0.0) + 1.0
        # 所有詞頻的總和
        total = sum(freq.values())
        # 將詞頻(TF)乘上逆向文件頻率(即IDF，inverse document frequency)
        for k in freq:
            kw = k.word if allowPOS and withFlag else k
            # 如果idf_freq字典中未記錄該詞，則以idf的中位數替代
            freq[k] *= self.idf_freq.get(kw, self.median_idf) / total
        # 現在freq變為詞彙出現機率乘上IDF

        if withWeight:
            # 回傳詞彙本身及其TF-IDF
            # itemgetter(1)的參數是鍵值對(因為是sorted(freq.items()))
            #  它回傳tuple的第1個元素(index從0開始)，即字典的值
            #  所以sorted會依value來排序
            # reverse=True:由大至小排列
            tags = sorted(freq.items(), key=itemgetter(1), reverse=True)
        else:
            # 僅回傳詞彙本身
            # freq.__getitem__的參數是字典的鍵(因為是sorted(freq))
            #  它回傳的是字典的值，所達到的效用是sort by value
            tags = sorted(freq, key=freq.__getitem__, reverse=True)
        if topK:
            # 僅回傳前topK個
            return tags[:topK]
        else:
            return tags
