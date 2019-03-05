from __future__ import absolute_import, unicode_literals
import os
import re
import sys
import jieba
import pickle
from .._compat import *
from .viterbi import viterbi

PROB_START_P = "prob_start.p"
PROB_TRANS_P = "prob_trans.p"
PROB_EMIT_P = "prob_emit.p"
CHAR_STATE_TAB_P = "char_state_tab.p"

#一個或多個漢字
re_han_detail = re.compile("([\u4E00-\u9FD5]+)")
#一個或多個 (.跟0-9)或英數字
re_skip_detail = re.compile("([\.0-9]+|[a-zA-Z0-9]+)")
#一個或多個漢字或英數字或+#&._
re_han_internal = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)")
#\s等於[ \t\n\r\f\v]:https://docs.python.org/3/library/re.html#re.compile
re_skip_internal = re.compile("(\r\n|\s)")

#一個或多個英數字
re_eng = re.compile("[a-zA-Z0-9]+")
#一個或多個小數點或數字 
re_num = re.compile("[\.0-9]+")

#長度為1的英數字
re_eng1 = re.compile('^[a-zA-Z0-9]$', re.U)

"""
載入了HMM的參數
包括初始機率向量，狀態轉移機率矩陣，發射機率矩陣及CHAR_STATE_TAB_P這個字典(它記錄各個漢字可能的狀態及詞性)

如果是使用Jython的話，會需要用到load_model這個函數，裡面使用pickle這個模組來載入.p檔
如果是使用純Python的話，直接使用from ... import ...即可
"""
def load_model():
    # For Jython
    start_p = pickle.load(get_module_res("posseg", PROB_START_P))
    trans_p = pickle.load(get_module_res("posseg", PROB_TRANS_P))
    emit_p = pickle.load(get_module_res("posseg", PROB_EMIT_P))
    state = pickle.load(get_module_res("posseg", CHAR_STATE_TAB_P))
    return state, start_p, trans_p, emit_p


if sys.platform.startswith("java"):
    char_state_tab_P, start_P, trans_P, emit_P = load_model()
else:
    from .char_state_tab import P as char_state_tab_P
    from .prob_start import P as start_P
    from .prob_trans import P as trans_P
    from .prob_emit import P as emit_P

"""
pair類別具有兩個屬性，分別是word及flag，它們代表詞彙本身及其詞性。
在POSTokenizer中的__cut_DAG_NO_HMM及__cut_DAG函數中，將會把分詞結果及詞性標注結果打包成pair類別的物件後回傳。
"""
class pair(object):

    def __init__(self, word, flag):
        self.word = word
        self.flag = flag

    def __unicode__(self):
        return '%s/%s' % (self.word, self.flag)

    def __repr__(self):
        return 'pair(%r, %r)' % (self.word, self.flag)

    def __str__(self):
        if PY2:
            return self.__unicode__().encode(default_encoding)
        else:
            return self.__unicode__()

    def __iter__(self):
        return iter((self.word, self.flag))

    def __lt__(self, other):
        return self.word < other.word

    def __eq__(self, other):
        return isinstance(other, pair) and self.word == other.word and self.flag == other.flag

    def __hash__(self):
        return hash(self.word)

    def encode(self, arg):
        return self.__unicode__().encode(arg)

"""
POSTokenizer類別中定義了__cut_DAG_NO_HMM及__cut_DAG函數，它們負責了詞性標注的核心算法。
"""
class POSTokenizer(object):

    def __init__(self, tokenizer=None):
        # 它需要借用jieba.Tokenizer的get_dict_file, get_DAG, calc等函數
        # 所以這裡才會定義了tokenizer這個屬性
        self.tokenizer = tokenizer or jieba.Tokenizer()
        # 這一句怎麼同時出現在__init__()及initialize()?
        self.load_word_tag(self.tokenizer.get_dict_file())

    def __repr__(self):
        return '<POSTokenizer tokenizer=%r>' % self.tokenizer

    def __getattr__(self, name):
        if name in ('cut_for_search', 'lcut_for_search', 'tokenize'):
            # may be possible?
            raise NotImplementedError
        # POSTokenizer並未實作cut_for_search, lcut_for_search, tokenize
        # 其餘的功能如cut, lcut等有被POSTokenizer覆寫，所以可以使用
        return getattr(self.tokenizer, name)

    def initialize(self, dictionary=None):
        self.tokenizer.initialize(dictionary)
        # 這一句怎麼同時出現在__init__()及initialize()?
        self.load_word_tag(self.tokenizer.get_dict_file())

    def load_word_tag(self, f):
        #這個函數接受一個開啟的file object當作輸入，然後將它的內容讀到一個dict內
        #即，從jieba/dict.txt中載入word_tag_tab
        
        #一個把詞彙對應到詞性的字典
        self.word_tag_tab = {}
        f_name = resolve_filename(f)
        for lineno, line in enumerate(f, 1):
            try:
                line = line.strip().decode("utf-8")
                if not line:
                    continue
                word, _, tag = line.split(" ")
                self.word_tag_tab[word] = tag
            except Exception:
                raise ValueError(
                    'invalid POS dictionary entry in %s at Line %s: %s' % (f_name, lineno, line))
        f.close()

    def makesure_userdict_loaded(self):
        #如果使用者有自定義詞彙，那麼makesure_userdict_loaded函數會將它們加入word_tag_tab。
        
        #在使用者有用add_word增加新詞時self.tokenizer.user_word_tag_tab才會不為空
        if self.tokenizer.user_word_tag_tab:
            #參考https://www.programiz.com/python-programming/methods/dictionary/update
            #字典1.update(字典2):如果字典2的key不在字典1中,則把該key加入字典1;
            #如果字典2的key己經存在字典1中,則更新字典1中該key的值
            self.word_tag_tab.update(self.tokenizer.user_word_tag_tab)
            self.tokenizer.user_word_tag_tab = {}

    """
    __cut會先呼叫viterbi這個函數，得到句中各字的分詞標籤及詞性。
    然後再將得到的結果打包成一對一對的(詞彙，詞性)pair。
    __cut是一個生成器，每次被呼叫時，就生成一對(詞彙，詞性)pair。
    注意__cut並未對英數字做特別處理，所以它跟viterbi函數一樣，只能處理sentence全是漢字的情況。
    """
    def __cut(self, sentence):
        #使用維特比算法找出最有可能的狀態序列pos_list及其機率prob
        #所謂狀態包含分詞標籤及詞性
        prob, pos_list = viterbi(
            sentence, char_state_tab_P, start_P, trans_P, emit_P)
        begin, nexti = 0, 0

        for i, char in enumerate(sentence):
            #[0]表示分詞標籤('B'，'M'，'E'，'S')
            pos = pos_list[i][0]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                #到詞尾時yield出該詞彙
                #pos_list[i][1]表示該詞的詞性，這裡以詞尾的詞性代表全詞的詞性
                yield pair(sentence[begin:i + 1], pos_list[i][1])
                nexti = i + 1
            elif pos == 'S':
                #單字成詞的情況下直接yield
                yield pair(char, pos_list[i][1])
                nexti = i + 1
        #nexti記錄上個詞彙詞尾的後一個位置
        if nexti < len(sentence):
            yield pair(sentence[nexti:], pos_list[nexti][1])

    """
    __cut_detail是__cut的wrapper，它與__cut同樣是一個會生成(詞彙，詞性)pair的生成器。
    但是它有對英數字做處理，並賦予它們詞性，所以它可以處理句中包含英數字的情況。
    """
    def __cut_detail(self, sentence):
        #re_han_detail:一個或多個漢字
        blocks = re_han_detail.split(sentence)
        for blk in blocks:
            if re_han_detail.match(blk):
                #如果該區段包含漢字，則直接使用__cut來切
                for word in self.__cut(blk):
                    yield word
            else:
                #非漢字的區段
                #re_num:小數點及數字
                #re_eng:英數字
                #re_skip_detail:re_num或re_eng
                tmp = re_skip_detail.split(blk)
                for x in tmp:
                    if x:
                        if re_num.match(x):
                            #'m':數詞
                            yield pair(x, 'm')
                        elif re_eng.match(x):
                            #'eng':外語
                            yield pair(x, 'eng')
                        else:
                            #'x':非語素字
                            yield pair(x, 'x')
    """
    此處代碼與jieba/__init__.py裡的__cut_DAG_NO_HMM雷同
    可以參考https://blog.csdn.net/keineahnung2345/article/details/86735757
    不同之處僅在於：
    self.get_DAG及self.calc變成self.tokenizer.get_DAG及self.tokenizer.calc
    if re_eng.match(l_word) and len(l_word) == 1:被改成了if re_eng1.match(l_word)
    yield的東西由一個詞彙變成一個pair
    """
    def __cut_DAG_NO_HMM(self, sentence):
        #__cut_DAG_NO_HMM是以匹配正則表達式re_eng1及查找字典self.word_tag_tab並用的方式來標注詞性。
        DAG = self.tokenizer.get_DAG(sentence)
        route = {}
        self.tokenizer.calc(sentence, DAG, route)
        x = 0
        N = len(sentence)
        buf = ''
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            #re_eng1:長度為1的英數字
            if re_eng1.match(l_word):
                buf += l_word
                x = y
            else:
                if buf:
                    #buf裡只有與re_eng1配對的字
                    #所以這裡可以將它的詞性設為英文
                    yield pair(buf, 'eng')
                    buf = ''
                #如果字典裡沒有l_word，就把它的詞性當成'x'(未知)
                yield pair(l_word, self.word_tag_tab.get(l_word, 'x'))
                x = y
        if buf:
            #buf裡只有與re_eng1配對的字
            #所以這裡可以將它的詞性設為英文
            yield pair(buf, 'eng')
            buf = ''

    """
    此處的代碼邏輯與jieba/__init__.py裡的__cut_DAG函數類似。
    它會呼叫__cut_detail，而__cut_detail又會呼叫__cut。
    __cut_DAG這個函數是以查找字典為主，維特比算法(即呼叫__cut_detail)為輔的方式來找出詞首詞尾。
    它與之前的__cut_detail還有__cut一樣，都是生成器，每次生成一對(詞彙，詞性)pair。
    """
    def __cut_DAG(self, sentence):
        DAG = self.tokenizer.get_DAG(sentence)
        route = {}

        self.tokenizer.calc(sentence, DAG, route)

        x = 0
        buf = ''
        N = len(sentence)
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if y - x == 1:
                buf += l_word
            else:
                #碰到多字詞，先處理之前的buf
                if buf:
                    if len(buf) == 1:
                        #單字詞
                        yield pair(buf, self.word_tag_tab.get(buf, 'x'))
                    elif not self.tokenizer.FREQ.get(buf):
                        #如果是未記錄於FREQ裡的buf，就使用維特比算法來找出詞首詞尾
                        recognized = self.__cut_detail(buf)
                        for t in recognized:
                            yield t
                    else:
                        #如果buf存在於FREQ裡，則把它拆成多個單字詞?
                        for elem in buf:
                            yield pair(elem, self.word_tag_tab.get(elem, 'x'))
                    buf = ''
                #處理當前的多字詞
                yield pair(l_word, self.word_tag_tab.get(l_word, 'x'))
            x = y

        #用一樣的方式來處理殘留的buf
        if buf:
            if len(buf) == 1:
                yield pair(buf, self.word_tag_tab.get(buf, 'x'))
            elif not self.tokenizer.FREQ.get(buf):
                recognized = self.__cut_detail(buf)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield pair(elem, self.word_tag_tab.get(elem, 'x'))

    def __cut_internal(self, sentence, HMM=True):
        self.makesure_userdict_loaded()
        sentence = strdecode(sentence)
        blocks = re_han_internal.split(sentence)
        if HMM:
            cut_blk = self.__cut_DAG
        else:
            cut_blk = self.__cut_DAG_NO_HMM

        for blk in blocks:
            if re_han_internal.match(blk):
                for word in cut_blk(blk):
                    yield word
            else:
                tmp = re_skip_internal.split(blk)
                for x in tmp:
                    if re_skip_internal.match(x):
                        yield pair(x, 'x')
                    else:
                        for xx in x:
                            if re_num.match(xx):
                                yield pair(xx, 'm')
                            elif re_eng.match(x):
                                yield pair(xx, 'eng')
                            else:
                                yield pair(xx, 'x')

    def _lcut_internal(self, sentence):
        return list(self.__cut_internal(sentence))

    def _lcut_internal_no_hmm(self, sentence):
        return list(self.__cut_internal(sentence, False))

    def cut(self, sentence, HMM=True):
        for w in self.__cut_internal(sentence, HMM=HMM):
            yield w

    def lcut(self, *args, **kwargs):
        return list(self.cut(*args, **kwargs))

"""
此處基於上述定義的POSTokenizer及pair類別，定義了幾個全局的變數及函數。
"""
# default Tokenizer instance

dt = POSTokenizer(jieba.dt)

# global functions

initialize = dt.initialize


def _lcut_internal(s):
    return dt._lcut_internal(s)


def _lcut_internal_no_hmm(s):
    return dt._lcut_internal_no_hmm(s)


def cut(sentence, HMM=True):
    """
    Global `cut` function that supports parallel processing.

    Note that this only works using dt, custom POSTokenizer
    instances are not supported.
    """
    global dt
    if jieba.pool is None:
        for w in dt.cut(sentence, HMM=HMM):
            yield w
    else:
        parts = strdecode(sentence).splitlines(True)
        if HMM:
            result = jieba.pool.map(_lcut_internal, parts)
        else:
            result = jieba.pool.map(_lcut_internal_no_hmm, parts)
        for r in result:
            for w in r:
                yield w


def lcut(sentence, HMM=True):
    return list(cut(sentence, HMM))
