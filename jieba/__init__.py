from __future__ import absolute_import, unicode_literals
__version__ = '0.39'
__license__ = 'MIT'

import re
import os
import sys
import time
import logging
import marshal
import tempfile
import threading
from math import log
from hashlib import md5
from ._compat import *
from . import finalseg

"""
這個函數的功用是移動（或說重命名）檔案
這裡使用if-else的寫法是為了處理重命名函數在不同作業系統上的相容性，確保_replace_file在不同的作業系統上皆能運作
代碼第一行的os.name == 'nt'代表當前的作業系統是Windows
"""

if os.name == 'nt':
    from shutil import move as _replace_file
else:
    _replace_file = os.rename

"""
這個函數的參數path是字典的名稱，它的作用是在字典名稱前加上當前路徑，然後把路徑正規化後回傳
"""
_get_abs_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), path))

DEFAULT_DICT = None
DEFAULT_DICT_NAME = "dict.txt"

"""
default_logger如字面上的意思，是這個腳本檔中預設的logger
"""
log_console = logging.StreamHandler(sys.stderr)
default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.DEBUG)
default_logger.addHandler(log_console)

DICT_WRITING = {}

pool = None

"""
這裡定義了數個正則表達式，它們會在分詞及載入字典時發揮作用
"""
"""
re_userdict:
錨點 ^ 和 $ :^表示匹配字串的開頭，$表示匹配字串的結尾
分組和捕獲():()會創造一個捕獲性分組，之後可以用re_userdict.match(line).groups()來得到己匹配的分組
lazy匹配，.+?：.+會貪心地(盡可能多地)匹配，加上?之後變成lazy匹配。如果實際去測試移除?之後的效果，會發現字串全部被分到第一個組別。
或運算符[]：如[0-9]匹配0到9中的其中一個字
數量符?:匹配零次或一次?前面的東西
re.compile會將pattern編譯成正則表達式物件。我們之後就可以用它來match或search其它串。
re.U會根據Unicode字集來解釋字元。

re_userdict會匹配一個字串，並將它分成三組。
第一組是配對一至多個任意字元，直到空白出現為止。
第二組是配對空白加上一至多個數字。
第三組是配對空白加上一至多個英文字母。
"""
re_userdict = re.compile('^(.+?)( [0-9]+)?( [a-z]+)?$', re.U)

"""
re_eng對應到單個英文或數字。
"""
re_eng = re.compile('[a-zA-Z0-9]', re.U)

"""
\u4E00表示的是'一'這個字，\u9FD5表示的是'鿕'，[\u4E00-\u9FD5]表示所有漢字
re_han_default的作用是與一個或多個漢字，英數字，+#&._%-等字元配對。
"""
# \u4E00-\u9FD5a-zA-Z0-9+#&\._ : All non-space characters. Will be handled with re_han
# \r\n|\s : whitespace characters. Will not be handled.
# re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)", re.U)
# Adding "-" symbol in re_han_default
re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-]+)", re.U)

re_skip_default = re.compile("(\r\n|\s)", re.U)
"""
re_han_cut_all的作用是與一個或多個漢字配對。
"""
re_han_cut_all = re.compile("([\u4E00-\u9FD5]+)", re.U)
re_skip_cut_all = re.compile("[^a-zA-Z0-9+#\n]", re.U)

def setLogLevel(log_level):
    global logger
    default_logger.setLevel(log_level)

class Tokenizer(object):
    """
    在Tokenizer類別中有__init__及initialize這兩個函數，他們發揮的都是初始化的作用。
    但是__init__函數是比較輕量級的，在該函數中只簡單地定義了幾個屬性。
    分詞所必需的字典載入則延後至initialize函數中完成。
    """
    def __init__(self, dictionary=DEFAULT_DICT):
        """
        當我們希望某一段代碼能完整地被執行而不被打斷時，我們可以使用threading.Lock來達成。
        
        但是initialize裡使用的是threading.RLock而非threading.Lock
        兩者之間的區別可以參考：class threading.RLock及What is the difference between Lock and RLock
        在上述連結的例子中，函數a跟函數b都需要lock，並且函數a會呼叫函數b。
        
        如果這時候使用threading.Lock將會導致程序卡死，因此我們必須使用threading.RLock。
        RLock的特性是可以重複地被獲取。
        """
        self.lock = threading.RLock()
        if dictionary == DEFAULT_DICT:
            self.dictionary = dictionary
        else:
            self.dictionary = _get_abs_path(dictionary)
        self.FREQ = {}
        self.total = 0
        self.user_word_tag_tab = {}
        self.initialized = False
        self.tmp_dir = None
        self.cache_file = None

    """
    這裡覆寫了object類別的__repr__函數。
    """
    def __repr__(self):
        return '<Tokenizer dictionary=%r>' % self.dictionary

    """
    從一個己開啟的字典file object中，獲取每個詞的出現頻率以及所有詞的出現次數總和。
    """
    # gen_pfdict接受的參數是一個以二進制、讀取模式開啟的檔案。
    def gen_pfdict(self, f):
        #記錄每個詞的出現次數
        lfreq = {}
        #所有詞出現次數的總和
        ltotal = 0
        #他們會在函數的最後被回傳

        #resolve_filename定義於_compat.py
        #它的作用是獲取一個己開啟的檔案的名字
        f_name = resolve_filename(f)
        #逐行讀取檔案f的內容
        for lineno, line in enumerate(f, 1):
            try:
                #因為是以二進制的方式讀檔，
                # 所以這裡用decode來將它由bytes型別轉成字串型別
                line = line.strip().decode('utf-8')
                #更新lfreq及ltotal
                word, freq = line.split(' ')[:2]
                freq = int(freq)
                lfreq[word] = freq
                ltotal += freq

                #把word的前ch+1個字母當成一個出現次數為0的單詞，加入lfreq這個字典中
                # 我們待會可以在get_DAG函數裡看到這樣做的用意

                #這裡的xragne在Python3也會被認識，這是因為在_compat.py中定義了xrange
                #，並將它指向Python3裡的range函數
                for ch in xrange(len(word)):
                    wfrag = word[:ch + 1]
                    if wfrag not in lfreq:
                        lfreq[wfrag] = 0
            #在使用.decode('utf-8')的過程中有可能拋出UnicodeDecodeError錯誤。
            #而我們可以由inspect.getmro(UnicodeDecodeError)這個函數來得知:
            #ValueError是UnicodeDecodeError的parent class。
            #所以可以接住UnicodeDecodeError這個異常
            except ValueError:
                raise ValueError(
                    'invalid dictionary entry in %s at Line %s: %s' % (f_name, lineno, line))
        #記得參數f是一個己開啟的檔案
        #這裡將這個檔案給關閉
        f.close()
        return lfreq, ltotal
    
    """
    initialize函數的功能是載入字典，雖然與__init__函數一樣都是用於初始化。
    但是它不像__init__是在物件創造時就被執行，而是在使用者要存取字典或是開始進行分詞的時候才會執行。

    initialize函數會調用前面介紹的get_dict_file，gen_pfdict，_get_abs_path，DICT_WRITING，default_logger等函數及變數。

    在initialize函數的定義中，使用到了tempfile，marshal套件以及threading中的RLock類別。
    
    在initialize結束後，self.FREQ才會被賦予有意義的值，而self.FREQ在分詞的時候會被用到。
    """
    def initialize(self, dictionary=None):
        """
        abs_path代表的是字典的絕對路徑
        如果使用者傳入了dictionary參數，則需要更新abs_path
        否則的話，就直接使用在__init__()中己經設好的self.dictionary
        """
        if dictionary:
            abs_path = _get_abs_path(dictionary)
            if self.dictionary == abs_path and self.initialized:
                #因為詞典己載入，所以返回
                return
            else:
                self.dictionary = abs_path
                self.initialized = False
        else:
            abs_path = self.dictionary

        #載入詞典的過程必須被完整執行，所以使用lock
        with self.lock:
            #這一段try-except的內容都是pass，似乎沒有作用
            try:
                with DICT_WRITING[abs_path]:
                    pass
            except KeyError:
                pass
            #如果self.intialized為True，代表字典己載入
	        #這時就直接返回
            if self.initialized:
                return

            default_logger.debug("Building prefix dict from %s ..." % (abs_path or 'the default dictionary'))
            t1 = time.time()
            #將cache_file設定快取檔案的名稱
            if self.cache_file:
                cache_file = self.cache_file
            # default dictionary
            elif abs_path == DEFAULT_DICT:
                cache_file = "jieba.cache"
            # custom dictionary
            else:
                cache_file = "jieba.u%s.cache" % md5(
                    abs_path.encode('utf-8', 'replace')).hexdigest()
            """
            tempfile.gettempdir的作用旨在尋找一個可以寫入暫存檔的目錄。
            """
            #將cache_file更新為其絕對路徑
            cache_file = os.path.join(
                self.tmp_dir or tempfile.gettempdir(), cache_file)
            #快取檔案的目錄
            # prevent absolute path in self.cache_file
            tmpdir = os.path.dirname(cache_file)

            load_from_cache_fail = True
            """
            載入cache_file
            首先檢查cache_file是否存在，並且是一個檔案
            如果不是的話則略過這部份;
            如果是的話則接著確認如果使用的是預設的字典DEFAULT_DICT
            如果不是使用預設的字典，則要確認cache_file的修改時間晚於自訂義字典的修改時間
            如果都符合條件，則從快取檔案中載入self.FREQ, self.total這兩個值,
            並將load_from_cache_fail設為False
            """
            if os.path.isfile(cache_file) and (abs_path == DEFAULT_DICT or
                #os.path.getmtime: 獲取檔案的最後修改時間
                os.path.getmtime(cache_file) > os.path.getmtime(abs_path)):
                default_logger.debug(
                    "Loading model from cache %s" % cache_file)
                try:
                    with open(cache_file, 'rb') as cf:
                        """
                        marshal.dump及marshal.load是用來儲存及載入Python物件的工具。
                        """
                        self.FREQ, self.total = marshal.load(cf)
                    load_from_cache_fail = False
                except Exception:
                    load_from_cache_fail = True

            #如果cache_file載入失敗，就重新讀取字典檔案，
            # 獲取self.FREQ, self.total然後生成快取檔案
            if load_from_cache_fail:
                #可能是怕程式中斷，所以先把lock存到DICT_WRITING這個字典裡
                #中斷後繼續執行時就可以不用再重新生成一個lock
                wlock = DICT_WRITING.get(abs_path, threading.RLock())
                DICT_WRITING[abs_path] = wlock
                #在這個程式區塊中，又需要一個lock，用來鎖住寫檔的這一區塊
                with wlock:
                    self.FREQ, self.total = self.gen_pfdict(self.get_dict_file())
                    default_logger.debug(
                        "Dumping model to file cache %s" % cache_file)
                    try:
                        # prevent moving across different filesystems
                        """
                        tempfile.mkstemp的作用旨在使用最安全的方式創建一個暫存檔。
                        它回傳的是一個file descriptor，以及該檔案的絕對路徑。
                        """
                        # tmpdir是剛剛決定好的快取檔案的路徑
                        # prevent moving across different filesystems
                        fd, fpath = tempfile.mkstemp(dir=tmpdir)
                        """
                        os.fdopen:
                        利用傳入的file descriptor fd，回傳一個開啟的檔案物件。
                        """
                        # 使用marshal.dump將剛拿到的
                        # (self.FREQ, self.total)倒入temp_cache_file
                        with os.fdopen(fd, 'wb') as temp_cache_file:
                            """
                            marshal.dump及marshal.load是用來儲存及載入Python物件的工具。
                            """
                            marshal.dump(
                                (self.FREQ, self.total), temp_cache_file)
                        #把檔案重命名為cache_file
                        _replace_file(fpath, cache_file)
                    except Exception:
                        default_logger.exception("Dump cache file failed.")

                try:
                    del DICT_WRITING[abs_path]
                except KeyError:
                    pass

            #之後會利用self.initialized這個屬性
            # 來檢查self.FREQ, self.total是否己被設為有意義的值
            self.initialized = True
            default_logger.debug(
                "Loading model cost %.3f seconds." % (time.time() - t1))
            default_logger.debug("Prefix dict has been built successfully.")

    """
    檢查self.FREQ及self.total是否己被設為有意義的值。
    如果還沒，則調用initialize函數從字典導入。
    """
    def check_initialized(self):
        if not self.initialized:
            self.initialize()

    """
    分詞核心函數
    """

    """
    calc函數接受sentence,DAG及route三個參數。
    其中sentece是要分詞的句子，DAG是由get_DAG函數得到的有向無環圖。
    route在被傳入時是一個空的字典。
    在calc這個函數中會利用sentence及DAG來填充route，在函數結束後，route記錄的是句中每個詞的範圍，以及他們的機率對數。
    """
    def calc(self, sentence, DAG, route):
        N = len(sentence)
        route[N] = (0, 0)
        #self.total在gen_pfdict中計算，代表的是字典中所有詞出現次數的總和
        logtotal = log(self.total)

        """
        由後往前建構route
        我們會使用動態規劃的方式，由後往前填充route，依序找出由sentence[idx]到句尾的最大切分組合的機率對數。。
        為什麼要由後往前填充呢?我們可以從代碼中看到，在計算route[idx]時會利用到route[x+1](x大於等於idx)的值。
        因為在計算前面的內容時會用到後面的內容，使用這種順序來填充，就能讓我們在一次循環裡完成整個過程。
        """
        for idx in xrange(N - 1, -1, -1):
            """
            route[idx]是一個tuple。
            tuple裡第一個元素代表的是sentence[i:]（從當前字元到句子末尾的字段）最有可能出現的切分方式的機率對數。
            第二個元素則是在該切分方式中，包含sentence[i]的詞的詞尾索引。
            """
	
            """
            for x in DAG[idx]：
            DAG這個字典記錄的是sentence中可以成詞的範圍。
            它的鍵代表詞首索引，值則代表以該鍵為詞首的所有詞的詞尾索引。
            在上面的代碼塊中，idx就是詞首索引，DAG[idx]是所有以sentence[idx]為詞首的詞的詞尾索引。x則是DAG[idx]中的單個元素。
	    所以sentence[idx:x + 1]就表示DAG裡存的詞。
            """
		
            """
            self.FREQ.get(sentence[idx:x + 1]) or 1：
            表示如果sentence[idx:x + 1]這個字段不存在FREQ中，或者它的詞頻為0，則以1取代，這樣在經過log後的結果便是0。
            """
	
            """
            log(self.FREQ.get(sentence[idx:x + 1]) or 1) - logtotal + route[x + 1][0]：
            第一項是句中第idx個字到第x個字的詞頻對數。
            將第一項與第二項相加，相當於把第一項正規化，得到的是機率對數。
            第三項是句中第x+1個字到句尾最大切分組合的機率對數。
            所以三項總和就是"如果新增x為切分點，這種組合的機率對數"。
            """
		
            """
            max會找出sentence[idx:]的最大切分組合的機率對數以及對應的切分點。這也是route[idx]被賦予的值。
            """
            route[idx] = max((log(self.FREQ.get(sentence[idx:x + 1]) or 1) -
                              logtotal + route[x + 1][0], x) for x in DAG[idx])

    """
    get_DAG這個函數的目的就是利用手上有的字典(self.FREQ)，將句子表示成一個有向無環圖(DAG)。
    在以下代碼中，DAG是一個把詞首索引對應到一個list的字典。list裡的每個元素是所有可以與該詞首成詞的詞尾索引。
    
    要注意的是get_DAG函數會將非漢字的字元都切成單字詞。
    因此在__cut_DAG_NO_HMM中，會需要re_eng來對get_DAG的結果進行後處理。
    """
    def get_DAG(self, sentence):
        #這段代碼裡會用到self.FREQ，所以需要確保對象己經初始化
        self.check_initialized()
        DAG = {}
        N = len(sentence)
        #如果是使用Python3
        #這裡的xrange將指向range(這是在_compat.py裡做的)
        for k in xrange(N):
            #用來儲存詞尾的索引
            tmplist = []
            #當前cursor的位置，用來找出詞尾
            i = k
            #當前的字段，表示句子的第k個字到第i個字
            frag = sentence[k]
            #在gen_pfdict中為self.FREQ賦予了值(可以參考https://blog.csdn.net/keineahnung2345/article/details/86977785#gen_pfdict_133)
            #self.FREQ中包含的是存在字典裡的詞，他們的詞頻由字典給出，
            #另外對於self.FREQ中的所有詞，它還把這些詞的前n個字都加入self.FREQ，並將它們的詞頻設為0
            while i < N and frag in self.FREQ:
                #因為只有當句子的片段frag在self.FREQ裡面時，才會繼續尋找
                #為了避免還沒看到詞尾就跳出迴圈，
                #所以在gen_pfdict裡才會將詞的前n個字都加入self.FREQ的原因
            
                #這裡檢查sentence裡的第k到第i個字(即frag)是否成詞（也就是檢查frag這個片段的詞頻是否大於0）
                if self.FREQ[frag]:
                    #如果frag成詞的話，就把i加入tmplist裡，表示句中的第k個字到第i個字可以成詞
                    #實際上的字段frag是sentence[k:i+1]
                    tmplist.append(i)
                #繼續看下一個字
                i += 1
                frag = sentence[k:i + 1]
            #如果沒有字可與sentence[k]成詞
            if not tmplist:
                #單字成詞
                tmplist.append(k)
            #DAG記錄的是sentence裡的第k個字可以跟句字裡的哪些字組成詞
            DAG[k] = tmplist
        return DAG

    """
    __cut_all函數會利用get_DAG函數建立的dag來找出句中所有可以成詞的部份。
    注意到以下分詞函數是在句中的某個字無法與其它字成詞時，才會輸出該單字詞。
    
    要注意的是__cut_all函數並未對非漢字的字元做處理。
    因此在cut及tokenizer函數中才需要一個re_han來過濾出__cut_all能處理的詞。
    """
    def __cut_all(self, sentence):
        #dag表示句中的第幾個字到第幾個字可以成詞
        dag = self.get_DAG(sentence)
        #前一個詞的詞尾，初始值為0-1=-1
        old_j = -1

        #dag是一個字典，鍵代表詞首索引，值則是一個list，裡面的每個元素代表能與鍵成詞的詞尾索引
        #用在這裡的話，便是：從第k個字開始，到L中的第j個字都可以成詞

        # iteritems定義於_compat.py
        for k, L in iteritems(dag):
            #從get_DAG裡可以知道，當L的長度為1時，表示第k個字單字成詞
            #單字成詞在一般情況下不輸出
            #只有當第k個字無法與之前的字組詞(不被前面的字所組的詞所包含)時才會輸出
            if len(L) == 1 and k > old_j:
                #這時的L[0]+1其實就是k+1
                yield sentence[k:L[0] + 1]
                #詞尾設為k
                old_j = L[0]
            else:
                for j in L:
                    #如果有多字的詞，就略過單個字的詞
                    #(詞尾j=詞首k表示一個單字詞)
                    if j > k:
                        yield sentence[k:j + 1]
                        #更新詞尾
                        old_j = j

    """
    利用calc函數算出route，因為英數字並不存在字典中，所以route裡的英數字會被切成一個一個的字元。
    在__cut_DAG_NO_HMM中，會由前往後掃描route裡的內容。使用re_eng來找出句中的英數字，如果碰到了，就把它們放到buf裡，碰到下個中文字時再輸出。
    """
    def __cut_DAG_NO_HMM(self, sentence):
        DAG = self.get_DAG(sentence)
        route = {}
        #注意因為字典裡沒有記錄英數字的詞頻，所以會把它們切成一個一個的字元
        self.calc(sentence, DAG, route)
        x = 0
        N = len(sentence)
        #用來暫存英數字
        buf = ''
        # 由前往後
        while x < N:
            #route[x][1]表示的是"以sentence[x]開頭，擁有最大概率的詞的詞尾的索引"
            y = route[x][1] + 1
            #sentence[x:y]:從句子中擷取出該詞
            l_word = sentence[x:y]
            #如果是英數字且長度為1
            if re_eng.match(l_word) and len(l_word) == 1:
                buf += l_word
                #接下來從y（當前詞的下一個字元）開始搜尋
                x = y
            else:
                # 之前收集的字元被集中到buf裡，這裡輸出並清出buf
                if buf:
                    yield buf
                    buf = ''
                # 把buf清空後（如果有的話），才輸山l_word
                yield l_word
                #接下來從y（當前詞的下一個字元）開始搜尋
                x = y
        # 最後一次的清空
        if buf:
            yield buf
            buf = ''

    def __cut_DAG(self, sentence):
        DAG = self.get_DAG(sentence)
        route = {}
        self.calc(sentence, DAG, route)
        x = 0
        buf = ''
        N = len(sentence)
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if y - x == 1:
                buf += l_word
            else:
                if buf:
                    if len(buf) == 1:
                        yield buf
                        buf = ''
                    else:
                        if not self.FREQ.get(buf):
                            recognized = finalseg.cut(buf)
                            for t in recognized:
                                yield t
                        else:
                            for elem in buf:
                                yield elem
                        buf = ''
                yield l_word
            x = y

        if buf:
            if len(buf) == 1:
                yield buf
            elif not self.FREQ.get(buf):
                recognized = finalseg.cut(buf)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield elem

    def cut(self, sentence, cut_all=False, HMM=True):
        '''
        The main function that segments an entire sentence that contains
        Chinese characters into separated words.

        Parameter:
            - sentence: The str(unicode) to be segmented.
            - cut_all: Model type. True for full pattern, False for accurate pattern.
            - HMM: Whether to use the Hidden Markov Model.
        '''
        sentence = strdecode(sentence)

        if cut_all:
            re_han = re_han_cut_all
            re_skip = re_skip_cut_all
        else:
            re_han = re_han_default
            re_skip = re_skip_default
        if cut_all:
            cut_block = self.__cut_all
        elif HMM:
            cut_block = self.__cut_DAG
        else:
            cut_block = self.__cut_DAG_NO_HMM
        blocks = re_han.split(sentence)
        for blk in blocks:
            if not blk:
                continue
            if re_han.match(blk):
                for word in cut_block(blk):
                    yield word
            else:
                tmp = re_skip.split(blk)
                for x in tmp:
                    if re_skip.match(x):
                        yield x
                    elif not cut_all:
                        for xx in x:
                            yield xx
                    else:
                        yield x

    def cut_for_search(self, sentence, HMM=True):
        """
        Finer segmentation for search engines.
        """
        words = self.cut(sentence, HMM=HMM)
        for w in words:
            if len(w) > 2:
                for i in xrange(len(w) - 1):
                    gram2 = w[i:i + 2]
                    if self.FREQ.get(gram2):
                        yield gram2
            if len(w) > 3:
                for i in xrange(len(w) - 2):
                    gram3 = w[i:i + 3]
                    if self.FREQ.get(gram3):
                        yield gram3
            yield w

    """
    分詞函數wrapper
    原有的分詞函數回傳的是generator型別的變數。
    下面以l開頭的函數們調用了原有的分詞函數，將它們的回傳值轉為list型別，提升了其易用性。
    """
    def lcut(self, *args, **kwargs):
        return list(self.cut(*args, **kwargs))

    def lcut_for_search(self, *args, **kwargs):
        return list(self.cut_for_search(*args, **kwargs))

    _lcut = lcut
    _lcut_for_search = lcut_for_search

    def _lcut_no_hmm(self, sentence):
        return self.lcut(sentence, False, False)

    def _lcut_all(self, sentence):
        return self.lcut(sentence, True)

    def _lcut_for_search_no_hmm(self, sentence):
        return self.lcut_for_search(sentence, False)

    """
    這個函數的作用是讀取字典檔案，開啟後回傳。
    它預設讀取dict.txt，但使用者也可以自定義字典。
    """
    def get_dict_file(self):
        if self.dictionary == DEFAULT_DICT:
            return get_module_res(DEFAULT_DICT_NAME)
        else:
            return open(self.dictionary, 'rb')

    """
    自定義詞典
    jieba支持自定義詞典，因為這不是核心功能，在此僅列出相關函數，並不多做介紹。
    """
    def load_userdict(self, f):
        '''
        Load personalized dict to improve detect rate.

        Parameter:
            - f : A plain text file contains words and their ocurrences.
                  Can be a file-like object, or the path of the dictionary file,
                  whose encoding must be utf-8.

        Structure of dict file:
        word1 freq1 word_type1
        word2 freq2 word_type2
        ...
        Word type may be ignored
        '''
        self.check_initialized()
        if isinstance(f, string_types):
            f_name = f
            f = open(f, 'rb')
        else:
            f_name = resolve_filename(f)
        for lineno, ln in enumerate(f, 1):
            line = ln.strip()
            if not isinstance(line, text_type):
                try:
                    line = line.decode('utf-8').lstrip('\ufeff')
                except UnicodeDecodeError:
                    raise ValueError('dictionary file %s must be utf-8' % f_name)
            if not line:
                continue
            # match won't be None because there's at least one character
            word, freq, tag = re_userdict.match(line).groups()
            if freq is not None:
                freq = freq.strip()
            if tag is not None:
                tag = tag.strip()
            self.add_word(word, freq, tag)

    def add_word(self, word, freq=None, tag=None):
        """
        Add a word to dictionary.

        freq and tag can be omitted, freq defaults to be a calculated value
        that ensures the word can be cut out.
        """
        self.check_initialized()
        word = strdecode(word)
        freq = int(freq) if freq is not None else self.suggest_freq(word, False)
        self.FREQ[word] = freq
        self.total += freq
        if tag:
            self.user_word_tag_tab[word] = tag
        for ch in xrange(len(word)):
            wfrag = word[:ch + 1]
            if wfrag not in self.FREQ:
                self.FREQ[wfrag] = 0
        if freq == 0:
            finalseg.add_force_split(word)

    def del_word(self, word):
        """
        Convenient function for deleting a word.
        """
        self.add_word(word, 0)

    def suggest_freq(self, segment, tune=False):
        """
        Suggest word frequency to force the characters in a word to be
        joined or splitted.

        Parameter:
            - segment : The segments that the word is expected to be cut into,
                        If the word should be treated as a whole, use a str.
            - tune : If True, tune the word frequency.

        Note that HMM may affect the final result. If the result doesn't change,
        set HMM=False.
        """
        self.check_initialized()
        ftotal = float(self.total)
        freq = 1
        if isinstance(segment, string_types):
            word = segment
            for seg in self.cut(word, HMM=False):
                freq *= self.FREQ.get(seg, 1) / ftotal
            freq = max(int(freq * self.total) + 1, self.FREQ.get(word, 1))
        else:
            segment = tuple(map(strdecode, segment))
            word = ''.join(segment)
            for seg in segment:
                freq *= self.FREQ.get(seg, 1) / ftotal
            freq = min(int(freq * self.total), self.FREQ.get(word, 0))
        if tune:
            add_word(word, freq)
        return freq

    """
    tokenize函數
    將cut函數回傳的字串包裝成(字串起始位置，字串終止位置，字串)的三元組後回傳。
    """
    def tokenize(self, unicode_sentence, mode="default", HMM=True):
        """
        Tokenize a sentence and yields tuples of (word, start, end)

        Parameter:
            - sentence: the str(unicode) to be segmented.
            - mode: "default" or "search", "search" is for finer segmentation.
            - HMM: whether to use the Hidden Markov Model.
        """
        if not isinstance(unicode_sentence, text_type):
            raise ValueError("jieba: the input parameter should be unicode.")
        start = 0
        if mode == 'default':
            for w in self.cut(unicode_sentence, HMM=HMM):
                width = len(w)
                yield (w, start, start + width)
                start += width
        else:
            for w in self.cut(unicode_sentence, HMM=HMM):
                width = len(w)
                if len(w) > 2:
                    for i in xrange(len(w) - 1):
                        gram2 = w[i:i + 2]
                        if self.FREQ.get(gram2):
                            yield (gram2, start + i, start + i + 2)
                if len(w) > 3:
                    for i in xrange(len(w) - 2):
                        gram3 = w[i:i + 3]
                        if self.FREQ.get(gram3):
                            yield (gram3, start + i, start + i + 3)
                yield (w, start, start + width)
                start += width

    def set_dictionary(self, dictionary_path):
        with self.lock:
            abs_path = _get_abs_path(dictionary_path)
            if not os.path.isfile(abs_path):
                raise Exception("jieba: file does not exist: " + abs_path)
            self.dictionary = abs_path
            self.initialized = False

"""
根據jieba文檔裡介紹的使用方法，我們可以直接調用jieba.cut來分詞，這是怎麼做到的呢？
在定義好Tokenizer類別後，__init__.py裡建立了一個Tokenizer類別的dt對象。
然後逐一定義全局函數，並將它們指向dt中相對應的函數。
如：cut = dt.cut這一句，它定義了一個全局函數cut，並把它指向dt對象的cut函數。
如此一來，我們就可以不用自己新建一個Tokenizer對象，而是直接使用jieba.cut來分詞。
"""
# default Tokenizer instance

dt = Tokenizer()

# global functions

get_FREQ = lambda k, d=None: dt.FREQ.get(k, d)
add_word = dt.add_word
calc = dt.calc
cut = dt.cut
lcut = dt.lcut
cut_for_search = dt.cut_for_search
lcut_for_search = dt.lcut_for_search
del_word = dt.del_word
get_DAG = dt.get_DAG
get_dict_file = dt.get_dict_file
initialize = dt.initialize
load_userdict = dt.load_userdict
set_dictionary = dt.set_dictionary
suggest_freq = dt.suggest_freq
tokenize = dt.tokenize
user_word_tag_tab = dt.user_word_tag_tab


def _lcut_all(s):
    return dt._lcut_all(s)


def _lcut(s):
    return dt._lcut(s)


def _lcut_no_hmm(s):
    return dt._lcut_no_hmm(s)


def _lcut_all(s):
    return dt._lcut_all(s)


def _lcut_for_search(s):
    return dt._lcut_for_search(s)


def _lcut_for_search_no_hmm(s):
    return dt._lcut_for_search_no_hmm(s)

"""
以下是並行分詞相關函數
"""
def _pcut(sentence, cut_all=False, HMM=True):
    parts = strdecode(sentence).splitlines(True)
    if cut_all:
        result = pool.map(_lcut_all, parts)
    elif HMM:
        result = pool.map(_lcut, parts)
    else:
        result = pool.map(_lcut_no_hmm, parts)
    for r in result:
        for w in r:
            yield w


def _pcut_for_search(sentence, HMM=True):
    parts = strdecode(sentence).splitlines(True)
    if HMM:
        result = pool.map(_lcut_for_search, parts)
    else:
        result = pool.map(_lcut_for_search_no_hmm, parts)
    for r in result:
        for w in r:
            yield w


def enable_parallel(processnum=None):
    """
    Change the module's `cut` and `cut_for_search` functions to the
    parallel version.

    Note that this only works using dt, custom Tokenizer
    instances are not supported.
    """
    global pool, dt, cut, cut_for_search
    from multiprocessing import cpu_count
    if os.name == 'nt':
        raise NotImplementedError(
            "jieba: parallel mode only supports posix system")
    else:
        from multiprocessing import Pool
    dt.check_initialized()
    if processnum is None:
        processnum = cpu_count()
    pool = Pool(processnum)
    cut = _pcut
    cut_for_search = _pcut_for_search


def disable_parallel():
    global pool, dt, cut, cut_for_search
    if pool:
        pool.close()
        pool = None
    cut = dt.cut
    cut_for_search = dt.cut_for_search
