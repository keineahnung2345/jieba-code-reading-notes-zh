from __future__ import absolute_import, unicode_literals
import re
import os
import sys
import pickle
from .._compat import *

MIN_FLOAT = -3.14e100

"""
載入HMM的參數
"""

PROB_START_P = "prob_start.p"
PROB_TRANS_P = "prob_trans.p"
PROB_EMIT_P = "prob_emit.p"

"""
參考http://www.52nlp.cn/%E4%B8%AD%E6%96%87%E5%88%86%E8%AF%8D%E5%85%A5%E9%97%A8%E4%B9%8B%E5%AD%97%E6%A0%87%E6%B3%A8%E6%B3%954
B: begin詞的首字
M: middle詞的中間字
E：end詞的尾字
S：single單字成詞
PrevStatus表示一個狀態之前可能是哪些狀態
"""
PrevStatus = {
    #在一個詞的首字之前，只可能是上一個詞的詞尾，或者是單字成詞
    'B': 'ES',
    #在一個詞的中間字之前，可能是當前詞的中間字，或是當前詞的首字
    'M': 'MB',
    #在單字成詞之前，可能是另外一個單字成詞，也可能是上一個詞的詞尾
    'S': 'SE',
    #在一個詞的詞尾之前，可能是詞首或是中間字
    'E': 'BM'
}

Force_Split_Words = set([])
def load_model():
    start_p = pickle.load(get_module_res("finalseg", PROB_START_P))
    trans_p = pickle.load(get_module_res("finalseg", PROB_TRANS_P))
    emit_p = pickle.load(get_module_res("finalseg", PROB_EMIT_P))
    return start_p, trans_p, emit_p

"""
從.p檔或.py檔載入start_P, trans_P, emit_P
"""
if sys.platform.startswith("java"):
    start_P, trans_P, emit_P = load_model()
else:
    from .prob_start import P as start_P
    from .prob_trans import P as trans_P
    from .prob_emit import P as emit_P
"""
start_P:4個狀態的初始log機率
{'B': -0.26268660809250016, #使用e^x換算回去，機率約為0.76
 'E': -3.14e+100, #0
 'M': -3.14e+100, #0
 'S': -1.4652633398537678} #機率約為0.23

 trans_P:4個狀態間的轉移log機率
{'B': {'E': -0.51082562376599,
       'M': -0.916290731874155},
 'E': {'B': -0.5897149736854513,
       'S': -0.8085250474669937},
 'M': {'E': -0.33344856811948514,
       'M': -1.2603623820268226},
 'S': {'B': -0.7211965654669841,
       'S': -0.6658631448798212}}

emit_P:發射概率
在4個狀態，觀察到不同字的機率
{'B': {'十': -6.007344000041945, #emit_P['B']有6857個item
       '囤': -12.607094089862704,
       '撷': -14.695424578959786,
       '柱': -9.88964863468285,
       '齏': -13.868746005775318},
 'E': {'僱': -12.20458319365197, # 7439
       '凜': -11.06728136003368,
       '妪': -13.50584051208595,
       '拟': -9.377654786484934,
       '焘': -11.21858978309201},
 'M': {'咦': -13.561365842482271, # 6409
       '疊': -9.63829300086754,
       '荑': -15.758590419818491,
       '趙': -10.120235750484746,
       '骛': -12.324603215333344},
 'S': {'妳': -13.847864212264698, # 14519
       '庝': -10.732052690793973,
       '弑': -12.34762559179559,
       '氜': -16.116547753583063,
       '筱': -13.957063504229689}}
"""

"""
參考李航統計機器學習中的章節10.4 - 預測算法。
維特比算法是在給定模型參數及觀察序列的情況下，用於求解狀態序列的算法。
注意這個函數只適用於obs(也就是觀察序列)全是漢字的情況。
"""
def viterbi(obs, states, start_p, trans_p, emit_p):
    """
    obs: 觀察序列
    states: 所有可能的狀態
    start_p: 李航書中的Pi，一開始在每個不同狀態的機率
    trans_p: 李航書中的A, 轉移矩陣
    emit_p: 李航書中的B, 每個狀態發射出不同觀察值的機率矩陣
    """

    """
    算法10.5 (1)初始化
    """

    # 李航書中的delta, 表示在時刻t狀態為y的所有路徑的機率最大值，用一個字典表示一個時間點
    #V如果以矩陣表示的話會是len(obs)*len(states)=T*4的大小
    V = [{}]  # tabular
    #李航書中的psi是一個矩陣。這裡採用不同的實現方式：使用path儲存各個狀態最有可能的路徑
    path = {}
    for y in states:  # init
        #y代表4種狀態中的一個
        #在時間點0有路徑{y}的機率是為：在狀態y的初始機率乘上狀態y發射出obs[0]觀察值的機率
        #因為這裡是機率對數，所以用加的
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT)
        #在一開始時各狀態y的最大機率路徑都只包含它自己
        path[y] = [y]
    """
    算法10.5 (2)由t=0遞推到t=1...T-1(因為index是由0開始，所以最後一個是T-1)
    """
    #遞推：由前一時刻的path算出當前時刻的path
    for t in xrange(1, len(obs)):
        V.append({})
        #基於path(時刻t-1各個狀態機率最大的路徑)得到的，代表時刻t各個狀態的機率最大的路徑
        newpath = {}
        for y in states:
            #在狀態y發射觀察值obs[t]的機率對數
            em_p = emit_p[y].get(obs[t], MIN_FLOAT)
            """
            V[t - 1][y0] + trans_p[y0].get(y, MIN_FLOAT) + em_p:
              前一個時間點在y0的路徑的機率最大值*由y0轉移到y的機率*在狀態y0發射出y的機率
            for y0 in PrevStatus[y]:
              這裡使用PrevStatus這個字典獲取狀態y前一個時間點可能在什麼狀態，而不是使用所有的狀態
            max([(a1, b1), (a2, b2)]):
              會先比較a1與a2,如果一樣，繼續比較b1與b2
            state:
              李航書中的psi_t(y):如果時刻t時在狀態y，那麼在時刻t-1時最有可能在哪一個狀態？
            """
            (prob, state) = max(
                [(V[t - 1][y0] + trans_p[y0].get(y, MIN_FLOAT) + em_p, y0) for y0 in PrevStatus[y]])
            #時刻t在狀態y的路徑的機率最大值
            V[t][y] = prob
            #時刻t狀態y機率最大的路徑為時刻t-1狀態為state機率最大的路徑 加上 當前狀態y
            #註：path是前一時刻（時刻t-1）在各個狀態機率最大的路徑
            newpath[y] = path[state] + [y]
        #時刻t各個狀態機率最大的路徑
        path = newpath

    """
    算法10.5 (3)終止
    """
    """
    for y in 'ES':限制最後一個狀態只能是這兩個
    len(obs) - 1:最後一個時間點，即T-1
    prob:時間T-1所有路徑的機率最大值
    state:時間T-1最有可能在哪一個狀態上
    """
    (prob, state) = max((V[len(obs) - 1][y], y) for y in 'ES')

    """
    李航書中本來還有一步最優路徑回溯，
    但是這裡因為path的實現方式跟書中不同，所以不必回溯。
    這裡的path直接記錄了len(states)條路徑
    我們只要從中選一條機率最大的即可
    """
    #path[state]:終點是state的路徑
    return (prob, path[state])

"""
viterbi函數返回的是狀態序列及其機率值。在__cut函數中，調用了viterbi，並依據狀態序列來切分傳入的句子。
要注意的是：__cut函數只能接受全是漢字的句子當作輸入。
所以我們等一下會看到，它還會有一個wrapper，用來處理句子中包含英數字或其它符號的情況。
"""
def __cut(sentence):
    #為什麼只有emit_P是global?
    global emit_P
    # 向viterbi函數傳入觀察序列，可能狀態值以及三個矩陣
    # 得到機率最大的狀態序列及其機率
    prob, pos_list = viterbi(sentence, 'BMES', start_P, trans_P, emit_P)
    begin, nexti = 0, 0
    # print pos_list, sentence
    # 利用pos_list(即狀態序列)來切分sentence
    for i, char in enumerate(sentence):
        pos = pos_list[i]
        if pos == 'B':
            begin = i
        elif pos == 'E':
            yield sentence[begin:i + 1]
            nexti = i + 1
        elif pos == 'S':
            yield char
            nexti = i + 1
    """
    如果sentence[-1]是'E'或'S'，那麼句中的最後一個詞就會被yield出來，而nexti就派不上用場
    如果sentence[-1]是'B'或'M'，那麼在迴圈中就不會yield出最後一個詞，
    所以到迴圈外後我們需要nexti（表示上個詞詞尾的下一個字的位置），然後用yield來產生句中的最後一個詞
    """
    if nexti < len(sentence):
        yield sentence[nexti:]

"""
一個或多個漢字
"""
re_han = re.compile("([\u4E00-\u9FD5]+)")
"""
[a-zA-Z0-9]+ : 一個或多個英數字
\.\d+ : ".加上一個或多個數字"
(?:) : ()會創造一個捕獲性分組，而在組內加上?:則會讓該分組無法捕獲。
(?:\.\d+)? : 配對該group零次或一次
%? : 配對%零次或一次
"""
re_skip = re.compile("([a-zA-Z0-9]+(?:\.\d+)?%?)")

"""
這個函數用於更新Force_Split_Words這個變數，
而Force_Split_Words會被接下來介紹的cut函數所使用。
如果用戶有使用jieba/__init__.py裡的add_word(word, freq=None, tag=None)來新增詞彙，
並且該詞彙原本不存在於字典裡，那麼在add_word執行的過程中，
就會調用finalseg.add_force_split(word)。
可以看到add_force_split的作用是更新Force_Split_Words這個集合。
而如果用戶沒有使用jieba.add_word，那麼Force_Split_Words都會是一個空集合。
"""
def add_force_split(word):
    global Force_Split_Words
    Force_Split_Words.add(word)

"""
這是__cut函數的wrapper，它會把句中的漢字/非漢字的部份分離。
如果某個字段是漢字，才會呼叫__cut函數來分詞。
"""
def cut(sentence):
    sentence = strdecode(sentence)
    blocks = re_han.split(sentence)
    for blk in blocks:
        if re_han.match(blk):
            #呼叫__cut函數切分漢字
            for word in __cut(blk):
                if word not in Force_Split_Words:
                    yield word
                else:
                    #Force_Split_Words中的字會被強制切分
                    for c in word:
                        yield c
        else:
            #非漢字的部份
            tmp = re_skip.split(blk)
            for x in tmp:
                if x:
                    yield x
