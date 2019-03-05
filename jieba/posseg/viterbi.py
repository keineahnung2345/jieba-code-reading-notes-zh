import sys
import operator
MIN_FLOAT = -3.14e100
MIN_INF = float("-inf")

# 處理Python2/3相容的問題
# 為何不復用_compat.py?
if sys.version_info[0] > 2:
    xrange = range

# 沒有被用到
def get_top_states(t_state_v, K=4):
    return sorted(t_state_v, key=t_state_v.__getitem__, reverse=True)[:K]


def viterbi(obs, states, start_p, trans_p, emit_p):
    """
    請參考李航書中的算法10.5(維特比算法)
    
    HMM共有五個參數，分別是觀察值集合(句子本身, obs)，
    狀態值集合(all_states, 即trans_p.keys())，
    初始機率(start_p)，狀態轉移機率矩陣(trans_p)，發射機率矩陣(emit_p)

    此處的states是為char_state_tab_P，
    這是一個用來查詢漢字可能狀態的字典

    此處沿用李航書中的符號，令T=len(obs)，令N=len(trans_p.keys())
    """
    
    """
    維特比算法第1步:初始化
    """
    #V:李航書中的delta，在時刻t狀態為i的所有路徑中之機率最大值
    V = [{}]  # tabular
    #李航書中的Psi，T乘N維的矩陣
    #表示在時刻t狀態為i的所有單個路徑(i_1, i_2, ..., i_t-1, i)中概率最大的路徑的第t-1個結點
    mem_path = [{}]
    #共256種狀態，所謂"狀態"是:"分詞標籤(BMES)及詞性(v, n, nr, d, ...)的組合"
    all_states = trans_p.keys()
    #obs[0]表示句子的第一個字
    #states.get(obs[0], all_states)表示該字可能是由哪些狀態發射出來的
    for y in states.get(obs[0], all_states):  # init
        #在時間點0，狀態y的log機率為：
        #一開始在y的log機率加上在狀態y發射obs[0]觀察值的log機率
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT)
        #時間點0在狀態y，則前一個時間點會在哪個狀態
        mem_path[0][y] = ''
        
    """
    維特比算法第2步:遞推
    """
    #obs: 觀察值序列
    for t in xrange(1, len(obs)):
        V.append({})
        mem_path.append({})
        #prev_states = get_top_states(V[t-1])
        #mem_path[t - 1].keys(): 前一個時間點在什麼狀態，這裡以x代表
        #只有在len(trans_p[x])>0(即x有可能轉移到其它狀態)的情況下，prev_states才保留x
        prev_states = [
            x for x in mem_path[t - 1].keys() if len(trans_p[x]) > 0]

        #前一個狀態是x(prev_states中的各狀態)，那麼現在可能在什麼狀態(y)
        prev_states_expect_next = set(
            (y for x in prev_states for y in trans_p[x].keys()))
        #set(states.get(obs[t], all_states)):句子的第t個字可能在什麼狀態
        #prev_states_expect_next:由前一個字推斷，當前的字可能在什麼狀態
        #obs_states:以上兩者的交集
        obs_states = set(
            states.get(obs[t], all_states)) & prev_states_expect_next

        #如果交集為空，則依次選取prev_states_expect_next或all_states
        if not obs_states:
            obs_states = prev_states_expect_next if prev_states_expect_next else all_states

        for y in obs_states:
            #李航書中的公式10.45
            #y0表示前一個時間點的狀態
            #max的參數是一個list of tuple: [(機率1，狀態1)，(機率2，狀態2)，...]
            #V[t - 1][y0]:時刻t-1在狀態y0的機率對數
            #trans_p[y0].get(y, MIN_INF):由狀態y0轉移到y的機率對數
            #emit_p[y].get(obs[t], MIN_FLOAT):在狀態y發射出觀測值obs[t]的機率對數
            #三項之和表示在時刻t由狀態y0到達狀態y的路徑的機率對數
            prob, state = max((V[t - 1][y0] + trans_p[y0].get(y, MIN_INF) +
                               emit_p[y].get(obs[t], MIN_FLOAT), y0) for y0 in prev_states)
            #挑選機率最大者將之記錄於V及mem_path
            V[t][y] = prob
            #時刻t在狀態y，則時刻t-1最有可能在state這個狀態
            mem_path[t][y] = state
    
    """
    維特比算法第3步:終止
    """
    #mem_path[-1].keys():最後一個時間點可能在哪些狀態
    #V[-1][y]:最後一個時間點在狀態y的機率
    #把mem_path[-1]及V[-1]打包成一個list of tuple
    last = [(V[-1][y], y) for y in mem_path[-1].keys()]
    # if len(last)==0:
    #     print obs
    #最後一個時間點最有可能在狀態state，其機率為prob
    #在jieba/finalseg/__init__.py的viterbi函數中有限制句子末字的分詞標籤需為E或S
    #這裡怎麼沒做這個限制?
    prob, state = max(last)

    """
    維特比算法第4步:最優路徑回溯
    """
    route = [None] * len(obs)
    i = len(obs) - 1
    while i >= 0:
        route[i] = state
        #時間點i在狀態state，則前一個時間點最有可能在狀態mem_path[i][state]
        state = mem_path[i][state]
        i -= 1
    return (prob, route)
