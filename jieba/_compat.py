# -*- coding: utf-8 -*-
# _compat.py裡定義了讀取字典時會用到的函數，它處理了Python2/3的相容性問題
import os
import sys

"""
函數名(*res):
如果函數有個帶*號的參數，這就代表在呼叫該函數時可以傳入任意個引數
"""
"""
__name__, __file__:
__file__變數指的是當前的.py檔案的路徑，而__name__則是當前由python import的模組的名稱。
其中__name__變數的值是會根據導入模組方式的不同而改變的。
"""
"""
pkg_resources.resource_stream:
pkg_resources.resource_stream函數有兩個參數，分別是package_or_requirement及resource_name。
如果傳入的package_or_requirement是一個模組的名字，那麼這個函數會以該模組名為參考，
找到resource_name，然後載入並回傳它的檔案物件。
"""
"""
os.path.normpath:
它會將傳入的參數path中多餘的/或\移除(即正規化)後回傳。
"""
"""
get_module_res:
開啟文字檔後回傳檔案物件
"""
try:
    import pkg_resources
    get_module_res = lambda *res: pkg_resources.resource_stream(__name__,
                                                                os.path.join(*res))
except ImportError:
    get_module_res = lambda *res: open(os.path.normpath(os.path.join(
                            os.getcwd(), os.path.dirname(__file__), *res)), 'rb')

"""
統一Python2/3函數的名稱

這裡首先判斷Python版本是否為Python2，並將它存到PY2這個變數裡。
接著是依據Python2/3的特性，一一定義text_type及stringe_types等。

在Python3中，xrange變成range，遍歷字典的方式也跟Python2有所不同。
這裡建立了數個函數，並將它們指向Python3中相對應功能的函數。這樣一來，我們就可以統一以Python2的方式來呼叫他們。
"""
PY2 = sys.version_info[0] == 2

default_encoding = sys.getfilesystemencoding()

if PY2:
    text_type = unicode
    string_types = (str, unicode)

    iterkeys = lambda d: d.iterkeys()
    itervalues = lambda d: d.itervalues()
    iteritems = lambda d: d.iteritems()

else:
    text_type = str
    string_types = (str,)
    xrange = range

    iterkeys = lambda d: iter(d.keys())
    itervalues = lambda d: iter(d.values())
    iteritems = lambda d: iter(d.items())

"""
在使用Python 3的情況下，如果傳入的sentence不是字串型別，而是bytes型別，
就將它以utf-8編碼轉換成字串。如果解碼失敗，則改以gbk編碼(簡體中文)來轉換。
所以這個函數的作用就是確保sentence是字串型別後回傳。
"""
def strdecode(sentence):
    if not isinstance(sentence, text_type):
        try:
            sentence = sentence.decode('utf-8')
        except UnicodeDecodeError:
            sentence = sentence.decode('gbk', 'ignore')
    return sentence

"""
使用f = open('xxx.txt', 'r')會得到一個_io.TextIOWrapper型別的對象f。
resolve_filename接受一個_io.TextIOWrapper型別的對象當作參數，獲取它的檔名後回傳。
"""
def resolve_filename(f):
    try:
        return f.name
    except AttributeError:
        return repr(f)
