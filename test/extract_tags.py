import sys
sys.path.append('../')

import jieba
import jieba.analyse
from optparse import OptionParser

"""
參考[Python2文檔 - 15.5. optparse — Parser for command line options]
(https://docs.python.org/2/library/optparse.html#creating-the-parser)
此處USAGE參數代表的是help message
"""
USAGE = "usage:    python extract_tags.py [file name] -k [top k]"

parser = OptionParser(USAGE)
"""
from https://docs.python.org/2/library/optparse.html#optparse.Option.dest:
If the option’s action implies writing or modifying a value somewhere, 
this tells optparse where to write it: 
dest names an attribute of the options object that optparse builds 
as it parses the command line.

總結一下，就是當使用者輸入-k xxx時，parser.parse_args()回傳的opt的topK屬性就會被設為xxx
"""
parser.add_option("-k", dest="topK")
"""
from https://docs.python.org/2/library/optparse.html#module-optparse:
parse_args() returns two values:
options, an object containing values for all of your options—e.g. if --file takes a single string argument, then options.file will be the filename supplied by the user, or None if the user did not supply that option
args, the list of positional arguments leftover after parsing options
parser.parse_args()會回傳options及args兩個物件
options代表可選參數，args代表位置參數
在本例中[file name]為位置參數，-k [top k]為可選參數

As it parses the command line, optparse sets attributes of the options 
object returned by parse_args() based on user-supplied command-line values.
parser.parse_args()會回傳一個opt物件，opt的topK屬性由使用者輸入的參數決定
"""
opt, args = parser.parse_args()


#代表使用者沒有輸入位置參數[file name]
if len(args) < 1:
    print(USAGE)
    sys.exit(1)

file_name = args[0]

# 使用opt.topK來獲取可選參數topK
if opt.topK is None:
    topK = 10
else:
    topK = int(opt.topK)

content = open(file_name, 'rb').read()

tags = jieba.analyse.extract_tags(content, topK=topK)

print(",".join(tags))
