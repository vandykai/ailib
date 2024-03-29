import re
import jieba
import random
import typing
from ailib.text.basic_data import ch_punctuation, en_punctuation
from ailib.text.text_is import is_en_word

# 清 除 中  english混合文本 中多余的空格->清除中 english混合文本中多余的空格
# 清除中英混合文本中多余的空格，转换中文空格为英文
def clean_space(text: str) -> str:
    text = re.sub(u"[ 　]+", " ", text)
    match_regex = re.compile(u'''[\u4e00-\u9fa5{0}]+ +(?![a-zA-Z])| +\d+|[a-zA-Z]+ +(?=[{1}])'''.format(ch_punctuation, ch_punctuation+en_punctuation))
    should_replace_list = match_regex.findall(text)
    order_replace_list = sorted(should_replace_list,key=lambda i:len(i),reverse=True)
    for word in order_replace_list:
        if word == u' ':
            continue
        new_word = word.strip()
        text = text.replace(word,new_word)
    return text

# 统计中英文总共词数
def count_word(text: str) -> int:
    ch_word_regex = re.compile(u'[\u4e00-\u9fa5]')
    en_word_regex = re.compile(r'[a-zA-Z\'\-]+')
    ch_word = ch_word_regex.findall(text)
    en_word = en_word_regex.findall(text)
    return len(ch_word) + len(en_word)

# 统计中文总共词数
def count_cn_word(text: str) -> int:
    ch_word_regex = re.compile(u'[\u4e00-\u9fa5]')
    ch_word = ch_word_regex.findall(text)
    return len(ch_word)

# 统计英文总共词数
def count_en_word(text: str) -> int:
    en_word_regex = re.compile(r'[a-zA-Z\'\-]+')
    en_word = en_word_regex.findall(text)
    return len(ch_word)

# 使用结巴分词随机切分句子中的短语，用来生产短文本, 可能产生空字符串
def jieba_random_cut(text: str) -> str:
    text = clean_space(text)
    text_list = jieba.lcut(text)
    text_list = list(filter(lambda x:x!=" ", text_list))
    a = random.randint(0, len(text_list)-1)
    b = random.randint(0, len(text_list))
    result = ""
    if a < b:
        result = " ".join(text_list[a:b])
    else:
        result = " ".join(text_list[b:a])
    result = clean_space(result)
    return result

# 使用结巴分词随机切分句子中的短语，用来生产范围内的短文本, 可能产生空字符串，length_range 取值同range, 包含下界不包含上界
def jieba_random_cut(text: str, length_range: tuple) -> str:
    length_range = (length_range[0], length_range[1]-1)
    text = clean_space(text)
    text_list = jieba.lcut(text)
    text_list = list(filter(lambda x:x!=" ", text_list))
    text_len_list = [count_word(word) for word in text_list]
    text_len = sum(text_len_list)
    text_len_cut_min = text_len-length_range[1]
    text_len_cut_max = text_len-length_range[0]
    # 字符长度不够，返回空
    if text_len_cut_max < 0:
        return ""
    # 找到可能开始点范围
    min_index, max_index = -1, -1
    sum_length = 0
    for i in range(len(text_len_list)):
        sum_length +=  text_len_list[i]
        if min_index < 0 and sum_length > text_len_cut_min:
            min_index = max(0, i)
        if sum_length > text_len_cut_max:
            max_index = max(0, i)
            break
    if max_index < 0:
        max_index = len(text_len_list)
    start_index = random.randint(0, max_index)
    # 找到可能结束点范围
    min_index, max_index = -1, -1
    sum_length = 0
    for i in range(start_index, len(text_len_list)):
        sum_length +=  text_len_list[i]
        if min_index < 0 and sum_length >= length_range[0]:
            min_index = max(0, i)
        if sum_length > length_range[1]:
            max_index = max(0, i-1)
            break
    if min_index < 0:
        min_index = len(text_len_list)-1
        max_index = len(text_len_list)-1
    elif max_index < 0:
        max_index = len(text_len_list)-1
    # 一不小心随机到无解了
    if max_index < min_index:
        return ""
    end_index = random.randint(min_index, max_index)
    result = " ".join(text_list[start_index:end_index+1])
    result = clean_space(result)
    return result

def file_de_duplication_line(file_name):
    all_line = []
    with open(file_name) as f:
        for line in f:
            line = line.strip()
            if line not in temp_line:
                all_line.append(line)
    with open(file_name, "w") as f:
        for line in all_line:
            f.write(line+"\n")

def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")