ch_punctuation = r"""＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。　"""
en_punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ """
ch_en_punctuation = ch_punctuation + en_punctuation

en_punctuation_regex = r'\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~\ '
ch_punctuation_regex = r'\＂\＃\＄\％\＆\＇\（\）\＊\＋\，\－\／\：\；\＜\＝\＞\＠\［\＼\］\＾\＿\｀\｛\｜\｝\～\｟\｠\｢\｣\､\、\〃\〈\〉\《\》\「\」\『\』\【\】\〔\〕\〖\〗\〘\〙\〚\〛\〜\〝\〞\〟\〰\〾\〿\–\—\‘\’\‛\“\”\„\‟\…\‧\﹏\﹑\﹔\·\！\？\｡\。\　'
ch_en_punctuation_regex = en_punctuation_regex + ch_punctuation_regex

# 字符集	字数	Unicode 编码
# 基本汉字	20902字	4E00-9FA5
# 基本汉字补充	74字	9FA6-9FEF
# 扩展A	6582字	3400-4DB5
# 扩展B	42711字	20000-2A6D6
# 扩展C	4149字	2A700-2B734
# 扩展D	222字	2B740-2B81D
# 扩展E	5762字	2B820-2CEA1
# 扩展F	7473字	2CEB0-2EBE0
# 扩展G	4939字	30000-3134A
# 康熙部首	214字	2F00-2FD5
# 部首扩展	115字	2E80-2EF3
# 兼容汉字	477字	F900-FAD9
# 兼容扩展	542字	2F800-2FA1D
# PUA(GBK)部件	81字	E815-E86F
# 部件扩展	452字	E400-E5E8
# PUA增补	207字	E600-E6CF
# 汉字笔画	36字	31C0-31E3
# 汉字结构	12字	2FF0-2FFB
# 汉语注音	43字	3105-312F
# 注音扩展	22字	31A0-31BA
# 〇	1字	3007
zh_unicode_basic = r'\u4E00-\u9FA5'
zh_unicode_all = r'\u4E00-\u9FA5\u9FA6-\u9FEF\u3400-\u4DB5\u20000-\u2A6D6\u2A700-\u2B734\u2B820-\u2CEA1\u2CEB0-\u2EBE0\u30000-\u3134A\u2F00-\u2FD5\u2E80-\u2EF3\uF900-\uFAD9\u2F800-\u2FA1D\uE815-\uE86F\uE400-\uE5E8\uE600-\uE6CF\u31C0-\u31E3\u2FF0-\u2FFB\u3105-\u312F\u31A0-\u31BA\u3007'

is_ch_word_regex = r'^[{}]+$'.format(zh_unicode_basic)
is_all_ch_word_regex = r'^[{}][{}{}]+$'.format(zh_unicode_basic, zh_unicode_basic, ch_en_punctuation_regex)
is_all_en_word_regex = r'^[a-zA-Z][a-zA-Z{}]+$'.format(ch_en_punctuation_regex)