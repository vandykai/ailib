import re
import tqdm
import typing


# latex公式 (mathjax语法格式，标记为'$$')
pat_formula = re.compile(r'\$\$.+?\$\$')

pat_clean_formula = re.compile(r'((\\not)?\\[a-zA-Z0-9]+)|\\(.)|_|^|\{|\}|\$\$')

# html标记  (标记<*> 转义符)
# pat_html_tags = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
pat_html_tags = re.compile(r'<.*?>|&([a-z0-9]+);|&#([0-9]{1,6});|&#x([0-9a-f]{1,6});')

pat_non_formula = re.compile(r'\$\$([a-zA-Z0-9]+)\$\$')
pat_all_lower = re.compile(r'[a-z]+')
pat_sub_spaces = re.compile(r'\xa0|\u200b')

# 普通符号      [\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e]       [!-/:-@\[-`\{-~]
# 汉字          [\u3400-\u4dbf\u4e00-\u9fff\u2f00-\u2fd5\u2e80-\u2ef3\uf900-\ufaff]
# 希腊字母      [\u0370-\u03ff]
# 数学符号      [\u2200-\u22ff\u2a00-\u2aff]
# 罗马数字      [\u2160-\u217f]
# 带圈数字      [\u2460-\u24ff]
# 常用符号      ，|（|）|：|、|；|？|。|《|》|“|”|！
# 全角符号      [\uff00-\uffef\u3000-\u303f]
# 几何形状      [\u25a0-\u25ff]

# ASCII字符元素：字母 数字 单词 符号
reg_str_element_ascii = r'[a-zA-Z0-9]+|[\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e]'
# 汉字 + 常用标点符号
reg_str_element_hanzi = r'[\u3400-\u4dbf\u4e00-\u9fff\u2f00-\u2fd5\u2e80-\u2ef3\uf900-\ufaff]|，|（|）|：|、|；|？|。|《|》|“|”|！'
# 希腊字母 + 数学符号 + 几何形状 + 罗马数字 + 带圈数字
reg_str_math_symbols = r'[\u0370-\u03ff\u2200-\u22ff\u25a0-\u25ff\u2a00-\u2aff\u2160-\u217f\u2460-\u24ff]'

pat_cut_element = re.compile('|'.join([reg_str_element_ascii, reg_str_element_hanzi, reg_str_math_symbols]))

# entity name -> enity char
map_html_entities = {
    'amp': '&', 'lt': '<', 'gt': '>', 
    'Agrave': 'À',  'Aacute': 'Á',  'Acirc': 'Â',  'Atilde': 'Ã', 
    'Auml': 'Ä',  'Aring': 'Å',  'AElig': 'Æ',  'Ccedil': 'Ç', 
    'Egrave': 'È',  'Eacute': 'É',  'Ecirc': 'Ê',  'Euml': 'Ë', 
    'Igrave': 'Ì',  'Iacute': 'Í',  'Icirc': 'Î',  'Iuml': 'Ï', 
    'ETH': 'Ð',  'Ntilde': 'Ñ',  'Ograve': 'Ò',  'Oacute': 'Ó', 
    'Ocirc': 'Ô',  'Otilde': 'Õ',  'Ouml': 'Ö',  'Oslash': 'Ø', 
    'Ugrave': 'Ù',  'Uacute': 'Ú',  'Ucirc': 'Û',  'Uuml': 'Ü', 
    'Yacute': 'Ý',  'THORN': 'Þ',  'szlig': 'ß',  'agrave': 'à', 
    'aacute': 'á',  'acirc': 'â',  'atilde': 'ã',  'auml': 'ä', 
    'aring': 'å',  'aelig': 'æ',  'ccedil': 'ç',  'egrave': 'è', 
    'eacute': 'é',  'ecirc': 'ê',  'euml': 'ë',  'igrave': 'ì', 
    'iacute': 'í',  'icirc': 'î',  'iuml': 'ï',  'eth': 'ð', 
    'ntilde': 'ñ',  'ograve': 'ò',  'oacute': 'ó',  'ocirc': 'ô', 
    'otilde': 'õ',  'ouml': 'ö',  'oslash': 'ø',  'ugrave': 'ù', 
    'uacute': 'ú',  'ucirc': 'û',  'uuml': 'ü',  'yacute': 'ý', 
    'thorn': 'þ',  'yuml': 'ÿ',  'nbsp': ' ',  'iexcl': '¡', 
    'cent': '¢',  'pound': '£',  'curren': '¤',  'yen': '¥', 
    'brvbar': '¦',  'sect': '§',  'uml': '¨',  'copy': '©', 
    'ordf': 'ª',  'laquo': '«',  'not': '¬',  'shy': '', 
    'reg': '®',  'macr': '¯',  'deg': '°',  'plusmn': '±', 
    'sup2': '²',  'sup3': '³',  'acute': '´',  'micro': 'µ', 
    'para': '¶',  'cedil': '¸',  'sup1': '¹',  'ordm': 'º', 
    'raquo': '»',  'frac14': '¼',  'frac12': '½',  'frac34': '¾', 
    'iquest': '¿',  'times': '×',  'divide': '÷',  'forall': '∀', 
    'part': '∂',  'exist': '∃',  'empty': '∅',  'nabla': '∇', 
    'isin': '∈',  'notin': '∉',  'ni': '∋',  'prod': '∏', 
    'sum': '∑',  'minus': '−',  'lowast': '∗',  'radic': '√', 
    'prop': '∝',  'infin': '∞',  'ang': '∠',  'and': '∧', 
    'or': '∨',  'cap': '∩',  'cup': '∪',  'int': '∫', 
    'there4': '∴',  'sim': '∼',  'cong': '≅',  'asymp': '≈', 
    'ne': '≠',  'equiv': '≡',  'le': '≤',  'ge': '≥', 
    'sub': '⊂',  'sup': '⊃',  'nsub': '⊄',  'sube': '⊆', 
    'supe': '⊇',  'oplus': '⊕',  'otimes': '⊗',  'perp': '⊥', 
    'sdot': '⋅',  'Alpha': 'Α',  'Beta': 'Β',  'Gamma': 'Γ', 
    'Delta': 'Δ',  'Epsilon': 'Ε',  'Zeta': 'Ζ',  'Eta': 'Η', 
    'Theta': 'Θ',  'Iota': 'Ι',  'Kappa': 'Κ',  'Lambda': 'Λ', 
    'Mu': 'Μ',  'Nu': 'Ν',  'Xi': 'Ξ',  'Omicron': 'Ο', 
    'Pi': 'Π',  'Rho': 'Ρ',  'Sigma': 'Σ',  'Tau': 'Τ', 
    'Upsilon': 'Υ',  'Phi': 'Φ',  'Chi': 'Χ',  'Psi': 'Ψ', 
    'Omega': 'Ω',  'alpha': 'α',  'beta': 'β',  'gamma': 'γ', 
    'delta': 'δ',  'epsilon': 'ε',  'zeta': 'ζ',  'eta': 'η', 
    'theta': 'θ',  'iota': 'ι',  'kappa': 'κ',  'lambda': 'λ', 
    'mu': 'μ',  'nu': 'ν',  'xi': 'ξ',  'omicron': 'ο', 
    'pi': 'π',  'rho': 'ρ',  'sigmaf': 'ς',  'sigma': 'σ', 
    'tau': 'τ',  'upsilon': 'υ',  'phi': 'φ',  'chi': 'χ', 
    'psi': 'ψ',  'omega': 'ω',  'thetasym': 'ϑ',  'upsih': 'ϒ', 
    'piv': 'ϖ',  'OElig': 'Œ',  'oelig': 'œ',  'Scaron': 'Š', 
    'scaron': 'š',  'Yuml': 'Ÿ',  'fnof': 'ƒ',  'circ': 'ˆ', 
    'tilde': '˜',  'ensp': '',  'emsp': '',  'thinsp': '', 
    'zwnj': '',  'ndash': '–',  'mdash': '—',  'lsquo': '‘', 
    'rsquo': '’',  'sbquo': '‚',  'ldquo': '“',  'rdquo': '”', 
    'bdquo': '„',  'dagger': '†',  'Dagger': '‡',  'bull': '•', 
    'hellip': '…',  'permil': '‰',  'prime': '′',  'Prime': '″', 
    'lsaquo': '‹',  'rsaquo': '›',  'oline': '‾',  'euro': '€', 
    'trade': '™',  'larr': '←',  'uarr': '↑',  'rarr': '→', 
    'darr': '↓',  'harr': '↔',  'crarr': '↵',  'lceil': '⌈', 
    'rceil': '⌉',  'lfloor': '⌊',  'rfloor': '⌋',  'loz': '◊', 
    'spades': '♠',  'clubs': '♣',  'hearts': '♥',  'diams': '♦'
}


def remove_html(text: str) -> str:
    """
    删除文本中html标签，把转义符替换成相应的符号
    """
    ret = ''
    pos = 0
    for tag in pat_html_tags.finditer(text):
        st, ed = tag.span()
        if st > pos:
            ret += text[pos:st]
        t1 = tag.group(1)
        t2 = tag.group(2)
        t3 = tag.group(3)
        if t1 is not None:
            if t1 in map_html_entities:
                ret += map_html_entities[t1]
        elif t2 is not None:
            ret += chr(int(t2))
        elif t3 is not None:
            ret += chr(int(t3, 16))
        pos = ed
    ret += text[pos:]
    return ret
    # return re.sub(pat_html_tags, ' ', text)


map_latex_word = {
    # 希腊字母
    '\\alpha': 'α', '\\beta': 'β', '\\gamma': 'γ', '\\delta': 'δ', '\\epsilon': 'ϵ', '\\zeta': 'ζ', '\\eta': 'η', '\\theta': 'θ', 
    '\\iota': 'ι', '\\kappa': 'κ', '\\lambda': 'λ', '\\mu': 'μ', '\\nu': 'ν', '\\xi': 'ξ', '\\omicron': 'ο', '\\pi': 'π', 
    '\\rho': 'ρ', '\\sigma': 'σ', '\\tau': 'τ', '\\upsilon': 'υ', '\\phi': 'ϕ', '\\chi': 'χ', '\\psi': 'ψ', '\\omega': 'ω', 
    '\\Alpha': 'Α', '\\Beta': 'Β', '\\Gamma': 'Γ', '\\Delta': 'Δ', '\\Epsilon': 'Ε', '\\Zeta': 'Ζ', '\\Eta': 'Η', '\\Theta': 'Θ', 
    '\\Iota': 'I', '\\Kappa': 'Κ', '\\Lambda': 'Λ', '\\Mu': 'Μ', '\\Nu': 'Ν', '\\Xi': 'Ξ', '\\Omicron': 'Ο', '\\Pi': 'Π', 
    '\\Rho': 'Ρ', '\\Sigma': 'Σ', '\\Tau': 'Τ', '\\Upsilon': 'Υ', '\\Phi': 'Φ', '\\Chi': 'Χ', '\\Psi': 'Ψ', '\\Omega': 'Ω', 
    '\\varepsilon': 'ε', '\\varkappa': 'ϰ',  '\\varphi': 'φ', '\\varpi': 'ϖ', '\\varrho': 'ϱ', '\\varsigma': 'ς', 
    '\\vartheta': 'ϑ', '\\varOmega': 'Ω', '\\varPhi': 'Φ', '\\varGamma': 'Γ',
    # 关系符号
    '\\allequal': '≌', '\\approx': '≈', '\\approxeq': '≊', '\\approxnotequal': '≆', '\\asymp': '≍', '\\backsim': '∽',  '\\cong': '≅', 
    '\\doteq': '≐', '\\doteqdot': '≑', '\\equiv': '≡', '\\geq': '≥', '\\ge': '≥', '\\gt': '>',  '\\geqq': '≧', '\\geqslant': '⩾', 
    '\\gg': '≫', '\\gneq': '⪈', '\\gneqq': '≩', '\\greaterequivlnt': '≳',  '\\lazysinv': '∾', '\\leq': '≤', '\\le': '≤', '\\lt': '<', 
    '\\leqq': '≦', '\\leqslant': '⩽', '\\lessequivlnt': '≲',  '\\ll': '≪', '\\lneq': '⪇', '\\lneqq': '≨', '\\neq': '≠', '\\ne': '≠', 
    '\\not': '/', '\\ngeq': '≱', '\\ngtr': '≯',  '\\nleq': '≰', '\\nless': '≮', '\\not\\approx': '≉', '\\not\\cong': '≇', 
    '\\not\\equiv': '≢', '\\not\\sim': '≁',  '\\not\\simeq': '≄', '\\nprec': '⊀', '\\nsucc': '⊁', '\\prec': '≺', '\\precapprox': '≾', 
    '\\preccurlyeq': '≼',  '\\preceq': '⪯', '\\precneqq': '⪵', '\\sim': '∼', '\\simeq': '≃', '\\succ': '≻', '\\succapprox': '≿',  
    '\\succcurlyeq': '≽', '\\succeq': '⪰', '\\succneqq': '⪶', '\\tildetrpl': '≋', '\\wr': '≀', '\\backsim ': '∽', '\\backsimeq': '⋍',
    # 集合符号
    '\\aleph': 'ℵ', '\\And': '&', '\\and': '∧', '\\backepsilon': '∍', '\\because': '∵', '\\beth': 'ℶ', '\\bigcap': '⋂', '\\bigcup': '⋃', 
    '\\bot': '⊥', '\\Box': '□', '\\cap': '∩', '\\cup': '∪', '\\emptyset': '∅', '\\varnothing': '⌀', '\\exists': '∃', '\\forall': '∀', 
    '\\in': '∈', '\\land': '∧', '\\lor': '∨', '\\models': '⊨', '\\neg': '¬', '\\nexists': '∄', '\\ni': '∋', '\\not\\exists': '∄', 
    '\\not\\in': '∉', '\\not\\ni': '∌', '\\not\\subset': '⊄', '\\not\\subseteq': '⊈', '\\not\\supset': '⊅', '\\not\\supseteq': '⊉', 
    '\\notin': '∉', '\\nsubseteq': '⊈', '\\nsupseteq': '⊉', '\\or': '∨', '\\setminus': '∖', '\\subset': '⊂', '\\subseteq': '⊆', 
    '\\subseteqq': '⫅', '\\subsetneq': '⊊', '\\subsetneqq': '⫋', '\\supset': '⊃', '\\supseteq': '⊇', '\\supseteqq': '⫆', 
    '\\supsetneq': '⊋',  '\\supsetneqq': '⫌', '\\therefore': '∴', '\\top': '⊤', '\\vdash': '⊢', '\\vee': '∨', '\\veebar': '⊻', 
    '\\wedge': '∧',  '\\varsubsetneq': '⊊', '\\varsubsetneqq': '⫋', '\\varsupsetneq': '⊋', '\\varsupsetneqq': '⫌',
    # 箭头
    '\\rightarrow': '→', '\\leftarrow': '←', '\\uparrow': '↑', '\\downarrow': '↓', '\\nwarrow': '↖', '\\searrow': '↘', '\\Rightarrow': '⇒', 
    '\\Leftarrow': '⇐', '\\Uparrow': '⇑', '\\Downarrow': '⇓', '\\nearrow': '↗', '\\swarrow': '↙', '\\mapsto': '↦', '\\to': '→', 
    '\\leftrightarrow': '↔', '\\Leftrightarrow': '⇔', '\\rightarrowtail': '↣', '\\twoheadrightarrow': '↠', '\\hookrightarrow': '↪', 
    '\\rightsquigarrow': '⇝', '\\rightleftharpoons': '⇌', '\\rightharpoonup': '⇀', '\\hookleftarrow': '↩', '\\leftarrowtail': '↢', 
    '\\twoheadleftarrow': '↞', '\\leftrightharpoons': '⇋', '\\rightharpoondown': '⇁', '\\arrowvert': '|', '\\Arrowvert': '‖', 
    '\\lvert': '|', '\\rvert': '|', '\\lVert ': '‖', '\\rVert': '‖',
    # 运算符
    '\\times': '×', '\\div': '÷', '\\infty': '∞', '\\nabla': '∇', '\\partial': '∂', '\\sum': '∑', '\\prod': '∏', 
    '\\coprod': '∐', '\\int': '∫', '\\iint': '∬', '\\iiint': '∭', '\\iiiint': '⨌', '\\oint': '∮', 
    '\\surfintegral': '∯', '\\volintegral': '∰', '\\Re': 'ℜ', '\\Im': 'ℑ', '\\wp': '℘',
    # 函数名
    '\\arccos': 'arccos', '\\arcsin': 'arcsin', '\\arctan': 'arctan', '\\arg': 'arg', '\\cos': 'cos', '\\cosh': 'cosh', '\\cot': 'cot', 
    '\\coth': 'coth', '\\csc': 'csc', '\\deg': 'deg', '\\det': 'det', '\\dim': 'dim', '\\exp': 'exp', '\\gcd': 'gcd', '\\hom': 'hom', 
    '\\inf': 'inf', '\\ker': 'ker', '\\lg': 'lg', '\\lim': 'lim', '\\liminf': 'lim inf', '\\limsup': 'lim sup', '\\ln': 'ln', 
    '\\log': 'log', '\\max': 'max', '\\min': 'min', '\\bmod': 'mod', '\\mod': 'mod', '\\Pr': 'Pr', '\\sec': 'sec', '\\sin': 'sin', 
    '\\sinh': 'sinh', '\\sup': 'sup', '\\tan': 'tan', '\\tanh': 'tanh',
    # 其他
    '\\angle': '∠', '\\backprime': '‵', '\\backslash': '\\', '\\bullet': '∙', '\\cdot': '⋅', '\\cdots': '…', '\\circ': '∘', '\\dotplus': '∔', 
    '\\ell': 'l', '\\flat': '♭', '\\langle': '⟨', '\\lceil': '⌈', '\\ldotp': '..', '\\ldots': '…', '\\lfloor': '⌊', '\\measuredangle': '∡', 
    '\\mid': '∣', '\\mp': '∓', '\\natural': '♮', '\\nmid': '∤', '\\nparallel': '∦', '\\odot': '⊙', '\\ominus': '⊖', '\\oplus': '⊕', 
    '\\oslash': '⊘', '\\otimes': '⊗', '\\parallel': '∥', '\\pm': '±', '\\propto': '∝', '\\rangle': '⟩', '\\rceil': '⌉', '\\rfloor': '⌋', 
    '\\sharp': '♯', '\\sphericalangle': '∢', '\\surd': '√',
}


def clean_latex(text: str) -> str:
    """
    对latex元素进行转义，保留公式中的普通文本
    """
    ret = ''
    pos = 0
    for item in pat_clean_formula.finditer(text):
        st, ed = item.span()
        if st > pos:
            ret += text[pos:st]
        k = item.group(1)
        s = item.group(3)
        if k is not None:
            if k in map_latex_word:
                ret += map_latex_word[k]
        elif s is not None:
            ret += s
        pos = ed
    ret += text[pos:]
    return ret


def remove_latex(text: str) -> str:
    """
    清除latex公式格式，对部分元素进行转义，保留公式中的普通文本
    """
    if '$' in text:
        ret = ''
        pos = 0
        for formula in pat_formula.finditer(text):
            st, ed = formula.span()
            if st > pos:
                ret += text[pos:st]
            # ff = re.sub(pat_clean_formula, ' ', text[st:ed])
            ff = clean_latex(text[st:ed])
            ret += ff
            pos = ed
        ret += text[pos:]
        return ret
    else:
        return text


def fenci_char(text: str) -> str:
    """
    按字符分词，连续的字母数字作一个元素，汉字、标点、符号等单个字符为一个元素，其他字符忽略

    TODO - 汉语分词?
    """
    text = remove_html(text)
    text = remove_latex(text)
    ret = list()
    for element in pat_cut_element.findall(text):
        ret.append(element)
    return ' '.join(ret)


pat_prep_word = re.compile('( _\\w+_ )')
map_prep_words = {
    ' _lp_ ': '(', ' _rp_ ': ')', ' _lbrt_ ': '[', ' _rbrt_ ': ']', ' _lbrc_ ': '{', ' _rbrc_ ': '}', ' _crt_ ': '^', 
    ' _pls_ ': '+', ' _mns_ ': '-', ' _mlt_ ': '*', ' _dvd_ ': '/', ' _lt_ ': '<', ' _mt_ ': '>', ' _eq_ ': '='
}

def restore_prep(text: str) -> str:
    """
    还原后端预算理替换的文本，例如把 ' _eq_ ' 替换为 '='
    """
    ret = ''
    pos = 0
    for item in pat_prep_word.finditer(text):
        st, ed = item.span()
        if st > pos:
            ret += text[pos:st]
        word = item.group(1)
        if word is not None:
            if word in map_prep_words:
                ret += map_prep_words[word]
            else:
                ret += word    # 不在转换列表中的保留原样
        pos = ed
    ret += text[pos:]
    return ret


def proc_file_content(src_file: str, dst_file: str) -> None:
    """
    将src_file内容按行处理之后保存至dst_file  (1. 删除html标记 2.char分词)
    """
    with open(src_file, 'r') as fin, open(dst_file, 'w') as fout:
        for line in tqdm.tqdm(fin.readlines(), ncols=68, desc=src_file):
            pline = fenci_char(line)
            if pline is not None and len(pline) > 0:
                fout.write(pline)
                fout.write('\n')


def get_char_histo(path: typing.Union[str, typing.Collection[str]]) -> typing.Dict[str, int]:
    """
    统计文件中字符
    """
    ret = dict()
    if type(path) is str:
        path = [path]
    for file in path:
        print(file)
        with open(file, 'r') as fin:
            for line in fin:
                for ch in line.strip(' \t\n'):
                    if ch in ret:
                        ret[ch] += 1
                    else:
                        ret[ch] = 1
    return ret


def get_formula_histo(path: typing.Union[str, typing.Collection[str]]) -> typing.Dict[str, int]:
    """
    统计文件中latex公式元素 (  '\\frac'  等)
    """
    ret = dict()
    pat_formula_words = re.compile(r'\\[a-zA-Z]+')
    if type(path) is str:
        path = [path]
    for file in path:
        print(file)
        with open(file, 'r') as fin:
            for line in fin:
                if '$$' in line:
                    for formula in pat_formula.findall(line):
                        # print(formula)
                        for word in pat_formula_words.findall(formula):
                            if word in ret:
                                ret[word] += 1
                            else:
                                ret[word] = 1
    return ret


# id  (128位的md5结果 <--> 32个hex字符)
pat_question_id = re.compile(r'[a-fA-F0-9]{32}')
# id候选 （排除掉包含id的长串）
pat_id_candidate = re.compile(r'[a-zA-Z0-9]{32,}')
def cout_id(text: str) -> typing.Tuple[int, int]:
    """
    统计文本中的id串，返回结果为 id串数量，id串总字符数
    """
    num = 0
    cnt = 0
    for word in pat_id_candidate.findall(text):
        if pat_question_id.fullmatch(word):
            num += 1
            cnt += len(word)
    return (num, cnt)


def count_formula(text: str) -> typing.Tuple[int, int]:
    """
    统计文本中的公式，返回结果为 公式数量，公式总字符数
    """
    formulas = pat_formula.findall(text)
    num_formula = len(formulas)
    total_chars = sum(len(f) for f in formulas)
    return num_formula, total_chars


def check_id(text: str) -> bool:
    """
    判断是否为id

    必须含有id子串，id串字符数量占比超过80%
    """

    num, cnt = cout_id(text)
    if num > 0 and cnt > len(text)*0.8:
        return True
    else:
        return False


def load_file_content(filename: typing.Union[str, typing.Collection[str]], pattern: str=None) -> typing.List[str]:
    """[summary]
        按行读取文本文件内容

    Arguments:
        filename {str} -- 要读取的文件路径

    Keyword Arguments:
        pattern {str} -- 描述内容格式的正则表达式 (default: {None})
        如果不为空，则取每个匹配实例中capture的文本内容

    Returns:
        typing.List[str] -- 返回读取到的值列表
    """
    default_pattern = '^\\s*(.*?)\\s*$'
    pat = None if pattern is None else re.compile(pattern)
    pat = re.compile(default_pattern) if pat is None else pat
    cap = 1 if pat.groups > 0 else 0
    ret = list()
    if type(filename) is str:
        files = [filename]
    else:
        files = filename
    for f in files:
        with open(f, 'r') as fin:
            cnt = 0
            for line in fin:
                for m in pat.finditer(line):
                    ret.append(m[cap])
                    cnt += 1
            print(f'load {cnt} items from {f}')
    return ret


def load_fenci_lines(file_name: str) -> typing.List[str]:
    with open(file_name, 'r') as fin:
        lines = [fenci_char(line) for line in tqdm.tqdm(fin.readlines(), desc=file_name, ncols=68)]
    return lines


def load_lines(file_name: str) -> typing.List[str]:
    with open(file_name, 'r') as fin:
        lines = fin.readlines()
    return lines


def load_lines_stripped(file_name: str) -> typing.List[str]:
    with open(file_name, 'r') as fin:
        lines = [line.strip(' \n\t') for line in tqdm.tqdm(fin.readlines(), desc=file_name, ncols=68)]
    return lines


def print_latex_conversion():

    tex_dict = [
        '\\beta', '\\aleph', '\\alpha', '\\And', '\\angle', '\\approx', '\\arccos', '\\arcsin', '\\arctan', 
        '\\arg', '\\arrowvert', '\\backepsilon', '\\backsim', '\\backslash', '\\because', '\\beta', '\\bigcap', 
        '\\bigcirc', '\\bigcup', '\\bigotimes', '\\bigstar', '\\blacklozenge', '\\blacksquare', '\\blacktriangle', 
        '\\blacktriangledown', '\\bmod', '\\bot', '\\Box', '\\bullet', '\\C', '\\cap', '\\cdot', '\\cdots', 
        '\\centerdot', '\\chi', '\\circ', '\\circledcirc', '\\clubsuit', '\\complement', '\\cong', '\\coprod', 
        '\\cos', '\\cot', '\\csc', '\\cup', '\\ddots', '\\deg', '\\delta', '\\Delta', '\\diamond', '\\Diamond', 
        '\\diamondsuit', '\\div', '\\divideontimes', '\\dots', '\\downarrow', '\\Downarrow', '\\ell', '\\epsilon', 
        '\\equiv', '\\eta', '\\exists', '\\forall', '\\frown', '\\gamma', '\\Gamma', '\\gcd', '\\ge', '\\geq', 
        '\\geqslant', '\\gets', '\\gg', '\\gt', '\\hbar', '\\heartsuit', '\\Im', '\\in', '\\inf', '\\infty', 
        '\\int', '\\iota', '\\kappa', '\\lambda', '\\Lambda', '\\land', '\\langle', '\\lbrace', '\\lceil', '\\ldots', 
        '\\le', '\\leftarrow', '\\Leftarrow', '\\leftrightarrow', '\\Leftrightarrow', '\\leftrightharpoons', '\\leq', 
        '\\leqslant', '\\lfloor', '\\lg', '\\lim', '\\ll', '\\ln', '\\lnot', '\\log', '\\longleftrightarrow', 
        '\\longrightarrow', '\\lor', '\\lozenge', '\\lrcorner', '\\lt', '\\mapsto', '\\max', '\\mho', '\\mid', 
        '\\min', '\\mod', '\\mp', '\\mu', '\\nabla', '\\ne', '\\nearrow', '\\neg', '\\neq', '\\ngtr', '\\nless', 
        '\\nmid', '\\not', '\\notin', '\\nparallel', '\\nRightarrow', '\\nsubseteq', '\\nsupseteq', '\\nu', '\\nwarrow', 
        '\\odot', '\\omega', '\\Omega', '\\ominus', '\\oplus', '\\otimes', '\\parallel', '\\partial', '\\perp', '\\phi', 
        '\\Phi', '\\pi', '\\Pi', '\\pm', '\\Pr', '\\prec', '\\prime', '\\prod', '\\propto', '\\psi', '\\Psi', 
        '\\rangle', '\\rbrace', '\\rceil', '\\Re', '\\rfloor', '\\rho', '\\rightarrow', '\\Rightarrow', 
        '\\rightharpoonup', '\\rightleftarrows', '\\rightleftharpoons', '\\searrow', '\\sec', '\\setminus', '\\sigma', 
        '\\Sigma', '\\sim', '\\simeq', '\\sin', '\\space', '\\spadesuit', '\\sphericalangle', '\\square', '\\star', 
        '\\subset', '\\subseteq', '\\subseteqq', '\\subsetneq', '\\subsetneqq', '\\succ', '\\sum', '\\supset', 
        '\\supseteq', '\\supsetneqq', '\\surd', '\\swarrow', '\\tan', '\\tau', '\\therefore', '\\theta', '\\Theta', 
        '\\thicksim', '\\times', '\\to', '\\top', '\\triangle', '\\triangledown', '\\triangleleft', '\\triangleq', 
        '\\triangleright', '\\ulcorner', '\\uparrow', '\\Uparrow', '\\updownarrow', '\\Updownarrow', '\\upsilon', 
        '\\varepsilon', '\\varGamma', '\\varnothing', '\\varOmega', '\\varphi', '\\varPhi', '\\varpi', '\\varsigma', 
        '\\varsubsetneq', '\\varsubsetneqq', '\\varsupsetneq', '\\varsupsetneqq', '\\vartriangle', '\\vdots', '\\vee', 
        '\\vert', '\\wedge', '\\wp', '\\xi', '\\zeta'
    ]

    # tex_cool = [item for item in tex_dict if item in map_latex_word]
    # tex_bad = [item for item in tex_dict if item not in map_latex_word]
    # print(tex_cool)
    # print(tex_bad)

    for item in tex_dict:
        print(f'{item} --> {clean_latex(item)}')

    return


# 将题干用标点符号划分为若干段
reg_line_spliter = re.compile('，|：|；|？|。|．|！')

def split_question(question: str) -> typing.List[str]:
    return [item for item in reg_line_spliter.split(question) if len(item) > 0]


if __name__ == '__main__':

    # print_latex_conversion()

    print(fenci_char(r'$$  \not\in j\in \left\{ 1,2,\cdots ,m \right\}$$'))
    print(fenci_char(r'$$\rm F_1$$中同时出现紫花与白花的现象'))

    print(fenci_char(r'<p>定义&ldquo;等和数列&rdquo;：在一个数列中'))

    # for subj in ['math', 'physics', 'chemistry', 'biology']:
    #     for itype in ['keypoint', 'source']:
    #         src_file = f'data/raw/seg_{subj}__label__{itype}.txt'
    #         dst_file = f'data/proc/contents/seg_{subj}_{itype}.txt'
    #         proc_file_content(src_file, dst_file)

    print(restore_prep(r'https: _dvd_  _dvd_ jyresource.speiyou.com _dvd_ # _dvd_ question _dvd_ search?sid _eq_ 2&gid _eq_ 1&grade _eq_ 6&sy _eq_ 20192020&prov _eq_ 1000,%E5%8C%97'))

    print('Done')

