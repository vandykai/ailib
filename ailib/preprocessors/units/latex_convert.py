import re
from .unit import Unit

class LatexConvert(Unit):
    """Process unit to convert latex to plain text."""
    latex_word_map = {
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
    # latex公式 (mathjax语法格式，标记为'$$')
    formula_regex = re.compile(r'\$\$.+?\$\$')
    clean_formula_regex = re.compile(r'((\\not)?\\[a-zA-Z0-9]+)|\\(.)|[_\s\{\}\$]+')

    def replace_latex_element(self, input_: str) -> str:
        """
        对latex元素进行转义，保留公式中的普通文本
        """
        ret = ''
        pos = 0
        for item in self.clean_formula_regex.finditer(input_):
            st, ed = item.span()
            if st > pos:
                ret += input_[pos:st]
            regular_latex_element = item.group(1)
            irregular_latex_element = item.group(3)
            if regular_latex_element is not None:
                ret += self.latex_word_map.get(regular_latex_element, regular_latex_element)
            elif irregular_latex_element is not None:
                ret += irregular_latex_element
            else:
                ret += ' '
            pos = ed
        ret += input_[pos:]
        return ret.strip()

    def transform(self, input_: str) -> str:
        """
        :param input_: raw textual input.

        :return str: latex converted text.
        """
        if '$' in input_:
            ret = ''
            pos = 0
            for formula in self.formula_regex.finditer(input_):
                st, ed = formula.span()
                if st > pos:
                    ret += input_[pos:st]
                latex_replaced = self.replace_latex_element(input_[st:ed])
                ret += latex_replaced
                pos = ed
            ret += input_[pos:]
            return ret
        else:
            return input_