from ailib.text.basic_data import ch_en_punctuation

english_word_char = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
english_word_conjunctions = ["'", "-"]
formula_num = "0123456789"
formula_operation_symbol = "＄％（）＊＋－／：＜＝＞［］｛｜｝｢｣〈〉「」$%()*+-./:<=>[]{}×"

def tokenizer_v1(content):
    words = []
    word = []
    formula = []
    for char in content:
        if (char in english_word_char) or (char in english_word_conjunctions and word):
            word.append(char)
            if formula:
                words.append("".join(formula).strip())
                formula.clear()
        elif char in ch_en_punctuation:
            if formula:
                if char in formula_operation_symbol:
                    formula.append(char)
                else:
                    words.append("".join(formula).strip())
                    formula.clear()
            elif word:
                words.append("".join(word))
                word.clear()
        elif char in formula_num:
            if word:
                words.append("".join(word))
                word.clear()
            formula.append(char)
        else:
            if word:
                words.append("".join(word))
                word.clear()
            if formula:
                words.append("".join(formula).strip())
                formula.clear()
            words.append(char)
    if word:
        words.append("".join(word))
        word.clear()
    if formula:
        words.append("".join(formula).strip())
        formula.clear()
    return words