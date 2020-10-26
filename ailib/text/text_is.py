from ailib.text.basic_data import ch_en_punctuation

def is_en_word(input_str):
    return all(('a'<= c and c <= 'z') or ('A'<= c and c <= 'Z') or (c == r"'") for c in input_str)

def is_contain_punctions(input_str):
    return not all(i not in ch_en_punctuation for i in input_str)

def is_all_punctions(input_str):
    return all(i in ch_en_punctuation for i in input_str)