import pickle

##定义trie字典树节点
class TrieNode:
    def __init__(self):
        self.value = []
        self.children = {}
#遍历树
class SearchIndex:
    def __init__(self, index, char=None, parent=None):
        self.index = index
        self.char = char
        self.parent = parent

#定义Trie字典树
class Trie:
    def __init__(self):
        self.root = TrieNode()

    #添加树节点
    def insert(self, key, value = None):
        node = self.root
        for char in key:
            if char not in node.children:
                child = TrieNode()
                node.children[char] = child
                node = child
            else:
                node = node.children[char]

        node.value = value if value else key

    #查找节点
    def search(self, key):
        node = self.root
        matches = []
        for char in key:
            if char not in node.children:
                break
            node = node.children[char]
            if node.value:
                matches.append(node.value)
        return matches

    def build_trie(self, trie_save_path, data_path):
        for line in open(data_path):
            word = line.strip().lower()
            word_split = word.split()
            if len(word_split) > 1:
                self.insert(word_split[0], word_split[1])
            else:
                self.insert(word)
        if trie_save_path:
            with open(trie_save_path, 'wb') as f:
                pickle.dump(self, f)

class MultiValueTrie:
    def __init__(self):
        self.root = TrieNode()

    #添加树节点
    def insert(self, key, value = None):
        node = self.root
        for char in key:
            if char not in node.children:
                child = TrieNode()
                node.children[char] = child
                node = child
            else:
                node = node.children[char]
        node.value.append(value if value else key)

    #查找节点
    def search(self, key):
        node = self.root
        matches = []
        for char in key:
            if char not in node.children:
                break
            node = node.children[char]
            if node.value:
                matches.extend(node.value)
        return matches

    def build_trie(self, trie_save_path, data_path):
        for line in open(data_path):
            word = line.strip().lower()
            word_split = word.split()
            if len(word_split) > 1:
                self.insert(word_split[0], word_split[1])
            else:
                self.insert(word)
        if trie_save_path:
            with open(trie_save_path, 'wb') as f:
                pickle.dump(self, f)
