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

    def build_trie(self, data_path, trie_save_path=None):
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

    #查找节点
    def maxdepth_search(self, key):
        node = self.root
        matches = []
        for char in key:
            if char not in node.children:
                break
            node = node.children[char]
            if node.value:
                matches.clear()
                matches.extend(node.value)
        return matches

    def wildcard_search(self, key, wildcard):
        nodes = [self.root]
        matches = []
        for char in key:
            temp_nodes = []
            for node in nodes:
                if char in node.children:
                    child_node = node.children[char]
                    temp_nodes.append(child_node)
                    if child_node.value:
                        matches.extend(child_node.value)
                if wildcard in node.children:
                    child_node = node.children[wildcard]
                    temp_nodes.append(child_node)
                    if child_node.value:
                        matches.extend(child_node.value)
            nodes = temp_nodes
            if len(nodes) == 0:
                break
        return matches

    def wildcard_match(self, key, wildcard):
        nodes = [self.root]
        matches = []
        for char in key:
            temp_nodes = []
            for node in nodes:
                if char in node.children:
                    child_node = node.children[char]
                    temp_nodes.append(child_node)
                if wildcard in node.children:
                    child_node = node.children[wildcard]
                    temp_nodes.append(child_node)
            nodes = temp_nodes
            if len(nodes) == 0:
                break
        for node in nodes:
            if node.value:
                matches.extend(node.value)
        return matches

    def traverse(self, d, node):
        if node.value:
            yield node.value, d
        for c, child in node.children.items():
            yield from self.traverse(d, child)

    def fuzzy_search(self, query, k, sub=False, prefix=False, k_fn=None):
        n = len(query)
        tab = [[0] * (n + 1) for _ in range(n + k + 1)]
        for j in range(n + 1):
            tab[0][j] = j
        if not k_fn:
            k_fn = lambda i: k
        yield from self._fuzzy_search(k_fn, tab, self.root, 1, query, sub, prefix)

    def _fuzzy_search(self, k_fn, tab, node, i, query, sub=False, prefix=False):
        k = k_fn(i)
        if sub and i >= len(query) + 1:
            d = tab[i - 1][len(query)]
            if d <= k:
                # yield sofar, d
                yield from self.traverse(d, node)
            return

        # Can't be more than length of query + k insertions. Don't allow
        # children
        if prefix and node.value:
            d = min(tab[i - 1])
            if d <= k:
                yield node.value, d, tab[i - 1].index(d)
        elif not prefix and node.value:
            d = tab[i - 1][len(query)]
            if d <= k:
                yield node.value, d, len(query)
        if i >= len(tab):
            return
        
        for key, child in node.children.items():
            tab[i][0] = i
            for j in range(1, len(tab[i])):
                sub_cost = 1 if key != query[j - 1] else 0
                sub = tab[i - 1][j - 1] + sub_cost
                insert = tab[i - 1][j] + 1
                delete = tab[i][j - 1] + 1
                tab[i][j] = min(sub, insert, delete)

            smallest = min(tab[i])
            if smallest <= k:
                yield from self._fuzzy_search(k_fn, tab, child,
                                       i + 1, query, sub, prefix)

    def build_trie(self, data_path, trie_save_path=None):
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
