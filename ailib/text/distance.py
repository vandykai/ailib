# 计算编辑距离
# 输入：分割后的句子
# 输出：距离，动态规划矩阵
# dp[i][j] is the levenshtein_distance between sFrom[0:i] and sTo[0:j] note that sFrom[0:0] = []
def levenshtein_distance(sFrom: list, sTo: list) -> list:
    nFrom = len(sFrom)
    nTo = len(sTo)
    dp = [[0] * (nTo + 1) for _ in range(nFrom + 1)]
    # 第一行
    for j in range(1, nTo + 1):
        dp[0][j] = dp[0][j-1] + 1
    # 第一列
    for i in range(1, nFrom + 1):
        dp[i][0] = dp[i-1][0] + 1
    for i in range(1, nFrom + 1):
        for j in range(1, nTo + 1):
            if sFrom[i-1] == sTo[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1] ) + 1     
    return dp[-1][-1], dp

# 计算从sFrom 变换到sTo所要进行的改变
# 输入：分割后的句子
# 输出：变换的词语列表，操作列表
def levenshtein_diff(sFrom: list, sTo: list):
    diff = []
    operation=[]
    _, dp = levenshtein_distance(sFrom, sTo)
    i = len(sFrom)
    j = len(sTo)
    while (i > 0 or j > 0):
        if i>0 and j>0 and sFrom[i-1] == sTo[j-1]:
            i = i-1
            j = j-1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            operation.append('insert:{}'.format(i))
            diff.append(sTo[j-1])
            j = j-1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            operation.append('delete:{}'.format(i-1))
            diff.append(sFrom[i-1])
            i = i-1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            operation.append('replace:{}'.format(i-1))
            diff.append(sFrom[i-1])
            diff.append(sTo[j-1])
            i = i-1
            j = j-1
    return diff,list(reversed(operation))

# 计算从sFrom 变换到sTo所改动的位置
# 输入：分割后的句子
# 输出：修改位置，sFrom中对应位置的词，为空代表插入，sTo中对应位置的词，为空代表删除，两个都不为空代表替换
def levenshtein_ops(sFrom: list, sTo: list):
    diff = []
    position=[]
    alter_from = []
    alter_to = []
    ops = []
    _, dp = levenshtein_distance(sFrom, sTo)
    i = len(sFrom)
    j = len(sTo)
    while (i > 0 or j > 0):
        if i>0 and j>0 and sFrom[i-1] == sTo[j-1]: # the same
            i = i-1
            j = j-1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1: # insert op
            position.append(i)
            alter_from.append("")
            alter_to.append(sTo[j-1])
            ops.append('i')
            j = j-1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1: # delete op
            position.append(i-1)
            alter_from.append(sFrom[i-1])
            alter_to.append("")
            ops.append('d')
            i = i-1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1: # relace op
            position.append(i-1)
            alter_from.append(sFrom[i-1])
            alter_to.append(sTo[j-1])
            ops.append('r')
            i = i-1
            j = j-1
    return list(reversed(position)),list(reversed(alter_from)),list(reversed(alter_to)), list(reversed(ops))

def apply_levenshtein_ops(sFrom: list, op_position: list, alter_from: list, alter_to: list, ops: list):
    fixed_from = []
    index_from = 0
    for pos, from_word, to_word, op in zip(op_position, alter_from, alter_to, ops):
        while index_from < pos:
            fixed_from.append(sFrom[index_from])
            index_from += 1
        if op == 'i': # insert
            fixed_from.append(to_word)
        elif op == 'd': # delete
            index_from += 1
        else: # replace
            fixed_from.append(to_word)
            index_from += 1
    while index_from < len(sFrom):
        fixed_from.append(sFrom[index_from])
        index_from += 1
    return fixed_from