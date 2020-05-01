
# -*- coding: utf-8 -*-
"""
    Example of CART on the Banknote dataset
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    CART决策树分类器(数据集 Banknote：1372行*4个特征，2类别)
    :copyright: (c) 2020 by the angusgao.

"""
from random import seed#初始化随机数发生器的种子值
from random import randrange#指定区间随机不重复抽取整数
from csv import reader#读取CSV文件

# 1-1 Load a CSV file
def load_csv(filename):
    '''
    读取CSV文件返回整体数据集列表

    Parameters
    ----------
    filename : String
        路径名称

    Returns
    -------
    dataset : list
        只含有效行的数据集列表.

    '''
    dataset = list()#空列表
    with open(filename, 'r') as file:#只读 返回文件对象
        csv_reader = reader(file)#reader(open(filename, 'r'))#所有行
        for row in csv_reader:
            if not row:#去除CSV文件空行
                continue
            dataset.append(row)
    return dataset

# 1-22 Convert string column to float
def str_column_to_float(dataset, column):
    '''

    Parameters
    ----------
    dataset :list
         整体数据集列表.
    column : int
        需要转换的指定列.

    Returns
    -------
    None.

    '''
    for row in dataset:
        row[column] = float(row[column].strip())#提取字符串值（去掉字符串周围的空白和''）强制转换为float


# 1-3 Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
      
    dataset_split = list()
    
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)#每一折样本数量
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))#0到n-1
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# 1-4 Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    '''
    

    Parameters
    ----------
    actual : list
        真实标签.
    predicted : list
        预测标签

    Returns
    -------
    float
        精度.

    '''
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# 1-5 Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    '''
    评价算法

    Parameters
    ----------
    dataset : list
        所有已知标签的样本集.
    algorithm : function
        参与评价的算法.
    n_folds : int
        折数.
    *args : 其他参数
        .取决于算法的额外参数

    Returns
    -------
    scores : TYPE
        DESCRIPTION.

    '''
    folds = cross_validation_split(dataset, n_folds)#调用1-3,将数据集均分成n等分，列表方式返回。N个版本的训练集与测试集
    
    scores = list()
    
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)# 去掉本折
        train_set = sum(train_set, [])# 将[[],[],[],[]]变为[]
        
        test_set = list()
        for row in fold:# 将本折作为测试集
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None# 去掉正确的类别标号
            
        predicted = algorithm(train_set, test_set, *args)# 接收的关于测试集的预测标签
        
        actual = [row[-1] for row in fold]# 真实的预测标签
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# 1-6 Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    '''
    

    Parameters
    ----------
    index : int
        切分特征的序号.
    value : int
        切分特征对应的切分阈值.
    dataset : list
        到达当前节点的训练样本集.

    Returns
    -------
    left : list
        左子集.
    right : list
        右子集.

    '''
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)# scikit中《=
        else:
            right.append(row)
    return left, right

# 1-7 Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    '''
    CART 二叉树 只有左右子集.
    针对切分特征中的切分点划分后的子集计算加权基尼指数

    Parameters
    ----------
    groups : list
        左右两个子集.
    classes : TYPE
        所有类别标号.

    Returns
    -------
    gini : float
        基尼指数.

    '''
    # 总的样本数--count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))#求和[[左子集数量],[右子集数量]]得总的样本集数量，权重的分母
    
    # 划分后的加权GINI指数--sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))#获取当前子集的样本数量
        
        # avoid divide by zero#没有进行排序选阈值
        if size == 0:#阈值为 最小值，最大值
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size# [所有类别标号].count()-->class_val的数量 作为分子
            score += p * p# 平方和
            
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)#不纯度 * 权重
        
    return gini

# 1-8 Select the best split point for a dataset
def get_split(dataset):
    '''
    利用 1-6 划分左右子集，1-7 计算基尼指数进行划分,进行切分特征和切分阈值的优选

    Parameters
    ----------
    dataset : list
        到达该节点的样本集.

    Returns
    -------
    dict
        {特征序号，特征切分阈值，[左右子集]}.

    '''
    class_values = list(set(row[-1] for row in dataset))#[到达该节点的类别标号]
    
    # 初始化最小基尼指数对应的特征序号，切分点，基尼指数值，分组列表
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    
    # 针对每个备选的特征
    for index in range(len(dataset[0])-1):#[1,2,3,4,标号]
        
        # 考查每个备选的切分阈值
        for row in dataset:
            groups = test_split(index, row[index], dataset)# 为了简单快速生成树，未sort,只是选了每个特征作为阈值。#生成左右两个子集
            gini = gini_index(groups, class_values)
            if gini < b_score:# 因为找的是最小基尼指数的点.
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
                
    # 在当前结点处，存放字典形式的信息：切分特征、切分点、划分的两个子集
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

# 1-9 Create a terminal node value
def to_terminal(group):
    '''
    基于到达当前节点的训练集，生成该节点的预测值

    Parameters
    ----------
    group : list
        左子集或者右子集.

    Returns
    -------
    预测类别
        数量最多的类别.如果一样多，返回标号最小的类别.

    '''
    outcomes = [row[-1] for row in group]# [类别标号]
    return max(set(outcomes), key=outcomes.count)#返回出现次数最多的类别。
# 1-9_2
def to_terminal_pro(group):
    outcomes = [row[-1] for row in group]
    probablity_list = []
    for i in set(outcomes):
        probablity = outcomes.count(i)/len(outcomes)
        probablity_list.append([i,probablity])
    return probablity_list
#to_terminal_pro([[0,1],[0,1],[0,0],[0,0],[0,1]])

# 1-10 Create child splits for a node or make terminal

def split(node, max_depth, min_size, depth):
    '''
    # 基于当前节点生成当前结点的后续子节点,或直接作为叶子结点生成预测结果（节点信息，3个限制条件）要么递归划分，要么直接作为叶子终结
    
    Parameters
    ----------
    
    node : dict
        当前结点的字典形式的描述.{'index':b_index, 'value':b_value, 'groups':b_groups}
    max_depth : TYPE
        最大深度.
    min_size : TYPE
        最小样本数
    depth : TYPE
        当前深度.

    Returns
    -------
    None.

    '''
    # (1)获取该结点的左右子集，为后续更新结点信息更新做准备
    left, right = node['groups']
    del(node['groups'])
    
    # (2)判断当前结点是否直接作为叶子结点--check for a no split
    # 如果是，不再分裂，直接返回
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    
    # (2)check for max depth 若已为最大深度，后续两个子结点直接作为叶子结点
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)#得到预测类别。
        return
    
    # (3)process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
        
    # (4)process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

# 1-11 Build a decision tree
def build_tree(train, max_depth, min_size):
    '''
    
    Parameters
    ----------
    train : list
        初始用于生成树模型的训练样本集.
    max_depth : int
        最大深度.
    min_size : int
        最小样本数.

    Returns
    -------
    root : list
        根节点.

    '''
    root = get_split(train) # 1-8
    split(root, max_depth, min_size, 1)#递归调用生成多重嵌入字典.
    return root #多重字典

# 1-12 Make a prediction with a decision tree
def predict(node, row):
    '''
    单极的测试，基于该node节点的特征和阈值对样本进行测试

    Parameters
    ----------
    node : dict 或者其他
        当前节点.
    row : list
        样本.

    Returns
    -------
    TYPE
        node（dict或者预测输出）.

    '''
    
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):# 用字典方式表示
            return predict(node['left'], row)
        else:
            return node['left'] # 见1-10
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# 1-13 Classification and Regression Tree Algorithm(核心函数)
def decision_tree(train, test, max_depth, min_size):
    '''
    充当algorithm

    Parameters
    ----------
    train : list
        训练集.
    test : list
        测试集.
    max_depth : int
        最大深度.
    min_size : int
        最小样本数.

    Returns
    -------
    关于测试集的预测结果.

    '''
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return(predictions)#关于测试集的预测结果
    
if __name__=='__main__':
    # Test CART on Bank Note dataset
    seed(1)

    #1 load and prepare data
    filename = 'data_banknote_authentication.csv'
    dataset = load_csv(filename)

    #2 convert string attributes to integers
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    #3 evaluate algorithm--5折交叉验证
    n_folds = 5
    max_depth = 5
    min_size = 10

    scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)

    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

    
