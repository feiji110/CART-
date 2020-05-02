# -*- coding: utf-8 -*-
"""
    基于Bagging的Sonar dataset分类
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    (208行*(60+1)列   208个样本，60个数值特征
     2类别(M--mine矿石  R--rock 岩石))
    :copyright: (c) 2020 by the angus.
"""
from random import seed
from random import randrange
from csv import reader

# 1-1. 定义函数，实现CSV文件读取 Load a CSV file
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
    dataset = list()
    # 以只读的方式打开指定名称的磁盘文件，返回文件对象file
    with open(filename, 'r') as file:
          
        # 返回文件内容的各行组成的list
        csv_reader = reader(file)  # reader(open(filename, 'r'))
        
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# 1-2. 定义函数，将列表dataset中指定的列转化为浮点型
#      Convert string column to float
# 删除字符串中开头、结尾处的空白符（包括'\n', '\r',  '\t',  ' ')
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
        row[column] = float(row[column].strip())
        

# 1-3.
def str_column_to_int(dataset, column):
    '''
    该函数将类别标记的字符串形式转化为整数Convert string column to integer

    Parameters
    ----------
    dataset : list
        样本集.
    column : int
        列序号.

    Returns
    -------
    lookup : dict
        {类别1：0,类别2：1,...}.

    '''
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    
    lookup = dict()
    for i, value in enumerate(unique):#枚举 (0, seq[0]), (1, seq[1])
        lookup[value] = i
        #lookup = {'M':0,'R',1}
    for row in dataset:
        row[column] = lookup[row[column]]
        
    return lookup

# 1-4. 
#      
def cross_validation_split(dataset, n_folds):
    '''
    随机打乱指定的list,均分成k等分,Split a dataset into k folds

    Parameters
    ----------
    dataset : list
        样本集.
    n_folds : int
        折数.

    Returns
    -------
    dataset_split : list
        分折的样本集.

    '''
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# 1-5. 
def accuracy_metric(actual, predicted):
    '''
    Calculate accuracy percentage 

    Parameters
    ----------
    actual : list
        真实类别标号列表.
    predicted : list
        预测标号列表.

    Returns
    -------
    float
        精度值 * 100.

    '''
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# 1-6. 
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    '''
    Evaluate an algorithm using a cross validation split

    Parameters
    ----------
    dataset : list
        样本集.
    algorithm : function
        算法.
    n_folds : int
        折数.
    *args : TYPE
        算法用的其他参数.

    Returns
    -------
    scores : list
        精度列表.

    '''
    #(1)数据集随机打乱，得到n折
    folds = cross_validation_split(dataset, n_folds)
    
    #(2)结合上述划分，得到每个版本的训练集、测试集
    scores = list()
    
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
            
        # 注意此处的algorithm 
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
        
    return scores

# 1-7. 
#      
def test_split(index, value, dataset):
    '''
    将到达当前结点的数据集分成左右子集
    Split a dataset based on an attribute and an attribute value

    Parameters
    ----------
    index : int
        切分特征序号.
    value : int
        切分阈值.
    dataset : list
        当前节点.

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
            left.append(row)
        else:
            right.append(row)
            
    return left, right

# 1-8.
def gini_index(groups, classes):
    '''
    计算划分后累积基尼指数
    Calculate the Gini index for a split dataset

    Parameters
    ----------
    groups : list
        左子集或者右子集.
    classes : list
        类别
        
    Returns
    -------
    gini : float
        基尼指数

    '''
    # (1)总样本数的估计：统计当前结点分裂后，到达两个子结点的样本数的和
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    
    # (2)sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        
        # avoid divide by zero
        if size == 0:
            continue
      
        score = 0.0
        # 估计各类别在每个子节点的训练集内出现的概率
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
            
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
        
    return gini

# 1-9. 
def get_split(dataset):
    '''
    Select the best split point for a dataset
    基于到达当前结点的训练集dataset选择特征以及切分阈值

    Parameters
    ----------
    dataset : list
        当前结点的训练集dataset.

    Returns
    -------
    dict
         {特征序号'index':b_index,切分阈值 'value':b_value, 左右子集'groups':b_groups}.

    '''
      
    # (1)获取该数据集内各个样本类别标记，生成该样本集不同类别标记的list
    class_values = list(set(row[-1] for row in dataset))
    
    # (2)初始化：切分特征序号、切分阈值、划分后最小基尼指数阈值、划分结果
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    
    # (3)遍历每个备选的特征序号
    for index in range(len(dataset[0])-1):
        
        #(4)逐行遍历当前数据集的每个样本
        for row in dataset:
        # for i in range(len(dataset)):
        # 	row = dataset[randrange(len(dataset))]
              
            #(5)以当前样本的相应特征取值为阈值，将样本集划分为左右两个子集，构成list:groups
            groups = test_split(index, row[index], dataset)
            
            # (6)结合该特征以及上述备选阈值，计算划分后的加权基尼指数
            gini = gini_index(groups, class_values)
            
            # (7)若划分后基尼指数的和低于目前最小阈值，就需要更新四个参数
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
                
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

# 1-10. 
def to_terminal(group):
    '''
    Create a terminal node value--叶子结点的类别预测值

    Parameters
    ----------
    group : list
        左右子集.

    Returns
    -------
    int
        最大类别预测标号.

    '''
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# 1-11. 
def split(node, max_depth, min_size, depth):
    '''
    构造当前结点的子节点或者直接以当前结点为叶子结点
    Create child splits for a node or make terminal

    Parameters
    ----------
    node : list
        root后者左右子集.
    max_depth : int
        限制树深.
    min_size : int
        叶子节点最小样本数.
    depth : int
        当前深度.

    Returns
    -------
    None.

    '''
    left, right = node['groups']
    del(node['groups'])
    
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)

    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

# 1-12 
def build_tree(train, max_depth, min_size):
    '''
    Build a decision tree

    Parameters
    ----------
    train : list
        训练集.
    max_depth : int
        树深.
    min_size : int
        最小样本数.

    Returns
    -------
    root : dict
        {特征序号'index':b_index,切分阈值 'value':b_value, 左右子集'groups':b_groups}.

    '''
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

# 1-13 
def predict(node, row): #传入root逐级预测
    '''
    Make a prediction with a decision tree

    Parameters
    ----------
    node : dict
        {}
    row : list
        单个样本行

    Returns
    -------
    node : dict
        {特征序号'index':b_index,切分阈值 'value':b_value, 左右子集'groups':b_groups}.
    node: int
        类别序号
    '''
    if row[node['index']] < node['value']:
          
        # 如果node['left']是字典类型，继续由后续子节点进行顺序决策
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else: # 否则其左子节点为叶子结点，直接得到预测类别
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']
      

# 1-14 自举采样。Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):# 63.2%
    '''
    有放回抽取的简化版本

    Parameters
    ----------
    dataset : list
        训练样本集.
    ratio : float
        自举样本集占总体样本集的比率

    Returns
    -------
    sample : list
        自举样本集.
    '''
    sample = list()
    n_sample = round(len(dataset) * ratio)#四舍五入，确定自举样本集数量
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

# 1-15 
def bagging_predict(trees, row):
    '''
    Make a prediction with a list of bagged trees
    返回具有最大票数的类别
    
    Parameters
    ----------
    trees : list
        多棵决策树的集合.
    row : list
        待决策样本.

    Returns
    -------
    TYPE
        具有最大次数的类别标号.

    '''
    predictions = [predict(tree, row) for tree in trees]#多少棵树对应多少个决策结果
    #[[类别标号],[类别标号],[类别标号],[类别标号]...]
    return max(set(predictions), key=predictions.count)

# 1-16 
def bagging(train, test, max_depth, min_size, sample_size, n_trees):
    '''
    Bootstrap Aggregation Algorithm

    Parameters
    ----------
    train : list
        训练集.
    test : list
        测试集.
    max_depth : int
        树深.
    min_size : int
        最小样本数.
    sample_size : int
        自举样本比例.
    n_trees : int
        树的棵树.

    Returns
    -------
    None.

    '''
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return(predictions)


# Test bagging on the sonar dataset

if __name__=='__main__':
      # (1)随机数发生器种子值
      seed(1)
      
      # (2)load and prepare data
      filename = 'sonar.all-data.csv'
      dataset = load_csv(filename)
      
      # (3)convert string attributes to integers
      for i in range(len(dataset[0])-1):
            str_column_to_float(dataset, i)
            
     # (4)convert class column to integers:将最后一列的类别转成0，1形式
      str_column_to_int(dataset, len(dataset[0])-1)
      
     # (5)evaluate algorithm
      n_folds = 5
      max_depth = 6
      min_size = 2
      sample_size = 0.50
      
      #MeanScores = []
      for n_trees in [1, 5, 10, 50]:
            scores = evaluate_algorithm(dataset, bagging, n_folds, max_depth, 
                                        min_size, sample_size, n_trees)
            print('\n\nTrees: %d' % n_trees)
            print('Scores: %s' % scores)
            print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
            #MeanScores.append(sum(scores)/float(len(scores))))
