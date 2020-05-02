# 基于随机森林的Sonar 数据集分类：208行*60个特征，2个类别
#数据集生成方式与Bagging相同
# 构造树更具多样性，更具特色，不只有单棵树生成时使用的样本集不同.还有随机抽取的备选特征子集
from random import seed
from random import randrange
from csv import reader
from math import sqrt


# 1-1. Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
        
        #读取指定名称的文件，返回文件生成器
		csv_reader = reader(file)
        
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# 1-2. Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
        
        # 删除指定列中字符串的开头、结尾
		row[column] = float(row[column].strip())

# 1-3. Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup



# 1-4. Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
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

# 1-5. Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# 1-6. Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
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
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# 1-7. Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# 1-8. Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	
    # sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini
            
# 1-9.************ Select the best split point for a dataset
# 注意特征抽取的随机性
def get_split(dataset, n_features):
    '''
    

    Parameters
    ----------
    dataset : list
        样本集.
    n_features : int
        备选特征的数量.

    Returns
    -------
    dict
        {'index':b_index, 'value':b_value, 'groups':b_groups}.

    '''
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()#存放备选特征
    while len(features) < n_features:
        index = randrange(len(dataset[0])-1)# 0-59
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups	
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

# 1-9. Select the best split point for a dataset
# 基于到达当前结点的训练集dataset选择特征以及切分阈值
def get_split_bagging(dataset):
      
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

# 1-10 Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# 1-11 Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
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
		node['left'] = get_split(left, n_features)
		split(node['left'], max_depth, min_size, n_features, depth+1)
	
    # process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right, n_features)
		split(node['right'], max_depth, min_size, n_features, depth+1)

# 1-12 Build a decision tree 注意区别于bagging
def build_tree(train, max_depth, min_size, n_features):
	root = get_split(train, n_features)
	split(root, max_depth, min_size, n_features, 1)
	return root

# 1-13 Make a prediction with a decision tree
def predict(node, row):#使用单科树时的预测结果
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# 1-14 Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample
def oobsample(dataset, sample):
    '''
    获取包外样本

    Parameters
    ----------
    dataset : list
        初始总的样本集.
    sample : list
        自举样本集.

    Returns
    -------
    oobsample : list
        包外样本.

    '''
    oobsample = list()
    for i in dataset:
        if i not in sample:
            oobsample.append(i)
    return oobsample
# 1-15 Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
	predictions = [predict(tree, row) for tree in trees]
	return max(set(predictions), key=predictions.count)

# 1-16 Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return(predictions)


#Test the random forest algorithm on sonar dataset
seed(2)

#1. load and prepare data
filename = 'sonar.all-data.csv'
dataset0 = load_csv(filename)
dataset=dataset0

#2 convert string attributes to integers
for i in range(0, len(dataset[0])-1):
	str_column_to_float(dataset, i)
    
#3 convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)

#4 evaluate algorithm
#4-1 
n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_features = int(sqrt(len(dataset[0])-1))

#4-2 单棵树--50棵树
for n_trees in [1, 5, 10, 50]:
	scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
	print('\n\nnumber of Trees: %d' % n_trees)
	print('Scores: %s' % scores)
	print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))