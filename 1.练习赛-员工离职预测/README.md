# 员工离职预测
- 简介
	- 比较基础的分类问题
	- 核心思路为**属性构造+逻辑回归**
- 过程
	- 数据获取
		- 报名这个比赛即可获取到这个数据[点击获取](!https://www.dcjingsai.com/common/cmpt/%E5%91%98%E5%B7%A5%E7%A6%BB%E8%81%8C%E9%A2%84%E6%B5%8B%E8%AE%AD%E7%BB%83%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html)
	- 数据探索
		- 无关项
			-  EmployeeNumber为编号，对建模是干扰项，删除即可。
			- StandardHours和Over18全数据集固定值，没有意义，删除。
		- 相关性高
			- 相关图
			  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20190908155706598.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3NzAy,size_16,color_FFFFFF,t_70)
			  发现有两项相关性极高，删除其中一个JobLevel
	- 数据预处理
		-  属性构造
			- 个人凭感觉制造了几个新特征
		-one-hot编码
			- 对几个字符串类型的属性，进行了one-hot编码
		- 数据挖掘建模
			- 我们看出有的特征的数字大，有的特征的数字小，进行标准化处理。
			-  采用多模型交叉验证选择较好的模型，在进一步调参
				- 代码
				```python
				# 多模型交叉验证
				import warnings
				warnings.filterwarnings('ignore')
				
				from sklearn.linear_model import LogisticRegression
				from sklearn.svm import SVC
				from sklearn.ensemble import RandomForestClassifier
				from sklearn.model_selection import cross_val_score
				models ={
				    'LR':LogisticRegression(solver='liblinear'),
				    'SVM':SVC(),
				    'RF':RandomForestClassifier(n_estimators=150),
				}
				for k,clf in models.items():
				    print('the model is %s'%k)
				    scores = cross_val_score(clf,train.iloc[:,1:],train['Attrition'])
				    print('Mean accuracy is {}'.format(np.mean(scores)))
				    print('-'*20)
				  ```
			- 对LR进行网格搜索进行调参
				```python
				# 数据分割 加网格搜索
				from sklearn.model_selection import train_test_split
				from sklearn.model_selection import GridSearchCV
				x_train,x_test,y_train,y_test = train_test_split(
				    df_train.iloc[:,1:],df_train['Attrition'],random_state=22)
				params={
				    'penalty':['l1', 'l2'],
				    'C':np.arange(1,4.1,0.2),
				}
				estimator = LogisticRegression(solver='liblinear')
				grid = GridSearchCV(estimator,param_grid=params,cv=10)
				grid.fit(x_train,y_train)
				print('最好的参数',grid.best_params_)
				score = grid.score(x_test,y_test)
				print('在测试集的得分',score)
				  ```
- 补充说明
	- 具体数据集和代码可以在我的[Github]中找到,1-pred即为提交文件。
	- 最终分数在0.90以上，但是对于新人的我已经很满意了
## 参考文章
[周先森爱吃素的博客](!https://blog.csdn.net/zhouchen1998/article/details/89054512)
