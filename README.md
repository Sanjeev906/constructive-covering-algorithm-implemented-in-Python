# constructive-covering-algorithm-implemented-in-Python
Python实现的构造性覆盖算法
----------
##### 目录文件说明：
- load\_\*    _加载不同的数据集，提供jupyter notebook 和 python格式文件_
- data_set    _存放数据集(UCI)_
    - data_transformed    _存放统一格式转化后的数据集，最后一列为标签，其余列为属性_
- minmax_out    _存放归一化后的数据，分为训练集和测试集_
- mod    _为训练和测试编写的相关模块_
    - core.py     _对数据进行训练的核心代码_
    - datatest.py    _对测试集进行测试，并输出结果_
- result    _存放训练结果_
    - result{num}.json    _每次迭代形成的覆盖_
    - full    _对每次迭代的覆盖的合并_
- references    _参考文献_

> 每次运行时，先加载对应数据集，再运行train.py或train.ipynb即可



| 样本集  | 覆盖数 | 平均正确率 |
| :-----: | :----: | :--------: |
|  Iris   |  119   |   96.40%   |
|  wine   |   50   |   71.88%   |
|   zoo   |  151   |   96.97%   |
| soybean-small |   70   |   99.96%   |