## Seaborn

Matplotlib虽然已经是比较优秀的绘图库了，但是它有个今人头疼的问题，那就是API使用过于复杂，它里面有上千个函数和参数，属于典型的那种可以用它做任何事，却无从下手。

Seaborn是基于Matplotlib核心库进行了更高级的API封装的绘图模块，可以轻松地画出更漂亮的图形，而且Seaborn的配色更加舒服，以及图形更加细腻。

官方文档：https://seaborn.pydata.org/api.html



### 学习目标

- seaborn的基本使用[知道]
  - pip 或者 conda 安装
- 绘制单变量分布图形[掌握]
  - displot/histplot
- 绘制双变量分布图形[掌握]
  - jointplot
- 绘制成对的双变量分布图形[掌握]
  - pairplot

### 基本安装

安装Seaborn模块，打开终端，输入以下命令：

```python
# 安装前先切换到jupyter notebook 所在的虚拟环境下。
# conda activate tfdemo
# 安装 
pip install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple
# 如果安装失败，使用以下命令来安装
# conda install seaborn
```

接下来，我们正式进入Seaborn库的学习，使用前先导入：

```python
# 使用前，要导入模块
import seaborn as sns
```



### 可视化数据的分布

当处理一组数据时，通常先要做的就是了解变量是如何分布的。

- 对于单变量的数据来说，采用直方图或核密度曲线是个不错的选择，
- 对于双变量来说，可采用多面板图形展现，比如 散点图、二维直方图、核密度估计图形等。

针对这种情况， Seaborn库提供了对单变量和双变量分布的绘制函数，如 displot()函数、 jointplot()函数。



### 绘制单变量分布

可以采用最简单的直方图描述单变量的分布情况。 Seaborn中提供了 displot()函数，它默认绘制的是一个带有核密度估计曲线的直方图。 

官方文档：https://seaborn.pydata.org/generated/seaborn.displot.html

displot()函数的语法格式：

```Python
seaborn.displot(data, x, y, bins=None, kde=True, rug=False, fit=None, color=None)
```

上述函数中常用参数的含义如下：

- (1) data：表示要观察的数据，可以是 Series、一维数组或列表。
- (2) x：x轴的刻度
- (3) y：y轴的刻度
- (2) bins：用于控制条形的数量。
- (3) kde：接收布尔类型，表示是否绘制高斯核密度估计曲线(KDE)。
- (4) rug：接收布尔类型，表示是否在支持的轴方向上绘制rugplot。

> **核密度估计**
>
> 核密度估计是在概率论中用来估计未知的密度函数，属于非参数检验方法之一，可以比较直观地看出数据样本本身的分布特征。

通过 displot() 函数绘制直方图，代码：

```python
import numpy as np

sns.set()
np.random.seed(0)  # 确定随机数生成器的种子,如果不使用每次生成图形不一样
data = np.random.randn(100)  # 生成随机数组

ax = sns.displot(data, bins=10, kde=True,  rug=True)  # 绘制直方图
# ax = sns.histplot(data, bins=10, kde=True, color="red")  # 也可以使用直方图绘制方法来绘制直方图
```

上述示例中，首先导入了用于生成数组的numpy库，然后使用 seaborn调用set()函数获取默认绘图，并且调用 random模块的seed函数确定随机数生成器的种子，保证每次产生的随机数是一样的，接着调用 randn()函数生成包含100个随机数的数组，最后调用 displot()函数绘制直方图。运行效果：

![image-20231220131325989](assets/image-20231220131325989.png)



从上图中看出：

- 直方图共有10个条柱，每个条柱的颜色为蓝色，并且有核密度估计曲线。
- 根据条柱的高度可知，位于-1-1区间的随机数值偏多，小于-2的随机数值偏少。

通常，采用直方图可以比较直观地展现样本数据的分布情况，不过直方图也存在一些问题，它会因为条柱数量的不同导致直方图的效果有很大的差异。为了解决这个问题，可以绘制核密度估计曲线进行展现。

通过 distplot()函数绘制核密度估计曲线，代码：

```python
# 创建包含500个位于[0，100]之间整数的随机数组
array_random = np.random.randint(0, 100, 500)
# 绘制核密度估计曲线
ax = sns.displot(array_random, kind="kde", rug=True)  # kind 绘制的图形效果，默认kind就是hist，其他的参数值就kde和ecdf两种不同的曲线。
```

上述代码中，首先通过 random.randint()函数返回一个最小值不低于0、最大值低于100的500个随机整数数组然后调用 displot()函数绘制核密度估计曲线。

运行效果：

![image-20231220132128854](assets/image-20231220132128854.png)

从上图中看出，图表中有一条核密度估计曲线，并且在x轴的上方生成了观测数值的小细条。



### 绘制双变量分布

两个变量的二元分布可视化也很有用。在 Seaborn中最简单的方法是**使用 jointplot()函数**，该函数可以创建一个多面板图形，比如散点图、二维直方图、核密度估计等，以显示两个变量之间的双变量关系及每个变量在单坐标轴上的单变量分布。

官方文档：https://seaborn.pydata.org/generated/seaborn.jointplot.html

jointplot()函数的语法：

```python
seaborn.jointplot(
    data=None,
    x=None,
    y=None,
    kind='scatter', 
    stat_func=None, 
    color=None, 
    ratio=5, 
    space=0.2, 
    dropna=True
)
```

上述函数中常用参数的含义如下：

+ data：表示要观察的数据，可以是 Series、一维数组或列表。
+ x和y：表示X轴和Y轴的描述提示，因为seaborn是基于matplotlib，所以默认不支持中文，如果要显示中文，需要先设置字体
+ kind：表示绘制图形的类型。
+ stat_func：用于计算有关关系的统计量并标注图。
+ color：表示绘图元素的颜色。
+ size：用于设置图的大小(正方形)。
+ ratio：表示中心图与侧边图的比例。该参数的值越大，则中心图的占比会越大。
+ space：用于设置中心图与侧边图的间隔大小。

#### 绘制散点图

调用 seaborn.jointplot()函数绘制散点图，代码：

```python
import pandas as pd
# 创建DataFrame对象
data = pd.DataFrame({"x": np.random.randn(500),"y": np.random.randn(500)})
# 绘制散布图
ax = sns.jointplot(x="x", y="y", data=data)
```

上述代码中，首先创建了一个 DataFrame对象dataframe_obj作为散点图的数据，其中x轴和y轴的数据均为500个随机数，接着调用jointplot函数绘制一个散点图，散点图x轴的名称为“x”，y轴的名称为“y”。

运行效果：

![image-20231220132745936](assets/image-20231220132745936.png)



#### 绘制二维直方图

**二维直方图类似于“六边形”图，主要是因为它显示了落在六角形区域内的观察值的计数，适用于较大的数据集。**当调用 jointplot()函数时，只要传入kind="hex"，就可以绘制二维直方图，代码：

```python
# 绘制二维直方图
ax = sns.jointplot(x="x", y="y", data=data, kind="hex")
```

运行效果：

![image-20231220132826607](assets/image-20231220132826607.png)

**从六边形颜色的深浅，可以观察到数据密集的程度，**另外，图形的上方和右侧仍然给出了直方图。



#### 绘制核密度估计图形

利用核密度估计同样可以查看二元分布，其用等高线图来表示。当调用jointplot()函数时只要传入kind="kde"，就可以绘制核密度估计图形，代码：

```python
sns.jointplot(x="x", y="y", data=darta, kind="kde")
```

上述示例中，绘制了核密度的等高线图，另外在图形的上方和右侧给出了核密度曲线图。运行效果：

![image-20231220132908584](assets/image-20231220132908584.png)



### 绘制成对的双变量分布

要想在数据集中绘制多个成对的双变量分布，则可以使用pairplot()函数实现，该函数会创建一个坐标轴矩阵，并且显示Dataframe对象中每对变量的关系。另外pairplot()函数也可以绘制每个变量在对角轴上的单变量分布。

接下来，通过pairplot()函数绘制数据集变量间关系的图形，示例代码如下

```python
# 加载seaborn中的数据集，通过鸢尾花数据集去展示，需要网络，如果慢的话，换个好点的网络即可。
dataset = sns.load_dataset("iris")
dataset.head()
```

运行效果：

![image-20231220134349989](assets/image-20231220134349989.png)



上述代码中，通过load_dataset函数加载了seaborn中内置的数据集，根据iris数据集绘制多个双变量分布。

```python
# 绘制多个成对的双变量分布
sns.pairplot(dataset)
```

结果效果：

![image-20231220134455988](assets/image-20231220134455988.png)



#### 关于load_dataset代码的错误说明

##### 错误提示

![image-20231220152353338](assets/image-20231220152353338.png)

![image-20231220152400913](assets/image-20231220152400913.png)

![image-20231220152407883](assets/image-20231220152407883.png)



##### 错误原因

上面三个错误，都是网络远程连接失败。load_dataset加载数据集时是通过网络请求下载回来的。所以上面的错误就是网络连接失败。



##### 解决方案

报错conda，pip在没有设置-i 参数指定到清华源时，遇到getaddrinfo报错，都是这么处理。

- 使用科学上网。
- 耐心点，多执行几次，迟早成功。
- 自己手动下载数据集下来到本地替换即可。

这里使用第三个方案，手动下载数据集。

github地址：https://github.com/mwaskom/seaborn-data/

下载数据包

![image-20231220185512214](assets/image-20231220185512214.png)



在jupyter中使用sns.get_data_home()方法获取当前保存数据集的目录

![image-20231220185605185](assets/image-20231220185605185.png)

打开这个目录，并把下载回来的数据集压缩包替换进去即可。

![image-20231220185731169](assets/image-20231220185731169.png)

把路径去除\斜杠，回车打开目录

![image-20231220185830343](assets/image-20231220185830343.png)

![image-20231220185915428](assets/image-20231220185915428.png)



