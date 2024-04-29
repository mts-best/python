

[AllenDowney/ThinkBayes: Code repository for Think Bayes. (github.com)](https://github.com/AllenDowney/ThinkBayes/tree/master?tab=readme-ov-file)

跟着这个学习python贝叶斯分析（书籍是开源的）

可以学习到怎么把数学和代码结合进行数据分析

学习notebook

# 第一节：贝叶斯的提出

## 数据下载

这个数据是根据职业，性别，政治倾向，和种种（7个属性），判断一个人的职业是不是银行出纳

```python
# Load the data file,从github仓库中下载gss_bayes.csv到当前文件夹

from os.path import basename, exists

def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve
        local, _ = urlretrieve(url, filename)
        print('Downloaded ' + local)
    
download('https://github.com/AllenDowney/ThinkBayes2/raw/master/data/gss_bayes.csv')
```

数据的dataframe的列和数据的说明

The `DataFrame` has one row for each person surveyed and one column for each variable I selected.

The columns are

* `caseid`: Respondent identifier.

* `year`: Year when the respondent was surveyed.

* `age`: Respondent's age when surveyed.

* `sex`: Male or female.

* `polviews`: Political views on a range from liberal to conservative.

* `partyid`: Political party affiliation, Democrat, Independent, or Republican.

* `indus10`: [Code](https://www.census.gov/cgi-bin/sssd/naics/naicsrch?chart=2007) for the industry the respondent works in.

Let's look at these variables in more detail, starting with `indus10`.

```python
import pandas as pd

gss = pd.read_csv('gss_bayes.csv')
gss.head()
#使用的pandas，选择银行相关的职业编码
banker = (gss['indus10'] == 6870)
banker.head()
#查看数量
banker.sum()
#计算百分比banker.sum()/len(banker)
banker.mean()
```

```python
#概率函数
def prob(A):
    """Computes the probability of a proposition, A."""    
    return A.mean()

prob(banker)

#计算其他属性
female = (gss['sex'] == 2)
#政党倾向，具体参照notebook
liberal = (gss['polviews'] <= 3)

```

## 联合概率(Conjunction)

使用&，把条件进行合并

```python
#条件顺序不影响
prob(democrat & banker)
prob(banker & democrat)
```

## 条件概率

条件概率是一种取决于条件的概率，但这可能不是最有用的定义。下面是一些例子：

如果一个受访者是自由主义者，那么他是民主党人的概率是多少？

如果受访者是银行家，那么受访者是女性的概率是多少？

如果受访者是女性，那么她是自由主义者的概率是多少？

让我们从第一个问题开始，我们可以这样解释："在所有自由派受访者中，民主党人占多大比例？"

我们可以分两步计算这个概率：

```python
democrat = (gss['partyid'] <= 1)
banker = (gss['indus10'] == 6870)
liberal = (gss['polviews'] <= 3)
#此时democrat和banker都是一个bool的列数据，我们可以像串并联，是否有断路一样，选出是true的项
#liberal是选择，若为true，对应的democrat被选出来
#通过这样的方式来表示条件概率
#找出liberal中是democrat的概率 
selected = democrat[liberal]
prob(selected)
#0.5206403320240125
```

一半多一点的自由主义者是民主党人。如果这一结果低于您的预期，请记住：

我们对 "民主党人 "的定义有些严格，不包括 "倾向 "民主的独立人士。

数据集中的受访者最早可追溯到 1974 年；与现在相比，在这一时期的早期，政治观点与党派归属之间的一致性较低。

让我们试试第二个例子："如果受访者是银行家，那么受访者是女性的概率是多少？我们可以将其解释为："在所有银行家受访者中，女性占多大比例？

同样，我们将使用括号运算符只选择银行家，并使用 prob 计算女性的比例。

```python
#另一个例子,在银行家中筛选女性
selected = female[banker]
prob(selected)
```

```python
#定义条件概率的函数
def conditional(proposition, given):
    """Probability of A conditioned on given."""
    return prob(proposition[given])
#可指定传参，按序的话，不需要，可读性更好
conditional(liberal, given=female)
```

## 条件概率不是交换概率

条件概率不具备**&交换**的性质

我们已经看到，连词是交换性的；也就是说，prob(A & B) 总是等于 prob(B & A)。

但条件概率不是交换的，也就是说，conditional(A, B) 与 conditional(B, A) 并不相同。

如果我们看一个例子，就会明白这一点。之前，我们计算了受访者是银行职员时，受访者是女性的概率。

```python
conditional(female, given=banker)
#0.7706043956043956
conditional(banker, given=female)
#0.02116102749801969
```

## 条件的联合

我们可以将条件概率和联立概率结合起来。例如，如果受访者是自由民主党人，那么她是女性的概率就是这样。

```python
#在自由和民主党中，女性的比例
conditional(female, given=liberal & democrat)
#0.576085409252669
#在银行从业者中，女性和自由主义者的比例  17% of bankers are liberal women.
conditional(liberal & female, given=banker)
#0.17307692307692307
```

## 三个定理

在接下来的几节中，我们将推导出连接概率和条件概率之间的三种关系：

定理 1：利用连接词计算条件概率。

定理 2：利用条件概率计算连合概率。

定理 3: 利用条件（A，B）计算条件（B，A）。

定理 3 也称为贝叶斯定理。

我将使用概率的数学符号来书写这些定理：

* $P(A)$ is the probability of proposition $A$.

* $P(A~\mathrm{and}~B)$ is the probability of the conjunction of $A$ and $B$, that is, the probability that both are true.

* $P(A | B)$ is the conditional probability of $A$ given that $B$ is true.  The vertical line between $A$ and $B$ is pronounced "given".

 

```python
#定理1
#计算银行家中女性的概率的几种等价方法
female[banker].mean()

conditional(female, given=banker)

prob(female & banker) / prob(banker)

#定理2
prob(liberal & democrat)

prob(democrat) * conditional(liberal, democrat)

#定理3
conditional(liberal, given=banker)
prob(liberal) * conditional(banker, liberal) / prob(banker)
```

即定理一

$$P(A|B) = \frac{P(A~\mathrm{and}~B)}{P(B)}$$

定理二

$$P(A~\mathrm{and}~B) = P(B) ~ P(A|B)$$

定理三

$$P(A~\mathrm{and}~B) = P(B~\mathrm{and}~A)$$

$$P(B) P(A|B) = P(A) P(B|A)$$

根据前两条得到第三个

$$P(A|B) = \frac{P(A) P(B|A)}{P(B)}$$

## 贝叶斯全概率公式

除了这三个定理，我们还需要一个东西来进行贝叶斯统计：总概率定律。下面是用数学符号表示的该定律的一种形式：

$$P(A) = P(B_1 \mathrm{and} A) + P(B_2 \mathrm{and} A)$$

A事件的条件存在多个，子条件B1和子条件B2

```python
#男性的序列
male = (gss['sex'] == 1)
#男性银行从业者和女性银行从业者
prob(male & banker) + prob(female & banker)
#0.014769730168391153
(prob(male) * conditional(banker, given=male) + prob(female) * conditional(banker, given=female))
#0.014769730168391153

```

有多个条件的时候

$$P(A) = \sum_i P(B_i) P(A|B_i)$$

```python
B = gss['polviews']
#统计这个数据列的所有值，并统计数量
B.value_counts().sort_index()
#计算某个值的概率
i = 4
#在政治观点为评级4的银行从业概率
prob(B==i) * conditional(banker, B==i)
#全概率公式，得到的所有银行从业者的概率
sum(prob(B==i) * conditional(banker, B==i)for i in range(1, 8))
prob(banker)
```

# 第二节：贝叶斯的使用

举个例子，我们使用了社会总体调查的数据和贝叶斯定理来计算条件概率。但是，由于我们拥有完整的数据集，我们其实并不需要贝叶斯定理。直接计算等式的左边已经很容易了，计算等式的右边也不难。

$$P(A|B) = \frac{P(A) P(B|A)}{P(B)}$$

但我们往往没有完整的数据集，在这种情况下，贝叶斯定理就更有用了。在本章中，我们将用贝叶斯定理解决几个与条件概率有关的更具挑战性的问题。

## 饼干问题：



>假设有两碗饼干。
>
>第 1 碗里有 30 块香草饼干和 10 块巧克力饼干。
>
>第 2 碗里有 20 块香草饼干和 20 块巧克力饼干。
>
>现在假设你随机选择其中一个碗，然后不看，随机选择一块饼干。如果饼干是香草味的，那么它来自第 1 碗的概率是多少？

我们想要的是，在我们得到一块香草饼干的情况下，我们从 1 号碗中选择的条件概率$P(B_1 | V)$.

但我们从问题的陈述中得到的是

V是拿到香草饼干

从 1 号碗中选择香草饼干的条件概率$P(V | B_1)$

从第 2 碗中选择香草饼干的条件概率$P(V | B_2)$

使用贝叶斯公式构建这两个的联系

$$P(B_1|V) = \frac{P(B_1)~P(V|B_1)}{P(V)}$$
$$
P(V)=P(B_1)~P(V|B_1)+P(B_2)~P(V|B_2)
$$

$$
P(B_1) = \frac{1}{2} \\
P(V|B_1)=\frac{3}{4}
$$

1.选择碗

2.选择碗中的香草饼干

从碗1中选择香草饼干的概率。$P(B_1)~P(V|B_1)$

$$P(V) = (1/2)~(3/4) ~+~ (1/2)~(1/2) = 5/8$$

由于我们选择任何一个碗的机会都相同，而且碗里的饼干数量也相同，所以我们选择任何一块饼干的机会都相同。

两个碗中有 50 块香草饼干和 30 块巧克力饼干

$$P(B_1|V) = (1/2)~(3/4)~/~(5/8) = 3/5$$

## 非同步贝叶斯

这里有另一种理解贝叶斯定理的方法：它为我们提供了一种更新假设概率的方法、$H$给出一些数据$D$

这种解释是 "非同步 "的，意思是 "与时间变化有关"；在这种情况下，假设的概率会随着我们看到新数据而变化。

$$P(H|D) = \frac{P(H)~P(D|H)}{P(D)}$$



$P(H)$是我们看到数据之前假设的概率，称为先验概率，或简称先验。

$P(H|D)$是我们看到数据后的假设概率，称为后验概率。

 $P(D|H)$是数据在假设条件下的概率，称为可能性。

$P(D)$是数据在任何假设下的总概率。

有时，我们可以根据**背景信息计算先验**。例如，饼干问题规定我们以相等的概率随机选择一个碗。

在其他情况下，先验是主观的；也就是说，有理智的人可能会因为使用不同的背景信息或对相同信息有不同的解释而产生分歧。

可能性通常是最容易计算的部分。在饼干问题中，我们得到了每个碗中饼干的数量，因此我们可以计算每个假设下数据的概率。



**计算数据的总概率**可能很棘手。它应该是**在任何假设下看到数据的概率**，但很难确定其含义。

大多数情况下，我们通过**指定一组假设**来简化问题：

**互斥**，即只有一个假设为真

**集体穷举**，即其中一个假设必须为真。

$$P(D) = P(H_1)~P(D|H_1) + P(H_2)~P(D|H_2)$$

$$P(D) = \sum_i P(H_i)~P(D|H_i)$$

本节使用数据和先验概率计算后验概率的过程称为贝叶斯更新。



## 贝叶斯表

贝叶斯表是进行贝叶斯更新的便捷工具。你可以在纸上或使用电子表格编写贝叶斯表，但在本节中，我将使用 Pandas DataFrame。

首先，我将制作一个空的 DataFrame，每个假设各占一行：

```python
import pandas as pd

table = pd.DataFrame(index=['Bowl 1', 'Bowl 2'])

table['likelihood'] = 3/4, 1/2
table
```

这里我们看到了与前一种方法的不同之处：我们计算的是两种假设的可能性，而不仅仅是碗 1：

从第 1 碗中得到香草饼干的概率是 3/4。

从第 2 碗中得到香草饼干的概率是 1/2。

你可能会注意到，这些可能性加起来并不等于 1。没关系，每个可能性都是基于不同假设的概率。它们没有理由相加等于 1，即使不等于 1 也没有问题。

下一步与贝叶斯定理类似，我们将先验值乘以似然值：

```python
table['unnorm'] = table['prior'] * table['likelihood']
table
```

我把结果称为非正态分布，因为这些值是 "**非正态化后验值**"。每个后验值都是一个先验值和一个似然值的乘积：

$$P(H_i)~P(D|H_i)$$

这就是贝叶斯定理的分子。如果我们将它们相加，就会得出

$$P(H_1)~P(D|H_1) + P(H_2)~P(D|H_2)$$

```python
prob_data = table['unnorm'].sum()
prob_data
```

我们可以这样计算后验概率：

```python
table['posterior'] = table['unnorm'] / prob_data
table
```

第 1 碗的后验概率是 0.6，这就是我们明确使用贝叶斯定理得到的结果。另外，我们还得到了第 2 碗的后验概率，即 0.4。

当我们将未归一化的后验概率相加并除以时，我们就迫使后验概率相加为 1。这个过程被称为 "归一化"，这就是为什么数据的总概率也被称为 "归一化常数"。

## 骰子问题

贝叶斯表还可以解决两个以上假设的问题。例如

> 假设我有一个盒子，里面有一个 6 面骰子、一个 8 面骰子和一个 12 面骰子。
>
> 在这个例子中，有三个先验概率相同的假设。数据是我的报告，结果是 1。
>
> 我选择 6 面骰子的概率是多少？

如果我选择 6 面骰，数据的概率是 1/6。如果我选择 8 面骰子，概率是 1/8，如果我选择 12 面骰子，概率是 1/12。

下面是一个贝叶斯表，用整数表示假设：

```python
table2 = pd.DataFrame(index=[6, 8, 12])
```

我会用分数来表示先验概率和可能性。这样它们就不会被舍入为浮点数。

```python
from fractions import Fraction

table2['prior'] = Fraction(1, 3)
table2['likelihood'] = Fraction(1, 6), Fraction(1, 8), Fraction(1, 12)
table2
```

对于两步条件概率事件， 我们把贝叶斯的表计算给定义为如下函数

```python
def update(table):
    """Compute the posterior probabilities."""
    table['unnorm'] = table['prior'] * table['likelihood']
    prob_data = table['unnorm'].sum()
    table['posterior'] = table['unnorm'] / prob_data
    return prob_data

#调用
prob_data = update(table2)
table2
```

6 面骰子的后验概率是 4/9，比其他骰子的概率 3/9 和 2/9 稍高一些。直观地说，6 面骰子的可能性最大，因为它产生我们看到的结果的可能性最高。

## 门问题

接下来，我们将使用贝叶斯表来解决概率论中最有争议的问题之一。

蒙蒂-霍尔（Monty Hall）问题基于一个名为 "让我们做个交易 "的游戏节目。如果你是该节目的参赛者，那么游戏是这样进行的：

主持人蒙特-霍尔（Monty Hall）向你展示三扇分别编号为 1、2 和 3 的紧闭的门，并告诉你每扇门后面都有一个奖品。

其中一个奖品很值钱（通常是汽车），另外两个奖品不太值钱（通常是山羊）。

游戏的目的是猜哪扇门里有汽车。如果你猜对了，汽车就归你。

假设你选的是 1 号门。在打开你选择的门之前，蒙蒂打开了 3 号门，露出了一只山羊。然后，蒙蒂让你选择坚持原来的选择，或者换到剩下的未打开的门。



策略：

为了最大限度地增加赢得汽车的机会，您应该坚持使用 1 号门，还是改用 2 号门？

要回答这个问题，我们必须对主持人的行为做出一些假设：

蒙蒂总是会打开一扇门，让你选择换门。

他从来不会打开你选的那扇门，也不会打开有车的那扇门。

如果你选择了有车的那扇门，他就会随机选择另外一扇门。



根据这些假设，你最好还是换。坚持就是胜利发生的概率是$1/3$，更换后获胜的概率是$2/3$ 。

如果你以前没有遇到过这个问题，你可能会觉得这个答案很意外。你并不孤单；许多人都有一种强烈的直觉，认为坚持还是转换并不重要。他们的理由是，还剩下两扇门，所以汽车在 A 门后面的几率是 50%。但这是错误的。

要想知道原因，使用贝叶斯表格会有所帮助。我们从三个假设开始：汽车可能在 1 号门、2 号门或 3 号门后面。根据问题的陈述，每扇门的先验概率都是 1/3。

```python
table3 = pd.DataFrame(index=['Door 1', 'Door 2', 'Door 3'])
table3['prior'] = Fraction(1, 3)
table3

```

数据是蒙蒂打开了 3 号门，露出了一只山羊。因此，让我们考虑一下每个假设下数据的概率：

如果汽车在 1 号门后面，蒙蒂随机选择 2 号门或 3 号门，那么他打开 3 号门的概率是1/2

如果汽车在 2 号门后面，蒙蒂就必须打开 3 号门，因此在此假设下数据的概率为1

如果汽车在 3 号门后面，蒙蒂没有打开门，那么在此假设下数据的概率为0

```python
table3['likelihood'] = Fraction(1, 2), 1, 0
table3
```

使用表计算一下

```
update(table3)
table3
```

正如这个例子所示，我们对概率的直觉并不总是可靠的。贝叶斯定理可以提供一种 "分而治之 "的策略：

1. 首先，写下假设和数据。

2. 接下来，计算先验概率。

3. 最后，计算每个假设下数据的可能性。

剩下的就交给贝叶斯表了。

## 总结

在本章中，我们分别使用贝叶斯定理和贝叶斯表解决了饼干问题。这两种方法并无实质区别，但贝叶斯表可以更容易地计算数据的总概率，特别是对于有两个以上假设的问题。

然后，我们解决了骰子问题（我们将在下一章再次看到这个问题）和蒙蒂-霍尔问题（你可能希望永远不要再看到这个问题）。

如果蒙蒂-霍尔问题让你头疼，那你并不孤单。但我认为，它展示了贝叶斯定理作为解决棘手问题的分而治之策略的威力。我希望它能让你了解答案为何如此。

当蒙蒂打开一扇门时，他提供了一些信息，我们可以利用这些信息更新我们对汽车位置的看法。部分信息是显而易见的。如果他打开 3 号门，我们就知道汽车不在 3 号门后面。但部分信息更为微妙。如果车在 2 号门后面，那么打开 3 号门的可能性就更大，而如果车在 1 号门后面，那么打开 3 号门的可能性就更小。因此，这些数据是有利于 2 号门的证据。我们将在以后的章节中再来讨论证据的概念。

在下一章中，我们将扩展 "饼干问题 "和 "骰子问题"，并从基本概率论迈向贝叶斯统计。

# 第三节：分布

在上一章中，我们使用贝叶斯定理解决了一个 饼干问题；然后，我们使用贝叶斯表再次解决了这个问题。在本章中，我们将冒着考验你的耐心的风险，再次使用 Pmf 对象（表示 "概率质量函数"）来解决这个问题。我将解释它的含义，以及为什么它对贝叶斯统计很有用。

我们将使用 贝叶斯表对象来解决一些更具挑战性的问题，并向贝叶斯统计法再迈进一步。但我们将从分布开始。

```python
#下载基础工具函数文件
from os.path import basename, exists

def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve
        local, _ = urlretrieve(url, filename)
        print('Downloaded ' + local)
    
download('https://github.com/AllenDowney/ThinkBayes2/raw/master/soln/utils.py')
```



```python
from utils import set_pyplot_params
set_pyplot_params()
```

## 分布

在统计学中，分布是一组可能的结果及其相应的概率。例如，如果您掷一枚硬币，有两种可能的结果，其概率大致相同。如果掷一个六面骰子，可能的结果集合是数字 1 到 6，每个结果的相关概率是 1/6。

为了表示分布，我们将使用一个名为 **empiricaldist** 的库。经验 "分布基于数据，而不是理论分布。我们将在全书中使用这个库。我将在本章介绍其基本功能，稍后我们将看到其他功能。

## 概率分布函数

如果分布中的结果是离散的，我们可以用概率质量函数（或 PMF）来描述分布，它是一个将每个可能的结果映射到其概率的函数。

empiricaldist 提供了一个名为 Pmf 的类来表示概率质量函数。要使用 Pmf，可以这样导入：

先下载这个库

```python
pip install empiricaldist
或者
conda install empiricaldist
```



```
from empiricaldist import Pmf
```

下面的示例制作了一个表示掷硬币结果的 Pmf。

```python
coin = Pmf()
coin['heads'] = 1/2
coin['tails'] = 1/2
coin
```

Pmf 会创建一个没有结果的空 Pmf。然后，我们可以使用括号运算符添加新的结果。在本例中，两个结果用字符串表示，它们的概率相同，都是 0.5。

您还可以从一系列可能的结果中提取 Pmf。

下面的示例使用 Pmf.from_seq 制作了一个表示六面骰子的 Pmf。

```python
die = Pmf.from_seq([1,2,3,4,5,6])
die
```

从序列构建统计信息

从字符串构建,会返回每种字母的统计值

```
letters = Pmf.from_seq(list('Mississippi'))
letters
```

由于字符串中的字母不是随机过程的结果，我将用更一般的术语 "数量 "来表示 Pmf 中的字母。

Pmf 类继承自 Pandas Series，因此任何可以用 Series 实现的功能，也可以用 Pmf 实现。

例如，您可以使用括号运算符查找数量并获得相应的概率。

```python
letters['s']
#0.36363636363636365

```

在 "密西西比 "Mississippi一词中，约 36% 的字母是 "s"。

但是，如果您想知道一个不在分布中的量的概率，就会出现 KeyError。

```python
try:
    letters['t']
except KeyError as e:
    print(type(e))
```

也可以以函数的形式来调用pmf

```python
letters('s')
letters('t')
#0而不会返回错误
```

使用括号，您还可以提供一系列量，并得到一系列概率

```python
die([1,4,7])
#array([0.16666667, 0.16666667, 0.        ])
```

Pmf 中的数量可以是字符串、数字或任何其他可以存储在 Pandas Series 索引中的类型。如果你熟悉 Pandas，这将有助于你使用 Pmf 对象。不过，我会在接下来的过程中向你解释你需要知道的东西。

## 饼干问题使用pmf

```python
prior = Pmf.from_seq(['Bowl 1', 'Bowl 2'])
prior

likelihood_vanilla = [0.75, 0.5]
posterior = prior * likelihood_vanilla
posterior

```

Bowl 1 0.375

Bowl 2 0.250

要使它们相加等于 1，我们可以使用 Pmf 提供的 normalize 方法。

```python
posterior.normalize()
posterior

#Bowl 1	0.6
#Bowl 2	0.4
posterior('Bowl 1')
#0.6
```

使用 Pmf 对象的一个好处是可以轻松地使用更多数据进行连续更新。例如，假设你把第一块饼干放回原处（这样碗中的内容就不会改变），然后从同一个碗中再次提取。如果第二块饼干也是香草味的，我们可以像这样进行第二次更新：

可以更方便的计算多轮的概率

```python
posterior *= likelihood_vanilla
posterior.normalize()
posterior
#Bowl 1	0.692308
#Bowl 2	0.307692
```

现在，1 号碗的后验概率几乎是 70%。但假设我们再做一次同样的事情，得到的是一块巧克力饼干。

下面是新数据的可能性：

```python
likelihood_chocolate = [0.25, 0.5]
posterior *= likelihood_chocolate
posterior.normalize()
posterior
#Bowl 1	0.529412
#Bowl 2	0.470588
```

现在，第 1 碗的后验概率约为 53%。两块香草饼干和一块巧克力饼干之后，后验概率接近 50/50。

## 101 Bowls

接下来，让我们用 101 个碗来解决饼干问题：

第 0 碗中有 0% 的香草饼干、

第 1 碗里有 1%的香草饼干、

第 2 碗含有 2% 的香草饼干、

以此类推，直到

第 99 碗含有 99% 的香草饼干，以及

第 100 碗则全部是香草饼干。

和之前的版本一样，只有香草和巧克力两种饼干。因此，0 号碗里全是巧克力饼干，1 号碗里 99% 都是巧克力，以此类推。

假设我们随机选择一个碗，随机选择一块饼干，结果它是香草味的。这块饼干来自第 1 碗的概率是多少？

为了解决这个问题，我将使用 np.arange 制作一个数组，代表 101 个假设，编号从 0 到 100。

```python

import numpy as np

hypos = np.arange(101)
#开始把概率全置为1，之后使用正则化，归一平均
prior = Pmf(1, hypos)
prior.normalize()
```

正如本例所示，我们可以用两个参数来初始化 Pmf。第一个参数是**先验概率**；第二个参数是一个**数量序列**。

在本例中，概率都是相同的，因此我们只需提供其中一个参数，它就会在所有假设中 "广播"。由于所有假设都具有相同的先验概率，因此这种分布是均匀的。

下面是前几个假设及其概率。

给出一个生成序列概率

```python
likelihood_vanilla = hypos/100
likelihood_vanilla[:5]
#array([0.  , 0.01, 0.02, 0.03, 0.04])
```

```python
posterior1 = prior * likelihood_vanilla
posterior1.normalize()
posterior1.head()
```



绘制概率分布图像

```python
from utils import decorate

def decorate_bowls(title):
    decorate(xlabel='Bowl',
             ylabel='PMF',
             title=title)
    
prior.plot(label='prior', color='C5')
posterior1.plot(label='posterior', color='C4')
decorate_bowls('Posterior after one vanilla cookie')
```

![image-20240429155801707](https://s2.loli.net/2024/04/29/Wn7o3ui4LaYM2Vx.png)

0 号碗的后验概率为 0，因为它不含香草饼干。碗 100 的后验概率最大，因为它含有最多的香草饼干。在两者之间，后验分布的形状是一条直线，因为可能性与碗的数量成正比。

现在假设我们把饼干放回去，从同一个碗中再次抽取，得到另一块香草饼干。这是第二块饼干之后的更新：

```python
posterior2 = posterior1 * likelihood_vanilla
posterior2.normalize()
#进行一次概率更新后，概率发生了变化
posterior2.plot(label='posterior', color='C4')
decorate_bowls('Posterior after two vanilla cookies')
```

![image-20240429155723143](https://s2.loli.net/2024/04/29/w3hGmxtjy89uebq.png)

两块香草饼干后，编号高的碗的后验概率最高，因为它们含有最多的香草饼干；编号低的碗的后验概率最低。

但假设我们再次抽签，得到一块巧克力饼干。更新如下

```python
likelihood_chocolate = 1 - hypos/100

posterior3 = posterior2 * likelihood_chocolate
posterior3.normalize()
posterior3.plot(label='posterior', color='C4')
decorate_bowls('Posterior after 2 vanilla, 1 chocolate')

```

![image-20240429155923998](D:\系统数据\git仓库\ThinkBayes2Notebooks\ThinkBayes2\笔记\贝叶斯的python分析.assets\image-20240429155923998.png)

这次比例在2：1的获得了最大后验概率

现在，第 100 碗已经被淘汰，因为它没有巧克力饼干。但数字高的碗仍然比数字低的碗更有可能，因为我们看到的香草饼干比巧克力饼干多。

事实上，后验分布的峰值在第 67 碗，这与我们观察到的数据中香草饼干的比例相对应2/3

后验概率最高的量称为 MAP，即 "最大后验概率"，"a posteriori "在拉丁语中是 "后验 "的意思。

要计算 MAP，我们可以使用数列方法 idxmax：

```python
posterior3.idxmax()
#67
posterior3.max_prob()
#67
```

正如你所猜测的，这个例子其实与碗无关，而是关于比例的估算。想象一下，你有一碗饼干。如果你抽出三块饼干，其中两块是香草味的，你认为碗中香草味饼干的比例是多少？我们刚刚计算的后验分布就是这个问题的答案。

我们将在下一章再讨论估计比例的问题。不过，首先让我们用 Pmf 来解决骰子问题。

## 骰子问题

```python
hypos = [6, 8, 12]
prior = Pmf(1/3, hypos)
prior
prior.qs
#array([ 6,  8, 12], dtype=int64)
prior.ps
#array([0.33333333, 0.33333333, 0.33333333])
```



与前面的例子一样，先验概率会在所有假设中广播。Pmf 对象有两个属性：

qs 包含分布中的数量；

ps 包含相应的概率。

现在我们可以进行更新了。下面是每个假设的数据可能性。



```python
likelihood1 = 1/6, 1/8, 1/12
posterior = prior * likelihood1
posterior.normalize()
posterior
#6	0.444444
#8	0.333333
#12	0.222222
```

现在假设我再掷一次同样的骰子，结果是 7：

```python
likelihood2 = 0, 1/8, 1/12

posterior *= likelihood2
posterior.normalize()
posterior
#6	0.000000
#8	0.692308
#12	0.307692
```

6 面骰子的可能性为 0，因为在 6 面骰子上不可能得到 7。其他两个可能性与上次更新的相同。

以下是更新内容：

我们获得了1，7两个结果，对应骰子的后验概率

## 更新骰子

以下函数是上一节更新函数的通用版本：

```python
def update_dice(pmf, data):
    """Update pmf based on new data."""
    hypos = pmf.qs
    likelihood = 1 / hypos
    impossible = (data > hypos)
    likelihood[impossible] = 0
    pmf *= likelihood
    pmf.normalize()
```

第一个参数是 Pmf，表示可能的骰子及其概率。第二个参数是掷骰子的结果。

第一行从 Pmf 中选择代表假设的量。由于假设是整数，我们可以用它们来计算可能性。一般来说，如果骰子有 n 个面，那么任何可能结果的概率都是 1/n。

不过，我们必须检查不可能出现的结果！如果结果超过了骰子的假设边数，那么该结果的概率为 0。

impossible 是一个布尔数列，每个不可能的结果都为 True。我用它作为可能性的索引，将相应的概率设为 0。

最后，我将 pmf 乘以可能性并进行归一化处理。

下面我们来看看如何使用这个函数计算上一节中的更新。我从一份全新的先验分布开始：

## 总结

我们需要试着学习和使用pmf，并使用函数对一个概率组事件进行归纳和计算



本章将介绍提供 Pmf 的 empiricaldist 模块，我们用它来表示一组假设及其概率。

empiricaldist 基于 Pandas；Pmf 类继承自 Pandas Series 类，并提供了概率质量函数特有的额外功能。我们将在全书中使用 Pmf 和 empiricaldist 中的其他类，因为它们简化了代码，使代码更具可读性。不过，我们也可以直接用 Pandas 来做同样的事情。

我们使用 Pmf 来解决 cookie 问题和骰子问题，这在上一章中已经介绍过。使用 Pmf 可以轻松地对多条数据执行顺序更新。

我们还解决了饼干问题的一个更普通的版本，有 101 个碗，而不是两个。然后我们计算了 MAP，也就是后验概率最高的量。

下一章，我将介绍欧元问题，我们将使用**二项分布**。最后，我们将实现从使用贝叶斯定理到进行贝叶斯统计的飞跃。

但首先，你可能需要做一些练习。（略）

# 第四节：估计比例



## 欧元问题

在《信息论、推理和学习算法》一书中，戴维-麦凯提出了这个问题：

"2002年1月4日星期五，《卫报》刊登了一篇统计声明：

>一枚比利时一欧元硬币在边缘旋转250次后，正面出现140次，反面出现110次。伦敦经济学院统计学讲师巴里-布莱特说："在我看来，这非常可疑。如果硬币没有偏差，出现如此极端结果的几率将低于 7%。

"但是，[麦凯问]这些数据是否能证明硬币是有偏见的，而不是公平的？

要回答这个问题，我们将分两步进行。首先，我们将利用二项分布来了解这 7% 的来源；然后，我们将利用贝叶斯定理来估计这枚硬币出现正面的概率。

## 二项分布(The Binomial Distribution)

假设我告诉你一枚硬币是 "公平 "的，即正面的概率是 50%。如果你旋转两次，有四种结果：HH、HT、TH 和 TT。这四种结果的概率相同，都是 25%。

如果我们把人头的总数加起来，有三种可能的结果：0、1 或 2。0 和 2 的概率是 25%，1 的概率是 50%。

更一般地说，假设人头的概率是p,我们旋转硬币n次。我们总共得到k次是人头面的概率

$$\binom{n}{k} p^k (1-p)^{n-k}$$

我们可以自己计算这个表达式，但也可以使用 SciPy 函数 binom.pmf。例如，如果我们掷硬币 n=2 次，正面的概率为 p=0.5，那么得到 k=1 个正面的概率就是这样：

```python
from scipy.stats import binom
#n实验次数
#实验发生概率
#k某类结果发生的次数
n = 2
p = 0.5
k = 1

binom.pmf(k, n, p)
```

我们也可以调用 binom.pmf，而不是为 k 提供一个单一的值，同时提供一个数组的值。

```python
import numpy as np
ks = np.arange(n+1)

ps = binom.pmf(ks, n, p)
ps
#array([0.25, 0.5 , 0.25])
```

结果是一个 NumPy 数组，其中包含 0、1 或 2 个头的概率。如果我们将这些概率放入 Pmf 中，结果就是给定 n 值和 p 值时 k 的分布。

下面就是它的样子：

```python
from empiricaldist import Pmf

pmf_k = Pmf(ps, ks)
pmf_k
```

>probs
>
>0	0.25
>
>1	0.50
>
>2	0.25

下面的函数计算给定 n 值和 p 值的二项分布，并返回表示结果的 Pmf。

```python
def make_binomial(n, p):
    """Make a binomial Pmf."""
    ks = np.arange(n+1)
    ps = binom.pmf(ks, n, p)
    return Pmf(ps, ks)

pmf_k = make_binomial(n=250, p=0.5)

from utils import decorate
pmf_k.plot(label='n=250, p=0.5')

decorate(xlabel='Number of heads (k)',
         ylabel='PMF',
         title='Binomial distribution')
```

![image-20240429163749905](https://s2.loli.net/2024/04/29/WifO6PlTj21qkeD.png)

```python
pmf_k.max_prob()
#125
pmf_k[125]
#0.0504122131473
pmf_k[140]
#0.008357181724918204
```

但是，尽管这是最有可能的数量，我们得到正好 125 个人头的概率也只有大约 5%。

在麦凯的例子中，我们得到了 140 个人头，这比 125 个人头的可能性还要小：

在麦凯引用的文章中，这位统计学家说："如果硬币没有偏差，得到如此极端结果的几率将小于 7%"。

我们可以用二项分布来检验他的数学。下面的函数获取一个 PMF，并计算数量大于或等于阈值的总概率。

```python
def prob_ge(pmf, threshold):
    """Probability of quantities greater than threshold."""
    #所有结果pmf.qs，大于threshold的标签为true
    ge = (pmf.qs >= threshold)
    #合计概率
    total = pmf[ge].sum()
    return total
```

下面是得到 140 个或更多人头的概率：

```python
prob_ge(pmf_k, 140)
#0.03321057562002164
```

Pmf 提供了一种进行相同计算的方法。

```python
pmf_k.prob_ge(140)
#ge大于等于
```

结果约为 **3.3%**，**低于所引用的 7%**。造成差异的原因是，统计学家将所有 "与 140 一样极端 "的结果包括在内，其中包括小于或等于 110 的结果。

要想知道这是怎么一回事，请回想一下，预期人头的数量是 125。如果我们得到 140，就超出了预期的 15。如果我们得到 110，我们就少了 15。

如下图所示，7% 是这两个 "尾数 "的总和。

```python
import matplotlib.pyplot as plt

def fill_below(pmf):
    qs = pmf.index
    ps = pmf.values
    plt.fill_between(qs, ps, 0, color='C5', alpha=0.4)

qs = pmf_k.index
fill_below(pmf_k[qs>=140])
fill_below(pmf_k[qs<=110])
pmf_k.plot(label='n=250, p=0.5')

decorate(xlabel='Number of heads (k)',
         ylabel='PMF',
         title='Binomial distribution')
```

![image-20240429164445339](https://s2.loli.net/2024/04/29/rNzLUaxYKnDjBCk.png)

下面是我们计算左尾总概率的方法。

```python
pmf_k.prob_le(110)
```

小于或等于 110 的结果概率也是 3.3%，因此 "极端 "结果为 140 的总概率为 6.6%。

这个计算的关键在于，如果硬币是公平的，这些极端结果是不可能出现的。

这很有趣，但并没有回答麦凯的问题。让我们来看看能否回答。

## 贝叶斯估计

任何给定的硬币在边缘旋转时都有一定的正面朝上概率，我称之为概率 x。如果一枚硬币是完全平衡的，我们预计 x 接近 50%，但对于一枚倾斜的硬币，x 可能会有很大不同。我们可以利用贝叶斯定理和观察到的数据来估计 x。

为了简单起见，我将从均匀先验开始，即假设 x 的所有值都具有相同的可能性。这可能不是一个合理的假设，所以我们稍后再考虑其他先验。

我们可以这样建立一个均匀先验

# 第五节：估算计数

# 第六节：赔率和加数

# 第七节：最小值、最大值和混合

# 第八节：泊松过程

# 第九节：决策分析

# 第十节：测试

# 第十一节：比较

第十二节：分类

第十三节：（推理）Inference

第十四节：生存分析（Survival Analysis）

事件时间分析、失效时间分析或时间至事件分析，是一种统计方法，用于分析时间直到特定事件发生的过程。这种方法最初是为了分析医学领域中的生存时间数据而开发的，用于研究患者在某种治疗或疾病观察期间存活的概率。

第十五节：标记与重获（Mark and Recapture）

第十六节: 逻辑回归

第十七节：回归（Regression）

第十八节：共轭先验（Conjugate Priors）



