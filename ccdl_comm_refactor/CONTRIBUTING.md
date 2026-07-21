# ccdl_comm开发规范
## 代码规范
1. python版本：3.10

2. 缩进采用4个空格，而不是TAB

3. 命名规范：
   1. 文件名、变量名、方法名：小写+下划线，如data_manager.py
   2. 类名：帕斯卡命名法，如DataManager
   3. 对方法的命名遵循以下原则：
      1. 如果类的成员函数仅以self作为输入，尽量以“属性”对类的成员函数进行命名，如parameters()，而不是get_parameters()
      2. 如果函数有除了self之外的输入，优先采用动词对函数命名，如generate_batch(data, device, indices=None)，而不是batch_generation(data, device, indices=None)
   4. 内部方法、变量前加_
   5. 使用被广泛接受的缩写（如obs, a2c, img等），否则写全称
   6. 杜绝无意义命名，如a、b、c之类

4. 顶级定义之间空两行, 方法定义之间空一行

5. 空格：
   1. 括号内不要有空格，如generate_batch(data, device, indices=[1,2,3])，不要写成generate_batch(data, device, indices=[1, 2, 3])
   2. 二元符号前后加空格，如x == y，而不是x==y
   3. 冒号、逗号后面加空格，如x, y = func()，而不是x,y = func()
   4. 表示默认参数或者关键字参数的时候，=前后不加空格，如DataManager(env=cart_pole, agent=agent, buffer_size=5000)，而不是DataManager(env = cart_pole, agent = agent, buffer_size = 5000)

6. 函数参数、返回值使用类型注解来提升代码开发规范，避免由于数据格式错误导致的不必要bug，详见这个[链接](https://docs.python.org/zh-cn/3/library/typing.html)

7. 注释：

   1. 除了极其简单明了、且外部不可见的函数之外，必须写注释，注释需要包含对函数功能的介绍、输入、输出的详细说明以及可能抛出的异常说明，格式如下：

      ```python
      def fetch_bigtable_rows(big_table, keys, other_silly_variable=None):
          """Fetches rows from a Bigtable.

          Retrieves rows pertaining to the given keys from the Table instance
          represented by big_table.  Silly things may happen if
          other_silly_variable is not None.

          Args:
              big_table: An open Bigtable Table instance.
              keys: A sequence of strings representing the key of each table row
                  to fetch.
              other_silly_variable: Another optional variable, that has a much
                  longer name than the other args, and which does nothing.

          Returns:
              A dict mapping keys to the corresponding table row data
              fetched. Each row is represented as a tuple of strings. For
              example:

              {'Serak': ('Rigel VII', 'Preparer'),
               'Zim': ('Irk', 'Invader'),
               'Lrrr': ('Omicron Persei 8', 'Emperor')}

              If a key from the keys argument is missing from the dictionary,
              then that row was not found in the table.

          Raises:
              IOError: An error occurred accessing the bigtable.Table object.
          """
          pass
      ```

   2. 类的定义下有一个用于描述该类的文档字符串。如果该类有外部可访问的属性，那么注释中应该有一个属性(Attributes)段。

      ```python
      class SampleClass(object):
          """Summary of class here.

          Longer class information....
          Longer class information....

          Attributes:
              likes_spam: A boolean indicating if we like SPAM or not.
              eggs: An integer count of the eggs we have laid.
          """

          pass
      ```

   3. 对于技巧性较强的代码段，需要使用注释单独说明

## 代码提交规范

### 代码提交遵循以下步骤：
1. 首先与master分支同步
2. 从master分支checkout一个新的分支
3. 修改代码
4. 本地跑通测试
5. 提交代码到GitHub（commit & push）
6. 提交pull request
7. pull request关闭之后，删除分支

### commit message规范

commit message遵循以下格式：`<type>(<scope>): <subject>`


* type为必须字段，用于说明git commit的类别，只允许使用下面的标识。
  * feat：新功能（feature）
  * fix：修复bug
  * docs：文档（documentation）。
  * style：格式（不影响代码运行的变动）。
  * refactor：重构（即不是新增功能，也不是修改bug的代码变动）。
  * perf：优化相关，比如提升性能、体验。
  * test：增加测试。
  * build：影响系统安装或外部依赖关系的更改，如setup.py等
  * ci：对CI配置文件和脚本的更改，比如GitHub action


* scope为可选字段，用于说明commit影响的范围，比如env、data_manager、algorithm等等，如果你的修改影响了不止一个scope，你可以使用*代替。
* subject为必须字段，详细描述修改内容

例如：

1. fix(env):修复并行环境存在的通讯错误
2. feat(data_manager):增加分布式数据采集的支持

### 通过工具来规范commit message

1. 安装依赖
```shell
pip install commitizen
pip install pre-commit
```
2. 在项目根目录下面运行`pre-commit install --hook-type commit-msg`，以生成git hooks
3. 每次commit的时候，将原来的`git commit`指令替换为`cz commit`，然后根据指示提交commit即可![img](https://tva1.sinaimg.cn/large/008i3skNly1guiatoauiej60ye07vdhg02.jpg)

### 注意事项
* 严禁未经pull request而直接merge到master分支
* 更严禁直接修改master分支
* 在commit和pull request时，详细说明修改内容
* 已发布的 Git tag 和 GitHub Release 必须保持不可变，不得删除后重建或移动到其他 commit
* 发布后的修复必须递增 patch 版本，例如从 `v0.1.0` 发布为 `v0.1.1`
