# 信号控制
基于Phaselight的信号控制，是基于车道级别的信息进行控制，表面看上去是拓扑无关，实际还是拓扑相关，所以可以改进。

## 改进措施
把拓扑结构信息引入，得到车道注意力emedding。
```python
def merge(single_line_embedding,all_lines_embedding,structure):
    line_embedding -> q
    all_lines_embedding -> v
    all_lines_embedding -> k
    (Mask(q@k))*v
    pass
```

# 环境设置
`python 3.9`+ `requirements.txt`

# 参考的仓库
不得不说，Eclipse的sumo的文档做的真是一坨大便。很多东西都不一样。

通过pip下载sumo：https://sumo.dlr.de/docs/Downloads.php
和installing是完全不一样的东西：https://sumo.dlr.de/docs/Installing/index.html。
这应该是两个人写的，并且两个人都不知道对方的存在。

1.解析路网：

基于cityflow和sumo的信号控制：https://github.com/DaRL-LibSignal/LibSignal
基于Cityflow的efficient traffic signal：https://github.com/LiangZhang1996/Efficient_XLight

2.修改仿真引擎


3.修改