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