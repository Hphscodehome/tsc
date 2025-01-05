import sumolib

# 加载网络文件
net = sumolib.net.readNet('/data/hupenghui/tsc/net.net.xml')

# 打印节点信息
print("Nodes:")
for node in net.getNodes():
    print(f"ID: {node.getID()}, X: {node.getCoord()[0]}, Y: {node.getCoord()[1]}")

# 打印边信息
print("\nEdges:")
for edge in net.getEdges():
    print(f"ID: {edge.getID()}, From: {edge.getFromNode().getID()}, To: {edge.getToNode().getID()}, Length: {edge.getLength()}")

# 打印车道信息
print("\nLanes:")
for edge in net.getEdges():
    for lane in edge.getLanes():
        print(f"Edge ID: {edge.getID()}, Lane ID: {lane.getID()}, Speed: {lane.getSpeed()}, Length: {lane.getLength()}")