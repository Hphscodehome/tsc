import sumolib

def parse_roadnet(net_file):
    """
    使用 sumolib 对指定的 roadnet (e.g., .net.xml) 文件进行解析。
    """
    # 使用 sumolib 读取网络文件
    net = sumolib.net.readNet(net_file)
    
    # 获取网络中的所有边
    edges = net.getEdges()
    print(f"文件 {net_file} 中包含边的数量: {len(edges)}")
    
    # 示例：打印每条边的 ID
    for edge in edges:
        print("Edge ID:", edge.getID())

if __name__ == "__main__":
    # 请将下面的文件路径替换为你实际的 roadnet 文件名
    net_file = "/data/hupenghui/LibSignal/data/raw_data/hangzhou_4x4_hetero/hangzhou_4x4_gudang_18041610_1h_m.net.xml"
    parse_roadnet(net_file)