#region other-package
import xml.etree.ElementTree as ET
from collections import defaultdict
#endregion

#region my-package
from datatype.define_datatype import NetConfig
#endregion

def net_2_struct(net):
    """
    从路网文件中提取信息：
    1.内部车道的形状，用于判断冲突拓扑信息
    2.交叉口的进车道和出车道，以及移动路径，和每个移动路径的进车道出车道。
    """
    net_file = net
    tree = ET.parse(net_file)
    root = tree.getroot()
    # 统计内部车道的信息，主要是内部车道的位置，方便后面计算冲突车道
    lane_2_shape = defaultdict(list)
    edge_2_lane = defaultdict(lambda : defaultdict(str))
    for edge in root.findall('edge'):
        edge_id = edge.get('id')
        edge_function = edge.get('function')
        for lane in edge.findall('lane'):
            lane_id = lane.get('id')
            lane_index = lane.get('index')
            lane_shape = lane.get('shape')
            if edge_2_lane[edge_id][lane_index] != '':
                raise 'error'
            edge_2_lane[edge_id][lane_index] = lane_id
            for ax in lane_shape.strip().split(' '):
                x,y = ax.strip().split(',')
                lane_2_shape[lane_id].append((float(x),float(y)))
    intersection_2_position = defaultdict(tuple)
    for junc in root.findall('junction'):
        if junc.get('type') == "traffic_light":
            intersection_2_position[junc.get('id')] = (float(junc.get('x')),float(junc.get('y')))
    # 每个交叉口要记录进edge和出edge，因为车辆的排队数量指标是从edge角度表示的
    # 每个交叉口也需要记录index与内部车道之间的对应关系，因为这和设置交叉口的状态有关。
    # 每个内部车道也需要记录进车道和出车道信息，因为内部车道的状态信息与进车道和出车道信息是相关的。
    intersection_2_updownstream = defaultdict(lambda : defaultdict(set))
    lane_2_updownstream = defaultdict(lambda : defaultdict(str))
    for connection in root.findall('connection'):
        from_edge = connection.get('from')
        to_edge = connection.get('to')
        from_lane = connection.get('fromLane')
        to_lane = connection.get('toLane')
        tl = connection.get('tl')
        linkIndex = connection.get('linkIndex')
        via = connection.get('via')
        if tl != None:
            intersection_2_updownstream[tl]['from'].add(edge_2_lane[from_edge][from_lane])
            intersection_2_updownstream[tl]['to'].add(edge_2_lane[to_edge][to_lane])
            intersection_2_updownstream[tl]['traffic_light'].add((int(linkIndex),via))
            if lane_2_updownstream[via]['from'] != "":
                raise "error"
            lane_2_updownstream[via]['from'] = edge_2_lane[from_edge][from_lane] 
            lane_2_updownstream[via]['to']= edge_2_lane[to_edge][to_lane]
    return NetConfig(lane_2_shape=lane_2_shape,
              intersection_2_updownstream=intersection_2_updownstream,
              lane_2_updownstream=lane_2_updownstream,
              intersection_2_position=intersection_2_position)
if __name__ == '__main__':
    net = '/data/hupenghui/tsc/net.net.xml'
    print(net_2_struct(net))
