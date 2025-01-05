import xml.etree.ElementTree as ET
net_file = '/data/hupenghui/tsc/net.net.xml'

# 假设我们将上述XML内容保存为 net.xml
tree = ET.parse(net_file)
root = tree.getroot()

# 1. 读取 <location> 信息
location_elem = root.find('location')
if location_elem is not None:
    net_offset = location_elem.get('netOffset')
    conv_boundary = location_elem.get('convBoundary')
    print("netOffset:", net_offset)
    print("convBoundary:", conv_boundary)

# 2. 遍历 <edge> 元素
for edge in root.findall('edge'):
    edge_id = edge.get('id')
    edge_function = edge.get('function')
    if edge_function != None:
        print(f"Edge ID: {edge_id}, function: {edge_function}")
    else:
        print(f"Edge ID: {edge_id}")

    # 2.1. 遍历 <lane> 子元素
    for lane in edge.findall('lane'):
        lane_id = lane.get('id')
        lane_index = lane.get('index')
        lane_speed = lane.get('speed')
        lane_length = lane.get('length')
        print(f"  Lane ID: {lane_id}, index: {lane_index}, speed: {lane_speed}, length: {lane_length}")

# 3. 遍历 <junction> 元素
for junction in root.findall('junction'):
    junction_id = junction.get('id')
    junction_type = junction.get('type')
    print(f"Junction ID: {junction_id}, type: {junction_type}")

    # junction 可能还有子元素 <request> 等，可以继续读取
    for request in junction.findall('request'):
        index = request.get('index')
        foes = request.get('foes')
        print(f"  Request index: {index}, foes: {foes}")

# 4. 遍历 <tlLogic>（交通灯配时逻辑）
for tl_logic in root.findall('tlLogic'):
    tl_id = tl_logic.get('id')
    tl_type = tl_logic.get('type')
    print(f"Traffic Light ID: {tl_id}, type: {tl_type}")

    # 读取其 <phase> 子元素
    for phase in tl_logic.findall('phase'):
        duration = phase.get('duration')
        state = phase.get('state')
        print(f"  Phase duration: {duration}, state: {state}")

# 假设已经读取出交叉口列表
from collections import defaultdict
intersection_2_edges = defaultdict(lambda : defaultdict(set))
# 每个交叉口要记录进edge和出edge，因为车辆的排队数量指标是从edge角度表示的
# 每个交叉口也需要记录index与内部车道之间的对应关系，因为这和设置交叉口的状态有关。
# 每个内部车道也需要记录进车道和出车道信息，因为内部车道的状态信息与进车道和出车道信息是相关的。
lane_2_edges = defaultdict(lambda : defaultdict(str))
# 5. 遍历 <connection> 元素
for connection in root.findall('connection'):
    from_edge = connection.get('from')
    to_edge = connection.get('to')
    tl = connection.get('tl')
    linkIndex = connection.get('linkIndex')
    via = connection.get('via')
    if tl != None:
        intersection_2_edges[tl]['from'].add(from_edge)
        intersection_2_edges[tl]['to'].add(to_edge)
        lane_2_edges[via]['from'] = from_edge
        lane_2_edges[via]['to'] = to_edge
        intersection_2_edges[tl]['traffic_light'].add((int(linkIndex),via))
        print(f"Connection from={from_edge} to={to_edge}, via={via}")
print(intersection_2_edges['0']['from'])
print(intersection_2_edges['0']['traffic_light'])
print(lane_2_edges)