class Intersection():
    def __init__(self,intersection_id,intersection_2_updownstream,lane_2_shape,lane_2_updownstream):
        super().__init__()
        # 根据id提取上下游车道信息
        self.id = intersection_id
        self.upstream_lanes = list(intersection_2_updownstream[self.id]['from'])
        self.downstream_lanes = list(intersection_2_updownstream[self.id]['to'])
        self.traffic_light = [item[1] for item in sorted(list(intersection_2_updownstream[self.id]['traffic_light']), key=lambda x:x[0])]
        
if __name__ == '__main__':
    True
