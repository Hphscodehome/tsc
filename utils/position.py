def judge_cross(line_A,line_B):
    line_A_start = line_A[0]
    line_A_end = line_A[1]
    line_B_start = line_B[0]
    line_B_end = line_B[1]
    return segments_intersect(line_A_start, line_A_end, line_B_start, line_B_end)

# 判断A,B,C的方向
def direction(A, B, C):
    return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

def on_segment(A, B, C):
    return (min(A[0], B[0]) <= C[0] <= max(A[0], B[0]) and
            min(A[1], B[1]) <= C[1] <= max(A[1], B[1]))

def segments_intersect(A, B, C, D):
    d1 = direction(A, B, C)
    d2 = direction(A, B, D)
    d3 = direction(C, D, A)
    d4 = direction(C, D, B)

    # 检查一般情况
    if d1 * d2 < 0 and d3 * d4 < 0:
        #交叉
        return True

    # 检查特殊情况
    if d1 == 0 and on_segment(A, B, C):
        return True
    if d2 == 0 and on_segment(A, B, D):
        return True
    if d3 == 0 and on_segment(C, D, A):
        return True
    if d4 == 0 and on_segment(C, D, B):
        return True

    return False

if __name__ == '__main__':
    line_A = [(0,0),(2,2)]
    line_B = [(2,0),(0,2)]
    print(judge_cross(line_A,line_B))

