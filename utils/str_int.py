trans_c_i = {
        'r': 0,
        'g': 1,
        'y': 2,
        'G': 1
    }
trans_i_c = {
    0: 'r',
    1: 'g',
    2: 'y'
}

def get_int(in_char):
    return trans_c_i[in_char]
def get_char(in_int):
    return trans_i_c[in_int]