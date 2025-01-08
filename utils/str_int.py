def get_int(in_char):
    trans = {
        'r': 0,
        'g': 1,
        'G': 2,
        'y': 3
    }
    return trans[in_char]