

def self_print(a_str):
    prefix = '#---- '
    suffix = ''.join([i for i in reversed(prefix)])
    print(prefix + a_str + suffix)
