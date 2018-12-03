from collections import deque

def read_data(filename):
    lst = []
    with open(filename, 'r') as f:
        for line in f:
            lst.append(float(line))
    return lst

def write_data(filename, lst):
    with open(filename, 'w') as f:
        for val in lst:
            f.write("{:.6f}\n".format(val))

def full_avg(in_lst):
    out_lst = []
    curr_sum = 0.0
    curr_count = 0
    for val in in_lst:
        curr_sum += val
        curr_count += 1
        out_lst.append(curr_sum / curr_count)
    return out_lst

def dyn_avg(in_lst, last_n):
    out_lst = []
    q = deque(maxlen=last_n)
    for val in in_lst:
        q.append(val)
        out_lst.append(sum(q) / len(q))
    return out_lst

def dyn_mult(sz, i, last_n):
    mult = 1.0
    dist = sz - i
    if dist > last_n:
        mult = 1.0 - last_n / (dist + 1)
    return mult

def dyn_mult1(sz, i, last_n):
    dist = sz - i
    if dist > 6*last_n:
        return dyn_mult(sz, i, last_n)
    elif dist > 5*last_n:
        return 0.1
    elif dist > 4*last_n:
        return 0.2
    elif dist > 3*last_n:
        return 0.4
    elif dist > 2*last_n:
        return 0.6
    elif dist > last_n:
        return 0.8
    else:
        return 1.0

def dyn_sum(in_lst, last_n):
    curr_sum = 0.0
    sz = len(in_lst)
    r = range(sz)
    for i in r:
        curr_sum += in_lst[i] * dyn_mult1(sz, i, last_n)
    return curr_sum


def full_dyn_avg(in_lst, last_n):
    out_lst = []
    curr_count = 0
    r = range(len(in_lst))
    for i in r:
        curr_count += 1
        out_lst.append(dyn_sum(in_lst[0:i+1], last_n) / curr_count)
    return out_lst

#lst = [2, 3, 2, 0, 3, 6, 7, 5, 4, 2, 0]
#print(full_dyn_avg(lst, 3))

wd = ""
filename = "6.txt"

lst = read_data(wd + filename)
out_lst = full_avg(lst)
write_data(wd + filename + "_avg.txt", out_lst)