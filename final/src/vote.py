import sys
from collections import Counter

def get_ans(filename):
    with open(filename, 'r') as f:
        next(f)
        l = f.read().splitlines()
        l = list(map(lambda x: x[-1], l))
    return l

def out_res(filename, ans):
    with open(filename, 'w') as f:
        print('id,Ans', file=f)
        for i, s in enumerate(ans):
            print(f'{i+1},{s}', file=f)

args = sys.argv
l = [get_ans(s) for s in args[2:]]
ans = [Counter(t).most_common()[0][0] for t in zip(*l)]

out_res(args[1], ans)
