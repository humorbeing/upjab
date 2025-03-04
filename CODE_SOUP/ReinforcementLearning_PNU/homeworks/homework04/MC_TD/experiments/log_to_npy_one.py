
td_path = 'logs/td_fix03.log'
# mc_every_path = 'logs/everyvisit.log'
# mc_first_path = 'logs/firstvisit.log'




with open(td_path, 'r') as f:
    td_log = f.readlines()

# with open(mc_every_path, 'r') as f:
#     every_log = f.readlines()

# with open(mc_first_path, 'r') as f:
#     first_log = f.readlines()


tl = td_log[1:]
# el = every_log[1:]
# fl = first_log[1:]

def get_n(s):
    _, r, _, e, _, score, _ = s.split('|')
    r = int(r)
    e = int(e)
    score = float(score)
    score = round(score, 5)
    return e, score

tmatrix = []
# ematrix = []
# fmatrix = []
# for r in range(1000):
for r in range(999):
    t_list = []
    # e_list = []
    # f_list = []
    for episode in range(1000):
        index = r * 1000 + episode
        tt = tl[index]
        # ee = el[index]
        # ff = fl[index]
        tepisode, tscore = get_n(tt)
        assert tepisode == episode
        # eepisode, escore = get_n(ee)
        # assert eepisode == episode
        # fepisode, fscore = get_n(ff)
        # assert fepisode == episode

        t_list.append(tscore)
        # e_list.append(escore)
        # f_list.append(fscore)
        print(f'{r}-{episode}')
    
    tmatrix.append(t_list)
    # ematrix.append(e_list)
    # fmatrix.append(f_list)


import numpy as np

ttt = np.array(tmatrix)
# eee = np.array(ematrix)
# fff = np.array(fmatrix)

np.save('logs/td03.npy', ttt)
# np.save('logs/every.npy', eee)
# np.save('logs/first.npy', fff)

print("end")