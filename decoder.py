#VITTERBI + TOKEN PASSING
#searching for a path with lowest cost in a HMM net
# LANG. MODEL:
# - only unigrams used
# ACUSTIC:
# - as: hmm transition probability
# - bs: output probabilities for each time step

import numpy as np
import time

# lang and acustic model weight params
beta = 10 #lang model weight
pena = np.log10(0.001) #penalization 

def load_data():
    
    As = {}
    Bs = [] #output probabilites of phones in time
    vocab = {}
    phone_order = []
    pw_apri = {}    #{word: prob} -- prior log probability
    data_start = False
    
    with open("acust_data/vocab.txt", encoding='cp1250') as f:
            line = tuple(f.readline().split())
            if(len(line) > 1):
                    vocab[line[1:]] = line[0] 
            while(line):
                line = tuple(f.readline().split())
                if(len(line) > 1):
                    vocab[line[1:]] = line[0]

    with open("acust_data/as.txt", encoding='cp1250') as f:
        line = f.readline().split()
        if(len(line) > 1):
            As[line[0]] = np.array(line[1:]).astype(float)
            phone_order.append(line[0])
        while(line):
            line = f.readline().split()
            if(len(line) > 1):
                As[line[0]] = np.array(line[1:]).astype(float)
                phone_order.append(line[0])
                
    with open("acust_data/bs.txt", encoding='cp1250') as f:
        line = f.readline().split()
        if(len(line) > 1):
                Bs.append(line)
        while(line):
            line = f.readline().split()
            if(len(line) > 1):
                Bs.append(line)
    
    #load language stuff -- unigrams from ARPA
    pw_apri = {}
    data_start = False
    with open("out_arpa.txt") as f:
        line = f.readline().split()
        while(True):
            line = f.readline().split()
            if(len(line) > 0):
                if(line[0] == '\\2-grams:'):
                    data_start = False
                    break
                if(data_start):
                    pw_apri[line[0]] = float(line[1])
                if(line[0] == '\\1-grams:'):
                    data_start = True
    print(len(pw_apri), "unigrams - aprior. prob. loaded")
    
    return As, Bs, vocab, pw_apri, phone_order

def viterbi(As, Bs, vocab, pw_apri, phone_order):
    '''Viterbi alg. + token passing'''

    def get_B(T, fonem):
        idx = phone_order.index(fonem)
        return float(Bs[T,idx])

    def find_optimal_path():
        #calculate network path with lowest cost
        fin_cost = []
        for i, c in enumerate(cost):
            fin_cost.append(c[-1] - np.log10(As[net[i][-1]][1]))
        path = []

        end = np.argmin(fin_cost)
        print("final cost:", min(fin_cost))                

        tok = tokens[end][-1]
        path.append(tok)
        while tok[-1] != None:
            tok = tokens_all[tok[-1]]
            path.append(tok)
            
        sent = []
        sent.append("".join(net[end])) #last
        for el in path:
            w = tuple(el[0])
            if w in vocab:
                sent.append(vocab[w])
            else:
                sent.append(el[0])
        sent.reverse()

        return sent

    def lang_model_prob(w):
        '''get prior log prob. from language model'''
        if w != tuple('#') and w in vocab: #TODO: what if not in pw_apri -> unk?
            word = vocab[w]
            return -beta * pw_apri[word] - pena
        elif w == tuple('#'): #w == '#'
            return 0
        else:
            print("WHAT?")
            return 1

    #net -> [words][phones]
    net = []
    net.append(tuple('#')) #sil
    for w in vocab:
        net.append(w)
            
    #init
    Bs = np.asarray(Bs)
    As = As
    N = len(As)
    T = len(Bs) #measurement count
    stat_costs = {}

    #token passing  - link list -> token := [word, id, previous]
    tokens = [] #tokens for each node of net
    tokens_past = [] #past tokens for each node of net
    tokens_all = [] #list of all tokens

    stat_costs = {}
    cost = [] #phi
    
    #INIT for T=0
    start = time.time()
    for i, w in enumerate(net):
        cost.append([])
        tokens.append([]) #tokens[id], id odpovida i (na zacatku)
        for j in range(0,len(w)):     
            if(j == 0):
                cost[i].append(-np.log10(get_B(0, w[j])) + lang_model_prob(w)) 
            else:
                cost[i].append(float('inf'))
            
            #init token for each state
            tok = ['<s>', len(tokens_all), None]
            tokens[i].append(tok) #init tokens 
            tokens_all.append(tok)

    stat_costs['#'] = [(cost[0][0])]
    stat_costs['a'] = [(cost[1][0])]
    stat_costs['aby'] = [(cost[2][:])]

    copyTime = []

    for t in range(1,T): 
        
        t1 = time.time()
        cost_past = cost
        tokens_past = tokens
        cost = [] #re-init without deep copy
        tokens = []
        for i, w in enumerate(net):
            cost.append([0] * len(cost_past[i]))
            tokens.append(tokens_past[i][:])
        t2 = time.time()
        copyTime.append(t2-t1)
        
        #cena prechodu z koncovych stavu z minula
        phi_last = []
        #nalezeni minima z predchozich -> phi[1] pro prvni stavy
        for i, w in enumerate(net):
            phi_last.append(cost_past[i][-1] - np.log10(As[w[-1]][1]))
        last_w_idx = np.argmin(phi_last)

        for i, w in enumerate(net):
            #first state -- min. of end states form last pass
            stay = cost_past[i][0] - np.log10(As[w[0]][0])
            tran = min(phi_last) + lang_model_prob(w) #lang model here!
            if(stay < tran):
                cost[i][0] = stay - np.log10(get_B(t, w[0]))
                tokens[i][0] = tokens_past[i][0]
            else: #prechod mezi slovy
                cost[i][0] = tran - np.log10(get_B(t, w[0]))
                #new token based on transition
                tok = ["".join(net[last_w_idx]), len(tokens_all), tokens_past[last_w_idx][-1][-2]]
                tokens_all.append(tok)
                tokens[i][0] = tok
            
            #loop over states of current word
            for j in range(1, len(w)):
                stay = cost_past[i][j] - np.log10(As[w[j]])[0]
                tran = cost_past[i][j-1] - np.log10(As[w[j-1]])[1]
                if(stay < tran): #same state
                    cost[i][j] = stay - np.log10(get_B(t, w[j])) 
                    tokens[i][j] = tokens_past[i][j]
                else: #state transition
                    cost[i][j] = tran - np.log10(get_B(t, w[j]))
                    tokens[i][j] = tokens_past[i][j-1]
                    
        stat_costs['#'].append((cost[0][0]))
        stat_costs['a'].append((cost[1][0]))
        stat_costs['aby'].append((cost[2][:]))
            
    print("#:", stat_costs['#'][-1], "a:", stat_costs['a'][-1], "aby:", stat_costs['aby'][-1])
    end = time.time()
    totalTime = end - start

    return find_optimal_path()

if __name__ == "__main__":
    As, Bs, vocab, pw_apri, phone_order = load_data()
    sent = viterbi(As, Bs, vocab, pw_apri, phone_order)
    print(sent)
