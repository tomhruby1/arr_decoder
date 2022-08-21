import numpy as np

def load_vocab(path, encoding='utf-8'):
    vocab = set()   #hashable
    with open(path, encoding=encoding) as f:
        line = f.readline()
        while(line):
            vocab.add(line[0:-1]) # no \n
            line = f.readline()
    print(str(len(vocab)) + ' words loaded')
    return vocab

def load_train_sents(path, encoding='utf-8'): 
    train_sents = []
    unigrams = {}
    uni_prob = {}
    with open('train.txt', encoding='cp1250') as fp:
        line = fp.readline()
        while(line):
            sent = line.split()
            train_sent = []
            train_sent.append("<s>")
            for i in range(len(sent)):
                if(sent[i] in vocab):
                    w = sent[i].strip()
                else:
                    w = "<unk>"  
                
                if w in unigrams:
                    unigrams[w] += 1
                else:
                    unigrams[w] = 1
                
                train_sent.append(w)
            train_sent.append("</s>")
            
            if("</s>" in unigrams):
                unigrams["</s>"] += 1
            else:
                unigrams["</s>"] = 1
            train_sents.append(train_sent)
            line = fp.readline()
    for x in unigrams:
        uni_prob[x] = np.log10(unigrams[x] / len(unigrams))
    print(len(unigrams), "unigrams")

    return train_sents, unigrams, uni_prob
    
def get_bigrams(train_sents):
    bigrams = {}
    for sent in train_sents:
        for i in range(len(sent) - 1):
            bigram = (sent[i], sent[i+1])
            if bigram in bigrams:
                bigrams[bigram] += 1
            else:
                bigrams[bigram] = 1
    print(len(bigrams), "bigrams")
    return bigrams
    
def get_trigrams(train_sents):
    trigrams = {}
    for sent in train_sents:
        for i in range(len(sent) - 2):
            trigram = (sent[i], sent[i+1], sent[i+2])
            if trigram in trigrams:
                trigrams[trigram] += 1
            else:
                trigrams[trigram] = 1
    #prorezani
    trig_fin = {}
    for t in trigrams:
        if trigrams[t] > 1:
            trig_fin[t] = trigrams[t]
            
    print(len(trig_fin), "trigrams")
    return trig_fin

def bigram_prob(bigrams):
    big_prob = {}
    for x in bigrams:
        N_tot = 0
        for h in bigrams:
            if(x[0] == h[0]):
                N_tot += bigrams[h]
        big_prob[x] = np.log10(bigrams[x] / N_tot)
    return big_prob     

def n_gram_prob(ngrams, bigrams, unigrams):
    for k in ngrams:
        if not type(k) is tuple:
            print("Error - ngram of type set expected!")
            return 
        else: break     
    prob = {}
    for x in ngrams:
        N_tot = 0 #tot. count
        if(len(x) == 2):
            N_tot = unigrams[x[1]] #hist
        elif(len(x) == 3):
            N_tot = bigrams[x[1:]]  #hist, which is saved in bigrams
        prob[x] = np.log10(ngrams[x] / N_tot)

    return  prob

def save_ARPA(uni, bi, tri, file="out_arpa.txt"):
    with open(file, 'w+') as f:
        f.write("\data\ \n")
        f.write("ngram 1="+str(len(uni))+"\n")
        f.write("ngram 2="+str(len(bi))+"\n")
        f.write("ngram 3="+str(len(tri))+"\n")
        f.write("\n\\1-grams: \n")
        for ngram, val in uni.items():
            f.write(ngram+" "+str(val)+"\n")
        f.write("\n\\2-grams: \n")
        for ngram, val in bi.items():
            f.write("".join([str(x)+" " for x in ngram]) +" "+str(val)+"\n")
        f.write("\n\\3-grams: \n")
        for ngram, val in tri.items():
            f.write("".join([str(x)+" " for x in ngram]) +" "+str(val)+"\n")
        f.write("\n\\end\\")

if __name__ == "__main__":
    vocab = load_vocab('cestina.txt', encoding='cp1250')
    
    #n-gram:= dict{'(n-gram)':count}
    train_sents, unigrams, uni_prob = load_train_sents('train.txt', encoding='cp1250')

    bigrams = get_bigrams(train_sents)
    trigrams = get_trigrams(train_sents)

    bi_prob = bigram_prob(bigrams)
    tri_prob = n_gram_prob(trigrams, bigrams, unigrams)
    print("P('<s>', '<unk>') =", bi_prob[('<s>', '<unk>')])
    print("P('<s>', 'valná', 'hromada') =", tri_prob[('<s>', 'valná', 'hromada')]) 
    save_ARPA(uni_prob, bi_prob, tri_prob)
