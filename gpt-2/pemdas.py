#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import random
import tensorflow as tf
import model, sample, encoder
from datetime import datetime
import time
import calendar


# coding: utf-8

# In[17]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re


#get_ipython().magic('matplotlib inline')


# In[18]:


def squished(s):
    s = s.upper()
    quished = ""
    prev = ""
    def isIgnoredChar(_):
        patterns = '^[a-zA-Z0-9_]*$'
        if re.search(patterns,  _):
            return False
        return True
    for _ in s:
        if isIgnoredChar(_) != True and _ != prev:
            quished = quished + _
            prev = _
            
    return quished

print(squished("bananna"))


# In[19]:


def U2V(s):
    ret = ""
    s = s.upper()
    for _ in s:
        if _ == "U":
            ret = ret + "V"
            continue
        if _ == "W":
            ret = ret + "V"
            continue
        ret = ret + _
        
    return ret

print(U2V("rainboW"))


# In[20]:


def PRINTWR(s):
    s = s.upper()
    squish = ""
    prev = ""

    def isIgnoredChar(_):
        patterns = '^[a-zA-Z0-9_]*$'
        if re.search(patterns,  _):
                return False
        
        return True
    
    for _ in s:
        if isIgnoredChar(_) != True and _ != prev:
            squish = squish + _
            prev = _
    i = 0
    level = ""
    for _ in squish:
        #TODO CONFIG TOWER WIDTH
        if i % 3 == 0:
            i = 0
            print(level)
            level = ""
        i = i + 1
            
        level = level + _
            
            #WE CAN MAKE THIS MORE DYNAMIC LATER FOR RANDO SORTA PADEND
    if len(level) == 1:
        print(level + "XZ")
    if len(level) == 2:
        print(level + "X")
    if len(level) == 3:
        print(level)
    print("\n" + squish + "\n\n")

# In[21]:


def MTRXTWER(s):
    s = s.upper()
    squish = ""
    prev = ""

    def isIgnoredChar(_):
        patterns = '^[a-zA-Z0-9_]*$'
        if re.search(patterns,  _):
                return False
        
        return True
    
    for _ in s:
        if isIgnoredChar(_) != True and _ != prev:
            squish = squish + _
            prev = _
    i = 0
    for _ in squish:
        print(_)
    print("")
    
MTRXTWER("WASHINGTON DC")
MTRXTWER("APPLE")
MTRXTWER("BANANNA")
MTRXTWER('                    - WHO ELSE SEEN THE LEPRECHAUN?SAY "YEAH"!')
print("test")



# In[22]:


def gpx(seq, i):
    grpz = []
    j = 0
    while (j < i):
        grpz.append([])
        j = j + 1

    for j, _ in enumerate(seq):
        k = j
        while k >= i:
            k = k - i
        grpz[k].append(_)

    return grpz

print(gpx("SEED", 2)[0])


# In[23]:


def lkpz(grpz):
    ret = {}

    for i, _ in enumerate(grpz):
        for __ in _:
            ret[__] = i
            
    return ret


# In[24]:


seqAZ = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
seqZA = seqAZ[::-1]
seqVVVAZ = "ABCDEFGHIJKLMNOPQRSTVXYZ"
seqVVVZA = seqVVVAZ[::-1]


# In[25]:


x = gpx(seqAZ, 8)
t = gpx(seqZA, 8)
VA = gpx(seqVVVAZ, 3)
VZ = gpx(seqVVVZA, 3)


# In[26]:


print(x[5])
print(t[5])


# In[27]:


print(x[0])
print(t[0])


# In[28]:


def wrdToDec(s, lkp):
    s = s.upper()
    fart = ""
    for _ in s:
        fart = fart + str(lkp[_])
        
    return int(fart)

def defaultGrpLen():
    return 8


def codzeifyWord(s, grpLen = defaultGrpLen()):
    s = s.upper()
    azLkp = lkpz(gpx(seqAZ, grpLen))
    zaLkp = lkpz(gpx(seqZA, grpLen))
    
    ret = { "WRD": s }
    
    azDec = wrdToDec(s, azLkp)
    ret["AZ"] = {"DEC": azDec, "BIN": str(bin(azDec))[2:]}
    
    zaDec = wrdToDec(s, zaLkp)
    ret["ZA"] = {"DEC": zaDec, "BIN": str(bin(zaDec))[2:]}
    
    return ret
    
def printCodze(z):
    print(z["WRD"])
    
    #TODO(QS): PAD FRONT FOR DEC?
    print(z["AZ"]["DEC"])
    print(z["ZA"]["DEC"])
    
    ab = z["AZ"]["BIN"]
    zb = z["ZA"]["BIN"]
    
    maxLen = len(ab)
    if len(zb) > maxLen:
        maxLen = len(zb)
        
    def getPad(padMe, padTo):
        pad = ""
        if len(padMe) < padTo:
            ws = padTo - len(padMe)
            i = 0
            while (i < ws):
                pad = pad + "0"
                i = i + 1
    
        return pad
    
    abb = getPad(ab, maxLen) + ab
    zbb = getPad(zb, maxLen) + zb
    
    print(abb)
    print(zbb)

    


# In[29]:


z = codzeifyWord("SALLY")

printCodze(z)


# In[30]:


X = 100*np.random.rand(6,6)

fig, ax = plt.subplots()
i = ax.imshow(X, cmap=cm.jet, interpolation='nearest')
fig.colorbar(i)

plt.show()

def trxtwr(s, i = 1):
    s = s.upper();
    mid = []    
    multi = []
    j = 0
    for D in squished(s):
        if i == 1:
            mid.append([D])
            continue
        j = j + 1
        multi.append(D)
        if len(multi) == i:
            mid.append(multi)
            print(multi)
            multi = []
            j = 0
    print("multi")
    print(multi)
    if len(multi) > 0:
        while len(multi) < i:
            multi.append("X")
            
        mid.append(multi)
    
    return mid

def trxtwr(s, i = 1):
    s = s.upper();
    mid = []    
    multi = []
    j = 0
    for D in squished(s):
        if i == 1:
            mid.append([D])
            continue
        j = j + 1
        multi.append(D)
        if len(multi) == i:
            mid.append(multi)
            print(multi)
            multi = []
            j = 0
    print("multi")
    print(multi)
    if len(multi) > 0:
        while len(multi) < i:
            multi.append("X")
            
        mid.append(multi)
    
    return mid

def trxtwrstr(twr):
    r = ""
    for l in twr:
        for ll in l:
            r = r + ll
        r = r + "\n"
    return r

def str2Dec(s, seq = seqAZ, grpLen = defaultGrpLen()):
    #azLkp = lkpz(gpx(seqAZ, grpLen))
    lkp = lkpz(gpx(seq, grpLen))
        
    return int(wrdToDec(s, lkp))

def dec2Binary(i):
    return str(bin(i))[2:]
        

def TD(twr):
    binLayers = []
    decLayers = []
    
    # after dec -> binary: treat binary representation as a decimal number and convert to binary again (latent level 0)
    # 27 -> 11101 -> 10101101011101
    embeded0 = []
    embeded01 = []
     
    mtxE0 = []

    for f in twr:
        for _ in f:
            d = str2Dec(_)
            decLayers.append(d)
    print("decimal")    
    print(decLayers)
    for _ in decLayers:
        print(str(_))
    for _ in decLayers:
        b = dec2Binary(_)
        binLayers.append(b)
    print("\nbinary")
    print(binLayers);
    for _ in binLayers:
        print(_)
    print("\nembeded0")
    for _ in(binLayers):
        embeded0.append(dec2Binary(int(_)))
    print(embeded0)
    for _ in embeded0:
        print(_)
    print("\nembeded01")
    for _ in embeded0:
        embeded01.append(dec2Binary(int(_)))
    print(embeded01)
    for _ in embeded01:
        row = []
        for __ in _:
            row.append(__)
        mtxE0.append(row)
        
    maxColCount = 0
    for _ in mtxE0:
        count = 0
        for __ in _:
            count = count +1
        if maxColCount < count:
            maxColCount = count
    for _ in mtxE0:
        toPad = maxColCount - len(_)
        while(toPad > 0):
            _.insert(0, "0")
            toPad = toPad - 1

    for _ in mtxE0:
        for __ in _:
            __ = float(__)
    
    mtxE0_array = np.array(mtxE0)
    
    fig, ax = plt.subplots()
    i = ax.imshow(mtxE0_array, cmap=cm.jet, interpolation='nearest')
    fig.colorbar(i)

    plt.show()
    
    print(mtxE0)
            


# In[63]:


print(dec2Binary(11101))


# In[64]:


VVV = "ABCDEFGHIJKLMNOPQRSTVXYZ"


# In[65]:


gVVV = gpx(VVV, 3)
gVVVy = gpx(VVV, 8)


# In[69]:


print(gVVV)
print(gVVVy)


# In[67]:


print(trxtwrstr(trxtwr("dinosaur", 3)))

magicVVVLookup = {
    "A": "20 07",
    "B": "10 17",
    "C": "00 27",
    "D": "21 06",
    "E": "11 16",
    "F": "01 26",
    "G": "22 05",
    "H": "12 15",
    "I": "02 25",
    "J": "23 04",
    "K": "13 14",
    "L": "03 24",
    "M": "24 03",
    "N": "14 13",
    "O": "04 23",
    "P": "25 02",
    "Q": "15 12",
    "R": "05 22",
    "S": "26 01",
    "T": "16 11",
    "V": "06 21",
    "X": "27 00",
    "Y": "17 10",
    "Z": "07 20"
}
def magicVVVDecTower(twr):
    ret = []
    for level in twr:
        ourLevel = []
        for char in level:
            ourLevel.append(magicVVVLookup[char])
        ret.append(ourLevel)
    return ret



def fortminute(oldtime = datetime.utcnow().__format__('%Y%m%d%H%M%S')):
    yyyymmdd = oldtime[:8]
    hhmmss = oldtime[8:]
    hr = hhmmss[:2]
    min = hhmmss[2:4]
    sec = hhmmss[4:]
    hr = int(hr)
    min  = int(min)
    sec = int(sec)
    seconds = (hr * (60 * 60)) + (min * 60) + sec

    return int((seconds/864)) - 1

def _seed(time = datetime.utcnow().__format__('%Y%m%d%H%M%S')):
    return time[:8] + str(fortminute(time))

def interact_model(
    model_name='124M',
    seed=int(_seed()),
    nsamples=8,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=50,
    top_p=0.93,
    models_dir='models',
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    print(seed)
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = random.randint(9, 93)
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )
        
        output2 = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        output3 = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

 #       print('seed: ' + str(seed))
 #       

        filename = str(seed) + str(calendar.timegm(time.gmtime())) + ".txt"
      
        o = open("../TXT/TXZ/GEN/" + filename, 'w')
        
        k = 0
        i = 0
        count = 0
        
        while True:
            raw_text = input("> ")
            while not raw_text:
                raw_text = input("> ")
            context_tokens = enc.encode(raw_text)
            generated = 0
            o.write("-----\n" + raw_text + "\n-----\n\n")

            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                generated2 = 0
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    context_tokens2 = enc.encode(text)
                    o.write(raw_text + " " + text + "\n----")

                    print(raw_text + " " + text + "\n----")

                    for _ in range(nsamples // batch_size): 
                        out2 = sess.run(output2, feed_dict={
                            context: [context_tokens2 for _ in range(batch_size)]
                        })[:, len(context_tokens2):]
                        for j in range(batch_size):
                            generated2 += 1

                            text2 = enc.decode(out2[i])

                            o.write(text + " " + text2 + "\n\n")
                            o.write("---\n\n")
                            o.flush()
            o.write("-------\n\n\n\n")
            o.flush()

            raw_text = U2V(squished(raw_text))
            context_tokens = enc.encode(raw_text)
            generated = 0
            o.write("-----\n" + raw_text + "\n-----\n\n")

            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                generated2 = 0
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    context_tokens2 = enc.encode(text)
                    o.write(raw_text + " " + text + "\n----")

                    print(raw_text + " " + text + "\n----")

                    for _ in range(nsamples // batch_size): 
                        out2 = sess.run(output2, feed_dict={
                            context: [context_tokens2 for _ in range(batch_size)]
                        })[:, len(context_tokens2):]
                        for j in range(batch_size):
                            generated2 += 1

                            text2 = enc.decode(out2[i])

                            o.write(text + " " + text2 + "\n\n")
                            o.write("---\n\n")
                            o.flush()
            o.write("-------\n\n\n\n")
            o.flush()
            
            twr = trxtwr(raw_text, 3)
            twrstr = trxtwrstr(trxtwr(raw_text, 3))
            raw_text = twrstr
            context_tokens = enc.encode(raw_text)
            generated = 0
            o.write("-----\n" + raw_text + "\n-----\n\n")

            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                generated2 = 0
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    context_tokens2 = enc.encode(text)
                    print(raw_text + " " + text + "\n----")
                    o.write(raw_text + " " + text + "\n----")

                    for _ in range(nsamples // batch_size): 
                        out2 = sess.run(output2, feed_dict={
                            context: [context_tokens2 for _ in range(batch_size)]
                        })[:, len(context_tokens2):]
                        for j in range(batch_size):
                            generated2 += 1

                            text2 = enc.decode(out2[i])

                            o.write(text + " " + text2 + "\n\n")
                            o.write("---\n\n")
                            o.flush()
            
            twr = trxtwr(raw_text, 3)
            twrstr = trxtwrstr(trxtwr(raw_text, 3))

            mtwr = magicVVVDecTower(twr)
            mtwrstr = trxtwrstr(mtwr)
            raw_text = twrstr + "\n" + mtwrstr
            context_tokens = enc.encode(raw_text)
            generated = 0
            o.write("-----\n" + raw_text + "\n-----\n\n")


            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                generated2 = 0
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    context_tokens2 = enc.encode(text)
                    print(raw_text + " " + text + "\n----")
                    o.write(raw_text + " " + text + "\n----")

                    for _ in range(nsamples // batch_size): 
                        out2 = sess.run(output2, feed_dict={
                            context: [context_tokens2 for _ in range(batch_size)]
                        })[:, len(context_tokens2):]
                        for j in range(batch_size):
                            generated2 += 1

                            text2 = enc.decode(out2[i])

                            o.write(text + " " + text2 + "\n\n")
                            o.write("---\n\n")
                            o.flush()
            o.write("-------\n\n\n\n")
            o.flush()
            
            
            print("done")
#        with open("kjv.txt") as f:
#            for line in f:
#                print(line)
#                if (len(line) == 1):
#                    continue
#                prompt = "> " + filterline(line) + "\n\n"
#                context_tokens = enc.encode(prompt)
#                o = sess.run(output, feed_dict={
#                    context: [context_tokens for _ in range(batch_size)]
#                })[:, len(context_tokens):]
#                for i in range(batch_size):


        o.close()



if __name__ == '__main__':
    fire.Fire(interact_model)


