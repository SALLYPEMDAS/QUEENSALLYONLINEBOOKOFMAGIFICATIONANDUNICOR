

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

%matplotlib inline
```


```python
import re

def TWR(s):
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

    
TWR("WASHINGTON DC")
TWR("APPLE")
TWR("BANANNA")
TWR('                    - WHO ELSE SEEN THE LEPRECHAUN?SAY "YEAH"!')
```

    
    WAS
    HIN
    GTO
    NDC
    
    WASHINGTONDC
    
    
    
    APL
    EXZ
    
    APLE
    
    
    
    BAN
    ANA
    
    BANANA
    
    
    
    WHO
    ELS
    ESE
    NTH
    ELE
    PRE
    CHA
    UNS
    AYE
    AHX
    
    WHOELSESENTHELEPRECHAUNSAYEAH
    
    



```python
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
```

    ['S', 'E']



```python
def lkpz(grpz):
    ret = {}

    for i, _ in enumerate(grpz):
        for __ in _:
            ret[__] = i
            
    return ret
```


```python
ABCLookup()["W"]
```




    6




```python
seqAZ = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
seqZA = seqAZ[::-1]
```


```python
x = gpx(seqAZ, 8)
t = gpx(seqZA, 8)
```


```python
print(x[5])
print(t[5])
```

    ['F', 'N', 'V']
    ['U', 'M', 'E']



```python
print(x[0])
print(t[0])
```

    ['A', 'I', 'Q', 'Y']
    ['Z', 'R', 'J', 'B']



```python
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

    
```


```python
z = codzeifyWord("SALLY")

printCodze(z)
```

    SALLY
    20330
    71661
    00100111101101010
    10001011111101101



```python
X = 100*np.random.rand(6,6)

fig, ax = plt.subplots()
i = ax.imshow(X, cmap=cm.jet, interpolation='nearest')
fig.colorbar(i)

plt.show()
```


![png](output_11_0.png)


TODO

* tower density
* simple dualistic density (word -> dec number -> each digit of dec number goes to binary tocreate a density
* nondualistic density (take the az and the za, and then do a digit from each letter in the word with both z and a to form a two digit decimal that then goes to binary for the density")
