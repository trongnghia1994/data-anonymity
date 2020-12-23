#libraries
from itertools import combinations, chain
from  more_itertools import unique_everseen
import sets
import time
import random
#import data and preprocess
data = open("./dataset/adult.data","r")

#sampling improvement for Apriori
samplingfactor = .6

#input data and each observation as a list within a data list structure
#add _i to data value indicating i'th variable
def clean(dta,samplingfactor):
    datalist = []
    for line in dta:
        linesep = line.split(", ")
        for attribute in linesep:
            newattribute = attribute  + "_" + str(linesep.index(attribute))
            linesep[linesep.index(attribute)] = newattribute
        datalist.append(linesep)

    random.shuffle(datalist) #randomizes data order
    datalist = [datalist[i] for i in range(0,int(round(samplingfactor*len(datalist))))] #return the first sampfactor
    return datalist



#Obtaining C1 - counts of all variable values
def gen_C1(dta):
    count_dict = {}
    for observation in dta:
        for val in observation:

            #if value in observation is not in dictionary, start its count at 1
            #if it is in the dictionary then increment its count by 1

            if val not in count_dict:
                count_dict[val]= 1
            elif val in count_dict:
                count_dict[val] += 1

    return count_dict


#Obtaining L1
def gen_L1(dict,min_supp,samplingfactor):
    L1 = []
    for var in dict:

        #if a value in the count dictionary passes the minimum support threshold
        #add it to list
        if float(dict[var])/float(nobservations) > min_supp*samplingfactor:
            L1.append(var)

    return L1


#print nobservations


#check if k-1 subset of k itemset is in L_k-1
def has_infeq_subsets(k,L,c):
    for i in list(combinations(c,k-1)):
            if i not in L:
                return False
            else:
                return True

#generate C2 from L1
def gen_C2(k,L):
    Ck = list(combinations(L,k)) #all combinations of 2 itemsets
    for c in Ck:
        if has_infeq_subsets(k,L,c):
            Ck.remove(c) #if it has infrequent subsets then remove candidate
        else:
            pass
    return Ck


def gen_Lk(Ck,min_supp,dta,samplingfactor):
    countdict = {}
    for c in Ck:
        countdict[c] = 0 #start every candidate count at 0

    for observation in dta:
        for c in Ck:
            if set(c).issubset(observation):
                countdict[c] += 1 #if candidate is subset of observation, increment count

    Lk = []
    for c in Ck:
        if float(countdict[c])/nobservations > min_supp*samplingfactor:
            Lk.append(c) #if minimum support threshold is passed append to list

    return Lk

#generate candidates for k > 2
#flatten tuples, get all possible combinations, and prune
def gen_Ck(k,L):
    flatten = [item for subtuple in L for item in subtuple] #flattens all candidates
    uniqueflatten = list(unique_everseen(flatten)) #gets out duplicate candidates
    Ck = list(combinations(uniqueflatten,k)) #creates list of all possible combinations of k length
    for c in Ck:
        if has_infeq_subsets(k,L,c):
            Ck.remove(c)
        else:
            pass
    return Ck


#Apriori Algorithm method
def Apriori(k,L,dta,min_supp,samplingfactor):
    l_of_L = [] #to store different Lk's
    while L != []: #while there are frequent itemsets
        if k == 2:
            Ck = gen_C2(k,L) #use specific 2-tuple candidate generator
        else:
            Ck = gen_Ck(k,L) #otherwise use normal one
        L = gen_Lk(Ck,min_supp,cleandata,samplingfactor)
        l_of_L.append(L)
        k += 1

    return l_of_L[0:len(l_of_L)-1] #return all Lk stored that are non-empty


###Test###

time1 = time.time()

cleandata = clean(data,samplingfactor)

#number of observations, used later for min_supp testing
nobservations = len(cleandata)

#getfirst candidate set
C1 = gen_C1(cleandata)

#generate L1
L1 = gen_L1(C1,.75,samplingfactor)
freqsets = Apriori(2,L1,cleandata,.6,samplingfactor)

print("The frequent itemsets are:" + "\n")
for i in freqsets:
    for j in i:
        print(j)

time2 = time.time()

testtime = time2 - time1
print("\n")
print("The runtime for this algorithm(with a sampling factor of .6 and a min_supp = .6) is" + " " + str(testtime) + " seconds.")
