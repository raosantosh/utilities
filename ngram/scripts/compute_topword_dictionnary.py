import sys
import numpy as np
from sets import Set


modelname = sys.argv[1]

with open(modelname) as f:
    lines=f.readlines()


triplets={}
queries={}

triplets_indexes={}
queries_indexes={}


query_index=0
triplet_index=0
for line in lines:
    query=line.strip().split(',')[0]
    if query not in queries_indexes:
        queries_indexes[query]=query_index
        queries[query]=1
    tokens=query.rstrip().split(' ')
    for token in tokens:
        token="#"+token+"#"
        for i in range(len(token) - 3 +1):
            if(token[i:i + 3] not in triplets):
                triplets[token[i:i + 3]]=[]
                triplets_indexes[token[i:i + 3]]=triplet_index
                triplet_index+=1
            triplets[token[i:i + 3]].append(query)

    query_index+=1


#print(queries_indexes)

nGramQuery = np.zeros((triplet_index,query_index))
for triplet in triplets:
    for query in triplets[triplet]:
        nGramQuery[triplets_indexes[triplet],queries_indexes[query]]=1.0

#nGramQuerymatrix = np.matrix(nGramQuery)
QQ = nGramQuery.transpose().dot(nGramQuery)


print(QQ.shape)

for query_a in queries:
    for query_b in queries:
        v = QQ[queries_indexes[query_a]][queries_indexes[query_b]]
        if v>7:
            if query_a!=query_b:
             print("Q1:"+query_a+' Q2:'+query_b+' #:'+str(v))

        #print(query_a+' '+query_b)#+str(QQ[queries_indexes[query_a]][queries_indexes[query_b]]))

print(QQ)