Data anonymity
- common.py: Common functions and constants
- Apriori.py => Find frequent itemsets, generate rules with support and confidence using Apriori algo
- m3ar.py => K-anonymity M3AR algorithm following the paper at https://www.researchgate.net/publication/224222001
- modified_m3ar.py => K-anonymity Modified M3AR algorithm
- gccg.py => GCCG algorithm http://ijns.jalaxy.com.tw/contents/ijns-v19-n6/ijns-2017-v19-n6-p1062-1071.pdf
- oka.py => OKA algorithm http://cscdb.nku.edu/pais08/papers/p-pais07.pdf
- Clustering_based_K_Anon: Other algorithms
- eval.py: Calculate metrics
- plot.py: Draw/Plot charts


HOW TO RUN
- cd <folder_path>
- Install all requirements
- Run python init.py => Preprocess and init the set of rules
- Go to each algorithm source file, change the necessary paths, then run each by:
    + python modified_m3ar.py
    + python m3ar.py
    + python gccg.py
    + python oka.py
