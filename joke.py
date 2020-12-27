import pickle

# Read list of rules from binary pickled file
with open('initial_rules.log', 'rb') as f:
    data = pickle.load(f)
    print(data)
