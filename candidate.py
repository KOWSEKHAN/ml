def candidate_elimination(data, attributes):
    S = [['?' for _ in range(len(attributes))]] 
    G = [['?' for _ in range(len(attributes))] for _ in range(len(attributes))]    
    for example in data:
        if example[-1] == 'Yes': 
            S = [s for s in S if matches(s, example[:-1])]
            for i in range(len(S)):
                S[i] = generalize(S[i], example[:-1])
        else: 
            G = [g for g in G if not matches(g, example[:-1])]
            for i in range(len(G)):
                G[i] = specialize(G[i], example[:-1])
    return S, G
def matches(hypothesis, example):
    return all(h == '?' or h == e for h, e in zip(hypothesis, example))
def generalize(specific, example):
    return [e if s == '?' else s for s, e in zip(specific, example)]
def specialize(general, example):
    return [e if g == '?' else g for g, e in zip(general, example)]
data = [['Sunny', 'Warm', 'High', 'No'],
        ['Sunny', 'Warm', 'Low', 'Yes'],
        ['Overcast', 'Cool', 'High', 'Yes']]
attributes = ['Outlook', 'Temperature', 'Humidity']
S, G = candidate_elimination(data, attributes)
print("Specific Hypotheses:", S)
print("General Hypotheses:", G)