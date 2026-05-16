# Calculate attention:
scores = query @ keys.T  # Dot product = similarity

weights = softmax(scores)  # Normalize to probabilities

output = weights @ values  # Weighted average
```


Complete Formula
The famous attention formula:

Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V