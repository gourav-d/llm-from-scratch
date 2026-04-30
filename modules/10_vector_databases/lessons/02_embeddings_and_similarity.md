# Lesson 02: Embeddings and Similarity

## Learning Objectives

By the end of this lesson, you will be able to:
1. Explain what an embedding model does (text in, vector out)
2. Calculate three types of similarity: Euclidean, Dot Product, Cosine
3. Explain WHY cosine similarity is usually the best choice for text
4. Understand why vectors with more dimensions carry more information

---

## GLOSSARY

```
Embedding Model:
  A neural network that converts text (or images) into a vector of numbers.
  Input:  "The dog chased the cat"
  Output: [0.23, -0.14, 0.88, 0.52, ...] (hundreds of numbers)
  The model is trained so that SIMILAR text produces SIMILAR vectors.

Embedding Vector:
  The output of an embedding model. A list of numbers representing "meaning."
  Also called: embedding, representation, latent vector.

Euclidean Distance:
  The straight-line distance between two points in space.
  For 2D: sqrt( (x2-x1)^2 + (y2-y1)^2 )  <- Pythagorean theorem
  For N dimensions: sqrt( sum of (a_i - b_i)^2 )
  Range: 0 (identical) to infinity (very different)

Dot Product:
  Multiply matching elements, then add them all together.
  [1, 2, 3] dot [4, 5, 6] = (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32
  In C#: v1.Zip(v2, (a, b) => a * b).Sum()
  Range: -infinity to +infinity

Magnitude (Norm):
  The "length" of a vector from the origin (0,0,...) to the vector tip.
  For 2D: sqrt(x^2 + y^2)
  For N dimensions: sqrt( sum of x_i^2 )
  Always positive. Used to "normalize" vectors.

Normalized Vector:
  A vector divided by its own magnitude.
  Result always has magnitude = 1.0 (it "points" in a direction but has no length).
  Formula: normalized = vector / magnitude(vector)

Cosine Similarity:
  The cosine of the angle between two vectors.
  Formula: (A dot B) / (|A| * |B|)
  Range: -1 to 1 (for text embeddings: usually 0 to 1)
  1.0  = identical direction (very similar)
  0.0  = perpendicular (unrelated)
  -1.0 = opposite directions (opposites)
  WHY: Cosine is NOT affected by vector magnitude -- only direction matters.
       A long vector and a short vector pointing the SAME way have cosine = 1.0.
       This makes it great for comparing text of different lengths.

Nearest Neighbor:
  The vector in your database that is MOST SIMILAR to the query vector.
  "Find the nearest neighbor to my query" = "find the most similar document."

k-NN (k Nearest Neighbors):
  Find the k most similar vectors (not just the single best one).
  "Give me the 5 most similar documents" = 5-NN search.
```

---

## Part 1: How an Embedding Model Works

An embedding model is a neural network (like the transformers from Module 04)
that has been trained to convert text into numbers.

```
Input text:  "The dog chased the cat"
                    |
                    v
         Embedding Model (neural network)
         - Tokenizes the text
         - Passes through transformer layers
         - Takes the final hidden state as the vector
                    |
                    v
Output vector: [0.23, -0.14, 0.88, 0.52, 0.11, -0.33, ...]
               (could be 384, 768, or 1536 numbers)
```

The model is trained on billions of text examples so that sentences with
similar MEANING end up with SIMILAR vectors.

### Key Property: Semantic Clustering

```
"The dog ran fast."           -> [0.80, 0.12, 0.65, ...]
"The puppy sprinted quickly." -> [0.78, 0.14, 0.63, ...]  <- very close!
"I bought a laptop."          -> [0.05, 0.92, 0.11, ...]  <- very different
```

This happens because the model learned:
  - "dog" and "puppy" appear in similar contexts in training data
  - "ran" and "sprinted" appear in similar contexts
  - "laptop" appears in very different contexts

---

## Part 2: Three Ways to Measure Similarity

### Method 1: Euclidean Distance

This is the straight-line distance -- the same as measuring distance on a map.

For two 2D vectors A = [1, 3] and B = [4, 7]:

```
distance = sqrt( (4-1)^2 + (7-3)^2 )
         = sqrt( 9 + 16 )
         = sqrt(25)
         = 5.0
```

Visualized:

```
         B (4, 7)
         |
       4 |  (the distance is the hypotenuse)
         |
         A (1, 3)
    ---- 3 ----
```

Higher distance = less similar.
Lower distance = more similar.
Distance of 0 = identical vectors.

For N-dimensional vectors (e.g. 768 dimensions):
  distance = sqrt( sum of (A[i] - B[i])^2 for all i )

Limitation: Euclidean distance is affected by vector LENGTH (magnitude).
A very long vector and a very short vector in the SAME direction will have
a large Euclidean distance -- even though they mean the same thing.
This is a problem for text embeddings of different-length documents.

---

### Method 2: Dot Product

Multiply each pair of matching elements, then sum the results.

For A = [1, 2, 3] and B = [4, 5, 6]:

```
dot product = (1 * 4) + (2 * 5) + (3 * 6)
            = 4 + 10 + 18
            = 32
```

Higher dot product = more similar (when vectors are normalized).
But like Euclidean, it is affected by vector length.

The dot product is FASTEST to compute -- just multiply and add.
When vectors are already normalized (magnitude = 1), dot product == cosine similarity.

---

### Method 3: Cosine Similarity (Most Common for Text)

Cosine similarity measures the ANGLE between two vectors.
It does NOT care about how long the vectors are -- only their direction.

```
cosine_similarity(A, B) = (A dot B) / (|A| * |B|)
```

Where |A| means the magnitude (length) of vector A.

Example with A = [1, 0] and B = [2, 0]:

```
dot product = (1*2) + (0*0) = 2
|A| = sqrt(1^2 + 0^2) = 1.0
|B| = sqrt(2^2 + 0^2) = 2.0
cosine = 2 / (1.0 * 2.0) = 2 / 2 = 1.0
```

Result: 1.0 -- perfectly similar!
Even though B is twice as long as A, they point in the same direction.

Example with A = [1, 0] and B = [0, 1]:

```
dot product = (1*0) + (0*1) = 0
cosine = 0 / (1.0 * 1.0) = 0.0
```

Result: 0.0 -- completely unrelated (perpendicular directions).

---

## Part 3: Cosine Similarity Step by Step

Let us calculate cosine similarity for two short text vectors manually.

Imagine we have tiny 3-dimensional embeddings (just for illustration):

```
"cat" vector:    A = [0.8, 0.2, 0.1]
"kitten" vector: B = [0.7, 0.3, 0.1]
"laptop" vector: C = [0.1, 0.1, 0.9]
```

### Step 1: Calculate dot products

```
A dot B = (0.8 * 0.7) + (0.2 * 0.3) + (0.1 * 0.1)
        = 0.56 + 0.06 + 0.01
        = 0.63

A dot C = (0.8 * 0.1) + (0.2 * 0.1) + (0.1 * 0.9)
        = 0.08 + 0.02 + 0.09
        = 0.19
```

### Step 2: Calculate magnitudes

```
|A| = sqrt(0.8^2 + 0.2^2 + 0.1^2) = sqrt(0.64 + 0.04 + 0.01) = sqrt(0.69) = 0.831

|B| = sqrt(0.7^2 + 0.3^2 + 0.1^2) = sqrt(0.49 + 0.09 + 0.01) = sqrt(0.59) = 0.768

|C| = sqrt(0.1^2 + 0.1^2 + 0.9^2) = sqrt(0.01 + 0.01 + 0.81) = sqrt(0.83) = 0.911
```

### Step 3: Calculate cosine similarity

```
cosine(A, B) = 0.63 / (0.831 * 0.768)
             = 0.63 / 0.638
             = 0.987   <- very similar! (cat and kitten are related)

cosine(A, C) = 0.19 / (0.831 * 0.911)
             = 0.19 / 0.757
             = 0.251   <- not very similar (cat and laptop are unrelated)
```

Result: "cat" and "kitten" have similarity 0.987 (nearly identical).
        "cat" and "laptop" have similarity 0.251 (unrelated).

---

## Part 4: Which Similarity Measure to Use?

```
Measure           | Formula                   | Best For
------------------+---------------------------+---------------------------------
Euclidean         | sqrt(sum of (A-B)^2)      | Image similarity, small datasets
Dot Product       | sum(A * B)                | When vectors are pre-normalized
Cosine Similarity | dot(A,B) / (|A| * |B|)   | Text search (most common choice)
```

For text:
  - Use COSINE SIMILARITY in almost all cases
  - It is not affected by vector length
  - A long document and a short document on the same topic will be similar
  - ChromaDB defaults to cosine similarity for this reason

For images:
  - Euclidean distance often works well
  - Image embeddings have similar magnitudes to each other

---

## Part 5: Dimensions and Information Capacity

More dimensions = more information the vector can encode.

```
3-dimensional vector:   [0.3, 0.7, 0.1]
  Can represent roughly a 3D sphere of "meaning space"

128-dimensional vector: [0.3, 0.7, 0.1, -0.2, 0.8, ...]
  Can represent much more nuanced differences in meaning

768-dimensional vector: (BERT, common LLM embedding size)
  Can distinguish subtle differences in meaning across many topics

1536-dimensional vector: (OpenAI text-embedding-ada-002)
  Very rich representations -- used in production AI applications
```

The tradeoff:
  - More dimensions = better quality search results
  - More dimensions = more storage, slower comparison

For this module, we use 384-dimensional vectors (a good practical size).

---

## Part 6: C#/.NET Analogy

In C#, cosine similarity would look like this:

```csharp
public static double CosineSimilarity(float[] a, float[] b)
{
    // Step 1: Dot product
    double dotProduct = 0;
    for (int i = 0; i < a.Length; i++)
        dotProduct += a[i] * b[i];

    // Step 2: Magnitude of A
    double magnitudeA = Math.Sqrt(a.Sum(x => (double)x * x));

    // Step 3: Magnitude of B
    double magnitudeB = Math.Sqrt(b.Sum(x => (double)x * x));

    // Step 4: Cosine similarity
    return dotProduct / (magnitudeA * magnitudeB);
}
```

In Python + NumPy, this is 3 lines (you will see this in Example 01).

---

## Key Takeaways

1. An embedding model takes text as input and outputs a vector (list of numbers).
   Similar text -> similar vectors.

2. Cosine similarity measures the ANGLE between vectors (0 to 1 for text).
   1.0 = identical meaning. 0.0 = completely different meaning.

3. Cosine similarity is preferred over Euclidean distance for text because
   it is not affected by document length.

4. More dimensions = richer meaning representation (but more storage needed).

5. The math: cosine(A, B) = dot(A, B) / (|A| * |B|)

---

## Next

Lesson 03: ChromaDB Hands-On
  - What is ChromaDB and why use it?
  - Creating a collection and adding documents
  - Running your first similarity query
