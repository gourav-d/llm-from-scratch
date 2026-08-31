# Missing Topics: Dropout, VAE, Flow-Based Models, and Generative AI Taxonomy

This document covers topics that were absent or only briefly mentioned in our earlier modules.

Topics covered:
1. Dropout in Transformers -- focused explanation
2. VAE (Variational Autoencoder) -- deep dive
3. Flow-Based Generative Models -- complete explanation
4. All Five Generative Model Types -- comparison table and summary

---

## Section 1: Dropout in Transformers

### What Dropout Is

During training, dropout randomly "switches off" a percentage of neurons (sets their output to 0)
on each forward pass. The switched-off neurons are chosen randomly, and a DIFFERENT random set
is chosen each time.

At inference time (when generating text), dropout is turned OFF. All neurons are active.

### Why Dropout Prevents Overfitting

**The Problem: Neuron Over-Reliance**

Without dropout, a neural network can develop "celebrity neurons" -- single neurons that the model
depends on heavily for certain predictions. If neuron #47 learns to detect "subject-verb agreement",
the rest of the network stops bothering to learn this because #47 is always reliable.

On the training data: works perfectly.
On new data: if the pattern differs slightly, the network fails because it over-relied on #47.
This is overfitting: the model memorized training data patterns instead of generalizing.

**The Solution: Chaos Engineering**

Dropout is like "chaos engineering" for neural networks.

In software engineering, chaos engineering means randomly killing servers to force your system
to build redundancy. If Server #3 can go down at any moment, your system must route around it.
Result: a resilient system that does not depend on any single component.

Dropout does the same thing to neurons. If neuron #47 can be dropped at any moment:
- The network cannot rely on it exclusively
- Other neurons MUST learn to detect subject-verb agreement as a backup
- Each neuron develops independent skills
- The network becomes ROBUST to individual neuron failures

**The technical term:** Dropout prevents "co-adaptation" -- neurons learning to rely on
each other instead of learning independently useful features.

### How Dropout Works Mathematically

**Simple version (training):**
- For each neuron, generate a random number 0 or 1 (with probability p of being 0)
- Multiply the neuron's output by this number (0 = dropped, 1 = kept)

```python
# Conceptual implementation (simplified):
import numpy as np

def dropout(x, rate=0.1):
    # rate = fraction to DROP (e.g., 0.1 = drop 10%)
    mask = np.random.binomial(1, 1 - rate, size=x.shape)  # 1=keep, 0=drop
    return x * mask

# Example with rate=0.5 (drop 50%):
# Input:  [1.0, 2.0, 3.0, 4.0, 5.0]
# Mask:   [  1,   0,   1,   0,   1]   <- random 50% dropped
# Output: [1.0, 0.0, 3.0, 0.0, 5.0]
```

**Inverted Dropout (what modern implementations actually use):**

There is a problem: during training, neurons are dropped so the remaining neurons output less
signal in total. But at inference, ALL neurons are active -- the signal magnitude is suddenly
larger than what the network was trained on.

Fix: during TRAINING, SCALE UP the remaining neurons by `1 / (1 - rate)`:

```python
def inverted_dropout(x, rate=0.2):
    # rate = 0.2 means 20% dropped, 80% kept
    mask = np.random.binomial(1, 1 - rate, size=x.shape)
    # Scale the kept neurons UP so total signal stays the same on average
    return x * mask / (1 - rate)

# Example with rate=0.2 (drop 20%, keep 80%):
# Input:  [1.0, 2.0, 3.0, 4.0, 5.0]
# Mask:   [  1,   0,   1,   1,   1]   <- 20% dropped
# Scaled: [1.0/0.8, 0, 3.0/0.8, 4.0/0.8, 5.0/0.8]
# Output: [1.25, 0.0, 3.75, 5.0, 6.25]
```

Now at inference: no scaling needed. ALL neurons are active with their trained weights.
The average signal magnitude matches what was used during training.

PyTorch's `nn.Dropout(p)` implements inverted dropout automatically.

### Where Dropout Is Used in the GPT Model

```
Input x (BS, SL, 384)
    |
    v
+--- Transformer Block Start ---+
|                                |
|  LayerNorm -> MultiHeadAttention
|      |                         |
|      v                         |
|  Attention Weights             |
|      |                         |
|  nn.Dropout(0.05) <-- HERE (1) |
|      |                         |
|  @ Values                      |
|      |                         |
|  combine heads                 |
|      |                         |
|  nn.Dropout(0.05) <-- HERE (2) |
|      |                         |
|  + residual                    |
|      |                         |
|  LayerNorm -> ForwardLayer     |
|      Linear: 384->2304         |
|      GELU                      |
|      Linear: 2304->384         |
|      nn.Dropout(0.05) <- HERE (3)
|      |                         |
|  + residual                    |
+--------------------------------+
    |
    v
Output (BS, SL, 384)
```

THREE dropout positions in each transformer block:
1. After attention weights (before multiplying with Values) -- drops attention connections
2. After combining multi-head outputs -- drops head combination connections
3. After FFN compression -- drops feed-forward connections

**Why three?** Each position is a different kind of "decision point" in the network.
Regularizing all three prevents the network from over-relying on any single pathway.

### Training vs Inference Mode

```python
# Training: dropout ON
model.train()   # automatically enables Dropout layers
output = model(input)  # ~5% of neurons randomly dropped

# Inference / Evaluation: dropout OFF
model.eval()    # automatically disables Dropout layers (all neurons active)
output = model(input)  # all 100% of neurons active
```

PyTorch's `model.train()` and `model.eval()` toggle ALL dropout (and batch norm) layers
automatically. You do NOT need to manually configure each layer.

In the Udemy notebook:
```python
model.eval()    # used before calculate_loss() and generate_sample()
model.train()   # used after evaluation, before the training loop resumes
```

### Typical Dropout Values

| Model Size | Dataset Size | Typical Dropout |
|---|---|---|
| Small (< 100M) | Small (< 1GB) | 0.05 to 0.1 |
| Medium (100M-1B) | Medium | 0.1 |
| Large (1B+) | Large (100GB+) | 0.0 to 0.1 |
| Dense networks (non-transformer) | Any | 0.2 to 0.5 |

**Why larger models use LESS dropout?**
Large models on large datasets have low risk of overfitting -- they have enough capacity
and data to generalize naturally. High dropout would slow learning unnecessarily.

Small models on small datasets (like our 19M model on wiki.txt) benefit more from dropout
to prevent memorizing the limited training data.

---

## Section 2: VAE -- Variational Autoencoder

### The Starting Point: Basic Autoencoder

Before explaining VAE, we need the simpler Autoencoder (AE).

**Autoencoder Architecture:**

```
Input Data (784 dims for 28x28 image)
    |
    v
[Encoder Network]    <- compresses data
    |
    v
Latent Code z (64 dims)   <- compact representation
    |
    v
[Decoder Network]    <- reconstructs data
    |
    v
Reconstructed Data (784 dims)
```

**What it learns:**
The encoder learns to compress data into a small "latent code" z.
The decoder learns to reconstruct the original from just that code.
Training loss = reconstruction error (how different is the output from the input?).

**C# analogy:** Like lossless data compression. ZIP/GZIP encodes a file to fewer bytes,
and decodes perfectly back. The autoencoder learns a LEARNED compression specific to your data type.

**The problem with basic AE:**
The latent space (the 64-dimensional space of all possible z values) has "holes".

Example: if z=[1.5, 2.3, ...] maps to "cat image" and z=[5.1, 0.8, ...] maps to "dog image",
what does z=[3.3, 1.5, ...] (the midpoint) decode to?
For a basic AE: garbage. There is no constraint making the midpoint meaningful.

This means you CANNOT generate new images by randomly sampling a z value --
most random z values will decode to meaningless noise.

### The VAE Fix: Encode to a Distribution

**VAE Architecture:**

```
Input Data
    |
    v
[Encoder Network]
    |
    v
Mean (mu) vector       <- "center" of the distribution for this input
Standard Dev (sigma)   <- "spread" of the distribution for this input
    |
    v  Reparameterization Trick: z = mu + sigma * epsilon
       (epsilon is random noise from N(0,1))
    |
    v
Sampled z  (still in latent space, but from a distribution, not a fixed point)
    |
    v
[Decoder Network]
    |
    v
Reconstructed Data
```

**The key difference:**
- Basic AE: input -> ONE specific point in latent space
- VAE: input -> DISTRIBUTION (a cloud of points centered at mu with spread sigma)

**Why does this enable generation?**
The training forces all these distributions to be close to N(0,1) (standard Gaussian).
Every input maps to a region near the origin of latent space with unit variance.
The entire latent space becomes FILLED with meaningful data.

Now, to generate a new sample:
1. Sample z from N(0,1) (any point near the origin)
2. Decode z with the decoder
3. Get a realistic new sample

Every point in the latent space decodes to something meaningful!

### The Reparameterization Trick

**Problem:** Sampling from a distribution is not differentiable (no gradient flows through random sampling).
You cannot backpropagate through `z = sample(mu, sigma)`.

**Solution:** Factor out the randomness:
```
z = mu + sigma * epsilon
where epsilon ~ N(0, 1) is sampled BEFORE the operation
```

Now:
- `epsilon` is a random constant (fixed for this forward pass)
- `z = mu + sigma * epsilon` is a DIFFERENTIABLE operation with respect to mu and sigma
- Gradients can flow back through z to the encoder

**C# analogy:** Like dependency injection of a random seed. Instead of `Random()` inside
your function (not testable/differentiable), you inject `epsilon` from outside,
making the function deterministic given epsilon.

### VAE Training Loss

VAE minimizes TWO loss terms:

```
Total Loss = Reconstruction Loss + KL Divergence

Reconstruction Loss:
    How similar is the decoded output to the original input?
    For images: Mean Squared Error or Binary Cross Entropy
    Penalizes: bad reconstructions

KL Divergence (Kullback-Leibler):
    How similar is the encoder's distribution (mu, sigma) to N(0, 1)?
    Formula: KL(q || p) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    Penalizes: distributions that are far from the standard Gaussian
```

**Why the KL term?**
Without it, the encoder could place every input's distribution in a completely different,
isolated region of latent space. The "holes" problem would persist.
The KL term forces all distributions toward N(0,1) -- filling the latent space.

**C# analogy:** Like adding a constraint to a C# generic type. The KL term constrains
the latent space to be a well-structured Gaussian, just as `where T : IComparable`
constrains what types can be used with a generic class.

### Simple VAE Code Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=64):
        super().__init__()

        # Encoder: input -> hidden -> mu and sigma
        self.encoder_hidden = nn.Linear(input_dim, 256)
        self.encoder_mu = nn.Linear(256, latent_dim)     # mean vector
        self.encoder_log_var = nn.Linear(256, latent_dim)  # log variance (log for stability)

        # Decoder: latent -> hidden -> reconstruction
        self.decoder_hidden = nn.Linear(latent_dim, 256)
        self.decoder_output = nn.Linear(256, input_dim)

    def encode(self, x):
        h = F.relu(self.encoder_hidden(x))  # shared hidden layer
        mu = self.encoder_mu(h)              # mean of distribution
        log_var = self.encoder_log_var(h)    # log variance of distribution
        return mu, log_var

    def reparameterize(self, mu, log_var):
        # Convert log variance to standard deviation
        sigma = torch.exp(0.5 * log_var)
        # Sample random noise (NOT through the computation graph -- external randomness)
        epsilon = torch.randn_like(sigma)
        # z = mu + sigma * epsilon  (differentiable with respect to mu and sigma)
        z = mu + sigma * epsilon
        return z

    def decode(self, z):
        h = F.relu(self.decoder_hidden(z))
        x_reconstructed = torch.sigmoid(self.decoder_output(h))  # output in [0,1] range
        return x_reconstructed

    def forward(self, x):
        # Encode: x -> (mu, log_var)
        mu, log_var = self.encode(x)
        # Sample: (mu, log_var) -> z
        z = self.reparameterize(mu, log_var)
        # Decode: z -> x_reconstructed
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var

    def compute_loss(self, x, x_reconstructed, mu, log_var):
        # Reconstruction loss: how well did we reconstruct the input?
        recon_loss = F.binary_cross_entropy(x_reconstructed, x, reduction='sum')
        # KL divergence: how close is our distribution to N(0,1)?
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_loss

    def generate(self, n_samples=10):
        # Generate new samples by sampling z from N(0,1) and decoding
        z = torch.randn(n_samples, 64)  # random points in latent space
        return self.decode(z)           # decode each to a realistic sample
```

### VAE Use Cases

| Domain | Application | Why VAE? |
|---|---|---|
| Drug discovery | Generate new molecular structures | Latent space interpolation explores chemical space |
| Image editing | Face attribute manipulation | "smile" direction in latent space |
| Anomaly detection | Fraud, manufacturing defects | High reconstruction error = unusual input |
| Data augmentation | Generate training examples | Sample new realistic variants of training data |
| Compression | Learned image compression | Encoder is a compressor; decoder is decompressor |

### VAE vs GAN

| Aspect | VAE | GAN |
|---|---|---|
| Quality | Blurry (averaging effect from L2/BCE loss) | Sharp (discriminator demands realism) |
| Diversity | Good (entire latent space sampled uniformly) | Risk of mode collapse (generates limited variety) |
| Training | Stable (single loss with two terms) | Unstable (two networks that can destabilize each other) |
| Latent structure | Interpretable, smooth (KL forces structure) | Less structured, harder to navigate |
| Exact likelihood | Approximate (ELBO lower bound) | None |
| Best for | Drug design, anomaly detection, interpolation | Photorealistic images, videos |

---

## Section 3: Flow-Based Generative Models

### The Core Idea: Invertible Functions

Flow-based models (also called "normalizing flows") are built on one key idea:

**Find a function f that is perfectly invertible:**
```
f(data) = noise         -- encode: data -> simple Gaussian noise
f_inverse(noise) = data -- decode: sample noise -> realistic data
```

If f is perfectly invertible:
- We can EXACTLY compute the probability of any data point (no approximations)
- We can generate new samples by sampling noise and applying f_inverse
- No need for a second network (discriminator) or probability approximations

### Why "Normalizing" Flows?

"Normalizing" refers to transforming a complex data distribution into a "normal" (Gaussian) distribution.

"Flows" refers to the series of invertible transformations applied step by step.

```
Complex data distribution:    Simple Gaussian distribution:
     +--------+                       +--------+
     |  /\_/\ |                       |   **   |
     | /     \|                       |  ****  |
     |/  cat  |    <-- f ----->       | ****** |
     |\      /|    <-- f^-1 --        |  ****  |
     | \_____/|                       |   **   |
     +--------+                       +--------+
     Irregular shape                  Smooth bell curve
```

Training: learn f to map data distribution to Gaussian.
Generation: sample from Gaussian and apply f_inverse.

### Why Can We Compute Exact Likelihood?

When you apply an invertible function f to a probability distribution, the probability changes
according to the Jacobian (the matrix of all partial derivatives).

The "change of variables formula":
```
log p(x) = log p(f(x)) + log |det J_f(x)|
```
Where:
- `p(x)` = probability of data point x
- `p(f(x))` = probability of the transformed point (easy if f(x) is Gaussian)
- `|det J_f(x)|` = the Jacobian determinant (how much f stretches/compresses volume near x)

For this to be computable, the Jacobian determinant must be easy to calculate.
This is the KEY constraint on flow model architectures.

**C# analogy:** Like a lossless compression algorithm with a built-in "compression ratio" field.
Not only can you compress and decompress, but you know exactly how much information is in each file.

### Coupling Layers: Making Invertibility Tractable

The most common flow architecture uses "coupling layers".

**Coupling layer idea:**
Split the input x into two parts: x = [x1, x2]

```
Input:  [x1, x2]
            |
            v
x1 is passed through unchanged
x2 is transformed using a function of x1:
    y2 = x2 * exp(s(x1)) + t(x1)   (element-wise scale and shift)
            |
            v
Output: [x1, y2]
```

**Why is this invertible?**
```
Given output [x1, y2]:
    x1 = x1   (unchanged)
    x2 = (y2 - t(x1)) * exp(-s(x1))   (invert the scale-shift)
```

Since x1 is known from the output, we can compute s(x1) and t(x1) and recover x2.
The function s() and t() can be any neural network (they don't need to be invertible themselves!).

**The Jacobian is triangular** for coupling layers, so its determinant is just the product
of the diagonal -- extremely fast to compute.

### Visual Architecture

```
Data (complex distribution)
    |
    v
[Coupling Layer 1]   -- invertible transform
    |
    v
[Coupling Layer 2]   -- invertible transform
    |
    v
[Coupling Layer 3]   -- invertible transform
    ...
    |
    v
[Coupling Layer N]   -- invertible transform
    |
    v
z (Gaussian noise)
```

Each coupling layer applies a simple invertible transform.
Composing many such layers = complex overall invertible transform.

To generate:
```
Sample z from N(0,1)
    |
    v
[Coupling Layer N inverse]
    ...
    |
    v
[Coupling Layer 1 inverse]
    |
    v
Generated data sample
```

### Famous Flow Models

**RealNVP (2016):**
First practical image flow model. Used coupling layers with convolutional networks for s() and t().

**Glow (2018, OpenAI):**
Extended RealNVP with 1x1 invertible convolutions for better mixing between splits.
Generated high-quality 256x256 face images. Showed exact likelihood estimation for images.

**WaveGlow (NVIDIA):**
Applied flows to raw audio waveforms.
Generated speech with lower latency than WaveNet (autoregressive).

**Normalizing Flows for Text:**
Less common than for continuous data (images, audio) because text is discrete.
Some work on applying flows to embeddings before discrete sampling.

### Comparison to Other Models

```
VAE:
   Uses ELBO (evidence lower bound) -- a lower bound on log likelihood.
   Cheaper to compute, but NOT exact. Decoder is separate from encoder.

GAN:
   Does NOT compute likelihood at all. Discriminator is discarded after training.
   Cannot evaluate "how likely is this data point?"

Flow:
   Computes EXACT log likelihood via change-of-variables formula.
   More expensive to compute, architecture constrained, but principled.
```

### Flow Model Weaknesses

1. **Architecture constraint:** f must be invertible with tractable Jacobian.
   Not every neural network architecture works. Standard CNNs/Transformers are NOT invertible.
   You must use special layers (coupling layers, 1x1 convolutions, etc.).

2. **Parameter inefficiency:** Coupling layers can only transform half the dimensions at a time.
   More layers needed to achieve the same capacity as unconstrained architectures.

3. **Scaling difficulty:** Getting flows to work at the scale of modern diffusion models
   has proven harder. Most SOTA image generation (2023+) uses diffusion models.

4. **Discrete data:** Flows are designed for continuous distributions.
   Text (discrete tokens) requires workarounds (dequantization, continuous relaxations).

### Simple Coupling Layer Code Example

```python
import torch
import torch.nn as nn

class CouplingLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        # s and t networks -- these do NOT need to be invertible
        # They operate on x1 (first half) to transform x2 (second half)
        self.scale_net = nn.Sequential(
            nn.Linear(half, 64), nn.ReLU(), nn.Linear(64, half), nn.Tanh()
        )
        self.translate_net = nn.Sequential(
            nn.Linear(half, 64), nn.ReLU(), nn.Linear(64, half)
        )

    def forward(self, x):
        # Split input in half
        x1, x2 = x.chunk(2, dim=-1)
        # Compute scale and translate from x1
        s = self.scale_net(x1)
        t = self.translate_net(x1)
        # Transform x2 using scale-and-shift (invertible: y2 = x2 * exp(s) + t)
        y2 = x2 * torch.exp(s) + t
        # Log determinant of Jacobian = sum of scale factors
        log_det = s.sum(dim=-1)
        return torch.cat([x1, y2], dim=-1), log_det

    def inverse(self, y):
        # Split output in half
        y1, y2 = y.chunk(2, dim=-1)
        # x1 is unchanged
        x1 = y1
        # Recover x2: invert y2 = x2 * exp(s) + t
        s = self.scale_net(x1)
        t = self.translate_net(x1)
        x2 = (y2 - t) * torch.exp(-s)
        return torch.cat([x1, x2], dim=-1)
```

---

## Section 4: All Five Generative Model Types -- Complete Comparison

### Summary: How Each Type Generates New Data

**Autoregressive (GPT, LLaMA, Claude):**
```
Start with <start> token
    |
    v
[Model] -> probability distribution over next token
    |
    v
Sample next token
    |
    v
Append to sequence, repeat
```
One token at a time. Each step uses ALL previous outputs as context.

---

**GAN (StyleGAN, DALL-E 1):**
```
Sample random noise z from N(0,1)
    |
    v
[Generator] -> fake image/text
    |
    v
[Discriminator] -> Real or Fake?
    |
    v
Generator improves to fool Discriminator
Discriminator improves to catch Generator
    |
    v (after training, use Generator only)
Sample z -> [Generator] -> output
```

---

**VAE (Molecular VAE, face interpolation):**
```
Training data -> [Encoder] -> (mu, sigma)
                                  |
                        z = mu + sigma * epsilon
                                  |
                              [Decoder] -> reconstruction

Generation:
    Sample z from N(0,1) -> [Decoder] -> new realistic sample
```

---

**Diffusion (DALL-E 2, Stable Diffusion):**
```
Training:
    Real image -> add noise (step 1) -> add noise (step 2) -> ... -> pure noise
    Train network to PREDICT the noise added at each step

Generation (reverse process):
    Start with pure random noise
    [Denoiser] -> slightly less noisy
    [Denoiser] -> slightly less noisy
    ...
    [Denoiser] -> realistic image
```

---

**Flow (Glow, RealNVP):**
```
Training:
    Real data -> [f: Coupling layers] -> z ~ N(0,1)
    Maximize log p(x) = log N(f(x)) + log |det J|

Generation:
    Sample z from N(0,1) -> [f_inverse] -> new realistic sample
```

---

### Master Comparison Table

| Property | Autoregressive | GAN | VAE | Diffusion | Flow |
|---|---|---|---|---|---|
| Generation method | Predict next token | Generator vs Discriminator | Sample from learned distribution | Reverse denoising | Invert transform |
| Likelihood computation | Exact | None | Approximate (ELBO bound) | Approximate (ELBO bound) | Exact |
| Sample quality (images) | Not used for images | Sharp | Blurry | State of the art | Good |
| Sample quality (text) | State of the art | Poor | Poor | Improving (2023+) | Not used for text |
| Training stability | Stable | Unstable (adversarial) | Stable | Stable | Stable |
| Generation speed | Slow (sequential tokens) | Fast (one pass) | Fast (one pass) | Slow (many denoising steps) | Fast (one pass) |
| Latent space | No fixed latent space | Unstructured z | Smooth Gaussian structure | Time as latent dimension | Perfectly structured |
| Architecture flexibility | Any architecture | Any architecture | Encoder + Decoder | U-Net / Transformer | Must be invertible |
| Scalability | Scales very well | Scales (with difficulty) | Scales | Scales very well | Hard to scale |
| Primary use today | Text, code (GPT-4, Claude) | Images, video (StyleGAN) | Drug discovery, anomaly | Images, audio (SD, DALL-E 2) | Audio (WaveGlow), density estimation |
| Famous examples | GPT-4, LLaMA, Claude, Gemini | StyleGAN, CycleGAN, DALL-E 1 | Molecular VAE, beta-VAE | DALL-E 2, Stable Diffusion, Midjourney, Sora | Glow, RealNVP, WaveGlow |

---

## Section 5: Coverage in This Course

### What Was Already Covered

| Type | Coverage in Our Modules |
|---|---|
| Autoregressive | Excellently covered -- entire M04 + M05 + M06 |
| GAN | M03 `08_types_of_neural_networks.md` -- dedicated section with examples |
| Diffusion | M03 (brief intro) + M16 (full module: MDLM, SEDD, Plaid, diffusion LMs) |

### What Was Missing

| Type | Previous Status | Now Covered |
|---|---|---|
| VAE | Only mentioned as "ELBO cross-reference" in M16 | Full Section 2 above |
| Flow-based models | Completely absent | Full Section 3 above |

### Where to Go Deeper

- **M16** (`modules/16_modern_architectures/`) -- Diffusion Language Models in depth
- **M13** (`modules/13_rlhf_alignment/`) -- RLHF, DPO, Constitutional AI
- **M03** (`modules/03_neural_networks/08_types_of_neural_networks.md`) -- GAN + Diffusion overview

For VAE and Flow hands-on practice, refer to the enhancement roadmap in `10_enhancement_roadmap.md`
for dataset and training suggestions.
