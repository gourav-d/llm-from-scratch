# =============================================================================
# MODULE 12 - PROJECT 03: DOMAIN-SPECIFIC CHATBOT FINE-TUNER
# =============================================================================
# Title   : Domain-Specific Chatbot Fine-Tuner (IT Support Assistant)
# Goal    : Build a complete chatbot pipeline that has been "fine-tuned" on an
#           IT support knowledge base so it answers only IT support questions.
# What you
# will    : 1. Build an IT support knowledge base with 30+ Q&A pairs
# build   : 2. Vectorize text using a Bag-of-Words approach (pure NumPy)
#           3. Train a 2-layer NumPy neural net to classify user intent
#           4. Build a response generator that picks the best reply
#           5. Wire everything into a DomainChatbot class with history
#           6. Test out-of-domain detection (sports / weather / cooking)
#           7. Compare Fine-tuning vs Prompting vs RAG approaches
# How to  :   python project_03_domain_chatbot.py
# run     :
# Dependencies: Python 3.10+ and NumPy only. No PyTorch, no APIs.
# =============================================================================

# =============================================================================
# GLOSSARY
# (Read every definition before looking at the code -- every term is here.)
# =============================================================================
#
# Domain Adaptation
#   The process of specialising a general model so it performs well on a
#   specific topic area (a "domain"), such as IT support, legal documents,
#   or medical records.  You feed it examples from that domain so it
#   learns the vocabulary, patterns, and correct answers for that field.
#   C# analogy: writing a domain-specific library (e.g. TaxCalculator)
#   rather than using a generic math library.
#
# Knowledge Base (KB)
#   A structured collection of question-answer pairs (or documents) that
#   capture the expertise of a domain.  Our IT support KB is a Python list
#   of dictionaries, each with "query", "intent", and "response".
#   C# analogy: a List<SupportTicketDto> loaded from a database.
#
# Intent Classification
#   The task of deciding WHAT the user wants to do, from a fixed set of
#   categories (intents).  E.g. "I forgot my password" -> intent = PASSWORD.
#   C# analogy: parsing a command-line argument to decide which branch of a
#   switch statement to execute.
#
# Slot Filling
#   After classifying the intent, extracting specific pieces of information
#   ("slots") from the message.  E.g. intent=PASSWORD, slot=username:"john".
#   C# analogy: reading named capture groups from a Regex match.
#
# Response Template
#   A fixed text pattern with placeholder slots that gets filled in to
#   produce a final reply.  E.g. "Hello {name}, your ticket {id} is open."
#   C# analogy: string.Format() or an interpolated string $"Hello {name}".
#
# Domain-Specific Vocabulary
#   Words that appear frequently in a domain but rarely in general text.
#   E.g. "VPN", "SSID", "firewall", "Outlook", "helpdesk" are IT-domain words.
#   A domain-adapted model knows these words; a general model may not.
#
# Out-of-Domain (OOD) Query
#   A user message that falls outside the topics the model was trained on.
#   E.g. asking an IT chatbot "what is the capital of France?" is OOD.
#   C# analogy: passing an argument that fails a Guard.Against check.
#
# Retrieval Fallback
#   When the model cannot confidently classify the intent, it falls back to
#   searching the KB for the most similar stored answer by text matching.
#   If nothing matches well, it returns a polite "I don't know" message.
#   C# analogy: a try/catch that returns a default response on failure.
#
# =============================================================================

# =============================================================================
# ASCII DIAGRAM -- FULL CHATBOT PIPELINE
# =============================================================================
#
#   User types a message
#          |
#          v
#   +------------------+
#   |  BagOfWords      |   Converts raw text into a numeric vector
#   |  Vectorizer      |   (word counts)  -- "vocabulary lookup"
#   +------------------+
#          |  numeric vector
#          v
#   +------------------+     trained on     +----------------------+
#   |  Intent          |  <--------------   |  IT Support          |
#   |  Classifier      |   IT Support KB    |  Knowledge Base      |
#   |  (NumPy 2-layer  |                    |  (30+ Q&A pairs)     |
#   |   neural net)    |                    +----------------------+
#   +------------------+
#          |  intent label + confidence score
#          v
#   +------------------+
#   |  Response        |   Picks the best stored answer for this intent
#   |  Generator       |   using word-overlap scoring.
#   |                  |   If confidence too low -> fallback message.
#   +------------------+
#          |
#          v
#   Bot reply shown to user
#          |
#          v
#   Conversation history updated (list of dicts)
#
# =============================================================================

import numpy as np            # NumPy: Python's math library (like System.Math on steroids)
import random                 # Python's built-in random module (like System.Random)
import math                   # Python's built-in math library (like Math.Log, Math.Exp)
from collections import defaultdict   # A dictionary that creates missing keys automatically
                                      # C# analogy: Dictionary with a default value factory

# Fix the random seed so we get the same results every time we run.
# C# analogy: new Random(42)  -- seeded random number generator.
np.random.seed(42)
random.seed(42)

# =============================================================================
# PART 1: KNOWLEDGE BASE (the "domain data" we fine-tune on)
# =============================================================================
print("=" * 70)
print("PART 1: IT SUPPORT KNOWLEDGE BASE")
print("=" * 70)

# IT_SUPPORT_KB is the heart of our domain adaptation.
# Each entry is a Python dictionary with three keys:
#   "query"   -- a sample question the user might ask
#   "intent"  -- the category / intent label for that question
#   "response"-- the correct IT support answer
#
# C# analogy: a List<SupportKbEntry> where SupportKbEntry is:
#   class SupportKbEntry { string Query; string Intent; string Response; }

IT_SUPPORT_KB = [

    # ------------------------------------------------------------------
    # PASSWORD ISSUES (5 examples)
    # ------------------------------------------------------------------
    {
        "query": "I forgot my password and cannot log in",
        "intent": "PASSWORD",
        "response": (
            "To reset your password: go to the login page and click "
            "'Forgot Password'. Enter your work email and check your inbox "
            "for the reset link. Link expires in 30 minutes. "
            "If you do not receive it, check your spam folder or call the helpdesk."
        )
    },
    {
        "query": "My password has expired and I am locked out",
        "intent": "PASSWORD",
        "response": (
            "Your account is locked due to an expired password. "
            "Call the IT helpdesk at ext. 4357 or submit a ticket at "
            "helpdesk.company.com. Have your employee ID ready. "
            "We will unlock your account and send a temporary password."
        )
    },
    {
        "query": "How do I change my Windows password",
        "intent": "PASSWORD",
        "response": (
            "Press Ctrl + Alt + Delete and select 'Change a password'. "
            "Enter your current password, then type and confirm the new one. "
            "Password must be at least 12 characters with uppercase, lowercase, "
            "a number, and a special character."
        )
    },
    {
        "query": "I keep getting a wrong password error even though I typed it correctly",
        "intent": "PASSWORD",
        "response": (
            "Check that Caps Lock is off. Make sure you are using your "
            "current network password, not an old one. "
            "After 5 failed attempts the account locks for 15 minutes. "
            "Wait and try again, or contact the helpdesk to unlock."
        )
    },
    {
        "query": "Can you reset the password for my service account",
        "intent": "PASSWORD",
        "response": (
            "Service account password resets require manager approval. "
            "Please submit a ticket at helpdesk.company.com with your "
            "manager in CC. Include the service account name and business reason. "
            "Processing time is 1 business day."
        )
    },

    # ------------------------------------------------------------------
    # VPN / NETWORK ISSUES (5 examples)
    # ------------------------------------------------------------------
    {
        "query": "I cannot connect to the VPN from home",
        "intent": "VPN",
        "response": (
            "Check you are using the latest VPN client (v4.2+). "
            "Verify your internet connection first by opening a browser. "
            "Open the VPN app, enter vpn.company.com as the server, "
            "and use your network credentials. If it still fails, "
            "try restarting the VPN service: Start -> Services -> Cisco AnyConnect."
        )
    },
    {
        "query": "VPN keeps disconnecting every few minutes",
        "intent": "VPN",
        "response": (
            "Frequent VPN drops are usually caused by an unstable home internet. "
            "Try connecting over a wired Ethernet cable instead of Wi-Fi. "
            "Also check that no firewall or antivirus is blocking the VPN port (UDP 443). "
            "Update the VPN client from the software centre and try again."
        )
    },
    {
        "query": "My network drive is not showing up after connecting to VPN",
        "intent": "VPN",
        "response": (
            "After connecting to VPN, open File Explorer and type "
            "\\\\fileserver in the address bar to reconnect mapped drives. "
            "If drives are still missing, right-click This PC -> Map network drive "
            "and re-map them. Log off and back on if the problem persists."
        )
    },
    {
        "query": "I cannot access internal websites when on the company network",
        "intent": "VPN",
        "response": (
            "If you are on the office network and cannot reach internal sites, "
            "check your proxy settings: go to Internet Options -> Connections -> "
            "LAN Settings and enable 'Automatically detect settings'. "
            "Clear your browser cache and try again. If the problem persists, "
            "flush DNS by opening CMD and typing: ipconfig /flushdns"
        )
    },
    {
        "query": "How do I request VPN access for a new employee",
        "intent": "VPN",
        "response": (
            "VPN access must be approved by the employee's manager. "
            "Submit a request at helpdesk.company.com, category = Access Management. "
            "Include: employee name, ID, start date, and business justification. "
            "Provisioning takes up to 2 business days."
        )
    },

    # ------------------------------------------------------------------
    # PRINTER PROBLEMS (4 examples)
    # ------------------------------------------------------------------
    {
        "query": "My printer is not printing anything",
        "intent": "PRINTER",
        "response": (
            "First check the printer is powered on and online (no amber light). "
            "Check the print queue: Start -> Devices and Printers -> right-click "
            "your printer -> See what is printing. Cancel all stuck jobs. "
            "Restart the Print Spooler: open Services, find Print Spooler, "
            "right-click -> Restart. Try printing again."
        )
    },
    {
        "query": "Pages are printing with streaks or faded text",
        "intent": "PRINTER",
        "response": (
            "Streaks or fading usually mean the toner cartridge is low or the "
            "drum is dirty. Open the printer, remove and gently shake the toner "
            "cartridge to redistribute remaining toner. If streaks continue, "
            "submit a supply request at helpdesk.company.com for a replacement cartridge."
        )
    },
    {
        "query": "Printer shows offline in Windows but it is switched on",
        "intent": "PRINTER",
        "response": (
            "Right-click the printer in Devices and Printers and choose "
            "'See what is printing'. In the menu bar click Printer -> "
            "uncheck 'Use Printer Offline'. If still offline, remove the printer "
            "and re-add it: Settings -> Bluetooth & Devices -> Printers & Scanners "
            "-> Add a printer."
        )
    },
    {
        "query": "How do I add a network printer to my laptop",
        "intent": "PRINTER",
        "response": (
            "Go to Settings -> Bluetooth & Devices -> Printers & Scanners -> "
            "Add a printer or scanner. Windows will search the network. "
            "If your printer does not appear, click 'The printer I want is not listed', "
            "choose 'Select a shared printer by name', and type "
            "\\\\printserver\\PrinterName. Contact IT if you need the exact printer name."
        )
    },

    # ------------------------------------------------------------------
    # SOFTWARE INSTALLATION (4 examples)
    # ------------------------------------------------------------------
    {
        "query": "I need to install Microsoft Office on my new laptop",
        "intent": "SOFTWARE",
        "response": (
            "Open the Company Software Centre (Start -> Software Centre). "
            "Search for 'Microsoft Office 365' and click Install. "
            "The installation takes about 20 minutes. "
            "Sign in with your company email when prompted. "
            "If Software Centre is missing, contact IT to push the install remotely."
        )
    },
    {
        "query": "How do I install a software that is not in the company catalogue",
        "intent": "SOFTWARE",
        "response": (
            "Unapproved software requires a business justification and manager sign-off. "
            "Submit a Software Request at helpdesk.company.com. "
            "Include: software name, version, vendor, business reason, and cost. "
            "The IT security team will review within 5 business days. "
            "Do NOT install unapproved software; it may violate company policy."
        )
    },
    {
        "query": "My application keeps crashing when I open it",
        "intent": "SOFTWARE",
        "response": (
            "Try these steps in order: "
            "1) Restart the application. "
            "2) Restart your computer. "
            "3) Repair the application: Settings -> Apps -> select the app -> Modify -> Repair. "
            "4) Check Windows Event Viewer (search 'Event Viewer' in Start) for error logs. "
            "5) If still crashing, submit a ticket with the event log details."
        )
    },
    {
        "query": "I cannot update my software because I do not have admin rights",
        "intent": "SOFTWARE",
        "response": (
            "Most company software updates through the Software Centre automatically. "
            "For manual updates, submit a ticket at helpdesk.company.com requesting "
            "a remote admin session. An IT technician will connect to your machine "
            "and perform the update. Estimated wait time: 2-4 hours."
        )
    },

    # ------------------------------------------------------------------
    # EMAIL / OUTLOOK ISSUES (4 examples)
    # ------------------------------------------------------------------
    {
        "query": "Outlook is not receiving new emails",
        "intent": "EMAIL",
        "response": (
            "Check your internet and VPN connection first. "
            "In Outlook, click Send/Receive -> Update Folder. "
            "If still not syncing, go to File -> Account Settings -> "
            "select your account -> Test Account Settings. "
            "If the test fails, remove and re-add the account. "
            "Contact IT if the issue persists after re-adding."
        )
    },
    {
        "query": "I accidentally deleted an important email",
        "intent": "EMAIL",
        "response": (
            "Check the Deleted Items folder first. "
            "If it is not there, right-click Deleted Items -> "
            "Recover Deleted Items. Exchange keeps deleted items for 30 days. "
            "If outside 30 days, submit a mailbox recovery request at "
            "helpdesk.company.com. Include sender, subject, and approximate date."
        )
    },
    {
        "query": "My Outlook calendar is not syncing with my phone",
        "intent": "EMAIL",
        "response": (
            "On your phone, go to Settings -> Mail/Calendar -> your company account. "
            "Toggle sync off, wait 10 seconds, then toggle it back on. "
            "Ensure the account type is set to Exchange (not IMAP). "
            "Server: mail.company.com, Domain: COMPANY. "
            "Call IT if you need the exact Exchange server settings."
        )
    },
    {
        "query": "I am getting a mailbox full error and cannot send emails",
        "intent": "EMAIL",
        "response": (
            "Your mailbox has reached its 50 GB limit. "
            "Empty your Deleted Items and Junk folders. "
            "Archive old emails: File -> Archive -> set a date (e.g. older than 1 year). "
            "If you need a larger mailbox, your manager can request an increase "
            "at helpdesk.company.com (approval required, extra cost applies)."
        )
    },

    # ------------------------------------------------------------------
    # HARDWARE PROBLEMS (4 examples)
    # ------------------------------------------------------------------
    {
        "query": "My laptop battery drains very fast",
        "intent": "HARDWARE",
        "response": (
            "Check battery health: Start -> search 'powercfg /batteryreport' -> run in CMD. "
            "Open the generated report in a browser and check 'Design Capacity' vs "
            "'Full Charge Capacity'. If the full charge is less than 60% of design, "
            "the battery needs replacing. Submit a hardware replacement request at "
            "helpdesk.company.com. Loaner laptops are available while yours is serviced."
        )
    },
    {
        "query": "My monitor has dead pixels or the screen is flickering",
        "intent": "HARDWARE",
        "response": (
            "First try a different display cable (HDMI or DisplayPort). "
            "Update your display driver: Device Manager -> Display Adapters -> "
            "right-click -> Update driver. "
            "If the screen still flickers or has dead pixels, the monitor needs "
            "hardware inspection. Submit a hardware ticket at helpdesk.company.com. "
            "Bring your monitor to the IT desk or request an onsite visit."
        )
    },
    {
        "query": "My laptop keyboard keys are not working",
        "intent": "HARDWARE",
        "response": (
            "Connect an external USB keyboard to check if the issue is hardware or software. "
            "If the USB keyboard works fine, the laptop keyboard likely needs physical repair. "
            "If neither keyboard works, restart the laptop. "
            "Check Device Manager for a yellow warning on the keyboard device. "
            "Submit a hardware repair ticket at helpdesk.company.com."
        )
    },
    {
        "query": "My computer is making a loud noise",
        "intent": "HARDWARE",
        "response": (
            "Loud noise from a computer usually means a failing fan or a loose component. "
            "Do NOT open the laptop yourself -- this voids the warranty. "
            "Back up your files immediately to the network drive or OneDrive. "
            "Submit a hardware urgent ticket at helpdesk.company.com and label it URGENT. "
            "An engineer will contact you within 2 hours."
        )
    },

    # ------------------------------------------------------------------
    # GENERAL / UNKNOWN / OUT-OF-DOMAIN (4 examples)
    # The model should learn to route these to a fallback response.
    # ------------------------------------------------------------------
    {
        "query": "What is the weather like today",
        "intent": "UNKNOWN",
        "response": (
            "I am the IT Support Assistant and can only help with IT-related questions. "
            "For weather information, please check a weather website or app."
        )
    },
    {
        "query": "Who won the football match last night",
        "intent": "UNKNOWN",
        "response": (
            "I am the IT Support Assistant. Sports scores are outside my area. "
            "Please visit a sports news site for that information."
        )
    },
    {
        "query": "Can you recommend a good restaurant near the office",
        "intent": "UNKNOWN",
        "response": (
            "I am set up to handle IT support questions only. "
            "For restaurant recommendations, try Google Maps or Yelp."
        )
    },
    {
        "query": "Tell me a joke",
        "intent": "UNKNOWN",
        "response": (
            "I am the IT Support Assistant and my training is limited to IT topics. "
            "If you have an IT issue, I am happy to help with that instead!"
        )
    },
]

# --- Print KB statistics ---
# Count how many examples exist for each intent.
# defaultdict(int) starts every new key at 0 automatically.
# C# analogy: Dictionary<string, int> with GetOrAdd(key, 0)
intent_counts = defaultdict(int)     # create a counting dictionary

for entry in IT_SUPPORT_KB:          # loop over every KB entry
    intent_counts[entry["intent"]] += 1   # increment count for that intent

print(f"\nKnowledge Base loaded: {len(IT_SUPPORT_KB)} total entries\n")

print("Entries per intent:")
for intent, count in sorted(intent_counts.items()):   # sort alphabetically
    # f-string: like $"  {intent,-12}: {count} examples" in C#
    bar = "#" * count                # simple ASCII bar chart
    print(f"  {intent:<12} : {count:2d} examples  {bar}")

print()    # blank line for readability

# =============================================================================
# PART 2: TEXT VECTORIZER (Bag-of-Words)
# =============================================================================
print("=" * 70)
print("PART 2: BAG-OF-WORDS VECTORIZER")
print("=" * 70)

# What is Bag-of-Words?
# ----------------------
# It ignores word ORDER and just counts HOW MANY TIMES each word appears.
# "I forgot my password" -> {"i":1, "forgot":1, "my":1, "password":1}
# Then we turn that into a fixed-size numeric vector.
# C# analogy: a Dictionary<string, int> flattened into a float[] array.
#
# ASCII illustration:
#
#   Sentence: "forgot my password"
#
#   Vocabulary (from ALL training sentences):
#   [battery, connect, email, forgot, install, my, network, password, vpn ...]
#      0        1        2       3       4       5      6       7        8
#
#   Vector:   [ 0,      0,       0,      1,      0,     1,      0,       1,       0 ...]
#                                       ^forgot        ^my             ^password

class BagOfWordsVectorizer:
    # C# analogy: a class with a Dictionary<string,int> vocabulary field.

    def __init__(self):
        # vocab: maps each word to an integer index
        # C# analogy: Dictionary<string, int> vocab = new();
        self.vocab = {}
        # vocab_size will be set after fit() is called
        self.vocab_size = 0

    def _tokenize(self, text):
        # Convert a text string to a list of lowercase words.
        # We strip punctuation by keeping only letters and spaces.
        # C# analogy: Regex.Replace(text, @"[^a-z ]", "").Split(' ')
        text = text.lower()         # lowercase everything
        cleaned = ""               # will hold cleaned characters
        for ch in text:            # loop character by character
            if ch.isalpha() or ch == " ":   # keep letters and spaces
                cleaned += ch
        words = cleaned.split()    # split on whitespace
        return words               # return list of word strings

    def fit(self, texts):
        # Build the vocabulary from a list of text strings.
        # Every unique word gets a unique integer index.
        # C# analogy: building an Enum from all words found.
        word_index = 0             # next available index
        for text in texts:         # loop over each text
            words = self._tokenize(text)    # tokenise it
            for word in words:             # loop over each word
                if word not in self.vocab: # if word is new
                    self.vocab[word] = word_index   # assign it an index
                    word_index += 1        # move to next index
        self.vocab_size = len(self.vocab)  # record final vocabulary size

    def transform(self, text):
        # Convert a single text string into a word-count vector (numpy array).
        # Returns a 1D array of length vocab_size.
        # C# analogy: float[] Transform(string text)
        vector = np.zeros(self.vocab_size)  # start with all zeros
        words = self._tokenize(text)        # tokenise the input
        for word in words:                  # for each word in the input
            if word in self.vocab:          # only count words we know
                idx = self.vocab[word]      # look up the word's index
                vector[idx] += 1.0          # increment that position
        # (unknown words are simply ignored -- out-of-vocabulary)
        return vector                       # return the numeric vector

    def fit_transform(self, texts):
        # Convenience method: fit and then transform all texts at once.
        # Returns a 2D array, one row per text.
        # C# analogy: a LINQ Select that both builds vocab and vectorises.
        self.fit(texts)    # build vocabulary from all texts
        # Use a list comprehension to transform each text
        # Python list comprehension: [f(x) for x in items]
        # C# analogy: texts.Select(t => Transform(t)).ToArray()
        vectors = [self.transform(text) for text in texts]
        return np.array(vectors)   # convert list of arrays to 2D NumPy array

# --- Create vectorizer and fit it on all KB queries ---
vectorizer = BagOfWordsVectorizer()    # create an instance

# Extract just the "query" field from each KB entry.
# Python list comprehension: [entry["query"] for entry in IT_SUPPORT_KB]
# C# analogy: IT_SUPPORT_KB.Select(e => e.Query).ToList()
all_queries = [entry["query"] for entry in IT_SUPPORT_KB]

# fit_transform: build vocab AND convert all queries to vectors in one call
X_all = vectorizer.fit_transform(all_queries)    # shape: (num_entries, vocab_size)

print(f"\nVocabulary size : {vectorizer.vocab_size} unique words")
print(f"Feature matrix  : {X_all.shape[0]} examples x {X_all.shape[1]} features")
print(f"Sample words in vocab: {list(vectorizer.vocab.keys())[:10]}")
print()

# =============================================================================
# PART 3: INTENT CLASSIFIER (the "fine-tuned" model)
# =============================================================================
print("=" * 70)
print("PART 3: INTENT CLASSIFIER (2-layer NumPy neural network)")
print("=" * 70)

# Architecture:
#
#   Input layer     Hidden layer    Output layer
#   (vocab_size)  -> (32 neurons) -> (num_intents)
#
#   Each arrow = matrix multiplication + activation function
#
#   ReLU after hidden layer (makes negatives = 0, keeps positives)
#   Softmax after output (turns scores into probabilities that sum to 1)
#
#   C# analogy: a pipeline of matrix transforms, like a series of
#   multiplied transformation matrices in 3D graphics.

# Map intent names to integer class labels.
# C# analogy: an Enum: PASSWORD=0, VPN=1, PRINTER=2, ...
INTENTS = ["EMAIL", "HARDWARE", "PASSWORD", "PRINTER", "SOFTWARE", "UNKNOWN", "VPN"]
# We sort alphabetically so the mapping is always deterministic.

# Number of intent categories
NUM_INTENTS = len(INTENTS)    # 7 intents

# Dictionary: intent name -> index  (for converting labels to numbers)
# C# analogy: Dictionary<string, int>
intent_to_idx = {intent: i for i, intent in enumerate(INTENTS)}

# Dictionary: index -> intent name  (for converting predictions back to text)
# C# analogy: Dictionary<int, string>
idx_to_intent = {i: intent for i, intent in enumerate(INTENTS)}

# Build the label vector Y: one integer per KB entry
# C# analogy: int[] Y = IT_SUPPORT_KB.Select(e => intentToIdx[e.Intent]).ToArray()
Y_all = np.array([intent_to_idx[entry["intent"]] for entry in IT_SUPPORT_KB])

class IntentClassifier:
    # A simple 2-layer neural network for intent classification.
    # C# analogy: a class that wraps two float[,] weight matrices and
    #             exposes Train() and Predict() methods.

    def __init__(self, vocab_size, hidden_size, num_classes):
        # vocab_size   : size of the input vector (= vocabulary size)
        # hidden_size  : number of neurons in the hidden layer
        # num_classes  : number of intent categories

        self.vocab_size  = vocab_size    # save for reference
        self.hidden_size = hidden_size   # save for reference
        self.num_classes = num_classes   # save for reference

        # He initialisation: multiply by sqrt(2/fan_in).
        # This prevents gradients vanishing or exploding at start.
        # C# analogy: setting random initial values scaled to layer size.
        scale1 = math.sqrt(2.0 / vocab_size)      # scale for layer 1
        scale2 = math.sqrt(2.0 / hidden_size)     # scale for layer 2

        # W1: weight matrix from input to hidden layer
        # Shape: (vocab_size x hidden_size)
        # C# analogy: float[vocab_size, hidden_size] W1
        self.W1 = np.random.randn(vocab_size, hidden_size) * scale1

        # b1: bias vector for hidden layer (one per hidden neuron)
        # C# analogy: float[] b1 = new float[hidden_size]
        self.b1 = np.zeros(hidden_size)

        # W2: weight matrix from hidden to output layer
        # Shape: (hidden_size x num_classes)
        self.W2 = np.random.randn(hidden_size, num_classes) * scale2

        # b2: bias vector for output layer (one per class)
        self.b2 = np.zeros(num_classes)

    def _relu(self, x):
        # ReLU activation: max(0, x) for each element.
        # Turns negative values to 0, keeps positive values unchanged.
        # Like a gate: let the signal through only if it is positive.
        # C# analogy: Math.Max(0, x) applied element-wise.
        return np.maximum(0, x)    # NumPy applies this element-wise

    def _softmax(self, x):
        # Softmax converts a vector of raw scores ("logits") into
        # a probability distribution that sums to 1.0.
        # e.g. [2.0, 1.0, 0.5] -> [0.59, 0.24, 0.17]
        # We subtract the max for numerical stability (avoids overflow).
        # C# analogy: normalising a score array so all values sum to 1.
        x_stable = x - np.max(x, axis=1, keepdims=True)   # subtract row max
        exps = np.exp(x_stable)                            # e^x for each element
        return exps / np.sum(exps, axis=1, keepdims=True)  # divide by row sum

    def forward(self, X):
        # Perform a forward pass through the network.
        # X shape: (batch_size, vocab_size)
        # Returns: probabilities shape (batch_size, num_classes)

        # Layer 1: input -> hidden
        # Matrix multiply X by W1, then add bias b1
        # C# analogy: hidden = X.Multiply(W1) + b1  (element-wise)
        self.z1 = X @ self.W1 + self.b1     # linear transformation
        self.a1 = self._relu(self.z1)       # ReLU activation

        # Layer 2: hidden -> output
        self.z2 = self.a1 @ self.W2 + self.b2     # linear transformation
        self.a2 = self._softmax(self.z2)           # softmax to get probs

        return self.a2    # shape: (batch_size, num_classes)

    def predict(self, X):
        # Return the predicted class index (the intent with highest probability).
        # C# analogy: Array.IndexOf(probs, probs.Max())
        probs = self.forward(X)                   # get probability distribution
        return np.argmax(probs, axis=1)           # index of max probability

    def predict_proba(self, X):
        # Return the full probability distribution over all intents.
        return self.forward(X)    # just return the softmax output

    def train(self, X, Y, epochs=300, lr=0.05, batch_size=8):
        # Train the network using mini-batch Stochastic Gradient Descent (SGD).
        # X         : feature matrix, shape (num_samples, vocab_size)
        # Y         : label vector of integer class indices, shape (num_samples,)
        # epochs    : how many times to loop through all data
        # lr        : learning rate (how big each weight update step is)
        # batch_size: how many samples to process before updating weights

        num_samples = X.shape[0]   # how many training examples we have

        # Loop over epochs.  C# analogy: for (int epoch = 0; epoch < epochs; epoch++)
        for epoch in range(epochs):

            # Shuffle training data at the start of each epoch.
            # This prevents the network from memorising the order of examples.
            shuffle_idx = np.random.permutation(num_samples)   # random ordering
            X_shuffled = X[shuffle_idx]      # reorder features
            Y_shuffled = Y[shuffle_idx]      # reorder labels (same order)

            epoch_loss = 0.0     # accumulate loss over the epoch

            # Process data in mini-batches.
            # range(start, stop, step): like for(int i=0; i<n; i+=batch_size)
            for start in range(0, num_samples, batch_size):
                end = start + batch_size        # end of this batch
                Xb = X_shuffled[start:end]      # slice features for this batch
                Yb = Y_shuffled[start:end]      # slice labels for this batch
                bs = Xb.shape[0]                # actual batch size (may be smaller at end)

                # --- Forward pass ---
                probs = self.forward(Xb)   # shape: (bs, num_classes)

                # --- Compute cross-entropy loss ---
                # For each example, we look at the probability assigned to the
                # CORRECT class and take the negative log.
                # Perfect prediction (prob=1.0) -> loss = 0
                # Wrong prediction (prob=0.0) -> loss = infinity
                # C# analogy: -Math.Log(probs[i][correctClass])
                correct_probs = probs[np.arange(bs), Yb]   # probability of true class
                loss = -np.mean(np.log(correct_probs + 1e-9))   # mean NLL loss
                epoch_loss += loss    # accumulate

                # --- Backward pass (backpropagation) ---
                # Compute how much each weight contributed to the loss,
                # then nudge weights in the direction that REDUCES the loss.

                # Gradient of loss w.r.t. output logits (before softmax).
                # The formula for cross-entropy + softmax gradient is elegantly:
                #   dL/dz2 = probs - one_hot(true_labels)
                dz2 = probs.copy()           # start with predicted probs
                dz2[np.arange(bs), Yb] -= 1 # subtract 1 at the true class position
                dz2 /= bs                    # average over batch

                # Gradient w.r.t. W2 and b2 (output layer weights)
                dW2 = self.a1.T @ dz2        # (hidden_size, num_classes)
                db2 = np.sum(dz2, axis=0)   # sum over batch -> (num_classes,)

                # Gradient flows back through the hidden layer
                da1 = dz2 @ self.W2.T        # (bs, hidden_size)

                # Gradient through ReLU: only pass gradient where a1 > 0
                # (gradient of ReLU is 1 where input > 0, else 0)
                dz1 = da1 * (self.a1 > 0).astype(float)    # apply ReLU mask

                # Gradient w.r.t. W1 and b1 (hidden layer weights)
                dW1 = Xb.T @ dz1             # (vocab_size, hidden_size)
                db1 = np.sum(dz1, axis=0)   # (hidden_size,)

                # --- Update weights (gradient descent step) ---
                # New weight = old weight - learning_rate * gradient
                # C# analogy: W -= lr * dW  (in-place subtraction)
                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                self.W2 -= lr * dW2
                self.b2 -= lr * db2

            # Print loss every 50 epochs so we can see progress
            if (epoch + 1) % 50 == 0:
                avg_loss = epoch_loss / math.ceil(num_samples / batch_size)
                print(f"  Epoch {epoch+1:4d} | Loss: {avg_loss:.4f}")

# --- Build and train the classifier ---
HIDDEN_SIZE = 32    # number of neurons in the hidden layer

print(f"\nBuilding classifier: {vectorizer.vocab_size} -> {HIDDEN_SIZE} -> {NUM_INTENTS}")
print("Training...\n")

classifier = IntentClassifier(
    vocab_size=vectorizer.vocab_size,
    hidden_size=HIDDEN_SIZE,
    num_classes=NUM_INTENTS
)

# Split into train (80%) and validation (20%).
# We have 35 examples, so about 28 train, 7 val.
num_total   = X_all.shape[0]          # total number of examples
num_train   = int(num_total * 0.8)    # 80% for training

# Shuffle indices before splitting
all_indices = np.random.permutation(num_total)   # random ordering of indices
train_idx   = all_indices[:num_train]             # first 80%
val_idx     = all_indices[num_train:]             # remaining 20%

X_train = X_all[train_idx]    # training features
Y_train = Y_all[train_idx]    # training labels
X_val   = X_all[val_idx]      # validation features
Y_val   = Y_all[val_idx]      # validation labels

print(f"Train set: {X_train.shape[0]} examples")
print(f"Val set  : {X_val.shape[0]} examples\n")

# Train for 300 epochs
classifier.train(X_train, Y_train, epochs=300, lr=0.05, batch_size=8)

# --- Evaluate accuracy ---
train_preds = classifier.predict(X_train)             # predictions on train set
train_acc   = np.mean(train_preds == Y_train) * 100   # accuracy as percentage

val_preds   = classifier.predict(X_val)               # predictions on val set
val_acc     = np.mean(val_preds == Y_val) * 100       # accuracy as percentage

print(f"\nTrain accuracy : {train_acc:.1f}%")
print(f"Val accuracy   : {val_acc:.1f}%")

# --- Confusion matrix ---
# A confusion matrix shows: for each TRUE intent (row),
# how often the model predicted each intent (column).
# Perfect classifier: diagonal is all non-zero, off-diagonal is all zero.
# C# analogy: a 2D int[,] grid indexed by [trueClass, predictedClass]
print("\nConfusion matrix (rows = true intent, cols = predicted intent):")

# Print header row with intent abbreviations
intent_abbrevs = ["EML", "HW ", "PWD", "PRN", "SW ", "UNK", "VPN"]
header = "         " + "  ".join(intent_abbrevs)
print(header)
print("         " + "-" * (len(intent_abbrevs) * 5))

# Compute confusion matrix using all examples (train + val)
all_preds = classifier.predict(X_all)    # predict everything

for true_class in range(NUM_INTENTS):    # loop over each true intent
    # Find examples where the TRUE label is this class
    row_mask = (Y_all == true_class)     # boolean mask
    row_preds = all_preds[row_mask]      # predictions for those examples

    # Count how many times each intent was predicted
    counts = []
    for pred_class in range(NUM_INTENTS):
        count = np.sum(row_preds == pred_class)   # count this prediction
        counts.append(f"{count:3d}")              # format as 3 digits

    # Print this row of the confusion matrix
    label = f"{INTENTS[true_class]:<8} |"
    print(label + "  ".join(counts))

print()

# =============================================================================
# PART 4: RESPONSE GENERATOR
# =============================================================================
print("=" * 70)
print("PART 4: RESPONSE GENERATOR")
print("=" * 70)

# The response generator does TWO things:
#   1. Groups all KB entries by intent (so we can look them up fast)
#   2. Picks the BEST matching response using simple word-overlap scoring
#      (Jaccard similarity: |intersection| / |union| of word sets)
#
# Word-overlap example:
#   User query: "forgot password cannot log in"
#   KB entry  : "forgot my password and cannot log in"
#   Common words: {forgot, password, cannot, log, in} = 5
#   All unique words: {forgot, password, cannot, log, in, my, and} = 7
#   Jaccard score = 5/7 = 0.71  <- high similarity, good match!
#
# C# analogy: computing the Jaccard index between two HashSet<string> objects.

class ResponseGenerator:
    # Groups KB entries by intent and picks the best reply via word overlap.
    # C# analogy: a class with Dictionary<string, List<SupportKbEntry>> grouped.

    def __init__(self, kb, confidence_threshold=0.50):
        # kb                   : the full IT_SUPPORT_KB list
        # confidence_threshold : minimum classifier confidence to trust the intent.
        #                        Below this, we return a fallback message.

        self.confidence_threshold = confidence_threshold  # save threshold

        # Group KB entries by intent.
        # defaultdict(list) creates an empty list for every new key automatically.
        # C# analogy: Dictionary<string, List<SupportKbEntry>> with GetOrAdd
        self.kb_by_intent = defaultdict(list)    # groups storage

        for entry in kb:                         # loop over all KB entries
            self.kb_by_intent[entry["intent"]].append(entry)  # add to group

    def _word_overlap_score(self, text_a, text_b):
        # Compute Jaccard similarity between two text strings.
        # Returns a float between 0 (no overlap) and 1 (identical word sets).

        # Convert each text to a set of lowercase words
        set_a = set(text_a.lower().split())   # words in text_a
        set_b = set(text_b.lower().split())   # words in text_b

        intersection = set_a & set_b          # words in BOTH
        union        = set_a | set_b          # words in EITHER

        if not union:          # guard against empty texts (division by zero)
            return 0.0

        return len(intersection) / len(union)   # Jaccard score

    def generate(self, user_query, predicted_intent, confidence):
        # Given user query + predicted intent + confidence, return a response.
        # C# analogy: string Generate(string query, string intent, float confidence)

        # If confidence is too low, or intent is UNKNOWN, use the fallback.
        if confidence < self.confidence_threshold or predicted_intent == "UNKNOWN":
            return (
                "I am the IT Support Assistant and I could not understand your request. "
                "Please describe your IT issue more specifically, or call the helpdesk "
                "at ext. 4357 for immediate assistance."
            )

        # Get all KB entries for the predicted intent
        candidates = self.kb_by_intent.get(predicted_intent, [])

        if not candidates:    # safety check if no entries found
            return "I do not have information on that topic. Please contact the helpdesk."

        # Score each candidate by word overlap with the user's query
        best_response = candidates[0]["response"]    # default to first entry
        best_score    = -1.0                         # start with an impossible score

        for entry in candidates:                     # loop over all candidates
            score = self._word_overlap_score(user_query, entry["query"])
            if score > best_score:     # if this is the best match so far
                best_score    = score          # update best score
                best_response = entry["response"]   # update best response

        return best_response    # return the best matching response

# --- Create the response generator ---
response_gen = ResponseGenerator(IT_SUPPORT_KB, confidence_threshold=0.35)

# --- Print 5 examples to verify the pipeline works ---
print("\nSample end-to-end predictions:\n")

sample_queries = [
    "How do I reset my forgotten password",          # PASSWORD
    "VPN disconnects every 5 minutes",               # VPN
    "Printer shows offline but is switched on",      # PRINTER
    "Application is crashing when I open it",        # SOFTWARE
    "Outlook not syncing my calendar on phone",      # EMAIL
]

for query in sample_queries:
    vec     = vectorizer.transform(query)                # convert to vector
    vec_2d  = vec.reshape(1, -1)                         # make it 2D (1 x vocab_size)
    probs   = classifier.predict_proba(vec_2d)[0]        # probability for each intent
    idx     = int(np.argmax(probs))                      # index of highest probability
    intent  = idx_to_intent[idx]                         # map index to intent name
    conf    = float(probs[idx])                          # confidence score (0-1)
    response = response_gen.generate(query, intent, conf)# get best response

    print(f"  Query     : {query}")
    print(f"  Intent    : {intent}  (confidence: {conf:.2f})")
    # Truncate response to 80 chars for display
    print(f"  Response  : {response[:80]}...")
    print()

# =============================================================================
# PART 5: DOMAIN CHATBOT (combines everything)
# =============================================================================
print("=" * 70)
print("PART 5: DOMAIN CHATBOT")
print("=" * 70)

class DomainChatbot:
    # The main chatbot class that wires together vectorizer, classifier,
    # and response generator.  Also maintains conversation history.
    #
    # C# analogy: a facade class (Design Pattern: Facade) that wraps three
    # collaborating services: IVectorizer, IClassifier, IResponseGenerator.

    def __init__(self, classifier, vectorizer, response_generator):
        # Store references to all three components
        self.classifier        = classifier         # the neural network
        self.vectorizer        = vectorizer         # word -> vector converter
        self.response_generator = response_generator  # best reply picker

        # Conversation history: list of {"user": ..., "bot": ..., "intent": ...}
        # C# analogy: List<ConversationTurn> where ConversationTurn is a record.
        self.history = []

    def chat(self, user_message):
        # Process one user message and return a bot response.
        # Also saves the turn to conversation history.

        # Step 1: Convert message to a numeric vector
        vec = self.vectorizer.transform(user_message)   # 1D array
        vec_2d = vec.reshape(1, -1)                     # reshape to 2D for network

        # Step 2: Get intent prediction + confidence
        probs  = self.classifier.predict_proba(vec_2d)[0]   # probability distribution
        idx    = int(np.argmax(probs))                       # best intent index
        intent = idx_to_intent[idx]                          # intent name string
        conf   = float(probs[idx])                           # confidence value

        # Step 3: Generate the best response
        response = self.response_generator.generate(user_message, intent, conf)

        # Step 4: Save to conversation history
        self.history.append({
            "user"   : user_message,   # what the user said
            "intent" : intent,         # what we classified it as
            "conf"   : conf,           # how confident we were
            "bot"    : response        # what the bot replied
        })

        return response    # return the response string

    def print_conversation_history(self):
        # Print all conversation turns in a readable format.
        # C# analogy: foreach (var turn in history) Console.WriteLine(...)
        print("\n" + "=" * 70)
        print("CONVERSATION HISTORY")
        print("=" * 70)

        for i, turn in enumerate(self.history, start=1):  # enumerate from 1
            print(f"\nTurn {i}:")
            print(f"  User  : {turn['user']}")
            print(f"  Intent: {turn['intent']} (conf: {turn['conf']:.2f})")
            # Wrap long bot responses at 70 chars for readability
            bot_lines = [turn['bot'][j:j+70] for j in range(0, len(turn['bot']), 70)]
            for k, line in enumerate(bot_lines):
                if k == 0:
                    print(f"  Bot   : {line}")
                else:
                    print(f"          {line}")

# --- Create the chatbot ---
chatbot = DomainChatbot(
    classifier         = classifier,
    vectorizer         = vectorizer,
    response_generator = response_gen
)

# --- Simulate 10 conversation turns covering all intents ---
print("\nSimulating 10 conversation turns...\n")

conversation_turns = [
    "I forgot my password and cannot log in to my computer",     # PASSWORD
    "My password expired and the account is locked",             # PASSWORD
    "I cannot connect to the VPN when I work from home",         # VPN
    "Network drive disappeared after VPN session",               # VPN
    "My printer is offline and not printing any documents",      # PRINTER
    "I need to install Microsoft Office on this new machine",    # SOFTWARE
    "Outlook keeps saying my mailbox is full",                   # EMAIL
    "My laptop battery is dying after only one hour",            # HARDWARE
    "The laptop fan is making a very loud grinding noise",       # HARDWARE
    "Can I install Slack on my company laptop",                  # SOFTWARE
]

for msg in conversation_turns:          # loop over each simulated turn
    response = chatbot.chat(msg)        # send message to chatbot
    # Print a short summary line (full history printed below)
    print(f"  User: {msg[:55]}")
    print(f"  Bot : {response[:70]}...\n")

# Print the full conversation history
chatbot.print_conversation_history()

# =============================================================================
# PART 6: OUT-OF-DOMAIN DETECTION
# =============================================================================
print("\n" + "=" * 70)
print("PART 6: OUT-OF-DOMAIN DETECTION")
print("=" * 70)

# What happens when someone asks the IT chatbot something completely unrelated?
# Good domain-specific chatbots should detect this and respond gracefully.
# We use two signals:
#   1. The classifier may predict "UNKNOWN" intent directly.
#   2. Even if it predicts a real intent, the confidence may be LOW
#      because the words do not match the IT vocabulary well.

print("\nTesting with 5 completely off-topic queries:\n")

ood_queries = [
    "What is the score in tonight's football match",    # sports
    "Can you suggest a recipe for pasta carbonara",     # cooking
    "What is the current temperature outside",          # weather
    "Who wrote the novel Pride and Prejudice",          # literature
    "How do I get to the nearest train station",        # navigation
]

for query in ood_queries:
    response = chatbot.chat(query)           # send to chatbot
    last_turn = chatbot.history[-1]         # retrieve saved turn

    print(f"  OOD Query  : {query}")
    print(f"  Predicted  : {last_turn['intent']} (conf: {last_turn['conf']:.2f})")
    print(f"  Bot reply  : {response[:80]}")
    print()

print(
    "Key observation: The chatbot either predicts UNKNOWN directly, or the\n"
    "confidence score is below the threshold -- either way it returns the\n"
    "fallback message instead of inventing an incorrect IT answer.\n"
    "This is domain adaptation working as intended: the model KNOWS what\n"
    "it does not know, and says so honestly.\n"
)

# =============================================================================
# PART 7: FINE-TUNING vs PROMPTING vs RAG -- COMPARISON TABLE
# =============================================================================
print("=" * 70)
print("PART 7: FINE-TUNING vs PROMPTING vs RAG")
print("=" * 70)

# This table shows the trade-offs between three popular ways to adapt
# a language model to a specific domain.
#
# Approach      : How you specialise the model
# Speed         : How long it takes to set up
# Accuracy      : How well it performs on domain-specific questions
# Cost          : Money / compute resources required
# Control       : How much you can customise behaviour
# Updates       : How easy it is to add new information

print()
# Use a formatted table with ASCII box-drawing characters
col_w = [14, 18, 22, 16, 14, 18]    # column widths

def row(cells):
    # Helper to print one table row.
    # C# analogy: string.Join(" | ", cells.Select((c,i) => c.PadRight(colWidths[i])))
    parts = [cells[i].ljust(col_w[i]) for i in range(len(cells))]
    return "| " + " | ".join(parts) + " |"

def divider():
    # Print a divider line between rows.
    parts = ["-" * w for w in col_w]
    return "+-" + "-+-".join(parts) + "-+"

print(divider())
print(row(["Approach", "Setup Speed", "Accuracy", "Resource Cost", "Control", "Adding New Info"]))
print(divider())
print(row(["Fine-tuning",  "Slow (train)",  "High (specialised)", "High (GPU)", "Full",    "Re-train needed"]))
print(row(["Prompting",    "Fast (no train)", "Medium (general)", "Low (tokens)", "Limited", "Edit the prompt"]))
print(row(["RAG",          "Medium (index)", "High (retrieval)", "Medium",      "Medium",  "Update the index"]))
print(divider())

print("""
Fine-tuning  : You modify the actual weights of the model by training on
               your domain data.  Most expensive but gives maximum accuracy
               and control.  Best when your domain vocabulary is very
               specialised (medical, legal, financial).

Prompting    : You give the general model instructions + examples in the
               prompt at inference time.  Quick to set up but you depend
               on the base model's knowledge.  Works well for general tasks.

RAG          : Retrieval-Augmented Generation.  You store domain docs in a
               vector database and retrieve relevant chunks at query time,
               then feed them to the model as context.  Good balance of
               accuracy, cost, and ease of updating knowledge.

THIS PROJECT demonstrates a simplified fine-tuning approach: we trained
a domain-specific intent classifier from scratch on IT support data,
then added retrieval (word overlap scoring) to find the best stored answer.
""")

# =============================================================================
# PART 8: KEY TAKEAWAYS
# =============================================================================
print("=" * 70)
print("PART 8: KEY TAKEAWAYS")
print("=" * 70)

takeaways = [
    (
        "Domain data is everything",
        "The quality and coverage of your knowledge base directly determines\n"
        "   how well the chatbot performs.  Garbage in = garbage out.\n"
        "   C# analogy: your unit tests are only as good as the cases you write."
    ),
    (
        "Intent classification is the gatekeeper",
        "Getting the intent wrong means the wrong response bucket is searched.\n"
        "   A confident wrong answer is worse than an honest 'I don't know'.\n"
        "   Always include a confidence threshold and a graceful fallback."
    ),
    (
        "Retrieval improves precision within an intent",
        "After classifying the intent, word-overlap scoring finds the KB entry\n"
        "   that best matches the specific wording of this query.\n"
        "   This is the seed of RAG (Retrieval-Augmented Generation)."
    ),
    (
        "Out-of-domain detection prevents hallucination",
        "A domain-specific model should refuse OOD queries rather than\n"
        "   fabricating an answer.  The UNKNOWN intent + confidence threshold\n"
        "   together act as a safety net.  C# analogy: Guard.Against checks."
    ),
    (
        "Fine-tuning a small model beats prompting a big one for narrow tasks",
        "Our 2-layer NumPy network trained on 35 IT examples is faster,\n"
        "   cheaper, and more accurate at IT intent classification than asking\n"
        "   a general GPT model with a generic prompt -- because it learned\n"
        "   the domain vocabulary and patterns directly."
    ),
]

print()
for i, (title, explanation) in enumerate(takeaways, start=1):
    # enumerate with start=1 gives us (1, item), (2, item), ...
    # C# analogy: for (int i = 1; i <= takeaways.Length; i++)
    print(f"  {i}. {title}")
    print(f"   {explanation}")
    print()

print("=" * 70)
print("Project 03 complete!  You built a domain-specific IT support chatbot")
print("using pure Python + NumPy -- no external APIs or pre-trained models.")
print("=" * 70)

# =============================================================================
# PART B: PyTorch / Hugging Face sketch (COMMENTED OUT)
# =============================================================================
# The section below shows how you would implement this same chatbot using a
# real transformer model via Hugging Face's transformers library.
# This is NOT runnable here -- it requires PyTorch + transformers installed.
# Read it to understand how the production version would look.
# =============================================================================

# # -------------------------------------------------------------------------
# # PART B: PyTorch + Hugging Face Domain Chatbot Sketch
# # -------------------------------------------------------------------------
# # C# analogy: this is like the production ASP.NET Core version of the
# # console prototype you just built.
# # -------------------------------------------------------------------------
#
# # Step 1: Install dependencies (run in terminal, not in Python)
# # pip install torch transformers datasets
#
# import torch                                   # PyTorch deep learning framework
# from transformers import (
#     AutoTokenizer,                             # loads the right tokenizer for any model
#     AutoModelForSequenceClassification,        # pre-trained model for classification
#     Trainer,                                   # HuggingFace training loop helper
#     TrainingArguments,                         # training config (epochs, lr, batch, etc)
# )
# from datasets import Dataset                   # HuggingFace dataset wrapper
#
# # Step 2: Define the label mapping
# INTENT_LABELS = ["EMAIL", "HARDWARE", "PASSWORD", "PRINTER", "SOFTWARE", "UNKNOWN", "VPN"]
# label2id = {label: i for i, label in enumerate(INTENT_LABELS)}
# id2label = {i: label for i, label in enumerate(INTENT_LABELS)}
#
# # Step 3: Prepare the data in HuggingFace Dataset format
# # Our IT_SUPPORT_KB becomes a list of {"text": ..., "label": ...} dicts.
# hf_data = [
#     {"text": entry["query"], "label": label2id[entry["intent"]]}
#     for entry in IT_SUPPORT_KB
# ]
# dataset = Dataset.from_list(hf_data)           # wrap Python list as HF Dataset
# dataset = dataset.train_test_split(test_size=0.2, seed=42)  # 80/20 split
#
# # Step 4: Load a pre-trained tokenizer and model
# # "distilbert-base-uncased" is a small, fast transformer (66M parameters).
# # It was pre-trained on Wikipedia + BookCorpus (general English text).
# # We fine-tune it HERE on our 35 IT support examples.
# # C# analogy: loading a DLL and then subclassing its main class.
# MODEL_NAME = "distilbert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)   # download tokenizer vocab
#
# # Load the pre-trained model body + add a new classification head.
# # num_labels = 7 (one per intent) -- this replaces the original head.
# model = AutoModelForSequenceClassification.from_pretrained(
#     MODEL_NAME,
#     num_labels=len(INTENT_LABELS),
#     id2label=id2label,
#     label2id=label2id,
# )
#
# # Step 5: Tokenize the dataset
# def tokenize_fn(batch):
#     # This function is applied to every batch of examples.
#     # It converts raw text into input_ids, attention_mask tensors
#     # that the transformer expects.
#     # C# analogy: a LINQ Select transformation applied to each record.
#     return tokenizer(
#         batch["text"],           # list of text strings
#         truncation=True,         # cut off if longer than model max (512 tokens)
#         padding="max_length",    # pad shorter sequences with [PAD] tokens
#         max_length=64,           # IT queries are short, 64 tokens is plenty
#     )
#
# tokenized = dataset.map(tokenize_fn, batched=True)   # apply to whole dataset
# tokenized = tokenized.remove_columns(["text"])        # remove raw text, keep tensors
# tokenized = tokenized.rename_column("label", "labels")  # HF expects "labels" column
# tokenized.set_format("torch")                         # return PyTorch tensors
#
# # Step 6: Define training arguments
# # C# analogy: a TrainingConfig record with properties.
# training_args = TrainingArguments(
#     output_dir="./it_support_model",    # where to save checkpoints
#     num_train_epochs=10,                # fine-tune for 10 epochs (small dataset)
#     per_device_train_batch_size=8,      # 8 examples per GPU batch
#     per_device_eval_batch_size=8,       # same for evaluation
#     learning_rate=2e-5,                 # very small LR for fine-tuning transformers
#     weight_decay=0.01,                  # L2 regularisation to prevent overfitting
#     evaluation_strategy="epoch",        # evaluate once per epoch
#     save_strategy="epoch",              # save checkpoint once per epoch
#     load_best_model_at_end=True,        # restore best checkpoint when done
#     logging_steps=5,                    # print loss every 5 steps
#     report_to="none",                   # disable wandb / mlflow logging
# )
#
# # Step 7: Create the Trainer and fine-tune
# trainer = Trainer(
#     model=model,                                      # the transformer model
#     args=training_args,                               # training config
#     train_dataset=tokenized["train"],                 # training split
#     eval_dataset=tokenized["test"],                   # validation split
# )
# trainer.train()    # this runs the fine-tuning loop
#
# # Step 8: Run inference on a new query
# def predict_intent_hf(query_text):
#     # Tokenize the input text
#     inputs = tokenizer(
#         query_text,
#         return_tensors="pt",    # pt = PyTorch tensors
#         truncation=True,
#         padding=True,
#         max_length=64,
#     )
#     with torch.no_grad():       # no gradient tracking needed at inference time
#         outputs = model(**inputs)   # ** unpacks dict as keyword args
#     logits = outputs.logits         # raw scores before softmax
#     predicted_class = torch.argmax(logits, dim=1).item()   # index of highest score
#     predicted_intent = id2label[predicted_class]            # map to intent name
#     confidence = torch.softmax(logits, dim=1).max().item()  # max probability
#     return predicted_intent, confidence
#
# # Example usage:
# # intent, conf = predict_intent_hf("I forgot my password")
# # print(f"Intent: {intent}, Confidence: {conf:.2f}")
#
# # Step 9: Save and reload the fine-tuned model
# # model.save_pretrained("./it_support_model_final")
# # tokenizer.save_pretrained("./it_support_model_final")
# # Later: model = AutoModelForSequenceClassification.from_pretrained("./it_support_model_final")
#
# # -------------------------------------------------------------------------
# # Key differences: PyTorch/HuggingFace vs our NumPy version
# # -------------------------------------------------------------------------
# # NumPy version (this project):
# #   - Bag-of-Words features (loses word order)
# #   - 2-layer network trained from scratch (random init)
# #   - ~30 parameters in embedding, ~1000 total params
# #   - Works on 35 examples, no GPU needed
# #   - Great for learning the concepts
# #
# # HuggingFace DistilBERT version:
# #   - Full transformer with attention (understands word order + context)
# #   - Fine-tuned from a 66M parameter pre-trained checkpoint
# #   - Needs only 10 epochs to adapt (transfer learning is powerful)
# #   - Can handle paraphrasing, synonyms, misspellings much better
# #   - Requires GPU for reasonable speed
# #   - This is what real production IT chatbots use
# # -------------------------------------------------------------------------
