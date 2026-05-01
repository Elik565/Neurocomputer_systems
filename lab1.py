import os
import re
import torch
import requests
from bs4 import BeautifulSoup
import trafilatura
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    set_seed
)
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')

model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

print(f"Модель: {model_name}")
print(f"Размер словаря: {tokenizer.vocab_size}")
print(f"Максимальная длина: {tokenizer.model_max_length}")

all_texts = [
"""The attention mechanism helps address problems found in the RNN-based encoder-
decoder setup. As illustrated in Fig. 2.2, an attention mechanism is like a memory
bank. When queried, it produces an output based on stored keys and values (Bah-
danau et al., 2014).
Fig. 2.2: The attention mechanism and its interplay among queries, keys, values, and
the resultant output vectors.
Let us consider the memory unit consisting of n key-value pairs (k1 , v1 ), ... , (kn , vn )
with ki ∈ Rdk and vi ∈ Rdv . The attention layer receives an input as query q ∈ Rdq
and returns an output o ∈ Rdv with the same shape as the value v.
The attention layer measures the similarity between the query and the key using
a score function 𝛼, which returns scores a1 , ... , an for keys k1 , ... , kn given by
ai = 𝛼(q, ki )
The score function 𝛼(q, k) exists in various forms, leading to multiple types of
attention mechanisms. The dot product-based scoring function is the simplest, re-
quiring.
2.2.1 Self-Attention
In self-attention, each input vector xi is projected onto three distinct vectors: query
qi , key ki , and value vi . These projections are performed via learnable weight ma-
trices WQ , WK , and WV , resulting in qi = xi Wq , ki = xi Wk , and vi = xi Wv ,
respectively. These weight matrices are initialized randomly and optimized during
training. The simplified matrix representation with each of the query, key, and value
matrices as a single computation is given.
2.3 Transformers
The Transformer model, which was introduced by Vaswani et al. (2017), is a corner-
stone in sequence-to-sequence tasks. The Transformer architecture, shown in Fig.
2.3, employs an encoder-decoder setup, each consisting of multiple identical layers
with the specifics of its essential components discussed in the following section.
2.3.1 Encoder
The encoder is responsible for processing the input sequence and compressing the
information into a context or memory for the decoder. Each encoder layer comprises
three main elements:
Multi-Head Attention: This component allows the model to focus on different
parts of the input for each attention head, thereby capturing various aspects of
the data.
• Feed-Forward Neural Network: A simple yet effective neural network that op-
erates on the attention vectors, applying nonlinear transformation and making it
available for the next encoder layer (and the decoder layer).
• Add & Norm: The Add & Norm layer aids in stabilizing the activations by com-
bining residual connections and layer normalization, ensuring smoother training
and mitigating the vanishing gradient problem in the encoder (and the decoder).
2.3.2 Decoder
The decoder takes the context from the encoder and generates the output sequence.
It is also composed of multiple layers and has many commonalities with the encoder,
but with minor changes:
• Masked Multi-Head Attention: Similar to multi-head attention but with a
masking mechanism to ensure that the prediction for a given word doesn’t de-
pend on future words in the sequence.
• Encoder-Decoder Attention: This layer allows the decoder to focus on relevant
parts of the input sequence, leveraging the context provided by the encoder.
• Feed-Forward Neural Network: Identical in architecture to the one in the en-
coder, this layer further refines the attention vectors in preparation for generating
the output sequence.
Next, we describe various components and sub-components of the Transformer
architecture.
2.3.3 Tokenization and Representation
In Transformer models, tokenization typically converts sentences into a machine-
readable format. This can be done at the level of words or subwords, depending on the
granularity required for the specific application. Each word in the sentence is treated
as a distinct token in word-level tokenization. These tokens are then mapped to their
corresponding vector representations, such as word embeddings, which serve as the
input to the Transformer model. This approach may face limitations when dealing
with out-of-vocabulary words. Subword-level approaches such as byte-pair encod-
ing (BPE) or WordPiece often address the limitations of word-level tokenization. In
these methods, words are broken down into smaller pieces or subwords, providing a
way to represent out-of-vocabulary terms and capture morphological nuances. These
subwords are then mapped to embeddings and fed into the Transformer.
For instance, the word “unhappiness” could be split into subwords such as “un”
and “happiness”. These subwords are then individually mapped to their embeddings.
This method increases the model’s ability to generalize and handle a broader range
of vocabulary, including words not seen during training.
A hybrid approach combining word and subword-level tokenization can also
leverage both. Such a strategy balances the comprehensiveness of subword-level rep-
resentations with the interpretability of word-level tokens.
2.3.4 Positional Encodings
Since the Transformer model processes all tokens in the input sequence in parallel,
it does not have a built-in mechanism to account for the token positions or order.
Positional encoding is introduced to provide the model with information about the
relative positions of the tokens in the sequence. The positional encoding is usually
added to the input embeddings before they are fed into the Transformer model.
If the length of the sentence is given by l and the embedding dimension/depth
is given by d, positional encoding P is a 2-d matrix of the same dimension, i.e.,
P ∈ Rl ×d . Every position can be represented with the equation in terms of i, which
is along the l, and j, which is along the d dimension as
Pi,2j = sin(i/10002j/d )(2.13)
Pi,2j+1 = cos(i/10002j/d )
2.3.5 Multi-Head Attention
Rather than a single self-attention head, multi-head attention employs h parallel
self-attention heads, enhancing the model’s representational capacity. In the original
Transformer model, h = 8 heads were used to allow the model to capture various as-
pects and dependencies within the input data, such as grammar and tense in machine
translation tasks.
Each head operates with its own set of learnable query, key, and value weight ma-
trices in multi-head attention. This results in distinct query, key, and value matrices
and unique output matrices for each head. These output matrices are concatenated
and subsequently linearly transformed using an additional weight matrix. The paral-
lel input-to-output transformations for all the heads are depicted in Fig. 2.5.
headi = attention(WQ i Q, WK i K, WV i V)(2.15)
multihead (Q, K, V) = WO concat(head1 , ... , headh )(2.16)
2.3.6 Position-Wise Feed-Forward Neural Networks
Following the attention mechanism, the next component in the architecture of the
Transformer model is the feed-forward neural network. This network transforms the
attention vectors further, rendering them compatible with the input to the subsequent
encoder or decoder layer. The feed-forward neural network often comprises two lay-
Masked Multi-Head Attention
In the Transformer model, the decoder aims to predict the next token (word or charac-
ter) in the sequence by considering both the encoder’s output and the tokens already
seen in the target sequence. The first layer of the decoder adopts a particular strategy:
it only has access to the tokens that come before the token it is currently trying to
predict. This mechanism is known as masked multi-head attention.
The masking is implemented using a particular weight matrix M. In this matrix,
entries corresponding to future tokens in the sequence are set to −∞, and those for
previous tokens are set to 0.
This masking is applied after calculating the dot product of the Query (Q) and
Key (KT ) matrices but before applying the softmax function. As a result, the softmax
output for future tokens becomes zero, effectively masking them from consideration.
This ensures that the decoder cannot peek into future tokens in the sequence, thereby
preserving the sequential integrity required for tasks such as language translation.
2.3.9 Encoder-Decoder Attention
The encoder-decoder attention mechanism serves as the bridge that connects the en-
coder and the decoder, facilitating the transfer of contextual information from the
source sequence to the target sequence. Conceptually, the encoder-decoder attention
layer works similarly to standard multi-head attention but with a critical difference:
the Queries (Q) come from the current state of the decoder, while the Keys (K) and
Values (V) are sourced from the output of the encoder. This mechanism allows the
model to focus on relevant portions of the source sequence while generating each to-
ken in the target sequence, thus capturing intricate relationships between the source
and target.
2.3.10 Transformer Variants
Numerous Transformer models have emerged, each featuring modifications to the
original Transformer discussed in the previous Sect. (Lin et al., 2022). These alter-
ations can be categorized into three types: architectural changes, pre-training meth-
ods, and applications, as illustrated in Fig. 2.6. We detail in the following sections
key variables between different Transformer variants. A selection are summarized at
the end in Table 2.1.
Normalization Methods
Training instability is challenging in the pre-training phase of LLMs. Normaliza-
tion methods are employed to stabilize training. Initially, BatchNorm was commonly
used but proved inefficient with variable-length sequence and small-batch data. Con-
sequently, LayerNorm (LN) was introduced to perform layer-wise normalization, re-
calculating the mean and variance for each layer’s activations. RMSNorm was later
proposed to enhance the training speed of LayerNorm by rescaling activations using
the root mean square of summed activations, demonstrating improved training speed
and performance in Transformer models.
The original Transformer utilizes full attention, conducting attention pairwise and
considering all token pairs in a sequence. It employs scaled dot-product attention
and multi-head attention, where queries, keys, and values are projected differently
in each head, with the concatenated output of each head forming the final output.
Sparse attention addresses the quadratic computational complexity challenge of full
attention, especially with long sequences.
! Practical Tips
Efficient Transformer variants, like locally banded sparse attention (e.g., Factorized
Attention in GPT-3), allow each query to attend to a subset of tokens based on po-
sitions, reducing complexity. Multi-query attention, where different heads share the
same linear transformation matrices on keys and values, offers computational savings
with minimal impact on model quality. Models such as PaLM and StarCoder utilize
multi-query attention. FlashAttention optimizes the speed and memory consump-
tion of attention modules on GPUs without compromising model quality. It orga-
nizes input into blocks and introduces recomputation to utilize fast memory (SRAM)
on GPUs efficiently. Integrated into platforms such as PyTorch, DeepSpeed, and
Megatron-LM, FlashAttention optimizes attention modules from an IO-aware per-
spective. For optimal generalization and training stability, pre-RMSNorm is recom-
mended for layer normalization, with SwiGLU or GeGLU as the activation function.
It is advised not to use layer normalization immediately after embedding layers to
avoid performance degradation. Some methods, such as Realformer and Predictive
Attention Transformer, reuse attention distributions from previous blocks to guide
the current block, creating more direct paths through the network. Transparent At-
tention eases optimization using a weighted sum of encoder representations from
all layers in cross-attention modules. Adaptive Computation Time (ACT) has been
introduced to tailor computation time based on input difficulty, leading to strategies
such as Universal Transformer and Conditional Computation Transformer, which ei-
ther refine representations iteratively or utilize gating mechanisms to optimize com-
putational resources.
In our network configurations, Sublayer refers to either a feed-forward
neural network (FFN) or a self-attention module within a Transformer layer. The
symbol d represents the size of the hidden states in the network. The position em-
bedding at a specific position i is denoted by pi. In the attention mechanism, Aij
signifies the attention score computed between a given query and its corresponding
key. The difference in positions between the query and the key is represented by ri −j ,
a learnable scalar value. Finally, the term R 𝜃,t refers to a rotary matrix, which rotates
by an angle determined by multiplying t by 𝜃.
The causal decoder architecture is designed for autoregressive tasks where the model
generates the output token by token. This architecture employs a unidirectional at-
tention mechanism, meaning that each token can only attend to previous tokens and
itself during the generation process. This is particularly useful for text generation
tasks where the model needs to generate coherent and contextually appropriate text.
For example, in text completion tasks, the model predicts the next token based on the
previous ones, ensuring that the generated text is coherent and contextually relevant.
Analysis of attention patterns across three primary architectures. In this
context, the blue, green, yellow, and gray rounded shapes represent attention within
prefix tokens, attention between prefix and target tokens, attention among target to-
kens, and masked attention, respectively.
2.5.3.3 Prefix Decoder
The prefix decoder architecture is a variation of the causal decoder where the model
can attend bi-directionally to a prefix of tokens while maintaining unidirectional at-
tention for the rest. This hybrid attention mechanism allows the model to have a
broader context while generating each token, making it effective for tasks that require
understanding both previous and subsequent tokens in a sequence. For instance, the
model can attend to the dialog history and the partially generated response in a dialog
system while generating the next token.
BERT (Encoder)
The Bidirectional Encoder Representation from Transformer (BERT) is a pre-trained
model that employs an attention mechanism to better comprehend linguistic con-
text (Devlin et al., 2019). BERT consists of multiple encoder segments, each con-
tributing to its robustness. Upon its introduction, BERT set new benchmarks for a
range of NLP tasks, such as question answering on the SQuAD v1.1 dataset and
natural language inference on the MNLI dataset. Unlike traditional language models
that process text sequences in a unidirectional manner, BERT’s bidirectional train-
ing approach offers a more comprehensive understanding of linguistic context and
sequence flow.
2.6.1.1 Dataset
BERT’s training data primarily comprise Wikipedia, accounting for approximately
2.5 billion words, and the BooksCorpus, which contains approximately 800 million
words.
2.6.1.2 Architecture
BERT is an encoder-only Transformer and offers various pre-trained models differ-
entiated by their architectural scale. Two examples include
BERT-BASE consists of 12 layers, 768 hidden nodes, 12 attention heads, and
110 million parameters.
• BERT-LARGE is a more extensive version with 24 layers, 1024 hidden nodes,
16 attention heads, and 340 million parameters.
The training of BERT-BASE utilized four cloud TPUs over four days, while
BERT-LARGE required 16 TPUs for the same duration.
2.6.1.3 Training
BERT operates in two phases–pre-training and fine-tuning–as shown in Fig. 2.10.
The model learns from unlabeled data across various tasks in the initial pre-training
phase. During the fine-tuning phase, the model starts with the parameters acquired
from the pre-training and then optimizes these parameters using labeled data specific
to the target tasks.
BERT’s training methodology combines two objectives: the masked language
model (MLM) and next sentence prediction (NSP). The combined loss function of
these techniques is minimized during training. For BERT, each training instance is
a pair of sentences that may or may not be sequential in the original document. The
special tokens [CLS] and [SEP] denote the beginning of the sequence and the sep-
aration between sentences, respectively. A subset of tokens in the training instance
is either masked with a [MASK] token or substituted with a random token. Before
being input into the BERT model, tokens are transformed into embedding vectors.
These vectors are then enhanced with positional encodings, and in BERT’s unique
approach, segment embeddings are added to indicate whether a token belongs to the
first or second sentence.
Once pre-trained, BERT can be adapted for various downstream tasks, whether
for individual texts or pairs of texts. General linguistic representations, derived from
BERT’s 350 million parameters trained on 250 billion tokens, have significantly ad-
vanced the state of the art in numerous NLP tasks. During the fine-tuning process,
additional layers can be incorporated into BERT. These layers and the pre-trained
BERT parameters are updated to align with the training data of specific downstream
tasks. The Transformer encoder, essentially a pre-trained BERT, accepts a sequence
of text and uses the [CLS] representation for predictions. For example, [CLS] is re-
placed with actual classification labels in sentiment analysis or classification tasks.
During this fine-tuning phase, the cross-entropy loss between the predictions and ac-
tual labels is minimized via gradient-based methods. The additional layers are trained
from scratch, and the pre-trained BERT parameters undergo updates.
Three different examples of prompt-based inference for English-to-French
language translation. In each case, the examples and prompts are passed to an LLM,
and the model is allowed to predict the most likely term to come next, in this case
“fromage”, thus accomplishing the prompt task. The three examples from top to
bottom illustrate zero-shot, one-shot, and few-shot inference.
The notion of prompting can be attributed to the work by Kumar et al. (2016),
which introduced the dynamic memory network (DMN). DMN comprises a neural
network architecture designed to process input sequences and questions, establish
episodic memories, and generate pertinent answers (Xiong et al., 2016). Tasks cor-
responding to questions (prompts) initiate an iterative attention mechanism, allowing
the model to concentrate on the inputs and outcomes of previous iterations. Radford
et al. (2019) revealed the potential of this approach for achieving expertise in various
natural language processing tasks without requiring explicit supervision, provided
that the models are trained on adequately extensive datasets.3.1 Introduction
Since these discoveries, a wealth of literature has developed, examining many dif-
ferent approaches and improvements to prompt-based inference and learning. This
chapter will introduce and systematically examine the critical aspects of prompt-
based inference, including the basic procedure, details of prompt shape, prompt opti-
mization, answer space engineering, and practical applications to various NLP tasks.¹
But first, to place prompting in its proper historical context, we will describe two
prominent approaches that have shaped the field in the last few years – supervised
learning and pre-trained model fine-tuning – and distinguish them from prompt-
based learning.      
Owing to the Transformer architecture, prompt tokens can be processed in paral-
lel within the prefill step, which results in relatively high latency (compared to the
decode step) and high compute utilization due to this parallelism. In contrast, the de-
code step is a sequential process in that the next token to be generated in a sequence
of output tokens depends on all previous tokens being generated first. This results in
relatively low per-output-token latency, but also low compute utilization due to the
sequential nature of the process. This means that the number of input tokens within
the prompt should not significantly impact inference latency, while the output length
will. For example, Tab. 8.2⁵ shows the impact of varying the input and output token
lengths on the response latency for OpenAI’s gpt-3.5-turbo model. Increasing
the number of input tokens from 51 to 232 while keeping the number of output to-
kens at 1 results in negligible latency change. However, using a similar input length
but increasing the output token length from 1 to 26 results in an almost 3x latency
increase, illustrating the imbalanced effect of input and output length on inference
latency.
With this imbalance in mind, what attributes of an LLM influence inference la-
tency? The first and most obvious is model size. The simple rule of thumb is that
more parameters result in greater latency. LLMs with more model parameters require
more computation to process inputs and generate outputs. In addition to model size,
model architecture is another important factor. The number of layers, the complexity
of layers, the attention mechanisms used in Transformer blocks, and the number and
location of Transformer blocks within the network influence inference latency.
Another important factor influencing inference latency in LLMs is the numeric
precision with which model parameters are stored. This aspect is discussed in de-
tail within the quantization sections in Chapter 4. However, in the context of open
vs closed-source LLMs, the customization difference between the two categories of
models is important. In the closed-source context, where customization is more re-
strictive, end-user quantization will be limited to whatever the model owner supports.
In contrast, in the open-source context, the end-user of the LLM is typically free to
test and implement whatever quantization approach works best for their use-case.
Since quantization represents a significant opportunity for inference latency decrease
and decreases in the memory and storage costs of running/hosting the LLM, any lack
of customization in closed-source LLMs should be considered strongly. In use cases
where the number of request-response cycles is expected to be low, this might be
less of an issue. Nevertheless, when the number of request-response cycles is high, a
closed-source LLM might become a problematic bottleneck within an application –
for example OpenAI APIs typically have rate-limits that apply to different end-points
and models.
In traditional inference architectures, it is largely up to the client to create
batches. Particularly in applications where users send one request at a time, the GPU
can be much more effectively utilized by dynamically aggregating multiple inputs
on the server. This comes with a latency cost in waiting for more inputs before a
complete batch is formed and computation can begin. Continuous batching addresses
this problem by putting newly received inputs into existing batches alongside other
inputs already in progress.354
8 LLMs in Production
this way, only the weights relating to new tokens must be computed. The attention
mechanism, the Transformer component with the highest order runtime complexity,
is often the largest performance bottleneck in the architecture. The ability to scale
down these computations can considerably increase the inference speed.
Multimodal pre-trained models use a multilayer Transformer architecture to ex-
tract and interact features from various modalities. One way to categorize these archi-
tectures is by their approach to multimodal information integration, distinguishing
them into single-stream and cross-stream types.
• Single-Stream Architecture: Multimodal inputs such as images and text are
treated equally and fused in a unified model. This process involves extracting
unimodal features from each modality, which are then tokenized and concate-
nated using separators, as shown in Fig. 9.2. These concatenated features serve
as inputs to a multimodal Transformer, which is instrumental in the fusion pro-
cess. The multi-head self-attention mechanism facilitates the interactive fusion
of unimodal features, leading to the generation of multimodal fusion features
(Li et al., 2020c). These features are typically derived from the class token of
the Transformer, which encapsulates information from various modalities and
enhances the model’s characterization capabilities.
• Cross-Stream Architecture: In this approach, features of different modalities
are extracted in parallel by independent models and then aligned using self-
supervised contrastive learning (discussed later) as shown in Fig. 9.3. This ap-
proach is distinct from single-stream architectures.
Multimodal Transformers facilitate cross-modal interactions, such as fusion and
alignment, through self-attention mechanisms and their variants. The self-attention
approaches are modality-agnostic, tokenization-agnostic, and embedding-agnostic,
showcasing the versatility of treating any token’s embeddings from any modality.
Given inputs XA and XB from two distinct modalities, Z (A) and Z (B) denote their
respective token embeddings. The following outlines these practices and their math-
ematical formulations in a two-modality context, although they are adaptable to mul-
tiple modalities:
1. Early Summation: Token embeddings from multiple modalities are weighted
and summed at each token position before processing by Transformer layers:
Z ← Tf (𝛼Z (A) ⊕ 𝛽Z (B)) = MHSA(Q (AB), K (AB), V (AB)),
where ⊕ indicates element-wise summation. This method offers simplicity and
effectiveness without increasing computational complexity (Gavrilyuk et al.,
2020).
2. Early Concatenation (Co-Transformer): Token embedding sequences from
different modalities are concatenated:
Z ← Tf (C (Z (A), Z (B))).
This all-attention or Co-Transformer approach allows a unified sequence treat-
ment, enhancing each modality’s encoding by contextualizing with other modal-
ities (Sun et al., 2019).
3. Hierarchical Attention (Multi-stream to One-stream): Independent Trans-
former streams first encode multimodal inputs; their outputs are then concate-
nated and fused:
Z ← Tf3 (C (Tf1 (Z (A)), Tf2 (Z (B)))).
This method represents a form of late interaction or fusion, acting as a particular
case of early concatenation (Li et al., 2021).
4. Hierarchical Attention (One-stream to Multi-stream): Concatenated multi-
modal inputs are encoded by a shared single-stream Transformer, followed by
separate streams for each modality:
C (Z (A), Z (B)) ← Tf1 (C (Z (A), Z (B))),
Z (A) ← Tf2 (Z (A)),
Z (B) ← Tf3 (Z (B)).
This structure, utilized in InterBERT, captures cross-modal interactions while
preserving unimodal representation independence (Lin et al., 2020).
5. Cross-Attention: In two-stream Transformers, exchanging query embeddings
across streams enables enhanced cross-modal interactions:
Z (A) ← MHSA(QB , KA , VA ),
Z (B) ← MHSA(QA , KB , VB ).
First proposed in VilBERT, this method maintains computational efficiency and
fosters cross-modal perception (Lu et al., 2019).
6. Cross-Attention to Concatenation: Cross-attention streams are concatenated
and further processed to model the global context:9.3 Multimodal LLM Framework
383
Z (A) ← MHSA(QB , KA , VA ),
Z (B) ← MHSA(QA , KB , VB ),
Z ← Tf (C (Z (A), Z (B))).
This hierarchical cross-modal interaction approach mitigates the drawbacks of
standalone cross-attention (Zhan et al., 2021).
9.3.3.1 Contrastive Learning
Before CLIP, vision-language models mainly used classifier or language model ob-
jectives. The classifier approach was limited to predefined classes, restricting the
model’s response diversity and adaptability to different tasks. The language model
objective, while more flexible, faced training challenges due to its focus on generat-
ing specific texts for each image.9 Multimodal LLMs
386
Contrastive learning, as implemented in CLIP, aims to overcome the limitations
of previous models by shifting the focus from predicting the exact text for each image
to determining whether a given text is more aptly associated with a specific image
than others (Radford et al., 2021). In practice, for a batch of N image-text pairs, CLIP
generates N text embeddings and N image embeddings. Let V1 , V2 , ... , VN represent
the embeddings for the N images, and L1 , L2 , ... , LN represent the embeddings for
the N texts. CLIP computes the cosine similarity scores for all N 2 possible pairings
of Vi , Lj . The training objective is to maximize the similarity scores for the N correct
pairings while minimizing the scores for the N 2 − N incorrect pairings.
Here, Li2t and Lt2i are image-to-text and text-to-image classification loss functions,
respectively. LCL is the total contrastive loss. Vi and Li represent the normalized
image and text embeddings, respectively. N is the batch size, and 𝜎 is the temperature
parameter.
9.3.3.2 Modality Matching Loss
Modality matching loss (MML) plays a critical role in pre-training large multimodal
models, mainly due to its ability to capture explicit or implicit alignment relation-
ships between different modalities. This loss function is applied in models such as
Unicoder-VL, which employs visual linguistic matching (VLM) for vision-language
pre-training (Li et al., 2020a). The VLM approach involves extracting both positive
and negative image-sentence pairs and training the model to discern whether these
pairs are aligned. The objective is to predict the matching scores of given sample
Here, (x , y ) represents the positive image-sentence pairs, and (x ′ , y ′ ) denotes
the negative pairs. The model predicts the probability p(aligned|x , y ) that a pair is
aligned and p(unaligned|x ′ , y ′ ) that it is not.
InterBERT introduces this variation with image-text matching using hard neg-
atives, termed ITM-hn (Lin et al., 2020). This approach selects negative samples9.3 Multimodal LLM Framework
Including hard negatives, identified by high TF-IDF similarity scores, makes
learning more challenging and effective, as the model must discern between closely
related but unaligned pairs.
9.3.3.3 Masked Language Modeling
Masked language modeling (MLM) is a prevalent objective in pre-training frame-
works, where researchers typically mask and fill input words randomly using spe-
cial tokens. This method leverages the context from surrounding words and as-
sociated image regions to predict the masked words. In SIMVLM, as developed
by Wang et al. (2021), this approach is combined with prefix language modeling
(PrefixLM). PrefixLM applies bidirectional attention to a prefix sequence and auto-
regressive factorization for the subsequent tokens. In this context, words are denoted
as w = {x1 , ... , xK } and image regions as v = {v1 , ... , vT }. For MLM, a certain
percentage p% of input words, represented as xm , are masked at randomly generated
indices m. The objective is to predict these masked words using the unmasked words
x¬m and all image regions v , by minimizing the negative log-likelihood:
LMLM (𝜃) = −E (x ,v ) log P 𝜃 (xm |x¬m , v ),
(9.10)
where 𝜃 are the trainable parameters.
In addition to MLM, PrefixLM in SIMVLM is another strategy for pre-training
vision-language representation. This technique focuses on predicting the continua-
tion of a text sequence given a prefix, formalized as:
LPreﬁxLM (𝜃) = −Ex ∼D log P 𝜃 (x ≥Tp |x<Tp ),
(9.11)
where x is the text sequence, D represents the pre-training data, and Tp is the length
of the prefix sequence of tokens.
9.3.3.4 Masked Object Classification
This technique involves selectively masking portions of visual images, typically by
setting their values to zero and then utilizing the labels predicted by an object detector
as ground truth for these masked regions.
The methodology behind MOC is somewhat analogous to the masked language
modeling (MLM) approach in NLP. In MOC, specific image regions are masked by9 Multimodal LLMs
388
altering their visual features with a certain probability p%. The primary objective is
to predict the object category for these masked image regions accurately, denoted as
vim . This process entails passing the encoder output of the masked image regions vim
through a fully connected (FC) layer, which computes the scores for T object classes
(Li et al., 2020a). These scores are then transformed into a normalized distribution
g 𝜃 (vim ) via a softmax function. The MOC objective is formally expressed as:
i ) represents the ground-truth label for the masked image region, and
where c (vm
CE denotes the cross-entropy loss function. Here, 𝜃 signifies the parameters of the
model, and the expectation E is over the distribution of words w and visual features
v . The MOC objective, therefore, focuses on enhancing the model’s ability to infer
and classify objects in partially observed or occluded visual contexts, reinforcing its
understanding of visual information.
9.3.3.5 Image-Text Matching (ITM)
The ITM process is integral in developing models that can understand and relate vi-
sual content to corresponding textual descriptions. A crucial aspect of ITM involves
generating negative training data, typically associating negative sentences with each
image and vice versa. The objective is to enhance the model’s discriminative capa-
bility in distinguishing between correctly matched image-text and mismatched pairs.
In the context of ITM, each image-text pair (v , t) is associated with a ground truth
label y , indicating whether the pair is correctly matched (positive) or not (negative).
The optimization of ITM is conducted using a binary classification loss function,
which assesses the model’s ability to predict these alignments accurately. The loss
function for ITM, denoted as LITM (𝜃), is mathematically formulated as:
LITM (𝜃) = −E (v ,t ) [y log s 𝜃 (v , t) + (1 − y ) log(1 − s 𝜃 (v , t))]
(9.13)
where s 𝜃 (v , t) represents the image-text similarity score computed by the model
with parameters 𝜃. The expectation E (v ,t ) is taken over the distribution of image-text
pairs. This loss function effectively measures the model’s proficiency in identifying
correct and incorrect alignments, thus refining its understanding of the complex re-
lationships between visual and textual modalities.
9.3.3.6 Image-Text Generation
Image-text Generation (ITG) is an essential component of vision-language-related
pre-training tasks. It focuses on training a model to generate text based on a given9.3 Multimodal LLM Framework
image, leveraging aligned image-text pairs. For instance, Xu et al. (2021) trained
the E2E-VLP model using the ITG objective. The ITG objective is formulated as
Here, X represents the visual sequence with context, and Y is the set of generated
text. The variable n indicates the length of tokens in the text y . This objective aims
to maximize the probability of correctly generating the sequence of text tokens yt
based on the preceding tokens y<t and the visual input x .
9.3.3.7 Video-Subtitle Matching (VSM)
Video-subtitle matching (VSM) in video-text pre-training, as exemplified in HERO,
focuses on two key alignment targets: local and global alignment (Li et al., 2020b).
Score functions quantify the alignment between video and subtitle content, with sep-
arate scores for local and global alignment. The loss functions, however, are designed
to optimize the model by minimizing the difference between these alignment scores
for correctly matched video-subtitle pairs (positive pairs) and maximizing it for in-
correctly matched pairs (negative pairs).
In HERO’s VSM implementation, two alignment targets are considered: local and
global.
Score Functions
In this model, sq represents the sampled query from all subtitle sentences, v is
the entire video clip, and Vtemp ∈ RNv ×d is the final visual frame representation
generated by a temporal Transformer. The query vector q ∈ Rd , start and end indices
yst , yed ∈ {1, ... , Nv }, and the probability vectors pst , ped ∈ RNv are derived from the
scores. The hinge loss function Lh is used for both positive and negative query-
video pairs, where (sq , v) is a positive pair and (sq , ^
v), (^sq , v) are negative pairs.
The margin hyper-parameter 𝛿 and balancing factors 𝜆1 , 𝜆2 are key components of
this framework.
9.3.3.8 Frame Order Modeling (FOM)
Frame order modeling (FOM) is conceptualized as a classification challenge within
the HERO model’s context, focusing on accurately predicting the chronological order
of a given set of video frames (Li et al., 2020b). The primary goal of FOM is to
determine the original sequence of timestamps for a subset of frames extracted from a
video, thereby testing the model’s understanding of temporal dynamics and narrative
flow in video content.
• R denotes the total number of frames that have been reordered and is subject to
classification.
• i represents the index within the reordered set, ranging from 1 to R.
• ti symbolizes the true timestamp position of the i th frame within the video,
which spans from 1 to Nv , where Nv is the total number of frames in the video.
• ri is the index corresponding to the reordered position of the i th frame.
• P is a probability matrix of dimensions Nv × Nv , where each element P [ri , ti ]
indicates the model’s predicted probability that the frame at reordered position
ri corresponds to timestamp ti .
9.3.4 MMLLM Tuning and Enhancements
Following the pre-training phase, MMLLMs can be further enhanced to improve
their adaptability, reasoning, and task generalization capabilities. This enhancement9.3 Multimodal LLM Framework
391
is achieved through various methodologies, three of which are presented here: multi-
modal instruction tuning (MM-IT), which refines models to follow instructions for a
broad spectrum of tasks; multimodal in-context learning (MM-ICL), which enables
models to apply preexisting knowledge to new tasks presented within input prompts;
and the multimodal chain-of-thoughts (MM-COT) approach, which enables more
transparent and logical reasoning by the model in solving complex problems.
9.3.4.1 Multimodal Instruction Tuning
Instruction tuning (IT) diverges from the data-heavy demands of traditional
supervised fine-tuning and the limited improvements of prompting methods
in few-shot scenarios by aiming to generalize task performance beyond initial
training data (Sect. 4.2). Building on this, multimodal instruction tuning (MM-
IT) adapts IT principles to enhance LLMs through fine-tuning multimodal
datasets structured around instructional tasks (Liu et al., 2024; Zhao et al.,
2023; Zhu et al., 2023). This approach empowers LLMs to handle new tasks
by interpreting instructions efficiently, markedly boosting zero-shot learning
abilities across various modalities.
Input Projector
Flamingo’s ability to handle visual inputs, including images and videos, necessitates
addressing the variability in feature outputs. This is achieved through the perceiver
resampler component, which standardizes outputs to a consistent 64 visual tokens, as
shown in Fig. 9.7. The modality alignment between language and visual modalities is
achieved by incorporating cross-attention (GATED XATTN-DENSE) layers among
the preexisting frozen language model layers, enhancing the attention mechanism
toward visual tokens during text token generation.
9.5.1.3 Pre-training: Core LLMs, Datasets and Task-Specific Objectives
The foundation of Flamingo is built upon the Chinchilla language model by freez-
ing nine of the pre-trained Chinchilla LM layers. The training regimen spans four
distinct datasets: M3W (Interleaved image-text), ALIGN (Image-text pairs), LTIP
(Image-text pairs), and VTP (Video-text pairs). This approach enables Flamingo to
predict subsequent text tokens y by considering both preceding text and visual to-
kens, quantified as:
The training loss function is defined as a weighted sum of the expected nega-
tive log-likelihoods of the generated text across the datasets, where 𝜆m signifies the
training weight for the m-th dataset:
where Dm and 𝜆 m represent the m-th dataset and its associated weighting, re-
spectively.
9.5.1.4 MMLLM Tuning and Enhancements
The Flamingo models exhibit exceptional performance in in-context learning, out-
classing state-of-the-art models fine-tuned for specific tasks despite relying on a sin-
gular set of model weights and a limited number of 32 task-specific examples – a
thousand times fewer task-specific training examples than existing state-of-the-art
approaches. The analysis presents support examples as pairs of images or videos
(visual inputs) with corresponding text (expected responses or task-specific infor-
mation, such as questions) to predict responses for new visual queries. The de-
fault prompts use are “Output: output” for tasks excluding question-answering, and
Experimental Design
There are many MMLLM to select from, so to narrow our choices we consider mod-
els small enough to be QLoRA-tuned in a Google Colab notebook and which are
already integrated with Huggingface so that we can easily take advantage of their
PEFT and fine-tuning routines. With these considerations, we choose as our model
the 9 billion parameter variant of IDEFICS (Image-aware Decoder Enhanced à la
Flamingo with Interleaved Cross-attentionS), an open-source text and image-to text
LLM modeled on Flamingo (Laurençon et al., 2023). The model takes arbitrarily
interleaved text and images as input and outputs a textual response.
The dataset we choose for this experiment is the 100 Sports Image Classification
dataset (100SIC) hosted at Kaggle¹. This set includes many small photos labeled by
sport for 100 different sports. It consists of approximately 13,000 training images and
500 test and validation images. For caption fine-tuning, we supplement this dataset
with a subset of the flickr30k dataset (Young et al., 2014), a 30,000+ item catalog of
image and caption pairs. We used the subset extracted by Shin Thant², who identified
flickr30k images of sports.
"""]

def clean_text(text):
    """Очистка текста от лишних символов"""
    if not text:
        return ""
    
    # Удаление лишних пробелов и переносов
    text = re.sub(r'\s+', ' ', text)
    
    # Удаление URL
    text = re.sub(r'http\S+', '', text)
    
    # Удаление специальных символов (оставляем буквы, цифры, знаки препинания)
    text = re.sub(r'[^а-яА-Яa-zA-Z0-9.,!?;:()\-\s]', '', text)
    
    # Удаление повторяющихся знаков препинания
    text = re.sub(r'([.,!?;:])\1+', r'\1', text)
    
    return text.strip()

def chunk_text(text, chunk_size=512, overlap=50):
    """Разбиение на фрагменты с перекрытием"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.split()) > 100:  # отбрасываем слишком короткие
            chunks.append(chunk)
    
    return chunks

def tokenize_function(examples):
    """Токенизация с padding и truncation"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors=None
    )

def tokenize_for_analysis(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,     
        max_length=512,
        return_tensors=None
    )

# Создание датасета из чанков
text_chunks = []
for source_text in all_texts:  # all_texts - список собранных текстов
    cleaned = clean_text(source_text)
    chunks = chunk_text(cleaned)
    text_chunks.extend(chunks)

# Создание Dataset
raw_dataset = Dataset.from_dict({"text": text_chunks})
print(f"Создано {len(raw_dataset)} фрагментов")

analysis_dataset = raw_dataset.map(tokenize_for_analysis, batched=True)
real_token_counts = [len(ids) for ids in analysis_dataset["input_ids"]]

#Токенизация
tokenized_dataset = raw_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]  # можно оставить для наглядности
)

# Добавляем исходный текст для удобства (опционально)
final_dataset = Dataset.from_dict({
    'text': text_chunks,
    'input_ids': tokenized_dataset['input_ids'],
    'attention_mask': tokenized_dataset['attention_mask']
})

# Сохраняем
variant = 21
topic_name = "Attention"
dataset_path = f'corpus_variant_{variant}'
final_dataset.save_to_disk(dataset_path)

# Создаем карточку датасета
dataset_info = {
    'variant': variant,
    'topic': topic_name,  # из задания
    'num_samples': len(final_dataset),
    'total_chars': sum(len(t) for t in text_chunks),
    'model_used': model_name,
    'sources': "Large_Language_Models_A_Deep_Dive_Bridging_Theory_and_Practice_Uday"  # список источников
}

import json
with open(f'{dataset_path}/info.json', 'w', encoding='utf-8') as f:
    json.dump(dataset_info, f, ensure_ascii=False, indent=2)

print(f"Датасет сохранен в папку {dataset_path}")
print(f"Информация: {dataset_info}")

# Создание pipeline для генерации
generator = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# Формулировка промптов по теме
prompts = [
    f"The main idea of attention mechanism in LLM is",
    f"How does the attention mechanism improve upon the limitations of the traditional RNN-based encoder-decoder architecture?",
    f"What is the masked language model (MLM) and the next sentence prediction (NSP)?"
]

print("=" * 60)
print(f"ТЕСТИРОВАНИЕ МОДЕЛИ НА ТЕМУ: {topic_name}")
print("=" * 60)

for prompt in prompts:
    result = generator(
        prompt,
        max_length=50,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )[0]['generated_text']
    
    print(f"Prompt: {prompt}")
    print(f"Generated: {result}")
    print("-" * 40)

# Сохраняем результаты
with open(f'{dataset_path}/inference_results.txt', 'w', encoding='utf-8') as f:
    f.write(f"Тема: {topic_name}\n")
    f.write(f"Модель: {model_name}\n\n")
    for prompt in prompts:
        result = generator(prompt, max_length=50)[0]['generated_text']
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Result: {result}\n\n")