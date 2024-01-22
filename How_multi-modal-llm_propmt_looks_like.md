## MiniGPT-v2: Large Language Model As a Unified Interface for Vision-Language Multi-task Learning
- https://arxiv.org/pdf/2310.09478.pdf
- We aim to project all into the language model space. However, for higher-resolution images such as 448x448, projecting all the image tokens results in a very long-sequence input (e.g., 1024 tokens) and significantly lowers the training and inference efficiency. Hence, we simply concatenate 4 adjacent visual tokens in the embedding space and project them together into one single embedding in the same feature space of the large language model, thus reducing the number of visual input tokens by 4 times.
- General input format. We follow the LLaMA-2 conversation template design and adapt it for the multi-modal instructional template. The template is denoted as follows,

$$[INST] <Img> < ImageFeature> </Img> [Task Identifier] Instruction [/INST]$$

- we have proposed six different task identifiers for visual question answering, image caption, grounded image captioning, referring expression comprehension, referring expression generation, and phrase parsing and grounding respectively.

<img width="600" alt="image" src="https://github.com/junuMoon/review/assets/52732827/2a0d5607-9640-4eb9-9956-81402d360c6f">

-  We represent the spatial location through the textual formatting of bounding boxes in our setting, specifically: “{< Xleft >< Ytop >< Xright >< Ybottom >}". Coordinates for X and Y are represented by integer values normalized in the range [0,100].
-  Stage 1: Pretraining. To have broad vision-language knowledge, our model is trained on a mix of weakly-labeled and fine-grained datasets. We give a high sampling ratio for weakly-labeled datasets to gain more diverse knowledge in the first-stage.
-  Stage 2: Multi-task training. To improve the performance of MiniGPT-v2 on each task, we only focus on using fine-grained datasets to train our model at this stage. We exclude the weakly-supervised datasets
-  Stage 3: Multi-modal instruction tuning. Subsequently, we focus on tuning our model with more multi-modal instruction datasets and enhancing its conversation ability as a chatbot.

## LLASM: LARGE LANGUAGE AND SPEECH MODEL
- https://arxiv.org/pdf/2308.15930.pdf
- The pre-training stage: We use Whisper to encode the raw audio data into embeddings first, then a modal adaptor is trained during the pre-training stage to align the audio embeddings and the text embeddings. The audio embeddings and the text embeddings are concatenated together to form interleaved input sequences to input to the large language model.
- The data sample of ASR data usually consists of a pair of speech audio and text utterances, especially, when we add a simple instruction to the data sample as the task instruction.
  - "Transcribe the following speech into text."
  - "Convert the spoken words into written text."
  - "Transform the speech into a written transcript."
  - "Record the oral communication as written text."

```
[INST]<<SYS>>
You are a helpful language and speech assistant. You are able to understand the speech content that the user provides, and assist the user with a variety of tasks using natural language
<</SYS>>

<au_start> <au_patch> * {len(audio_token)} <au_end>
{Simple Instruction: e.g. Convert the spoken words into written text}
[/INST]
LLASM RESPONSE TOKENS
```

## Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models
- https://arxiv.org/pdf/2311.07919.pdf
