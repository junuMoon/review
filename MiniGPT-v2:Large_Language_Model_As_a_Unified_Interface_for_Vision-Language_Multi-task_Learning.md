# MiniGPT-v2: Large Language Model As a Unified Interface for Vision-Language Multi-task Learning
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
