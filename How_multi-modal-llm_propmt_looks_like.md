## MiniGPT-v2: Large Language Model As a Unified Interface for Vision-Language Multi-task Learning
- https://arxiv.org/pdf/2310.09478.pdf
- We aim to project all into the language model space. However, for higher-resolution images such as 448x448, projecting all the image tokens results in a very long-sequence input (e.g., 1024 tokens) and significantly lowers the training and inference efficiency. Hence, we simply concatenate 4 adjacent visual tokens in the embedding space and project them together into one single embedding in the same feature space of the large language model, thus reducing the number of visual input tokens by 4 times.
- General input format. We follow the LLaMA-2 conversation template design and adapt it for the multi-modal instructional template. The template is denoted as follows,

$$[INST] <Img> < ImageFeature> </Img> [Task Identifier] Instruction [/INST]$$

```python
class LlavaMetaForCausalLM(ABC):
    ...

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue
    ...
```

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
- Multi-task Training Format Framework Motivated by Whisper (Radford et al., 2023), to incorporate different kinds of audio, we propose a multitask training format framework as follows:
  - Transcription Tag: The initiation of prediction is denoted using a transcription tag. The `<|startof- transcripts|>` is employed to indicate the tasks involve accurately transcribing the spoken words and capturing the linguistic content of a speech recording, such as speech recognition and speech translation tasks. For other tasks, the `<|startofanalysis|>` tag is utilized.
  - Audio Language Tag: Then, we incorporate a language tag that indicates the spoken language in the audio. This tag uses a unique token assigned to each language present in our training set, eight languages in totally. In the case where an audio segment does not contain any speech, such as natural sounds and music, the model is trained to predict a `<|unknown|>` token.
  - Task Tag: The subsequent tokens specify the task. We categorize the collected audio tasks into five categories: `<|transcribe|>, <|translate|>, <|caption|>, <|analysis|>, and <|question-answer|>` tasks. For question-answer (QA) tasks, we append the corresponding questions after the tag.
  - Text Language Tag: The tag token specifies the language of output text sequences.
  - Timestamps Tag: The presence of a `<|timestamps|> or <|notimestamps|>` token determines whether the model needs to predict timestamps or not. Different from the sentence-level timestamps used in Whisper, **the inclusion of the <|timestamps|> tag requires the model to perform fine-grained word-level timestamp prediction, abbreviated as SRWT (Speech Recognition with Word-level Timestamps)**. The prediction of these timestamps is interleaved with the transcription words: the start time token is predicted before each transcription token, while the end time token is predicted after. According to our experiments, **SRWT improves the ability of the model to align audio signals with timestamps. This improved alignment contributes to a comprehensive understanding of speech signals by the model, resulting in notable advancements across many tasks such as speech recognition and audio QA tasks.**
  - Output Instruction: Lastly, we provide output instruction to further specify the task and desired format for different subtasks, and then the text output begins.
- To handle multi-audio dialogue and multiple audio inputs effectively, we introduce the convention of labeling different audios with `"Audio id:"`, where id corresponds to the order of the audio input dialogue. In terms of dialogue format, we construct our instruction tuning dataset using the ChatML (Openai) format. In this format, each interaction’s statement is marked with two special tokens (`<im_start> and <im_end>`) to facilitate dialogue termination.

<img width="883" alt="image" src="https://github.com/junuMoon/review/assets/52732827/fbfea011-e557-41b7-a199-40f2d93d2231">

```
<im_start>user
Audio 1: <audio>emov-db/141-168-0155.wav</audio>what does the speaker say?<im_end> <im_start>assistant
The speaker says in English, "Won’t you draw up, gentlemen.".<im_end>
<im_start>user
What’s the mood of the person?<im_end>
<im_start>assistant
Based on the voice, the mood of the person is disgusted.<im_end>
```

- The purpose of SRWT is twofold: firstly, to improve the model’s ability to align audio signals with fine-grained timestamps; secondly, to support grounding of speech and audio, and grounding-based QA tasks in Qwen-Audio-Chat, such as finding the starting and ending time of an audio segment mentioning a person’s name or identifying whether a sound occurs in the given audio

<img width="714" alt="image" src="https://github.com/junuMoon/review/assets/52732827/41386420-da15-4942-82c4-6f193e7b82b9">
