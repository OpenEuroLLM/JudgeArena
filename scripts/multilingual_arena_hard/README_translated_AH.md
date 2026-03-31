# ArenaHard-EU Dataset Card

## Dataset Description

ArenaHard-EU is a comprehensive multilingual benchmark for evaluating Large Language Models (LLMs) across 35 European and neighboring languages. This dataset extends the original Arena-Hard benchmark through machine translation, enabling robust multilingual LLM evaluation.

### Key Features

- **35 Languages**: Covers all official EU languages plus co-official languages, candidate member languages, and Scandinavian languages
- **500+ Prompts per Language**: Challenging instructions spanning diverse topics and complexities
- **Consistent Structure**: Maintains the original Arena-Hard format for easy integration with existing evaluation pipelines

A similar dataset is [m-ArenaHard](https://huggingface.co/datasets/CohereLabs/m-ArenaHard) which contains 23 languages translated with google-translate.

## Supported Languages

The dataset includes the following 35 languages (ISO 639-3 codes in parentheses):

**Official EU Languages:**
Bulgarian (bul), Croatian (hrv), Czech (ces), Danish (dan), Dutch (nld), Estonian (est), Finnish (fin), French (fra), German (deu), Greek (ell), Hungarian (hun), Irish (gle), Italian (ita), Latvian (lav), Lithuanian (lit), Maltese (mlt), Polish (pol), Portuguese (por), Romanian (ron), Slovak (slk), Slovene (slv), Spanish (spa), Swedish (swe)

**Co-official & Regional Languages:**
Basque (eus), Catalan (cat), Galician (glg)

**Candidate & Neighboring Countries:**
Albanian (sqi), Bosnian (bos), Georgian (kat), Icelandic (isl), Macedonian (mkd), Norwegian (nor), Serbian (srp), Turkish (tur), Ukrainian (ukr)

## Usage

### Loading the Dataset
```python
from datasets import load_dataset

# Load a specific language subset
dataset = load_dataset("openeurollm/ArenaHard-EU", "fra")  # French

# Access the data
for example in dataset["train"]:
    print(example["prompt"])
```

### Using with JudgeArena for Model Evaluation

**TODO coming soon**

Evaluate and compare two models using the JudgeArena framework:
```bash
python judgearena/generate_and_evaluate.py \
  --dataset arena-hard-EU \
  --model_A gpt4_1106_preview \
  --model_B VLLM/utter-project/EuroLLM-9B \
  --judge_model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --language fra \
  --n_instructions 100
```

For more information, visit the [JudgeArena repository](https://github.com/OpenEuroLLM/JudgeArena).

## Dataset Structure

### Data Fields

Each example contains the following fields:

- `question_id` (string): Unique identifier for the example
- `cluster` (string): Thematic category/topic of the instruction
- `category` (string): Source dataset from the original Arena-Hard compilation
- `prompt` (string): The instruction or question text in the target language
- `lang` (string): ISO 639-3 language code

### Example
```json
{
  "question_id": "0001",
  "cluster": "coding",
  "category": "arena-hard",
  "prompt": "Écrivez une fonction Python qui calcule la suite de Fibonacci.",
  "lang": "fra"
}
```

## Dataset Creation


### Code Availability

The complete translation pipeline is open-source and can be used to extend this dataset to additional languages:

👉 [Translation Script](https://github.com/OpenEuroLLM/JudgeArena/blob/main/scripts/multilingual_arena_hard/translate_arena_hard.py)


### Translation Process

- **Source**: Original Arena-Hard benchmark dataset
- **Translation Model**: DeepSeek-3.1
- **Method**: High-quality neural machine translation with post-processing validation


## License

This dataset is released under the **Apache 2.0 License**, allowing for both academic and commercial use.

## Contributions & Feedback

We welcome contributions to expand language coverage or improve translation quality. Please open an issue or pull request in the [JudgeArena repository](https://github.com/OpenEuroLLM/JudgeArena).

## Acknowledgments

This dataset was created as part of the OpenEuroLLM initiative to promote multilingual AI research and development across European languages.
