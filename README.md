# ABSA-LLM

Comparative study of **large language models** on **Aspect-Based Sentiment Analysis (ABSA)** for restaurant reviews. Each notebook runs the same prompting setup—zero-shot, one-shot, and RAG-augmented (3-shot)—so you can compare models under a shared evaluation protocol.

## What this project does

Given a review sentence and a target aspect (e.g. *food*, *service*), the task is to predict sentiment toward that aspect as **positive**, **negative**, or **neutral**.

This repo does **not** fine-tune models. Everything is **prompt-based inference**: instructions + optional examples are sent to an LLM, and the reply is parsed into one of the three labels.

```
Review + aspect  →  prompt (zero / one / RAG)  →  LLM  →  {positive, negative, neutral}
```

For the RAG setting, similar training examples are retrieved from a vector index and injected into the prompt as in-context demonstrations.

## Dataset

All notebooks target the **SemEval-2016 Task 5** restaurant domain:

- File: `ABSA16_Restaurants_Train_SB1_v2.xml`
- Source: [SemEval-2016 ABSA](https://alt.qcri.org/semeval2016/task5/)

The XML is parsed into `(sentence, aspect, polarity)` tuples. `NULL` targets and `conflict` labels are skipped where applicable. You need to download/upload the XML yourself—it is not bundled in this repository.

## Unified benchmark protocol

Every local-model notebook follows the same design choices so results are comparable across architectures:

| Component | Setting |
|-----------|---------|
| Task | ABSA polarity for a given aspect |
| Labels | `positive`, `negative`, `neutral` |
| Prompting | Zero-shot, one-shot (fixed example), RAG (k=3 retrieved examples) |
| RAG embedder | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS |
| Retrieval | Top-3 similar training examples |
| Generation | Deterministic (`temperature=0`, `do_sample=False`) |
| Output parsing | Regex over allowed label tokens |
| Message format | Single **user** message (no separate system role) |

RAG examples are drawn from a held-out portion of the training XML. Exact train/test splits and sample counts differ slightly per notebook (e.g. 50 vs 100 vs 200 eval samples), so treat cross-notebook accuracy numbers as indicative, not a strict leaderboard.

## Notebooks

| Notebook | Model | Runtime |
|----------|-------|---------|
| [`Gemini3_1_Flash_Lite.ipynb`](Gemini3_1_Flash_Lite.ipynb) | Google Gemini 3.1 Flash Lite (API) | Colab + API key |
| [`Gemma_2_2B_RAG.ipynb`](Gemma_2_2B_RAG.ipynb) | `google/gemma-2-2b-it` | Colab GPU (T4) |
| [`Gemma_3_4BRAG.ipynb`](Gemma_3_4BRAG.ipynb) | `google/gemma-3-4b-it` | Colab GPU + HF token |
| [`Llama_3_2_3B_RAG.ipynb`](Llama_3_2_3B_RAG.ipynb) | `meta-llama/Llama-3.2-3B-Instruct` | Colab GPU + HF token |
| [`Mistral_7B_Instruct_v0_3_RAG.ipynb`](Mistral_7B_Instruct_v0_3_RAG.ipynb) | `mistralai/Mistral-7B-Instruct-v0.3` | Colab GPU |
| [`Phi3_mini_RAG.ipynb`](Phi3_mini_RAG.ipynb) | `microsoft/Phi-3-mini-4k-instruct` | Colab GPU |
| [`Qwen2_5_7B.ipynb`](Qwen2_5_7B.ipynb) | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | Colab GPU |

Each notebook is self-contained: install dependencies, load data, build the FAISS index, run the three strategies, and plot accuracy.

## Quick start (Google Colab)

1. Open any notebook in Colab (badge links are at the top of several notebooks).
2. Set runtime to **GPU** (T4 is enough for 2B–7B quantized models).
3. Upload `ABSA16_Restaurants_Train_SB1_v2.xml` to the runtime working directory.
4. For gated models (Gemma, Llama), add a [Hugging Face token](https://huggingface.co/settings/tokens) in Colab **Secrets** as `HF_TOKEN`, or paste it where the notebook indicates.
5. For Gemini, set your own Google AI API key in the client setup cell.
6. Run all cells top to bottom.

## Dependencies

Typical stack across notebooks:

- `transformers`, `accelerate`, `bitsandbytes` (local LLMs)
- `sentence-transformers`, `faiss-cpu` (RAG)
- `pandas`, `numpy`, `matplotlib`, `seaborn` (data + plots)
- `scikit-learn` (accuracy / F1 in some notebooks)
- `google-genai` (Gemini notebook only)

Install lines are in the first code cell of each notebook.

## Example results (from saved notebook runs)

These numbers come from executed cells in this repo. Splits and sample sizes are not identical across notebooks.

| Model | Zero-shot | One-shot | RAG (3-shot) |
|-------|-----------|----------|--------------|
| Gemini 3.1 Flash Lite | 94.0% | 94.0% | 96.0% |
| Mistral-7B-Instruct-v0.3 | 92.0% | 96.0% | 94.0% |
| Phi-3-mini-4k-instruct | 78.7% | 92.0% | 90.6% |
| Gemma-2-2B-it | 86.0% | 84.0% | 76.0% |
| Gemma-3-4B-it | 88.0% | 82.0% | 84.0% |
| Llama-3.2-3B-Instruct | 84.0% | 76.0% | 86.0% |
| Qwen2.5-7B-Instruct (4-bit) | 90.5% | 66.0% | 73.0% |

Re-run the notebooks to reproduce or update these figures on your own hardware.

## Project layout

```
ABSA-LLM/
├── Gemini3_1_Flash_Lite.ipynb      # API-based Gemini eval
├── Gemma_2_2B_RAG.ipynb            # Small open model baseline
├── Gemma_3_4BRAG.ipynb             # Larger Gemma variant
├── Llama_3_2_3B_RAG.ipynb          # Meta Llama 3.2 instruct
├── Mistral_7B_Instruct_v0_3_RAG.ipynb
├── Phi3_mini_RAG.ipynb             # Microsoft Phi-3 mini
├── Qwen2_5_7B.ipynb                # Qwen 2.5 7B (4-bit)
└── README.md
```

## Citation

If you use the SemEval dataset, cite the original task:

> Pontiki, M., et al. (2016). SemEval-2016 Task 5: Aspect Based Sentiment Analysis. *Proceedings of SemEval-2016*.

## Author

[Mohammad Reza Namvar Nejad](https://github.com/MohammadRezaNamvarNejad) — [ABSA-LLM](https://github.com/MohammadRezaNamvarNejad/ABSA-LLM)
