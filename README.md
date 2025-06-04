# DeepSeek-R1-Distill-Fine-tuning-for-medical-use


Welcome to the DeepSeek-R1 Medical Fine-Tuning project! This repository contains a Jupyter notebook demonstrating the fine-tuning of the DeepSeek-R1-Distill-Qwen-1.5B language model for medical question-answering using Low-Rank Adaptation (LoRA). The goal is to adapt a general-purpose large language model to excel in clinical reasoning, diagnostics, and treatment planning, enabling it to tackle complex medical questions with step-by-step reasoning. This README provides an overview, setup instructions, usage details, and evaluation guidance for anyone interested in exploring or extending this work.

Project Overview

The notebook, titled "deepseek-r1-distill-fine-tuning-for-medical-use.ipynb", fine-tunes the DeepSeek-R1-Distill-Qwen-1.5B model using the Unsloth library for efficiency and the FreedomIntelligence/medical-o1-reasoning-SFT dataset from Hugging Face. The dataset includes 1000 medical questions, chain-of-thought (CoT) reasoning steps, and expert responses, teaching the model to think like a medical professional. LoRA is applied to update a small fraction (0.28%) of the model’s parameters, making training feasible on limited hardware like a Kaggle Tesla T4 GPU. The process involves loading the model with 4-bit quantization, formatting the dataset with a custom prompt, training for 60 steps, and testing on medical scenarios like urinary incontinence and endocarditis. While results show coherent reasoning, accuracy needs improvement, suggesting potential for further tuning.

Requirements

To run this project, you need Python 3.11 or higher and a GPU for efficient training (e.g., NVIDIA Tesla T4, as used in the notebook). The notebook was developed in a Kaggle environment with internet enabled and GPU support. Key dependencies include Unsloth for optimized fine-tuning, Transformers for model handling, Datasets for data loading, TRL for the training framework, and Weights & Biases (WandB) for logging. You’ll also need Hugging Face and WandB accounts for authentication, with tokens stored securely (e.g., via Kaggle secrets).

Installation

First, clone this repository to your local machine or Kaggle environment using the command: git clone . Next, install the required packages. Run the following in your terminal or notebook cell: !pip install unsloth. Then, force-reinstall Unsloth from its GitHub source for the latest updates: !pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git. Additional libraries like transformers, datasets, trl, and wandb are imported directly in the notebook. Ensure you have your Hugging Face token and WandB API key ready, ideally stored in a secure method like UserSecretsClient in Kaggle. The notebook’s first cell handles these installations, so you can execute it directly if running interactively.

Usage

To use this project, open the Jupyter notebook "deepseek-r1-distill-fine-tuning-for-medical-use.ipynb" in a compatible environment like Kaggle or a local Jupyter setup with GPU support. Start by executing the initial cells to install dependencies and log in to Hugging Face and WandB using your tokens. The notebook loads the DeepSeek-R1 model with a max sequence length of 2048 and 4-bit quantization for memory efficiency. It then downloads the medical dataset, formats it with a prompt structure combining questions, CoT placeholders, and responses, and applies LoRA for fine-tuning. Training runs for 60 steps with a batch size of 2, gradient accumulation, and a learning rate of 5e-4, logged via WandB. After training, the model is tested on two medical questions, and results are saved locally as "DeepSeek-R1-Medical-COT-1.5B". To run, execute each cell sequentially, ensuring your GPU is active. Modify the test questions or training parameters (e.g., max_steps) to experiment further.


Results

The fine-tuned model generates coherent chain-of-thought reasoning, as seen in test cases. For a 61-year-old woman with urine loss during coughing, it predicted “less fluid loss” due to detrusor issues, missing the likely diagnosis of stress urinary incontinence. For a 59-year-old man with fever and aortic valve vegetation, it incorrectly flagged endocarditis as a predisposing factor instead of a prior valve abnormality. These results, logged via WandB, show a decreasing loss trend (from 6.8911 to 6.5796), indicating learning, but highlight the need for longer training or a larger dataset. The model and tokenizer are saved locally and merged for future use.

Contributing

Contributions are welcome! If you’d like to improve the model, try increasing training steps, expanding the dataset, or tweaking LoRA parameters like rank (r) or alpha. You can also add new test cases, refine the prompt structure, or integrate additional evaluation metrics. To contribute, fork the repository, make your changes, and submit a pull request with a clear description of your updates. Please ensure your code follows Python best practices and includes comments for clarity.

License

This project is licensed under the MIT License. You are free to use, modify, and distribute the code and model weights, provided you include the license and copyright notice. See the LICENSE file in the repository for full details.

Acknowledgments

Thanks to Unsloth for providing an efficient fine-tuning framework, Hugging Face for hosting the model and dataset, and Weights & Biases for experiment tracking. The FreedomIntelligence team deserves credit for the medical-o1-reasoning-SFT dataset, which powers this project. This work was tested in a Kaggle environment, leveraging their GPU resources.

