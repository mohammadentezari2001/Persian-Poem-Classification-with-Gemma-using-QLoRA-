# Persian Poem Classification with Gemma (using QLoRA)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)

This project implements a fine-tuning pipeline for the **Google Gemma (270M)** Large Language Model to classify Persian poems by their poets. 

The benchmark focuses on Parameter-Efficient Fine-Tuning (PEFT) methods, specifically comparing **4-bit QLoRA** against **8-bit Quantized LoRA** with varying adapter ranks ($r=8$ and $r=16$).

## Features

*   **Model**: Fine-tunes `google/gemma-3-270m`.
*   **Quantization Support**: 
    *   **QLoRA (4-bit)**: Uses `bitsandbytes` for 4-bit normal float quantization.
    *   **8-bit Quantization**: Uses standard 8-bit loading for memory efficiency.
*   **Text Processing**: Includes custom handling for **Persian/Farsi text** in visualizations (reshaping and BiDi support) to ensure correct rendering in plots.
*   **Benchmarking**: Automatically runs experiments across different configurations and outputs a summary CSV.
*   **Visualization**: Generates confusion matrices for each experimental run.

## Dataset Structure

The script expects the dataset to be placed in a folder named `Dataset` in the project root. The CSV files must contain at least two columns: `text` (the poem) and `poet` (the label).

```text
├── main.ipynb
├── requirements.txt
├── Dataset/
│   ├── Train.csv
│   ├── Test.csv
│   └── Val.csv
└── ...
```

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/mohammadentezari2001/Persian-Poem-Classification-with-Gemma-using-QLoRA-.git
    cd Persian-Poem-Classification-with-Gemma-using-QLoRA

    ```

2.  **Install dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

    *Required libraries include: `torch`, `transformers`, `peft`, `bitsandbytes`, `datasets`, `pandas`, `scikit-learn`, `matplotlib`, `arabic-reshaper`, `python-bidi`.*

## Configuration

Hyperparameters are centralized in the `CONFIG` dictionary within `main.py`. You can modify these settings directly in the code:

```python
CONFIG = {
    "model_name": "google/gemma-3-270m",
    "sample_size": 5000,    # Number of samples to use (speed optimization)
    "epochs": 5,
    "batch_size": 8,
    "lr": 2e-4,
    "r": [8, 16],           # LoRA ranks tested in the loop
    # ...
}
```

## Usage

Run the main script to start the benchmarking process:

```bash
python main.ipynb
```

### The Training Loop
The script will automatically execute 4 sequential experiments:
1.  **QLoRA (4-bit)** with Rank $r=8$
2.  **QLoRA (4-bit)** with Rank $r=16$
3.  **Quantized (8-bit)** with Rank $r=8$
4.  **Quantized (8-bit)** with Rank $r=16$

## Outputs

After execution, the following artifacts are generated:

1.  **Results Summary**: `fine_tuning_benchmark_results.csv`
    *   Contains comparison metrics: Method, Rank, Training Time, Max Memory Usage, and Test F1 Score.
2.  **Model Checkpoints**: Saved in `./lora_results/{method_name}_r{rank}`.
3.  **Confusion Matrices**: Saved as PNG images inside each result folder (e.g., `./lora_results/QLoRA_4bit_r8/confusion_matrix.png`).

## Example Results format

The final CSV will look similar to this:

| method | r | trainable_params | training_time_sec | max_memory_gb | test_f1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| QLoRA_4bit | 8 | 150,000 | 320.5 | 2.1 | 0.85 |
| Quant_8bit | 16 | 300,000 | 340.2 | 2.8 | 0.86 |

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgements

*   Hugging Face for the `transformers` and `peft` libraries.
*   Google DeepMind for the Gemma model.
*   `arabic_reshaper` and `python-bidi` for solving RTL text rendering issues in Matplotlib.



