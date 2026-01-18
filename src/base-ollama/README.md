# Evaluation with Ollama (Beginner's Guide)

This guide will help you run the SecQA evaluation using models hosted on **Ollama**.

## Prerequisites

1.  **Install Ollama**:
    - Download and install Ollama from [ollama.com](https://ollama.com/).
    - Open your terminal and verify it's working by typing:
      ```bash
      ollama --version
      ```

2.  **Pull the Model**:
    - You need to download the model weights first. For example, to use Llama 3.2 (3B):
      ```bash
      ollama pull llama3.2:3b
      ```

3.  **Install Python Requirements**:
    - Ensure you have the `requests` and `tqdm` libraries installed:
      ```bash
      pip install requests tqdm
      ```

## Running the Evaluation

1.  **Ensure Ollama is running**:
    - The Ollama application should be open (look for the icon in your menu bar).

2.  **Run the script**:
    - Navigate to the project root and run:
      ```bash
      python3 src/base-ollama/eval_ollama.py
      ```

3.  **Check Results**:
    - Once finished, the script will show the final accuracy in the terminal.
    - Detailed question-by-question results will be saved to `eval_results_ollama.json`.

## Changing the Model

If you want to test a different model (e.g., `phi` or `mistral`):
1.  Pull the model: `ollama pull phi`
2.  Open `src/base-ollama/eval_ollama.py`.
3.  Change the `MODEL_NAME` variable at the top:
    ```python
    MODEL_NAME = "phi"
    ```
4.  Run the script again.
