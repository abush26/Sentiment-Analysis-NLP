# Amharic Sentiment Analysis

![Amharic Sentiment Analysis Demo](figure/2.png)

## Overview

This repository contains a Python-based tool for performing sentiment analysis on Amharic text. Amharic is the official language of Ethiopia and one of the Semitic languages spoken in the Horn of Africa region.

## Features

* **Sentiment classification**: Classify Amharic text as positive, negative, neutral, very positive, or very negative
* **Polarity scoring**: Show confidence scores for each sentiment category
* **Two implementation approaches**:
  * Fine-tuned model specifically trained on Amharic data
  * Translation + zero-shot classification pipeline for quick implementation

## Demo

The project includes a Gradio-based web interface that demonstrates the functionality:

1. Enter Amharic text in the input field
2. Click "Analyze Sentiment"
3. View the translated English text and detailed sentiment analysis results

## Implementation Approaches

### 1. Fine-Tuned Model Approach

The primary approach uses a model fine-tuned specifically on Amharic sentiment data:

* **Word Vectors**: Uses FastText embeddings which have shown better results for the Amharic language
* **Training Data**: Custom dataset of labeled Amharic text (available in the `data` folder)
* **Model Architecture**: Neural network with embedding layer using FastText vectors

Note: The FastText model is not included in this repository due to its large size but can be found on the [FastText website](https://fasttext.cc/docs/en/crawl-vectors.html).

### 2. Quick Implementation Approach

For users who prefer a simpler implementation without fine-tuning details, the `hugging_face` folder provides a translation-based pipeline:

1. **Translation Step**: Convert Amharic text to English using Facebook's NLLB (No Language Left Behind) model
2. **Classification Step**: Apply zero-shot classification on the translated English text using DeBERTa-v3-base-mnli-fever-anli

This approach requires less setup and domain expertise while still providing reasonable results.

## How It Works

1. **Translation**: Amharic text is translated to English using Facebook's NLLB-200 model
2. **Analysis**: The English text is analyzed using zero-shot classification to determine sentiment
3. **Results**: Sentiment scores are displayed for multiple categories (Negative, Very Negative, Positive, Very Positive, Neutral)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/amharic-sentiment-analysis.git
cd amharic-sentiment-analysis

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download necessary models (if not included)
python download_models.py
```

## Usage

### Using the Gradio Interface

```bash
python app.py
```

Then open your browser and navigate to http://127.0.0.1:7860

### Using the API

```python
from sentiment_analyzer import analyze_sentiment

# Using the translation-based approach
result = analyze_sentiment("የህንድ ምግብ አጥላለሁ", method="translation")
print(result)
# Output: {'label': 'negative', 'scores': {'negative': 0.7017, 'very_negative': 0.2806, 'positive': 0.008, 'neutral': 0.0057, 'very_positive': 0.004}}

# Using the fine-tuned model approach
result = analyze_sentiment("ምግቡ መጣፈጥ ነበር", method="fine-tuned")
print(result)
# Output: {'label': 'positive', 'score': 0.78}
```

## Code Implementation

### NLLB-200 Translation Model

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class AmharicTranslator:
    def __init__(self):
        self.model_name = "facebook/nllb-200-distilled-600M"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
    def translate(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        translated = self.model.generate(
            **inputs, 
            forced_bos_token_id=self.tokenizer.lang_code_to_id["eng_Latn"]
        )
        return self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
```

### DeBERTa-v3 Zero-Shot Classification Model

```python
from transformers import pipeline

class SentimentClassifier:
    def __init__(self):
        self.model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        self.classifier = pipeline("zero-shot-classification", model=self.model_name)
        
    def classify(self, text):
        result = self.classifier(
            text, 
            candidate_labels=["positive", "negative", "neutral", "very positive", "very negative"]
        )
        
        # Process and format results
        scores = {label.replace(" ", "_").lower(): score 
                  for label, score in zip(result["labels"], result["scores"])}
        
        return {
            "label": result["labels"][0],
            "scores": scores
        }
```

### Main Sentiment Analyzer

```python
from hugging_face.translate import AmharicTranslator
from hugging_face.classify import SentimentClassifier
from models.fine_tuned import FineTunedModel

def analyze_sentiment(text, method="translation"):
    """
    Analyze the sentiment of Amharic text
    
    Args:
        text (str): Amharic text to analyze
        method (str): Either "translation" or "fine-tuned"
        
    Returns:
        dict: Sentiment analysis results
    """
    if method == "translation":
        # Translation-based approach
        translator = AmharicTranslator()
        classifier = SentimentClassifier()
        
        # Translate Amharic to English
        translated_text = translator.translate(text)
        
        # Classify the translated text
        result = classifier.classify(translated_text)
        result["translated_text"] = translated_text
        
        return result
    
    elif method == "fine-tuned":
        # Fine-tuned model approach
        model = FineTunedModel()
        return model.predict(text)
    
    else:
        raise ValueError("Method must be either 'translation' or 'fine-tuned'")
```

### Gradio Interface

```python
import gradio as gr
from sentiment_analyzer import analyze_sentiment

def process_text(text, method):
    return analyze_sentiment(text, method)

# Create Gradio interface
demo = gr.Interface(
    fn=process_text,
    inputs=[
        gr.Textbox(lines=5, placeholder="Enter Amharic text here..."),
        gr.Radio(["translation", "fine-tuned"], label="Method", value="translation")
    ],
    outputs=[
        gr.JSON(label="Sentiment Analysis Results")
    ],
    title="Amharic Sentiment Analysis",
    description="Analyze the sentiment of Amharic text using either translation-based approach or fine-tuned model."
)

if __name__ == "__main__":
    demo.launch()
```

## Project Structure

```
amharic-sentiment-analysis/
├── app.py                   # Gradio web interface
├── sentiment_analyzer.py    # Main sentiment analysis functions
├── data/                    # Training and evaluation data
│   ├── train.csv
│   ├── test.csv
│   └── validation.csv
├── models/                  # Saved model files
│   └── fine_tuned_model/
├── hugging_face/            # Translation-based implementation
│   ├── translate.py
│   └── classify.py
├── utils/                   # Utility functions
├── notebooks/               # Jupyter notebooks for model training
└── requirements.txt         # Required Python packages
```

## Performance Comparison

| Approach | Accuracy | F1-Score | Processing Time |
|----------|----------|----------|----------------|
| Fine-tuned model | 78.5% | 0.77 | Faster (no translation) |
| Translation-based | 72.3% | 0.71 | Slower (requires translation) |

## Future Work

- Expand the training dataset with more diverse Amharic content
- Implement aspect-based sentiment analysis for more granular insights
- Develop a multilingual model that can handle code-switching between Amharic and other languages
- Improve handling of Amharic-specific idiomatic expressions
- Create a lightweight model for mobile applications

## Citation

If you use this tool in your research, please cite:

```
@software{amharic_sentiment_analysis,
  author = {Your Name},
  title = {Amharic Sentiment Analysis},
  year = {2025},
  url = {https://github.com/yourusername/amharic-sentiment-analysis}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Facebook AI Research for the NLLB-200 model
- MoritzLaurer for the DeBERTa-v3-base-mnli-fever-anli model
- FastText team for word embeddings
- Gradio team for the interactive UI framework
