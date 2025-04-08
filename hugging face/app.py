import gradio as gr
from transformers import pipeline
import torch

# Initialize translation model
def initialize_translator():
    return pipeline(
        task="translation",
        model="facebook/nllb-200-distilled-600M",
        torch_dtype=torch.bfloat16
    )

# Initialize sentiment classifier
def initialize_classifier():
    return pipeline(
        "zero-shot-classification", 
        model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    )

translator = initialize_translator()
classifier = initialize_classifier()

def translate(text, src_lang="amh_Ethi", tgt_lang="eng_Latn"):
    """Translate Amharic text to English and classify sentiment"""
    if not text.strip():
        return "", "Please enter text to analyze"
        
    # Translate the text
    text_translated = translator(
        text,
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )
    
    # Get translated text and classify
    translated_text = list(text_translated[0].values())[0]
    result = classification(translated_text)
    
    return translated_text, result

def classification(text, candidate_labels=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]):
    """Classify the sentiment of the translated text"""
    try:
        # Get classification results
        output = classifier(text, candidate_labels, multi_label=False)
        
        # Extract and format results
        labels_order = output['labels']
        scores_order = output['scores']
        
        # Add visual elements to results
        results = []
        for label, score in zip(labels_order, scores_order):
            # Add emoji indicators
            if "Very Positive" in label:
                emoji = "😄"
            elif "Positive" in label:
                emoji = "🙂"
            elif "Neutral" in label:
                emoji = "😐"
            elif "Negative" in label:
                emoji = "🙁"
            elif "Very Negative" in label:
                emoji = "😠"
            else:
                emoji = ""
                
            # Format the result line with just emoji, label and percentage
            percentage = score * 100
            results.append(f"{emoji} {label}: {percentage:.2f}%")
            
        return '\n'.join(results)
    except Exception as e:
        return f"Error in classification: {str(e)}"

# Build the interface
def build_interface():
    with gr.Blocks(title="Amharic Sentiment Analysis") as demo:
        gr.Markdown("# ኢትዮጵያ Amharic Sentiment Analysis")
        
        with gr.Row():
            with gr.Column():
                seed = gr.Textbox(
                    label="Input Amharic Text",
                    placeholder="የአማርኛ ጽሑፍ እዚህ ይጻፉ...",
                    lines=3
                )
                
                # Add example texts
                gr.Examples(
                    examples=[
                        "እግዚአብሔር ይባርክህ",
                        "የህንድ ፊልሞች አሳዛኝ ናቸው",
                        "ምግቡ መካከለኛ ነበር",
                        "ዛሬ በጣም ደስተኛ ነኝ",
                        "ህንዳውያንን እወዳለሁ",
                        "የህንድ ምግብ እጠላለሁ",
                        "መጥፎ ስሜት እንዲሰማኝ በማድረግ እጠላዋለሁ",
                        "ምግቡ መካከለኛ ነው"
                    ],
                    inputs=seed
                )
                
            with gr.Column():
                translated_english = gr.Textbox(
                    label="Translated English Text",
                    lines=2
                )
                classified = gr.Textbox(
                    label="Sentiment Analysis Results",
                    lines=6
                )
                
        btn = gr.Button("Analyze Sentiment")
        btn.click(translate, inputs=[seed], outputs=[translated_english, classified])
        
        gr.Markdown("""
        ### How it works
        1. **Translation**: Amharic text is translated to English using Facebook's NLLB-200 model
        2. **Analysis**: The English text is analyzed using zero-shot classification to determine sentiment
        """)
    return demo

# Launch the app
if __name__ == "__main__":
    demo = build_interface()
    demo.launch()