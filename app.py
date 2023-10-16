import gradio as gr
from gradio.components import Text
from pipeline import transcription_classification_pipeline



demo = gr.Interface(
    title="Sentimment Analysis - FRENCH",
    fn=transcription_classification_pipeline,
    inputs = [gr.Audio(source="microphone", type="filepath"), 
              gr.Audio(source="upload", type="filepath")
            ],
    outputs = [Text(label="Transcription"), Text(label="Prediction")]
)


demo.launch()
