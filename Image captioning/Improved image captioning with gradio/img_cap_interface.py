import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.array):
    #convert the raw image into workable state for the ai
    raw_image = Image.fromarray(input_image).convert('RGB')

    inputs = processor(images=raw_image, return_tensors='pt')

    #Generate output
    outputs = model.generate(**inputs, max_length=50)

    #Decode the image
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption

iface = gr.Interface(
    fn=caption_image, 
    inputs=gr.Image(), 
    outputs="text",
    title="Image Captioning",
    description="Caption generation for provided images"
)

iface.launch()