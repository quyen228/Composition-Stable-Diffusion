from PIL import Image, ImageFilter
import torch
import argparse
from torchvision import transforms
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import gradio as gr

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of preprocessing daa.")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to source directory.",
    )

    parser.add_argument(
        "--instance_prompt",
        type=str,
        required=True,
        help="Path to output directory.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt.",
    )

    args = parser.parse_args()

    return args


def inference_lora(img_path, instance_prompt, prompt):        
    
    init_image = Image.open(img_path).convert("RGB")
    init_size = init_image.size
    init_image = init_image.resize((512, 512))

    inputs_clipseg = processor_clipseg(text=[instance_prompt], images=[init_image], padding="max_length", return_tensors="pt").to(device)
    outputs = model_clipseg(**inputs_clipseg)
    preds = outputs.logits.unsqueeze(0)[0].detach().cpu()
    mask_image = transforms.ToPILImage()(torch.sigmoid(preds)).convert("L").resize((512, 512))
    mask_image = mask_image.filter(ImageFilter.MaxFilter(11))
    
    image = pipe(prompt=prompt, image=init_image, 
                mask_image=mask_image
                ).images[0]
 
    return image.resize(init_size)



if __name__ == '__main__':
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "runwayml/stable-diffusion-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_name)    
    pipe = pipe.to(device)

    # os.makedirs(f'{args.out_path}', exist_ok=True)

    # clipseg for image segmentation
    processor_clipseg = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model_clipseg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model_clipseg.to(device)

    
    # out_img = inference_lora(args.image_path, args.instance_prompt, args.prompt)

    with gr.Blocks() as demo:
        gr.Markdown("Inpainting image with image and prompt")
        # with gr.Tab("Non Fine-tune"):
        # Define button
        image_file = gr.Image(type="filepath", label="Input Image")
        instance_prompt = gr.Textbox(label="Instance Prompt")    
        prompt = gr.Textbox(label="Prompt")
        out_img = gr.Image(type="pil", show_download_button=True, label="Output Image")

        inpating_button = gr.Button("Inpanting Image")    
        # Click action
        inpating_button.click(inference_lora, inputs=[image_file, instance_prompt, prompt], outputs=out_img) 

        # with gr.Tab("Fine-tune"):
        #     file_output = gr.Image()
        #     upload_button = gr.UploadButton("Click to Upload a File", file_types=["image"], file_count="multiple")
        #     upload_button.upload(upload_image, upload_button, file_output)
        
        demo.launch(share=True, debug=True)
        