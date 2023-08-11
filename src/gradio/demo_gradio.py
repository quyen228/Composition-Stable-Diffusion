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
        "--model_path",
        type=str,        
        help="Model path",
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
    if args.model_path:
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        pipe.unet.load_attn_procs(args.model_path)

    # os.makedirs(f'{args.out_path}', exist_ok=True)

    # clipseg for image segmentation
    processor_clipseg = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model_clipseg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model_clipseg.to(device)
    

    with gr.Blocks() as demo:
        gr.Markdown("Inpainting image with image and prompt")
        instance_prompt = gr.Textbox(label="Instance Prompt", value="sofa")    
        prompt = gr.Textbox(label="Prompt", value="a photo with sks sofa")
        
        with gr.Row():
            # Define button
            image_file = gr.Image(type="filepath", label="Input Image", value="/home/tndquyen/Documents/Composition-Stable-Diffusion/test/sofa_test/H-5168-12_8c8cc93e-163e-47a6-b8d0-4d3253f0b86b_900x.jpg")
            out_img = gr.Image(type="pil", show_download_button=True, label="Output Image")            
            
        inpating_button = gr.Button("Inpanting Image")    
        # Click action
        inpating_button.click(inference_lora, inputs=[image_file, instance_prompt, prompt], outputs=out_img)         
        
    demo.launch(share=True, debug=True)
        