export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export INSTANCE_DIR="data/sofa"
export Test_DIR="data/sofa_test/H-5168-12_8c8cc93e-163e-47a6-b8d0-4d3253f0b86b_900x.jpg"
export OUT_DIR="out/sofa"
export INSTANCE_PROMPT="sofa"
export MODEL_DIR="logs/sofa1"

# preprocess data
python src/preprocess/preprocess.py --instance_data_dir $INSTANCE_DIR \
                     --instance_prompt $INSTANCE_PROMPT

# CUDA_VISIBLE_DEVICES=0
accelerate launch --num_processes 1 src/train/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$MODEL_DIR \
  --instance_prompt=$INSTANCE_PROMPT \
  --resolution=256 \
  --train_batch_size=1 \
  --learning_rate=1e-6 \
  --max_train_steps=1000 \
  --checkpointing_steps 1000

# python inference_lora.py --image_path $Test_DIR \
#                     --model_path $MODEL_DIR \
#                     --out_path $OUT_DIR \
#                     --instance_prompt $INSTANCE_PROMPT


python src/gradio/demo_gradio.py --image_path $Test_DIR \
                    --model_path $MODEL_DIR \
                    --out_path $OUT_DIR \
                    --instance_prompt $INSTANCE_PROMPT                    