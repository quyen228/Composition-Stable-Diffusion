export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export INSTANCE_DIR="data/T-shirt2"
export Test_DIR="test/T-shirt_test"
export OUT_DIR="out/T-shirt2"
export INSTANCE_PROMPT="T-shirt"
export MODEL_DIR="logs/T-shirt2"

# preprocess data
python src/preprocess/preprocess.py --instance_data_dir $INSTANCE_DIR \
                     --instance_prompt $INSTANCE_PROMPT

CUDA_VISIBLE_DEVICES=0, 1
accelerate launch --num_processes 2 src/train/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$MODEL_DIR \
  --instance_prompt=$INSTANCE_PROMPT \
  --resolution=256 \
  --train_batch_size=1 \
  --learning_rate=1e-6 \
  --max_train_steps=5000 \
  --checkpointing_steps 1000

python src/inference/inference_lora.py --image_path $Test_DIR \
                    --model_path $MODEL_DIR \
                    --out_path $OUT_DIR \
                    --instance_prompt $INSTANCE_PROMPT
