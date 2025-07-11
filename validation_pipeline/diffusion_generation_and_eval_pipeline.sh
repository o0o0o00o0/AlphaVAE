#!/bin/bash

ROOT="./data/Diffusion_test_data"

GENERATE_ROOT="./results/validation/T2I_results"
VAE_dir="./models/finetune_VAE"
diffusion_dir="${VAE_dir}/finetune_diffusion"

# input
DATA_ROOT="$ROOT/images"
PROMPT_ROOT="$ROOT/prompts"
# output
OUTPUT_ROOT="$GENERATE_ROOT/generated_outputs"
FINAL_JSON="$GENERATE_ROOT/final_outputs.json"

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$PROMPT_ROOT"

CATEGORIES=("animal" "fruit" "furniture" "mixed" "plant" "portrait" "toy" "transparent")

for i in "${!CATEGORIES[@]}"; do
  CATEGORY=${CATEGORIES[$i]}
  CATEGORY_PROMPT_DIR="$PROMPT_ROOT/$CATEGORY"
  CATEGORY_PROMPT_FILE="$CATEGORY_PROMPT_DIR/prompt.txt"
  CATEGORY_OUTPUT_DIR="$OUTPUT_ROOT/$CATEGORY"

  mkdir -p "$CATEGORY_PROMPT_DIR"
  mkdir -p "$CATEGORY_OUTPUT_DIR"
  
  python inference/infer_t2i.py \
      --pretrained_model_name_or_path "./models/FLUX.1-dev" \
      --pretrained_vae_model "${VAE_dir}" \
      --lora_path "${diffusion_dir}" \
      --prompt_file "$CATEGORY_PROMPT_FILE" \
      --output_dir "$CATEGORY_OUTPUT_DIR" \
      --num_images_per_prompt 1 \

  echo "✅ Finished: $CATEGORY"
done

wait
echo "🎉 All categories processed. Assembling final JSON..."


> "$FINAL_JSON"
CATEGORIES=("animal" "fruit" "furniture" "plant" "portrait" "toy" "transparent" "mixed")
echo "[" > "$FINAL_JSON"
first=true
for CATEGORY in "${CATEGORIES[@]}"; do
  CATEGORY_PROMPT_FILE="$PROMPT_ROOT/$CATEGORY/prompt.txt"
  CATEGORY_GT_DIR="$DATA_ROOT/$CATEGORY"
  CATEGORY_GEN_DIR="$OUTPUT_ROOT/$CATEGORY"

  if [ "$first" = true ]; then
    first=false
  else
    echo "," >> "$FINAL_JSON"
  fi

  echo "  {" >> "$FINAL_JSON"
  echo "    \"GT_file_path_dir\": \"$CATEGORY_GT_DIR\"," >> "$FINAL_JSON"
  echo "    \"gen_file_path_dir\": \"$CATEGORY_GEN_DIR\"," >> "$FINAL_JSON"
  echo "    \"prompt_file\": \"$CATEGORY_PROMPT_FILE\"" >> "$FINAL_JSON"
  echo -n "  }" >> "$FINAL_JSON"
done
echo "" >> "$FINAL_JSON"
echo "]" >> "$FINAL_JSON"

# eval

INPUT_GT_DIR="${DATA_ROOT}"
INPUT_INFER_DIR="${OUTPUT_ROOT}"

BLEND_INPUT_GT_DIR="./validation_pipeline/local_blended_GT_data"
BLEND_INPUT_INFER_DIR="./validation_pipeline/local_blended_infer_data"

CONFIG_FILE="${GENERATE_ROOT}/final_outputs.json"
BLEND_SCRIPT="./validation_pipeline/scripts/blend_rgba.py"
EVAL_SCRIPT="./validation_pipeline/scripts/t2i_eval_pipe.py"
JSON2CSV_SCRIPT="./validation_pipeline/scripts/t2i_dump2csv.py"   

REPORTS_PATH="${GENERATE_ROOT}/image_quality_report.json"
CSV_OUTPUT="${GENERATE_ROOT}/image_quality_report.csv"

COLORS=("gray" "black" "white" "red" "green" "blue" "yellow" "cyan" "magenta")

for COLOR in "${COLORS[@]}"; do
    echo "🎨 Blending color: $COLOR"

    GT_COLOR_DIR="$BLEND_INPUT_GT_DIR/$COLOR"
    INFER_COLOR_DIR="$BLEND_INPUT_INFER_DIR/$COLOR"

    if [ -d "$GT_COLOR_DIR" ] && [ -d "$INFER_COLOR_DIR" ]; then
        echo "⚠️  Skipping $COLOR: already blended."
        echo "---------------------------"
        continue
    fi

    echo "👉 Processing GT images..."
    python "$BLEND_SCRIPT" --input_dir "$INPUT_GT_DIR" --output_dir "$BLEND_INPUT_GT_DIR" --color "$COLOR"

    echo "👉 Processing OUTPUT images..."
    python "$BLEND_SCRIPT" --input_dir "$INPUT_INFER_DIR" --output_dir "$BLEND_INPUT_INFER_DIR" --color "$COLOR"

    echo "✅ Done: $COLOR"
    echo "---------------------------"
done


echo "🗂️ Checking if 'rgba' folders already exist..."

GT_RGBA_DIR="$BLEND_INPUT_GT_DIR/rgba"
INFER_RGBA_DIR="$BLEND_INPUT_INFER_DIR/rgba"

if [[ -d "$GT_RGBA_DIR" && -d "$INFER_RGBA_DIR" ]]; then
    echo "⚠️ 'rgba' folders already exist. Skipping copy."
else
    echo "📥 Copying original GT and Infer images to 'rgba' folder..."
    mkdir -p "$GT_RGBA_DIR"
    mkdir -p "$INFER_RGBA_DIR"

    cp -r "$INPUT_GT_DIR/"* "$GT_RGBA_DIR"
    cp -r "$INPUT_INFER_DIR/"* "$INFER_RGBA_DIR"

    echo "✅ Done copying 'rgba' images."
fi

echo "---------------------------"

echo "📊 Starting evaluation..."
python "$EVAL_SCRIPT" \
    --config_file "$CONFIG_FILE" \
    --blend_GT_dir "$BLEND_INPUT_GT_DIR" \
    --blend_infer_dir "$BLEND_INPUT_INFER_DIR" \
    --output_file "$REPORTS_PATH"
echo "✅ Evaluation completed. Results saved to $REPORTS_PATH"


echo "📄 Converting JSON to CSV..."
python "$JSON2CSV_SCRIPT" \
    --json_file "$REPORTS_PATH" \
    --output_csv "$CSV_OUTPUT"

echo "✅ CSV saved to $CSV_OUTPUT"

rm -rf $BLEND_INPUT_GT_DIR
rm -rf $BLEND_INPUT_INFER_DIR