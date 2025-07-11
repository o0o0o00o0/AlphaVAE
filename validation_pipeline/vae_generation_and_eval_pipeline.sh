model_name=./models/finetune_VAE
output_dataset=./results/validation/VAE_results

dataset_root_dir=./data/VAE_test_data
for test_dir in ${dataset_root_dir}/*; do
    test_dir_name=$(basename ${test_dir})
    echo "test_dir_name: ${test_dir_name}"
    MASTER_PORT=$((RANDOM % (30000 - 20000 + 1) + 20000))
    input_dataset_dir=${test_dir}
    output_dataset_dir=${output_dataset}/${test_dir_name}

    for input_dir in ${input_dataset_dir}/*; do
        input_dir_name=$(basename ${input_dir})
        output_dir=${output_dataset_dir}/${input_dir_name}
        python inference/infer_vae.py \
        --pretrained_vae_path ${model_name}  \
        --input_dir ${input_dir} \
        --output_dir ${output_dir} \
        --resolution 1024 \
        --dtype bf16
    done
done

echo "📂 Processing model: $model_name"

INPUT_GT_DIR=$dataset_root_dir
INPUT_INFER_DIR=$output_dataset

BLEND_INPUT_GT_DIR="./validation_pipeline/local_blended_eval_data"
BLEND_INPUT_INFER_DIR="./validation_pipeline/local_blended_infer_data"

BLEND_SCRIPT="./validation_pipeline/scripts/blend_rgba.py"
EVAL_SCRIPT="./validation_pipeline/scripts/i2i_eval_pipe.py"

REPORTS_PATH="$output_dataset/image_quality_report.json"

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


echo "🗂️ Copying original GT and Infer images to 'rgba' folder..."

GT_RGBA_DIR="$BLEND_INPUT_GT_DIR/rgba"
INFER_RGBA_DIR="$BLEND_INPUT_INFER_DIR/rgba"

mkdir -p "$GT_RGBA_DIR"
mkdir -p "$INFER_RGBA_DIR"

cp -r "$INPUT_GT_DIR/"* "$GT_RGBA_DIR"
cp -r "$INPUT_INFER_DIR/"* "$INFER_RGBA_DIR"

echo "✅ Done copying 'rgba' images."
echo "---------------------------"


echo "📊 Starting evaluation..."
python "$EVAL_SCRIPT" \
    --input_root "$BLEND_INPUT_GT_DIR" \
    --output_root "$BLEND_INPUT_INFER_DIR" \
    --output_json "$REPORTS_PATH"
echo "✅ Evaluation completed. Results saved to image_quality_report.json"


CSV_OUTPUT="$output_dataset/image_quality_report.csv"
JSON2CSV_SCRIPT="./validation_pipeline/scripts/result_json2csv.py"

echo "📄 Converting JSON to CSV..."
python "$JSON2CSV_SCRIPT" \
    --input "$REPORTS_PATH" \
    --output "$CSV_OUTPUT"

echo "✅ CSV saved to $CSV_OUTPUT"

rm -rf ${BLEND_INPUT_GT_DIR}
rm -rf ${BLEND_INPUT_INFER_DIR}
