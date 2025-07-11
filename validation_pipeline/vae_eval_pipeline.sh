INPUT_GT_DIR="./data/VAE_test_data"
INPUT_INFER_DIR="/path/to/your/vae_results"

BLEND_INPUT_GT_DIR="./validation_pipeline/local_blended_eval_data"
BLEND_INPUT_INFER_DIR="./validation_pipeline/local_blended_infer_data"

BLEND_SCRIPT="./validation_pipeline/scripts/blend_rgba.py"
EVAL_SCRIPT="./validation_pipeline/scripts/i2i_eval_pipe.py"

REPORTS_PATH="${INPUT_INFER_DIR}/image_quality_report.json"

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


CSV_OUTPUT="${INPUT_INFER_DIR}/image_quality_report.csv"
JSON2CSV_SCRIPT="./scripts/result_json2csv.py"

echo "📄 Converting JSON to CSV..."
python "$JSON2CSV_SCRIPT" \
    --input "$REPORTS_PATH" \
    --output "$CSV_OUTPUT"

echo "✅ CSV saved to $CSV_OUTPUT"

rm -rf ${BLEND_INPUT_GT_DIR}
rm -rf ${BLEND_INPUT_INFER_DIR}

