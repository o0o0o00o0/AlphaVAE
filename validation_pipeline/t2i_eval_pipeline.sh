#!/bin/bash

INPUT_GT_DIR="./data/Diffusion_test_data"
INPUT_INFER_DIR="/path/to/your/t2i_results"

OUTPUT_ROOT="path/to/your/output_root"

BLEND_INPUT_GT_DIR="./validation_pipeline/local_blended_GT_data"
BLEND_INPUT_INFER_DIR="./validation_pipeline/local_blended_infer_data"

BLEND_SCRIPT="./validation_pipeline/scripts/blend_rgba.py"
EVAL_SCRIPT="./validation_pipeline/scripts/t2i_eval_pipe.py"
JSON2CSV_SCRIPT="./validation_pipeline/scripts/t2i_dump2csv.py"   

CONFIG_FILE="${OUTPUT_ROOT}/final_outputs.json"
REPORTS_PATH="${OUTPUT_ROOT}/image_quality_report.json"
CSV_OUTPUT="${OUTPUT_ROOT}/image_quality_report.csv" 

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