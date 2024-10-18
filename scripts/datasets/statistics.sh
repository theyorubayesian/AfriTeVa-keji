#!/bin/bash

calculate_xp3x_statistics() {
    XP3X_DIR=$1
    output_file="$XP3X_DIR/statistics.json"

    json_output="{}"

    for lang_dir in "$XP3X_DIR"/*; do
        lang=${lang_dir##*/}

        if [ "$lang" = "paths.json" ]; then
            continue
        fi

        lines=$(cat "$lang_dir"/*.jsonl | wc -l)

        json_output=$(echo "$json_output" | jq --arg lang "$lang" --argjson lines "$lines" '. + {($lang): $lines}')
    done

    echo "$json_output" > "$output_file"
}

calculate_flan_collection_statistics() {
    FLAN_COLLECTION_DIR=$1
    output_file="$FLAN_COLLECTION_DIR"/statistics.json

    json_output="{}"

    for submix_dir in "$FLAN_COLLECTION_DIR"/*; do
        lines=$(cat "$submix_dir"/*.json | wc -l)

        submix="${submix_dir##*/}"
        submix="${submix/'_filtered'}"

        json_output=$(echo "$json_output" | jq --arg submix "$submix" --argjson lines "$lines" '. + {($submix): $lines}')
    done

    echo "$json_output" > "$output_file"
}

calculate_aya_collection_statistics() {
    AYA_COLLECTION_DIR="$1"

    output_file="$AYA_COLLECTION_DIR/statistics.json"

    json_data=$(jq -n '{}')

    for dataset_dir in "$AYA_COLLECTION_DIR"/*; do
        dataset_name=$(basename "$dataset_dir")

        # Initialize JSON object for dataset
        dataset_json=$(jq -n '{}')

        for split in train validation test; do
            split_dir="$dataset_dir/$split"

            if [ -d "$split_dir" ]; then
                # Initialize JSON object for split
                split_json=$(jq -n '{}')

                for file in "$split_dir"/*.jsonl; do
                    if [ -f "$file" ]; then
                        lang=$(basename "$file" .jsonl)
                        line_count=$(wc -l < "$file")

                        # Add language line count to split JSON
                        split_json=$(echo "$split_json" | jq --arg lang "$lang" --argjson line_count "$line_count" '. + {($lang): $line_count}')
                    fi
                done

                # Add split JSON object to dataset JSON
                dataset_json=$(echo "$dataset_json" | jq --arg split "$split" --argjson split_json "$split_json" '. + {($split): $split_json}')
            fi
        done

        # Add dataset JSON object to main JSON data
        json_data=$(echo "$json_data" | jq --arg dataset "$dataset_name" --argjson dataset_json "$dataset_json" '. + {($dataset): $dataset_json}')
    done

    # Save JSON data to output file
    echo "$json_data" > "$output_file"
}

calculate_octopack_statistics() {
    OCTOPACK_DIR="$1"
    output_file="${OCTOPACK_DIR}/statistics.json"

    json_output="{}"

    for language_file in "$OCTOPACK_DIR"/*; do
        language="${language_file##*_}"
        language="${language/'.jsonl'/}"

        lines=$(wc -l < "$language_file")
        json_output=$(echo "$json_output" | jq --arg language "$language" --argjson lines "$lines" '. + {($language): $lines}')
    done

    echo "$json_output" > "$output_file"
}

calculate_oig_small_chip2_statistics() {
    OIG_SMALL_CHIP_DIR="$1"
    json_output="{}"
    lines=$(wc -l < "$OIG_SMALL_CHIP_DIR"/train*.json)
    json_output=$(echo "$json_output" | jq --arg split "train" --argjson lines "$lines" '. + {($split): $lines}')
    echo "$json_output" > "$OIG_SMALL_CHIP_DIR/statistics.json"
}

calculate_tasksource_instruct_statistics() {
    local TASKSOURCE_DIR="$1"
    json_output="{}"

    for split in "train" "validation" "test"; do
        lines=$(cat "$TASKSOURCE_DIR"/$split*.json | wc -l)
        json_output=$(echo "$json_output" | jq --arg split "$split" --argjson lines "$lines" '. + {($split): $lines}')
    done

    echo "$json_output" > "$TASKSOURCE_DIR/statistics.json"
}
# calculate_oig_small_chip2_statistics /share/jimmylin/collections/OIG-small-chip2
calculate_tasksource_instruct_statistics /share/jimmylin/collections/tasksource_instruct