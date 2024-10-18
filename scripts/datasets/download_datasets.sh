#!/bin/bash
source scripts/datasets/utils.sh

download_aya_collection() {
    DOWNLOAD_DIR="$1"
    DOWNLOAD_CONFIG="all"   # TODO: @theyorubayesian Allow subset downlaod

    AYA_TARESCO_URL="https://huggingface.co/datasets/taresco/aya_african_subset/resolve/main/data/dataset_name/dataset_split/language.jsonl"
    AYA_DATASET_LANGUAGES=(
        "afrikaans" "algerian_arabic" "amharic"
        "egyptian_arabic" "english" "french"
        "hausa" "igbo" "kinyarwanda"
        "moroccan_arabic" "mozambican_portuguese" "nyanja"
        "plateau_malagasy" "portuguese" "shona"
        "somali" "swahili" "tunisian_arabic"
        "shona" "southern_sotho" "xhosa"
        "yoruba" "zulu" "bemba"
        "central_kanuri" "fon" "twi" "wolof"
    )
    HUMAN_ANNOTATED=("aya-dataset")
    TEMPLATED_DATASETS=(
        "afriqa-inst" "afrisenti-inst" "amharic_qa"
        "joke-explaination-inst" "masakhanews-inst"
        "mintaka-inst" "ntx-llm-inst" "nusax-senti-inst"
        "scirepeval-biomimicry-inst" "soda-inst" "uner-llm-inst"
        "wiki-split-inst" "xlel_wd-inst" "xwikis-inst"
    )
    TRANSLATED_DATASETS=(
        "adversarial_qa_(t)" "cnn-daily-mail_(t)" "dolly-v2_(t)"
        "flan-coqa_(t)" "flan-cot-submix_(t)" "flan-gem-wiki-lingua_(t)"
        "flan-lambada_(t)" "flan-unified-qa_(t)" "hotpotqa_(t)"
        "joke-explaination-inst_(t)" "mintaka-inst_(t)" "mlqa-en_(t)"
        "nq-open_(t)" "paws-wiki_(t)" "piqa_(t)" "soda-inst_(t)"
        "wiki_qa_(t)" "wiki-split-inst_(t)" "xlel_wd-inst_(t)"
    )

    if [[ $DOWNLOAD_CONFIG == "all" ]]; then
        DATASETS=("${HUMAN_ANNOTATED[@]}" "${TEMPLATED_DATASETS[@]}" "${TRANSLATED_DATASETS[@]}")
    fi

    for dataset in "${DATASETS[@]}"; do
        dataset_dir="$DOWNLOAD_DIR/$dataset"
        dataset_url="${AYA_TARESCO_URL/dataset_name/"$dataset"}"

        for language in "${AYA_DATASET_LANGUAGES[@]}"; do

            language_url="${dataset_url/language/"$language"}"

            for split in "train" "validation" "test"; do
                split_dir="$dataset_dir/$split"
                mkdir -p "$split_dir"

                language_file_path="$split_dir/$language.jsonl"

                if [ -f "$language_file_path" ]; then
                    echo "$dataset:$language/$split already exists"
                else
                    URL="${language_url/dataset_split/"$split"}"

                    # check if url exists
                    if wget --header="Authorization: Bearer $HUGGINGFACE_CLI_TOKEN" --spider "${URL}" 2>/dev/null; then
                        echo "Downloading $dataset:$language/$split ......"
                        wget  --header="Authorization: Bearer $HUGGINGFACE_CLI_TOKEN"  -O "$language_file_path" "$URL"
                    else
                        echo " Online URL ${URL} not found !"
                    fi
                fi
            done
        done
    done
}

download_flan_collection() {
    DOWNLOAD_DIR="$1"

    TARESCO_URL="https://huggingface.co/datasets/taresco/flan_subset/resolve/main/data/train-file_idx-of-total_files.parquet"

    declare -A SUBMIX_MAP=(
        [flan2021_submix_filtered]=11
        [niv2_submix_filtered]=18
        [cot_submix_filtered]=1
    )

    for submix in "${!SUBMIX_MAP[@]}"; do
        mkdir -p "$DOWNLOAD_DIR/$submix"

        total_files=${SUBMIX_MAP[$submix]}
        padded_total_files=$(printf "%05d" "$total_files")

        URL=${TARESCO_URL/flan_subset/"$submix"}
        URL=${URL/total_files/"$padded_total_files"}

        for i in $(seq 0 $((total_files - 1))); do
            idx=$(printf "%05d" $i)
            _URL=${URL/file_idx/$idx}
            
            wget -P "$DOWNLOAD_DIR/$submix" "$_URL"
        done

        for file in "$DOWNLOAD_DIR/$submix"/*.parquet; do
            convert_parquet_to_jsonl "$file" && rm "$file"
        done
    done

    DPI_URL="https://huggingface.co/datasets/DataProvenanceInitiative/flan_subset/resolve/main/data"

    T0_SUBMIX_DIR="$DOWNLOAD_DIR/t0_submix"
    DIALOG_SUBMIX_DIR="$DOWNLOAD_DIR/dialog_submix"
    mkdir -p  "$T0_SUBMIX_DIR" "$DIALOG_SUBMIX_DIR"

    TO_SUBMIX_URL="${DPI_URL/flan_subset/t0_submix_original}/train-00000-of-00001-0a6693a25fc4a25e.parquet"
    wget -P "$T0_SUBMIX_DIR" "$TO_SUBMIX_URL" && convert_parquet_to_jsonl "$T0_SUBMIX_DIR"/*.parquet && rm "$T0_SUBMIX_DIR"/*.parquet

    DIALOG_SUBMIX_URL="${DPI_URL/flan_subset/dialog_submix_original}/train-00000-of-00001-0aecb489ddece98b.parquet"
    wget -P "$DIALOG_SUBMIX_DIR" "$DIALOG_SUBMIX_URL" && convert_parquet_to_jsonl "$DIALOG_SUBMIX_DIR"/*.parquet && rm "$DIALOG_SUBMIX_DIR"/*.parquet
}

download_xp3x_filtered() {
    DOWNLOAD_DIR="$1"

    BASE_XP3X_URL="https://huggingface.co/datasets/taresco/xP3x_african_subset/resolve/main"

    mkdir -p "$DOWNLOAD_DIR"
    XP3X_PATHS_URL="$BASE_XP3X_URL/paths.json"
    wget --header="Authorization: Bearer $HUGGINGFACE_CLI_TOKEN" -O "$DOWNLOAD_DIR/paths.json" "$XP3X_PATHS_URL"

    jq -r 'to_entries[] | "\(.key) \(.value[])"' "$DOWNLOAD_DIR/paths.json" | while read -r key url; do
        url="$(echo "$url" | sed 's/?/%3F/g' | sed 's/,/%2C/g')"
        mkdir -p "$DOWNLOAD_DIR/${key}"

        if [ ! -f "$DOWNLOAD_DIR/${key}/${url/"data/${key}/"/}" ]; then
            wget --header="Authorization: Bearer $HUGGINGFACE_CLI_TOKEN" \
            -x -O "$DOWNLOAD_DIR/${key}/${url/"data/${key}/"/}" "${BASE_XP3X_URL}/${url}"
        fi
    done
}

download_octopack_osst() {
    local DOWNLOAD_DIR="$1"
    mkdir -p "$DOWNLOAD_DIR"
    wget -P "$DOWNLOAD_DIR" https://huggingface.co/datasets/bigcode/oasst-octopack/resolve/main/oasst_octopack.jsonl

    for lang in "en" "fr" "pt-BR"; do
        jq -c "select(.lang | IN(\"${lang}\"))" "$DOWNLOAD_DIR/oasst_octopack.jsonl" > "$DOWNLOAD_DIR/oasst_octopack_${lang}.jsonl"
    done
    
    rm "$DOWNLOAD_DIR/oasst_octopack.jsonl"
}

download_oig_small_chip2() {
    local DOWNLOAD_DIR="$1"
    mkdir -p "$DOWNLOAD_DIR"
    wget -P "$DOWNLOAD_DIR" "https://huggingface.co/datasets/0-hero/OIG-small-chip2/resolve/main/data/train-00000-of-00001-34df73631d1c6428.parquet" && \
        convert_parquet_to_jsonl "$DOWNLOAD_DIR/train-00000-of-00001-34df73631d1c6428.parquet" && \
        rm "$DOWNLOAD_DIR/train-00000-of-00001-34df73631d1c6428.parquet"
}

# Needs to be filtered for tasks that overlap evaluation data
# Or evaluation task categories - textual entailment, co-reference resolution and
# setence comparison tasks
download_tasksource_instruct() {
    local DOWNLOAD_DIR="$1"
    mkdir -p "$DOWNLOAD_DIR"
    git clone https://huggingface.co/datasets/taresco/tasksource_instruct_filtered "$DOWNLOAD_DIR-temp" && \
        cd "$DOWNLOAD_DIR-temp" && git lfs pull && cd .. && \
        mv "$DOWNLOAD_DIR"-temp/data/* "$DOWNLOAD_DIR" &&
        rm -rf "$DOWNLOAD_DIR-temp"

    for split in "$DOWNLOAD_DIR"/*; do
        convert_parquet_to_jsonl "$split" && rm "$split"
    done
}

# A subset of the Flan collection 
# No code datasets included.
# Sample a maximum of 20K for each of these sources.

# ShareGPT - Not Included
export HUGGINGFACE_CLI_TOKEN="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"
# download_xp3x_filtered /share/jimmylin/collections/xP3x
# download_octopack_osst /share/jimmylin/collections/octopack_osst
# download_oig_small_chip2 /share/jimmylin/collections/OIG-small-chip2
download_tasksource_instruct /share/jimmylin/collections/tasksource_instruct