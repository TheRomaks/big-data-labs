# txt_to_csv.py
import pandas as pd

def parse_poem_song_txt(input_file="text.txt", output_csv="songs_n_poems.csv"):
    documents = []
    current_class = None
    current_lines = []

    with open(input_file, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")           # сохраняем оригинальные переносы
            stripped = line.strip()

            if stripped in ("0", "1"):
                # сохраняем предыдущий блок, если он был
                if current_lines and current_class is not None:
                    full_text = "\n".join(current_lines).strip()
                    documents.append({
                        "raw_text": full_text,
                        "label": int(current_class),
                        "class_name": "песня" if current_class == "0" else "стих",
                        "source": input_file
                    })
                current_lines = []
                current_class = stripped
                continue

            if not stripped and current_lines:
                # пустая строка → завершаем текст
                full_text = "\n".join(current_lines).strip()
                if current_class is not None:
                    documents.append({
                        "raw_text": full_text,
                        "label": int(current_class),
                        "class_name": "песня" if current_class == "0" else "стих",
                        "source": input_file
                    })
                current_lines = []
                continue

            current_lines.append(line)

    # последний блок
    if current_lines and current_class is not None:
        full_text = "\n".join(current_lines).strip()
        documents.append({
            "raw_text": full_text,
            "label": int(current_class),
            "class_name": "песня" if current_class == "0" else "стих",
            "source": input_file
        })

    df = pd.DataFrame(documents)
    df = df[["raw_text", "label", "class_name", "source"]]

    # короткий предпросмотр для удобства
    df["preview"] = df["raw_text"].str.replace(r"\n", " ", regex=True).str[:90] + "..."

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Сохранено {len(df)} текстов → {output_csv}")
    print("\nРаспределение:")
    print(df["class_name"].value_counts())
    print("\nПервые 3 записи:")
    print(df.head(3)[["label", "class_name", "preview"]])

if __name__ == "__main__":
    parse_poem_song_txt()