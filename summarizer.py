import torch
from pathlib import Path
import argparse
from transformers import pipeline
from tqdm.auto import tqdm
import nltk
from nltk.corpus import stopwords
from gtts import gTTS
import os


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)


def read_and_split(file_path):
    with open(file_path, 'r') as f:
        text = f.read()

    sub_text = [' '.join(line.split('\n')) for line in text.split('<\section>') if line.strip() != '']
    sub_text = [remove_stopwords(line) for line in sub_text]
    right_length = []
    batch_len = 400
    for line in sub_text:
        sp = line.split(' ')
        for i in range(0, len(sp), batch_len):
            if i + batch_len > len(sp):
                right_length.append(' '.join(sp[i:]))
            else:
                right_length.append(' '.join(sp[i:i + batch_len]))
    return right_length


def summarize(list_of_text, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipeline("summarization", model=model, device=0 if device == "cuda" else -1)
    total_summary = []
    for line in tqdm(list_of_text):
        min_length = max(len(line) // 10, 10)
        summary = pipe(line, max_length=400, min_length=min_length, do_sample=False)[0]["summary_text"]
        total_summary.append(summary)
    return total_summary


def directory_maker():
    if Path.cwd().joinpath("data").exists() is False:
        Path.cwd().joinpath("data").mkdir()
    if Path.cwd().joinpath("summaries").exists() is False:
        Path.cwd().joinpath("summaries").mkdir()


def read_aloud(file_name):
    out_string = ''
    with open(f"{file_name}.txt", 'r') as f:
        text = f.read()
    for line in text.split('\n'):
        line = line.strip()
        if line == '':
            continue
        else:
            out_string += line + '. '

    gTTS(out_string, lang='en').save(f"{file_name}.mp3")
    os.system("mpg321 " + f"{file_name}.mp3")


def main():
    directory_maker()
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default='academic-practitioner-divide')
    parser.add_argument("--summarize", type=bool, default=True)
    parser.add_argument('--model', type=str, default='facebook/bart-large-cnn')
    parser.add_argument('--read_aloud', type=bool, default=True)
    args = parser.parse_args()
    if args.summarize:
        print('Summarizing')
        total_text = read_and_split(file_path = Path.cwd().joinpath("data", args.file_name + '.txt'))
        total_summary = summarize(total_text, model=args.model)

        with open(Path.cwd().joinpath("summaries", args.file_name + '-summary.txt'), 'w') as f:
            for line in total_summary:
                f.write(line + '\n')
        print('Finished summarizing')

    if args.read_aloud:
        print('Reading aloud')
        read_aloud(Path.cwd().joinpath("summaries", args.file_name + '-summary'))


if __name__ == '__main__':
    main()
