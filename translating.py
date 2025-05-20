import os
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model_en_da():
    """
    Load the Helsinki English to Danish machine translation model

    Returns:
        enda: the model, en -> da
    """
    enda = pipeline("translation", model="Helsinki-NLP/opus-mt-en-da")
    print("English to Danish translation model loaded.")
    return enda

def translate(model):
    """
    Running the translation model
    """
    filepaths = []
    for files in os.listdir("train_files"):
        filepaths.append(os.path.join(files))
        
    for files in filepaths:
        traindir = os.path.join("train_files", files)
        tldir = os.path.join("DONE", files)
        file = open(traindir,'r', encoding="utf8") # read file to be translated
        print("Translating " + files)
        for i in file.readlines():
            result = model(i)[0]['translation_text'] # translate
            with open(tldir[:-3] + "-tl.da", "a", encoding="utf8") as f: # make a new file and add the translation to it. add -tl and the correct language name to filename
                f.write(result + "\n")

def main():
    enda = load_model_en_da()
    translate(enda) # translate from english to danish

if __name__ == "__main__":
    main()