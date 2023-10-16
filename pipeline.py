from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


model_name = 'serge-wilson/sentiment_analysis_fr'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

#Creation des pipelines
classifier = pipeline("text-classification", model = model,tokenizer = tokenizer) #pipeline pour la classification
transcriber = pipeline("automatic-speech-recognition", model="bhuang/asr-wav2vec2-french")  #pipeline pour la transcription


def transcription_classification_pipeline(audio):
  """
    Cette fonction fonction prend en entrée un audio et renvoie la transcription et la classe prédite
  """

  #On passe l'argument "audio" au pipeline transcriber, on repurère le text et on le stocke dans la variable transcription
  transcription = transcriber(audio)["text"]

  #On passe la variable "transcription" au pipeline classifier et on stocke la valeur de retour(resultat) dans la variable "result"
  result = classifier(transcription, truncation=True)[0]

  #On recupère le label du resultat
  predicted_label = result.get("label")

  return transcription, predicted_label.capitalize()