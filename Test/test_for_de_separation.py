from keras.models import load_model

model_save_path = "../Models/ModelForDe/model.h5"
tokenizer_save_path = "../Models/ModelForDe/tokenizer.pickle"

# Load the model
loaded_model = load_model(model_save_path)

# Optionally, load the tokenizer
with open(tokenizer_save_path, 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)


# Load and predict on sentences from "test_sentences_de.txt"
test_sentences_path = "../Data/test_sentences_de.txt"

with open(test_sentences_path, 'r', encoding='utf-8') as file:
    test_sentences = file.readlines()

treshold = 3.5015692567696988e-09

# Predictions
for sentence_to_predict in test_sentences:
  print(f'\nInput Sentence: {sentence_to_predict}')
  X_new = tokenizer.texts_to_sequences([sentence_to_predict])
  X_new = pad_sequences(X_new, maxlen=X.shape[1])

  prediction = model.predict(X_new)[0][0]
  prediction_label = 'Error Detected! Separate it!' if prediction < treshold else 'Correct :)'

  print(f'Prediction: {prediction_label} (Probability: {prediction})')
