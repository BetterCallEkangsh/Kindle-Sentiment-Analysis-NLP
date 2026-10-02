# Run this as a new cell at the END of Kindle_Sentiment_Analysis.ipynb
import joblib

model.wv.save("w2v.kvmodel")        # your trained 200-d Word2Vec vectors
joblib.dump(rnf, "rf_model.joblib") # your trained Random Forest

# Colab: download both files, then put them next to app.py
from google.colab import files
for f in ["w2v.kvmodel", "w2v.kvmodel.vectors.npy", "rf_model.joblib"]:
    try:
        files.download(f)
    except Exception as e:
        print(f, e)
