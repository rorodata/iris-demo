from __future__ import print_function
import joblib
import os
from sklearn.datasets import load_iris

MODEL_FILE = os.getenv("MODEL_PATH", "iris_model.pkl")

model = None
iris_classes = load_iris()['target_names']

def get_model():
    global model
    if model is None:
        model = load_model()
    return model

def load_model():
    if not os.path.exists(MODEL_FILE):
        print("Unable to find the model file iris_model.pkl", file=sys.stderr)
        return None
    return joblib.load(MODEL_FILE)

def predict(sepal_length, sepal_width, petal_length, petal_width):
    model = get_model()
    if not model:
        return 'error-no-model'

    row = [sepal_length, sepal_width, petal_length, petal_width]
    dataset = [row]
    result = model.predict(dataset)
    class_name = iris_classes[result[0]]
    print("predict: {} -> {}".format(row, class_name))
    return class_name
