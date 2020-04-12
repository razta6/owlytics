from os.path import join as pjoin

import torch.nn.functional as F

from deep_ts import LSTMClassifier
from utils import *
import time

import joblib

def predict_knn(X, src_path):
    cls = joblib.load(src_path)
    y_pred = cls.predict(X)
    return y_pred

def predict_rf(X, src_path):
    cls = joblib.load(src_path)
    X = get_cwt_features(X)
    y_pred = cls.predict(X)
    return y_pred

def predict_lstm(X, src_path, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X.shape[1]
    output_dim = 5

    model = LSTMClassifier(input_dim, params["hidden_dim"], params["layer_dim"], output_dim, device)
    model.load_state_dict(torch.load(src_path))

    model.eval()

    test_dl = DataLoader(create_test_dataset(X), batch_size=64, shuffle=False)

    y_pred = []

    print('Predicting on test dataset')
    for batch, _ in test_dl:
        batch = batch.permute(0, 2, 1)
        out = model(batch.to(device))
        y_hat = F.log_softmax(out, dim=1).argmax(dim=1)
        y_pred += y_hat.tolist()

    return np.array(y_pred)

def main(test_path, model, params):
    if model not in ["knn_minkowski", "knn_dtw", "rf", "lstm"]:
        raise ValueError("Model not supported")

    src_path = pjoin("ckpts", f"{model}.{'pth' if model=='lstm' else 'pkl'}")

    features, labels = parse_dataset(test_path)
    features = preprocess_data(features, smoothing_window=params["smoothing_window"], downsample_window=params["downsample_window"])

    start_time = time.time()

    if "knn" in model:
        y_pred = predict_knn(features, src_path)
    if model=="rf":
        y_pred = predict_rf(features, src_path)
    if model=="lstm":
        y_pred = predict_lstm(features, src_path, params)

    end_time = time.time()

    print(f"Inference duration: {end_time-start_time:.2f} seconds")
    y_pred = label_decoder(y_pred)
    pd.DataFrame(y_pred).to_csv(pjoin("preds", f"{model}.csv"), index=False)

    cm = results_report(labels, y_pred, plot_cm=True)

    return y_pred, cm


# if __name__ == "__main__":
#     main()