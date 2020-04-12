from os.path import join as pjoin
import joblib
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from dtaidistance import dtw
from torch import nn

from deep_ts import LSTMClassifier, Trainer
from utils import *

def train_knn(X, y, dest_path, params=None, model='auto'):
    metric = model.split("_")[-1]
    if params is None:
        if metric=="dtw":
            params = {
                "smoothing_window": 1,
                "downsample_window": 1,
                "n_neighbors": 1,
                "dwt_window": 60
            }
        else:
            params = {
                "smoothing_window": 1,
                "downsample_window": 1,
                "n_neighbors": 1,
            }

    print("Start training")
    if metric=="dtw":
        cls = KNeighborsClassifier(n_neighbors=params["n_neighbors"], metric=dtw.distance_fast, metric_params={"window": params["dwt_window"]})
    else:
        cls = KNeighborsClassifier(n_neighbors=params["n_neighbors"], metric=metric)

    cls.fit(X, y)

    print(f"Saving to file: {dest_path}")
    joblib.dump(cls, dest_path)

    return cls

def train_rf(X, y, dest_path, params=None):

    if params is None:
        params = {
            "smoothing_window": 1,
            "downsample_window": 1,
            "decomp_level": 1,
            "wavelet": "db2",
            "max_depth": 16,
            "n_estimators": 100,
        }


    X = get_cwt_features(X, params["decomp_level"], params["wavelet"])

    print("Start training")
    cls = RandomForestClassifier(max_depth=params["max_depth"], n_estimators=params["n_estimators"])

    cls.fit(X, y)

    print(f"Saving to file: {dest_path}")
    joblib.dump(cls, dest_path)

    return cls

def train_lstm(X, y, dest_path, params=None):
    if params is None:
        params = {
            "smoothing_window": 1,
            "downsample_window": 1,
            "layer_dim": 5,
            "hidden_dim": 512
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trn_ds_list, val_ds_list = create_datasets(X, y.values, cv=0)
    batch_size = X.shape[0]
    data_loaders = [create_loaders(trn_ds, val_ds, batch_size) for trn_ds, val_ds in zip(trn_ds_list, val_ds_list)]

    for k, (trn_dl, val_dl) in enumerate(data_loaders):
        input_dim = X.shape[1]
        output_dim = 5

        n_epochs = 1000
        patience = 100
        lr = 0.0005
        iterations_per_epoch = len(trn_dl)

        model = LSTMClassifier(input_dim, params["hidden_dim"], params["layer_dim"], output_dim, device)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.RMSprop(model.parameters(), lr=lr)
        sched = CyclicLR(opt, cosine(t_max=iterations_per_epoch * 2, eta_min=lr / 100))

        Trainer(model, trn_dl, val_dl, n_epochs, sched, opt, criterion, device, patience, dest_path)

    return model


def main(train_path, model, params=None):
    if model not in ["knn_minkowski", "knn_dtw", "rf", "lstm"]:
        raise ValueError("Model not supported")


    dest_path = pjoin("ckpts", f"{model}.{'pth' if model=='lstm' else 'pkl'}")

    features, labels = parse_dataset(train_path)
    if params is None:
        params = {
            "smoothing_window": 1,
            "downsample_window": 1,
        }
        features = preprocess_data(features, smoothing_window=params["smoothing_window"], downsample_window=params["downsample_window"])
    else:
        features = preprocess_data(features, smoothing_window=params["smoothing_window"],
                                   downsample_window=params["downsample_window"])

    labels = label_encoder(labels)

    start_time = time.time()

    if "knn" in model:
        cls = train_knn(features, labels["1"], dest_path, params, model=model)
    if model=="rf":
        cls = train_rf(features, labels["1"], dest_path, params)
    if model=="lstm":
        cls = train_lstm(features, labels["1"], dest_path, params)

    end_time = time.time()

    print(f"Training duration: {end_time-start_time:.2f} seconds")

    return cls



# if __name__ == "__main__":
#     main()