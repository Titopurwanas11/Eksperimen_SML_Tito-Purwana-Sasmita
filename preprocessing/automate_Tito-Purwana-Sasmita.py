from pandas._libs.hashtable import duplicated
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_database(input_path: str) -> pd.DataFrame:
    """
    Membaca dataset raw Bank Marketing.
    Dataset menggunakan delimiter titik koma (;).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File tidak ditemukan: {input_path}")

    df = pd.read_csv(input_path, sep=";")
    return df

def preprocess_data(
    input_path: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Tahap Preprocessing otomatis pada dataset Bank Marketing
    """
    os.makedirs(output_dir, exist_ok=True)

    df = load_database(input_path)
    print("Dataset berhasil dimuat.")
    print(f"Shape awal: {df.shape}")

    #validasi kolom target
    if "y" not in df.columns:
        raise ValueError("Kolom target 'y' tidak ditemukan dalam dataset")
    
    # Hapus data duplikat
    duplicated_count = df.duplicated().sum()
    if duplicated_count > 0:
        df = df.drop_duplicates()
        print(f"Data duplikat dihapus: {duplicated_count}")
    else:
        print("Tidak ditemukan data duplikat.")

    # hapus kolom duration
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])
        print("Kolom 'duration' dihapus dari dataset.")
    else:
        print("Kolom 'duration' tidak ditemukan dalam dataset.")
    
    # Encode Taget
    df["y"] = df["y"].map({
        "no": 0,
        "yes": 1
    })

    if df["y"].isnull().sum() > 0:
        raise ValueError("Terdapat nilai target selain 'yes' dan 'no'.")


    #Pisahkan Target
    x = df.drop(columns=["y"])
    y = df["y"]

    numeric_features = x.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = x.select_dtypes(include=["object"]).columns.tolist()

    print("Fitur numerik:", numeric_features)
    print("Fitur ketgorikal:", categorical_features)

    #Pembagian data latih dan data uji
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size = test_size,
        random_state = random_state,
        stratify= y
    )

    print(f"X_train shape: {x_train.shape}")
    print(f"X_test shape: {x_test.shape}")

    #pipeline preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    try:
        onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", onehot_encoder)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    #fit transform
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)

    #Mengambil Nama fitur setelah encoding

    numeric_features_names = numeric_features

    fitted_onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    categorical_features_names = fitted_onehot.get_feature_names_out(
        categorical_features
    ).tolist()

    feature_names = numeric_features_names + categorical_features_names


    #mengubah ke dataframe

    x_train_processed_df = pd.DataFrame(
        x_train_processed,
        columns=feature_names
    )

    x_test_processed_df = pd.DataFrame(
        x_test_processed,
        columns=feature_names
    )

    train_processed = x_train_processed_df.copy()
    train_processed["target"] = y_train.reset_index(drop=True)

    test_processed = x_test_processed_df.copy()
    test_processed["target"] = y_test.reset_index(drop=True)

    #meyimpan output
    train_output_path = os.path.join(output_dir, "train_preprocessed.csv")
    test_output_path = os.path.join(output_dir, "test_preprocessed.csv")
    preprocessor_output_path = os.path.join(output_dir, "preprocessor.joblib")
    feature_names_output_path = os.path.join(output_dir, "feature_names.json")
    metadata_output_path = os.path.join(output_dir, "metadata.json")

    train_processed.to_csv(train_output_path, index=False)
    test_processed.to_csv(test_output_path, index=False)

    joblib.dump(preprocessor, preprocessor_output_path)

    with open(feature_names_output_path, "w") as f:
        json.dump(feature_names, f, indent=4)

    metadata = {
        "raw_dataset_shape": list(df.shape),
        "train_shape": list(train_processed.shape),
        "test_shape": list(test_processed.shape),
        "target_column": "target",
        "dropped_columns": ["duration"],
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "test_size": test_size,
        "random_state": random_state,
        "target_mapping": {
            "no": 0,
            "yes": 1
        },
        "train_target_distribution": train_processed["target"].value_counts(normalize=True).to_dict(),
        "test_target_distribution": test_processed["target"].value_counts(normalize=True).to_dict()
    }

    with open(metadata_output_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nPreprocessing selesai.")
    print(f"Train dataset disimpan di      : {train_output_path}")
    print(f"Test dataset disimpan di       : {test_output_path}")
    print(f"Preprocessor disimpan di       : {preprocessor_output_path}")
    print(f"Feature names disimpan di      : {feature_names_output_path}")
    print(f"Metadata preprocessing disimpan: {metadata_output_path}")

    return train_processed, test_processed, preprocessor


if __name__ == "__main__":
    # Path otomatis berdasarkan lokasi file script ini
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    input_path = os.path.join(
        project_root,
        "bank_raw",
        "bank-full.csv"
    )

    output_dir = os.path.join(
        current_dir,
        "bank_preprocessing"
    )

    preprocess_data(
        input_path=input_path,
        output_dir=output_dir,
        test_size=0.2,
        random_state=42
    )
