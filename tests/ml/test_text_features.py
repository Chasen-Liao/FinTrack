import pandas as pd

from backend.ml.features_v2 import TextSvdFeatureTransformer


def test_text_svd_transformer_fits_only_training_rows():
    train = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "text": ["earnings beat demand", "chip demand rises", "margin demand improves"],
        }
    )
    test = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-05"]),
            "text": ["unseen litigation topic"],
        }
    )

    transformer = TextSvdFeatureTransformer(max_features=20, min_df=1, n_components=2)
    transformer.fit(train)
    transformed = transformer.transform(test)

    assert transformer.fit_dates_ == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert transformed["trade_date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-05"]
    assert "text_svd_0" in transformed.columns
    assert "text_svd_1" in transformed.columns
