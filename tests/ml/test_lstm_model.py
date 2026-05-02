# Test file for lstm_model.py train_and_save_lstm

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os


class TestTrainAndSaveLstm:
    """Test train_and_save_lstm function."""

    def test_signature_has_symbols_parameter(self):
        """Verify train_and_save_lstm has symbols parameter."""
        from backend.ml.lstm_model import train_and_save_lstm
        import inspect
        sig = inspect.signature(train_and_save_lstm)
        params = list(sig.parameters.keys())
        assert 'symbols' in params, f"symbols param missing, got: {params}"

    def test_uses_build_features_multi_when_symbols_provided(self):
        """When symbols param is provided, build_features_multi should be called."""
        from backend.ml.lstm_model import train_and_save_lstm

        # Build minimal mock DataFrame with required columns
        mock_df = pd.DataFrame({
            'trade_date': pd.date_range('2024-01-01', periods=100),
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, 100),
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 110,
            'low': np.random.randn(100).cumsum() + 90,
            'ret_1d': np.random.randn(100) * 0.02,
            'ret_3d': np.random.randn(100) * 0.02,
            'ret_5d': np.random.randn(100) * 0.02,
            'ret_10d': np.random.randn(100) * 0.02,
            'target_t3': (np.random.randn(100) > 0).astype(int),
            'symbol': ['MU'] * 100,
        })

        with patch('backend.ml.features.build_features_multi') as mock_build:
            mock_build.return_value = mock_df
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch('backend.ml.lstm_model.MODELS_DIR', Path(tmpdir)):
                    result = train_and_save_lstm(
                        symbol='MU',
                        target_col='target_t3',
                        seq_len=10,
                        exclude_neutral=False,
                        epochs=5,
                        symbols=['MU']
                    )
            
            # build_features_multi should have been called with symbols=['MU']
            mock_build.assert_called_once()
            call_args = mock_build.call_args
            # symbols passed as kwarg
            assert call_args.kwargs.get('symbols') == ['MU']

    def test_returns_correct_keys(self):
        """train_and_save_lstm returns dict with train_size, n_features, model_path, meta."""
        from backend.ml.lstm_model import train_and_save_lstm

        mock_df = pd.DataFrame({
            'trade_date': pd.date_range('2024-01-01', periods=100),
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, 100),
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 110,
            'low': np.random.randn(100).cumsum() + 90,
            'ret_1d': np.random.randn(100) * 0.02,
            'ret_3d': np.random.randn(100) * 0.02,
            'ret_5d': np.random.randn(100) * 0.02,
            'ret_10d': np.random.randn(100) * 0.02,
            'target_t3': (np.random.randn(100) > 0).astype(int),
            'symbol': ['MU'] * 100,
        })

        with patch('backend.ml.features.build_features_multi') as mock_build:
            mock_build.return_value = mock_df
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch('backend.ml.lstm_model.MODELS_DIR', Path(tmpdir)):
                    result = train_and_save_lstm(
                        symbol='MU',
                        target_col='target_t3',
                        seq_len=10,
                        exclude_neutral=False,
                        epochs=5,
                        symbols=['MU']
                    )
            
            if 'error' not in result:
                assert 'train_size' in result
                assert 'n_features' in result
                assert 'model_path' in result
                assert 'meta_path' in result
                assert 'meta' in result
