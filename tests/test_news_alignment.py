from backend.pipeline.alignment import _resolve_trade_date_from_published


def test_resolve_trade_date_moves_after_close_news_to_next_trading_day():
    idx = {"2026-05-08": 0, "2026-05-11": 1}

    result = _resolve_trade_date_from_published("2026-05-08T21:30:00Z", idx)

    assert result == "2026-05-11"


def test_resolve_trade_date_keeps_regular_session_news_on_same_trading_day():
    idx = {"2026-05-08": 0, "2026-05-11": 1}

    result = _resolve_trade_date_from_published("2026-05-08T15:00:00Z", idx)

    assert result == "2026-05-08"
