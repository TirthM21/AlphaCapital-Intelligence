from src.data.universe_fetcher import StockUniverseFetcher


def test_cached_index_size_validation_rejects_polluted_nifty_50_cache():
    fetcher = StockUniverseFetcher()
    symbols = [f"SYM{i}.NS" for i in range(500)]
    assert not fetcher._is_cached_index_size_plausible("NIFTY 50", symbols)


def test_cached_index_size_validation_accepts_expected_nifty_50_size():
    fetcher = StockUniverseFetcher()
    symbols = [f"SYM{i}.NS" for i in range(50)]
    assert fetcher._is_cached_index_size_plausible("NIFTY 50", symbols)
