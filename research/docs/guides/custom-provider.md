# Build a Custom Data Provider

1) Subclass the base provider in `src/data_providers/base_provider.py`
2) Implement `fetch(start, end, symbols, interval)` to return a dataframe
3) Optionally implement `stream()` for live data
4) Wire into configuration and test with a small range

Example skeleton:

```python
from src.data_providers.base_provider import BaseProvider

class MyProvider(BaseProvider):
    def fetch(self, start, end, symbols, interval):
        # call API, build dataframe with required schema
        return df
```

