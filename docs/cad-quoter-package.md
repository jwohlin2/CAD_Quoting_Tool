# cad-quoter

This package contains the CAD quoting domain, pricing, geometry, and LLM helpers that
previously lived directly in the `CAD_Quoting_Tool` repository.  It is intended to be
published to an internal package index so that other applications can reuse the quoting
logic without vendoring the source tree.

The package exposes the `cad_quoter` namespace.  After building and publishing, install it
into your environment via your private index:

```bash
pip install cad-quoter --extra-index-url <your-internal-index>
```

Once installed, the main `appV5.py` GUI and associated scripts can `import cad_quoter`
without requiring an in-tree copy.
