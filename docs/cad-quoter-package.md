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

## Local development

During local development against the monorepo, install the project in editable mode and
pull in the packaging extras so Python resolves the `cad_quoter` package directly from
`src/` and has the build/publish tooling on hand:

```bash
pip install -e .[dev]
```

If you are working on the desktop UI, follow that with `pip install -r requirements.txt`
so the application runtime dependencies (including `cad-quoter` itself) are available.

## Building and publishing

Use the standard Python packaging workflow when cutting a new internal release:

```bash
rm -rf build dist src/cad_quoter.egg-info
python -m build
python -m twine upload --repository cad-internal dist/*
```

The `cad-internal` repository name matches the entry in your `~/.pypirc`. Substitute your
team’s private index alias if it differs. After the upload succeeds, downstream projects
can upgrade by running `pip install --upgrade cad-quoter --extra-index-url <your-internal-index>`.
