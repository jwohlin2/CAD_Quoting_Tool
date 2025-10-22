# Application settings and environment overrides

The desktop application now ships with a single consolidated settings file at
`cad_quoter/resources/app_settings.json`. It contains the minimum set of
defaults required across all environments:

* top-level UI preferences such as `rate_mode` and `last_variables_path`
* `pricing_defaults.params` with quoting parameters that used to live in
  `params_v1.json`
* `pricing_defaults.rates` with the labour and machine buckets previously stored
  in `rates_v1.json`

## Providing environment specific overrides

Set the `CAD_QUOTER_APP_SETTINGS` environment variable to point at a JSON file
on disk when you need to override any of the bundled defaults. The override file
only needs to declare the keys you want to change. For example, the following
`/etc/cad_quoter/prod_settings.json` file tightens the margin and updates the
labour rate while leaving every other setting untouched:

```json
{
  "pricing_defaults": {
    "params": {
      "MarginPct": 0.42
    },
    "rates": {
      "labor": {
        "Programmer": 105.0
      }
    }
  }
}
```

Launch the UI with:

```bash
CAD_QUOTER_APP_SETTINGS=/etc/cad_quoter/prod_settings.json python appV5.py
```

Settings from the override file are merged with the packaged defaults, allowing
multiple environments (development, staging, production) to share the same base
file while specifying their unique adjustments.
