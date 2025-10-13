    try:
        canonical_bucket_summary_local = canonical_bucket_summary
    except NameError:
        canonical_bucket_summary_local = None

        summary_entry = (
            canonical_bucket_summary_local.get(canon_key)
            if isinstance(canonical_bucket_summary_local, dict)
            else None
        )
