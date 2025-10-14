from cad_quoter.utils import compact_dict, jdump, json_safe_copy, sdict
from cad_quoter.llm_suggest import (
    build_suggest_payload,
    sanitize_suggestions,
    get_llm_quote_explanation,
)
            llm_explanation = get_llm_quote_explanation(
                res,
                model_path,
                debug_enabled=APP_ENV.llm_debug_enabled,
                debug_dir=APP_ENV.llm_debug_dir,
            )
