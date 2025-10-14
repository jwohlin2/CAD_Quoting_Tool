from cad_quoter.vendors import ezdxf as _ezdxf_vendor

        ezdxf = _ezdxf_vendor.require_ezdxf()
        odafc = _ezdxf_vendor.require_odafc()
            odafc_mod = _ezdxf_vendor.require_odafc()
            readfile = getattr(odafc_mod, "readfile", None)
