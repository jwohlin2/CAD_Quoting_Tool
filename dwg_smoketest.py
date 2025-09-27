import sys

import ezdxf

from appV5 import convert_dwg_to_dxf_2018


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python dwg_smoketest.py <dwg_path> <ODAFileConverter.exe>")
        sys.exit(1)

    dwg_path = sys.argv[1]
    oda_exe = sys.argv[2]
    dxf_path = convert_dwg_to_dxf_2018(dwg_path, oda_exe)
    print("DXF:", dxf_path)

    doc = ezdxf.readfile(dxf_path)
    print("Model entities:", len(list(doc.modelspace())))
    print("Layouts:", list(doc.layouts.names_in_taborder()))
    print("Units:", doc.header.get("$INSUNITS"))


if __name__ == "__main__":
    main()
