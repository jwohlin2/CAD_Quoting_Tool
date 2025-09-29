import sys

from cad_quoter.config import configure_logging, logger

import ezdxf

from appV5 import convert_dwg_to_dxf_2018


def main() -> None:
    configure_logging()
    if len(sys.argv) < 3:
        logger.error("Usage: python dwg_smoketest.py <dwg_path> <ODAFileConverter.exe>")
        sys.exit(1)

    dwg_path = sys.argv[1]
    oda_exe = sys.argv[2]
    dxf_path = convert_dwg_to_dxf_2018(dwg_path, oda_exe)
    logger.info("DXF path: %s", dxf_path)

    doc = ezdxf.readfile(dxf_path)
    logger.info("Model entities: %d", len(list(doc.modelspace())))
    logger.info("Layouts: %s", list(doc.layouts.names_in_taborder()))
    logger.info("Units: %s", doc.header.get("$INSUNITS"))


if __name__ == "__main__":
    main()
