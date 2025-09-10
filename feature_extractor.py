import cadquery as cq
import ezdxf
from ezdxf import bbox
from stl import mesh
import numpy as np
import os
import argparse
import sys

def analyze_step_file(file_path):
    """
    Analyzes a .STEP file to extract its features and parameters.

    Args:
        file_path (str): The path to the .STEP file.

    Returns:
        tuple: A tuple containing the extracted parameters (dict) and a status message (str).
    """
    try:
        solid = cq.importers.importStep(file_path)
        bb = solid.val().BoundingBox()
        parameters = {
            "file_name": file_path,
            "length_mm": bb.xlen,
            "width_mm": bb.ylen,
            "height_mm": bb.zlen,
            "volume_mm3": solid.val().Volume(),
            "num_faces": len(solid.faces().vals()),
            "num_edges": len(solid.edges().vals()),
        }
        return parameters, "Success: STEP Analysis complete!"

    except Exception as e:
        return None, f"Error processing STEP file {file_path}: {e}"

def analyze_dxf_file(file_path):
    """
    Analyzes a .DXF file to extract its features and parameters.

    Args:
        file_path (str): The path to the .DXF file.

    Returns:
        tuple: A tuple containing the extracted parameters (dict) and a status message (str).
    """
    try:
        doc = ezdxf.readfile(file_path)
        msp = doc.modelspace()
        
        try:
            extents = bbox.extents(msp)
            length = extents.size.x
            width = extents.size.y
            height = extents.size.z
        except (IndexError, ValueError, ezdxf.DXFError):
            length = 0
            width = 0
            height = 0

        parameters = {
            "file_name": file_path,
            "length_mm": length,
            "width_mm": width,
            "height_mm": height,
            "num_entities": len(msp),
        }
        return parameters, "Success: DXF Analysis complete!"

    except Exception as e:
        return None, f"Error processing DXF file {file_path}: {e}"

def analyze_stl_file(file_path):
    """
    Analyzes a .STL file to extract its features and parameters.

    Args:
        file_path (str): The path to the .STL file.

    Returns:
        tuple: A tuple containing the extracted parameters (dict) and a status message (str).
    """
    try:
        stl_mesh = mesh.Mesh.from_file(file_path)
        xmin, xmax, ymin, ymax, zmin, zmax = stl_mesh.min_[0], stl_mesh.max_[0], stl_mesh.min_[1], stl_mesh.max_[1], stl_mesh.min_[2], stl_mesh.max_[2]
        parameters = {
            "file_name": file_path,
            "length_mm": xmax - xmin,
            "width_mm": ymax - ymin,
            "height_mm": zmax - zmin,
            "volume_mm3": stl_mesh.get_mass_properties()[0],
            "num_triangles": len(stl_mesh.vectors),
        }
        return parameters, "Success: STL Analysis complete!"

    except Exception as e:
        return None, f"Error processing STL file {file_path}: {e}"

def print_parameters(extracted_data):
    """Prints the extracted parameters in a readable format."""
    if extracted_data:
        print("\n--- Extracted Parameters ---")
        for key, value in extracted_data.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        print("--------------------------")

# --- Main execution ---
if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Launching GUI...")
        os.system("python gui.py")
        sys.exit(0)

    parser = argparse.ArgumentParser(description='Analyze CAD files for quoting.')
    parser.add_argument('file_path', type=str, help='The path to the CAD file (.step, .dxf, or .stl)')
    args = parser.parse_args()

    cad_file = args.file_path
    file_ext = os.path.splitext(cad_file)[1].lower()
    
    extracted_data, message = None, None
    if file_ext == ".step" or file_ext == ".stp":
        extracted_data, message = analyze_step_file(cad_file)
    elif file_ext == ".dxf":
        extracted_data, message = analyze_dxf_file(cad_file)
    elif file_ext == ".stl":
        extracted_data, message = analyze_stl_file(cad_file)
    else:
        print(f"Unsupported file type: {file_ext}")

    if message:
        print(message)
    if extracted_data:
        print_parameters(extracted_data)