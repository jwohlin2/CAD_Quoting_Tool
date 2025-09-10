import cadquery as cq
import math

def analyze_cad_file(file_path):
    """
    Analyzes a .STEP or .STL file to extract a rich set of geometric features.

    Args:
        file_path (str): The path to the CAD file.

    Returns:
        dict: A dictionary containing the extracted features, or None on error.
    """
    try:
        # Load the solid from the file
        solid = cq.importers.importStep(file_path)

        # --- Basic Dimensions ---
        bb = solid.val().BoundingBox()
        
        # --- Advanced Feature Analysis ---
        
        # 1. Hole Detection: Find circular faces and get their radii
        hole_radii = []
        for face in solid.faces().vals():
            if face.geomType() == "PLANE" and len(face.Edges()) == 1:
                edge = face.Edges()[0]
                if edge.geomType() == "CIRCLE":
                    hole_radii.append(edge.radius())
        
        # 2. Smallest Corner Radius (Approximation)
        # We check the radius of all non-straight edges. This is a heuristic.
        min_radius = float('inf')
        for edge in solid.edges().vals():
            if edge.geomType() == "CIRCLE" and edge.radius() < min_radius:
                # We ignore large holes for this calculation
                if edge.radius() > 0.001 and edge.radius() < (bb.xlen / 4):
                     min_radius = edge.radius()

        # 3. Complexity Score (Heuristic)
        # A simple score based on the ratio of faces to volume.
        # A higher number suggests more complex geometry.
        volume = solid.val().Volume()
        num_faces = len(solid.faces().vals())
        complexity_score = (num_faces / volume) * 100 if volume > 0 else 0

        # --- Assemble the data package for the LLM ---
        features = {
            "GEO-01_Length_mm": bb.xlen,
            "GEO-02_Width_mm": bb.ylen,
            "GEO-03_Height_mm": bb.zlen,
            "Calculated_Volume_cm3": volume / 1000,
            "Feature_Face_Count": num_faces,
            "Feature_Hole_Count": len(hole_radii),
            "Feature_Detected_Hole_Radii_mm": [round(r, 3) for r in hole_radii],
            "GEO-06_Smallest_Internal_Radius_mm": round(min_radius, 3) if min_radius != float('inf') else "Not Found",
            "Heuristic_Complexity_Score": round(complexity_score, 2)
        }
        
        print("✅ Advanced analysis complete!")
        return features

    except Exception as e:
        print(f"❌ Error processing file {file_path}: {e}")
        return None

# --- Main execution for testing ---
if __name__ == "__main__":
    # Replace with the path to your CAD file
    cad_file = "your_part.step" 
    
    extracted_data = analyze_cad_file(cad_file)

    if extracted_data:
        import json
        print("\n--- Extracted Data for LLM ---")
        print(json.dumps(extracted_data, indent=4))
        print("----------------------------")
