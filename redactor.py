import ezdxf

def redact_dxf_file(file_path, output_path, words_to_redact):
    """
    Finds and replaces sensitive words in a .DXF file.

    Args:
        file_path (str): Path to the input DXF file.
        output_path (str): Path to save the redacted DXF file.
        words_to_redact (list): A list of strings to search for and replace.
    """
    try:
        # Load the DXF document
        doc = ezdxf.readfile(file_path)
        msp = doc.modelspace() # Modelspace contains the main drawing entities

        print(f"Scanning {file_path} for sensitive words...")
        redaction_count = 0

        # We look for text-based entities (TEXT, MTEXT, etc.)
        for entity in msp.query('TEXT MTEXT'):
            original_text = entity.dxf.text
            new_text = original_text
            
            # Check each sensitive word
            for word in words_to_redact:
                if word.lower() in new_text.lower():
                    # Simple replacement. You can make this more sophisticated.
                    new_text = new_text.replace(word, "[REDACTED]")
                    redaction_count += 1
            
            # If changes were made, update the entity
            if new_text != original_text:
                entity.dxf.text = new_text

        if redaction_count > 0:
            # Save the changes to a new file
            doc.saveas(output_path)
            print(f"✅ Redaction complete! Found and replaced {redaction_count} instances.")
            print(f"   Redacted file saved to: {output_path}")
        else:
            print("ℹ️ No sensitive words found. No new file was created.")

    except IOError:
        print(f"❌ Error: Could not read file at {file_path}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


# --- Main execution ---
if __name__ == "__main__":
    # Replace with your file paths and sensitive data
    input_dxf = "your_drawing.dxf"
    output_dxf = "redacted_drawing.dxf"
    
    # Add any customer names or sensitive info you want to remove
    sensitive_info = ["Acme Corp", "Confidential", "Project X"]

    redact_dxf_file(input_dxf, output_dxf, sensitive_info)