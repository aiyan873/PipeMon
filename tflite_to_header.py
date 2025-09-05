# =====================
# Upload a TFLite model and convert to C++ header
# =====================
from google.colab import files

# Step 1: Upload your .tflite file
uploaded = files.upload()  # <-- Drag & drop your .tflite here

# Get the filename from uploaded dict
tflite_path = list(uploaded.keys())[0]
print(f"  Uploaded file: {tflite_path}")

# Step 2: Define conversion function
def tflite_to_header(
    tflite_path,
    header_path,
    var_name="g_pipe_leak_model",
    progmem=True
):
    with open(tflite_path, "rb") as f:
        data = f.read()

    # Format bytes nicely (12 per line)
    hex_array = ",\n  ".join(
        [", ".join([f"0x{b:02x}" for b in data[i:i+12]]) for i in range(0, len(data), 12)]
    )

    header_guard = "PIPE_LEAK_MODEL_DATA_H_"
    progmem_str = "PROGMEM " if progmem else ""
    progmem_include = "#include <avr/pgmspace.h>\n" if progmem else ""

    header_content = f"""#ifndef {header_guard}
#define {header_guard}

#include <cstddef>
#include <cstdint>
{progmem_include}// TensorFlow Lite model data converted to C++ array
alignas(8) const unsigned char {var_name}[] {progmem_str}= {{
  {hex_array}
}};

const int {var_name}_len = {len(data)};

#endif  // {header_guard}
"""

    with open(header_path, "w") as f:
        f.write(header_content)

    print(f"   Header file written: {header_path}")
    print(f"   Array name : {var_name}")
    print(f"   Length     : {len(data)} bytes")
    print(f"   PROGMEM    : {'enabled' if progmem else 'disabled'}")

    # Step 3: Auto-download header file
    files.download(header_path)

# Step 4: Convert uploaded TFLite → C++ header
tflite_to_header(tflite_path, "pipe_leak_model_data.h", progmem=True)

