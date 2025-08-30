import subprocess

arr = [
12856,
12857,
12858,
12859,
12860,
12861,
12862,
12863,
12864,
12865,
12866,
12867,
12868,
12869,
12870,
12871,
12872,
12873,
12874,
12875,
12876,
12877,
12878,
12879
]

image_widths = {}
image_heights ={}

for i in arr:
    # Run the exiftool command and capture the output
    result = subprocess.run(
        f'exiftool wider_{i}.jpg | grep -i "Width" -A 1',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
  # Check if the command was successful
    if result.returncode == 0:
        # Extract the width value from the output
        output = result.stdout.strip()
        if "Width" in output:
            width = output.split(":")[1].strip().split("\n")[0]
            image_widths[i] = int(width)  # Store the width as an integer
        if "Height" in output:
            height = int(output.split(":")[-1].strip())
            image_heights[i] = int(height)  # Store the height as an integer
    else:
        print(f"Failed to process wider_{i}.jpg: {result.stderr}")

# Print the resulting dictionary
print(image_widths)
print(image_heights)

for idx, i in enumerate(arr):
    print(f"""
{{
    src: pic{i},
    localSrc: BASE_PATH + "{i}.jpg",
    width: {image_widths[i]},
    height: {image_heights[i]},
}},""")