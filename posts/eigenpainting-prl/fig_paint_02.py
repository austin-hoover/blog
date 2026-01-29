import argparse
import os
import glob
import pathlib
from PIL import Image
from pprint import pprint



def save_gif(filenames: list[str], path: str, duration: float = 100.0) -> None:
    frames = [Image.open(f).convert("RGB") for f in filenames]
    
    palette = frames[0].convert("P", palette=Image.ADAPTIVE, colors=256)
    
    frames_new = [
        f.quantize(palette=palette, dither=Image.FLOYDSTEINBERG)
        for f in frames
    ]
    
    frames_new[0].save(
        path,
        save_all=True,
        append_images=frames_new[1:],
        duration=duration,
        loop=0,
        disposal=2
    )


parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=float, default=85.0)
args = parse.parse_args()


path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
os.makedirs(output_dir, exist_ok=True)

run_names = [
    "eig",
    "eig_n",
    "corr",
    "corr_n",
    "anticorr",
    "anticorr_n",
]

input_folders = []
for run_name in run_names:
    input_folder = os.path.join("outputs/fig_paint_01", run_name)
    input_folders.append(input_folder)

for input_folder, run_name in zip(input_folders, run_names):
    input_filenames = glob.glob(input_folder + "/*.png")
    input_filenames = sorted(input_filenames)

    output_filename = os.path.join(output_dir, f"paint_{run_name}.gif")
    print(output_filename)

    save_gif(input_filenames, output_filename)
