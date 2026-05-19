from PIL import Image
import glob

path = "/Users/grnho/Desktop/image-processing-fundamentals/data/plane/*"

for file in glob.glob(path):
    print(file)
    a = Image.open(file)
    rotated45 = a.rotate(45, expand = True)
    rotated45.save(file+"_rotated45.png", "PNG")
