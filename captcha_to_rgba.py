from PIL import Image

# Open an image
img = Image.open("C:/Users/MC/Desktop/PFE S5/Code/data/Train_Labels/banknote/0bb281b845f0eb07c8c42289208ef5d2_text_image.png")

# Convert to RGBA
rgba_img = img.convert("RGBA")

# Show the image
rgba_img.show()

# Save the converted image
rgba_img.save("image_rgba.png")
