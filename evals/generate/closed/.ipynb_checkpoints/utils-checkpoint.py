

import io
import base64


from PIL import Image
from PIL.PngImagePlugin import PngImageFile


def base64_encode_pil_image(
    pil_image:PngImageFile
):
    """
    Encodes a PIL Image object to a Base64 string.
    Args:
        pil_image (PIL.Image.Image): The PIL Image object to encode.
                                      This can be a PIL.PngImagePlugin.PngImageFile object.
    Returns:
        str: The Base64 encoded string representation of the image.
             Returns None if an error occurs.
    """
    try:
        # Create an in-memory bytes buffer
        buffered = io.BytesIO()

        pil_image.save(buffered, format="PNG")

        img_byte = buffered.getvalue()

        # Encode the bytes to Base64
        base64_bytes = base64.b64encode(img_byte)

        # Decode the Base64 bytes to a string (e.g., UTF-8 or ASCII)
        base64_string = base64_bytes.decode('utf-8') # Or 'ascii'

        return base64_string
    except Exception as e:
        print(f"An error occurred: {e}") ; return None


def anthropic_encode_pil_image(image):
    """
    Converts a PIL.PngImagePlugin.PngImageFile object to a Base64 encoded string.

    Args:
        pil_image_object (PIL.PngImagePlugin.PngImageFile): The PIL image object.

    Returns:
        str: The Base64 encoded string representation of the image.
    """

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    # Encode as base64
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    
    
    return image_base64


    