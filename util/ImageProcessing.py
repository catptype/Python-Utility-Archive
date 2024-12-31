import cv2
import random
import warnings
import numpy as np
from PIL import Image
from typing import Tuple, Literal

class Handler:

    @staticmethod
    def get_stucture(structure:Literal['ellipse', 'rect'], kernel:Tuple[int,int]) -> np.ndarray:
        if structure.lower() == 'rect':
            stuctureKernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
        elif structure.lower() == 'ellipse':
            stuctureKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
        else:
            raise ValueError("stucture could be 'rect' or 'ellipse'")
        
        return stuctureKernel

class PillowImageProcessing:

    @staticmethod
    def resize_aspect_height(image:Image.Image, target_height:int) -> Image.Image:
        # Get the original image dimensions
        height = image.height
        width = image.width
        
        # Calculate the scaling factor to maintain aspect ratio
        scale = target_height / height
        new_width = round(width * scale)
        
        # Resize image
        resized_image = image.resize(size=(new_width, target_height), resample=Image.LANCZOS)
        return resized_image

    @staticmethod
    def resize_aspect_width(image:Image.Image, target_width:int) -> Image.Image:
        # Get the original image dimensions
        height = image.height
        width = image.width
        
        # Calculate the scaling factor to maintain aspect ratio
        scale = target_width / width
        new_height = round(height * scale)
        
        # Resize
        resized_image = image.resize(size=(target_width, new_height), resample=Image.LANCZOS)
        return resized_image
    
class CvImageProcessing:

    @staticmethod
    def is_color_numpy(image:Image.Image, threshold:float=0.005, fast_mode:bool=True) -> bool:
        """
        Original code for diff_sum
        R = image_np[:,:,0]
        G = image_np[:,:,1]
        B = image_np[:,:,2]

        RG = np.count_nonzero(abs(R-G))
        RB = np.count_nonzero(abs(R-B))
        GB = np.count_nonzero(abs(G-B))

        diff_sum = float(RG + RB + GB)
        """
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if fast_mode:
            image.thumbnail((100, 100), Image.ANTIALIAS)  # Resize image with aspect ratio
        
        image_np = np.array(image)
        
        diff_sum = np.sum(np.abs(np.diff(image_np, axis=2)))
        ratio = diff_sum / image_np.size
        return ratio > threshold

    @staticmethod
    def is_color(image:Image.Image, threshold:float=0.005, fast_mode:bool=True) -> bool:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if fast_mode:
            image.thumbnail((100, 100), Image.ANTIALIAS)  # Resize image with aspect ratio
        
        pixels = image.load()
        width, height = image.size

        diff_sum = 0
        total_pixels = width * height * 3

        for y in range(height):
            for x in range(width - 1):
                rg_diff = abs(pixels[x, y][0] - pixels[x, y][1])
                rb_diff = abs(pixels[x, y][0] - pixels[x, y][2])
                gb_diff = abs(pixels[x, y][1] - pixels[x, y][2])
                diff_sum += rg_diff + rb_diff + gb_diff

        ratio = diff_sum / total_pixels  # Multiply by 3 to account for RGB channels

        return ratio > threshold

    @staticmethod
    def add_noise(base_image:np.ndarray, noise_level=3):
        image_height = base_image.shape[0]
        image_width = base_image.shape[1]
        noise = np.random.normal(0, noise_level, (image_height, image_width, 3)).astype(np.float32)
        image_with_noise = np.clip(base_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return image_with_noise

    @staticmethod
    def ghosting_effect(image:np.ndarray, alpha:float=0.5, beta:float=0.5, blur_strength:int=11) -> np.ndarray:
        if alpha + beta > 1:
            warnings.warn(
                "The sum of alpha and beta exceeds 1. The resulting image may be clipped.",
                UserWarning
            )
        # Create a blurred version of the image
        blurred_image = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
        
        # Create the ghosting effect by blending the original and blurred images
        ghosted_image = cv2.addWeighted(image, alpha, blurred_image, beta, 0)
        
        return ghosted_image
    
    @staticmethod
    def brightness(image_hsv:np.ndarray, brightness_factor:float) -> np.ndarray:
        """
        output image BGR
        """
        # Split the channels
        h, s, v = cv2.split(image_hsv)

        # Change the brightness
        v = np.clip(v * brightness_factor, 0, 255).astype(np.uint8)

        # Merge the channels
        adjusted_hsv = cv2.merge([h, s, v])

        # Convert back to BGR
        adjusted_image = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)

        return adjusted_image

    @staticmethod
    def rotate(image:np.ndarray, angle_degrees:int) -> np.ndarray:
        # Convert angle to radians
        angle_radians = np.radians(angle_degrees)

        # Calculate new size of the output image
        height, width = image.shape[:2]
        cos_val = abs(np.cos(angle_radians))
        sin_val = abs(np.sin(angle_radians))
        new_width = int(height * sin_val + width * cos_val)
        new_height = int(height * cos_val + width * sin_val)

        # Define the rotation transformation matrix
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle_degrees, 1)

        # Adjust the translation component of the matrix to prevent cropping
        rotation_matrix[0, 2] += (new_width - width) / 2
        rotation_matrix[1, 2] += (new_height - height) / 2

        # Apply the rotation transformation
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

        return rotated_image

    @staticmethod
    def shear(image:np.ndarray, x_factor:float, y_factor:float, invert:bool=False) -> np.ndarray:
        if not (0.0 <= x_factor <= 1.0 and 0.0 <= y_factor <= 1.0):
            raise ValueError("Factor must be range from 0.0 to 1.0")

        if invert:
            image = cv2.flip(image, 1)

        # Calculate new size of the output image
        max_x, max_y = image.shape[1], image.shape[0]
        shift_x = x_factor * max_y
        shift_y = y_factor * max_x
        new_width = max_x + abs(shift_x)
        new_height = max_y + abs(shift_y)

        # Define the shear transformation matrix
        matrix_shear = np.float32([[1, x_factor, 0],
                                [y_factor, 1, 0]])

        # Apply the shear transformation
        sheared_image = cv2.warpAffine(image, matrix_shear, (int(new_width), int(new_height)))

        if invert:
            sheared_image = cv2.flip(sheared_image, 1)

        return sheared_image
    
    @staticmethod
    def skew(image:np.ndarray, x_angle:int, y_angle:int, invert:bool=False) -> np.ndarray:
        
        if not (x_angle >= 0 and y_angle >= 0):
            raise ValueError("Only accept positive integer")
        
        if invert:
            image = cv2.flip(image, 1)
        # Get image dimensions
        height, width = image.shape[:2]

        # Define the skew angles in radians
        x_skew_angle = np.deg2rad(x_angle)
        y_skew_angle = np.deg2rad(y_angle)

        # Calculate the dimensions of the new image
        new_width = int(width + abs(height * np.tan(x_skew_angle)) + 0.5)
        new_height = int(height + abs(width * np.tan(y_skew_angle)) + 0.5)

        # Define the transformation matrices
        x_skew_matrix = np.array([[1, np.tan(x_skew_angle), 0],
                                [0, 1, 0]])

        y_skew_matrix = np.array([[1, 0, 0],
                                [np.tan(y_skew_angle), 1, 0]])

        # Perform skew transformation
        skewed_image = cv2.warpAffine(image, x_skew_matrix, (new_width, height))
        skewed_image = cv2.warpAffine(skewed_image, y_skew_matrix, (new_width, new_height))

        if invert:
            skewed_image = cv2.flip(skewed_image, 1)

        return skewed_image

    @staticmethod
    def resize_aspect_height(image:np.ndarray, target_height:int) -> np.ndarray:
        # Get the original image dimensions
        height, width = image.shape[:2]
        
        # Calculate the scaling factor to maintain aspect ratio
        scale = target_height / height

        new_width = round(width * scale)
        
        image = Image.fromarray(np.uint8(image))
        resized_image = image.resize(size=(new_width, target_height), resample=Image.LANCZOS)
        resized_image = np.array(resized_image)

        return resized_image

    @staticmethod
    def resize_aspect_width(image:np.ndarray, target_width:int) -> np.ndarray:
        # Get the original image dimensions
        height, width = image.shape[:2]
        
        # Calculate the scaling factor to maintain aspect ratio
        scale = target_width / width

        new_height = round(height * scale)
        
        image = Image.fromarray(np.uint8(image))
        resized_image = image.resize(size=(target_width, new_height), resample=Image.LANCZOS)
        resized_image = np.array(resized_image)

        return resized_image

    @staticmethod
    def gray(image:np.ndarray) -> np.ndarray:
        color_image = Image.fromarray(np.uint8(image))
        gray_image  = color_image.convert('L')
        gray_image = np.array(gray_image)
        return gray_image

    @staticmethod
    def denoise_gray(image:np.ndarray, h:int=10, block_size:int=7, search_window:int=21) -> np.ndarray:
        image = cv2.fastNlMeansDenoising(image, None, h=h, templateWindowSize=block_size, searchWindowSize=search_window)
        return image
    
    @staticmethod
    def denoise_color(image:np.ndarray, h:int=10, block_size:int=7, search_window:int=21) -> np.ndarray:
        image = cv2.fastNlMeansDenoisingColored(image, None, h=h, templateWindowSize=block_size, searchWindowSize=search_window)
        return image
    
    @staticmethod
    def cannyedge(image:np.ndarray, threshold1:int=100, threshold2:int=200) -> np.ndarray:
        image = cv2.Canny(image, threshold1=threshold1, threshold2=threshold2, apertureSize=3, L2gradient=True)
        return image

    @staticmethod
    def blur(image:np.ndarray, kernel:Tuple[int,int]=(3, 3)) -> np.ndarray:
        image = cv2.GaussianBlur(image, kernel, 0)
        return image
    
    @staticmethod
    def deblur_color(image:np.ndarray, kernel_size:Tuple[int,int]=(3, 3), balance:float=0.1) -> np.ndarray:
        from skimage.restoration import wiener
        # Separate the image into its color channels
        image = image.astype(np.float32) / 255.0

        channels = cv2.split(image)
        
        # Create a Gaussian kernel for deblurring
        kernel = cv2.getGaussianKernel(kernel_size[0], 0) @ cv2.getGaussianKernel(kernel_size[1], 0).T
        kernel = np.float32(kernel)
        
        # Apply Wiener deconvolution to each channel
        deblurred_channels = []
        for channel in channels:
            deblurred_channel = wiener(channel, kernel, balance=balance)
            deblurred_channels.append(deblurred_channel)
        
        # Merge the channels back into a single image
        deblurred_image = cv2.merge(deblurred_channels)
        deblurred_image = np.clip(deblurred_image * 255.0, 0, 255).astype(np.uint8)
    
        return deblurred_image

    @staticmethod
    def sharpening(image:np.ndarray) -> np.ndarray:
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        image = cv2.filter2D(image, -1, kernel)
        return image
    
    @staticmethod
    def sharpening_lapacian(image:np.ndarray) -> np.ndarray:
        kernel = np.array([[0, -1, 0],
                           [-1,  5, -1],
                           [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)
        return image
    
    @staticmethod
    def binary(image:np.ndarray) -> np.ndarray:
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #  
        return image
    
    @staticmethod
    def adaptive_binary(image:np.ndarray, c:int=2) -> np.ndarray:
        mean_val = np.mean(image)
        block_size = int(mean_val / 4.5) if mean_val / 4.5 > 1 else 1

        if block_size % 2 == 0: # Handleing error for blocksize even nunmber
            block_size += 1

        image = cv2.adaptiveThreshold(
            image, 
            maxValue=255, 
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # cv2.ADAPTIVE_THRESH_GAUSSIAN_C or cv2.ADAPTIVE_THRESH_MEAN_C
            thresholdType=cv2.THRESH_BINARY_INV, 
            blockSize=block_size,
            C=c,
        )

        return image
    
    @staticmethod
    def adaptive_tophat(image:np.ndarray, adaptive_num:float=3.5, stucture:Literal['ellipse', 'rect']='ellipse') -> np.ndarray:
        # Default 3.5
        mean_val = np.mean(image)
        kernel_size = int(mean_val / adaptive_num) if mean_val / adaptive_num > 1 else 1        
        # print("TOPHAT KERNEL=", kernel_size, mean_val, adaptive_num)
        stuctureKernel = Handler.get_stucture(stucture, (kernel_size, kernel_size))
            
        image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, stuctureKernel)
        return image
    
    @staticmethod
    def adaptive_blackhat(image:np.ndarray, adaptive_num:float=3.0, stucture:Literal['ellipse', 'rect']='ellipse') -> np.ndarray:
        # Default 3.0
        mean_val = np.mean(image)
        kernel_size = int(mean_val / adaptive_num) if mean_val / adaptive_num > 1 else 1        
        # print("BLACKHAT KERNEL=", kernel_size, mean_val, adaptive_num)
        stuctureKernel = Handler.get_stucture(stucture, (kernel_size, kernel_size))
        image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, stuctureKernel)
        return image
    
    @staticmethod
    def invert(image:np.ndarray) -> np.ndarray:
        image = 255 - image
        return image
    
    @staticmethod
    def calculate_blurriness(image:np.ndarray) -> float:
        """
        Low value mean very blur
        """
        laplacian = cv2.Laplacian(image, cv2.CV_64F).var()
        return laplacian
    
    @staticmethod
    def calculate_noise(image:np.ndarray, color:bool=False, average_sigmas:bool=True) -> float:
        """
        High value means low noise
        Reference: https://stackoverflow.com/questions/2440504/noise-estimation-noise-measurement-in-image
        """
        from skimage.restoration import estimate_sigma
        if color:
            channel_axis = -1
        return estimate_sigma(image, channel_axis=channel_axis, average_sigmas=average_sigmas)
    
class SpecialEffect:
    
    @staticmethod
    def add_light_reflection(image, alpha=0.5):
        # Ensure the image is in RGBA format
        if image.shape[2] == 3:  # If the image doesn't have an alpha channel
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

        # Get image dimensions
        height, width, channel = image.shape

        def linear():
            left_half = width // 2
            right_half = width - left_half
            gradient_left = np.linspace(0, 1, left_half).astype(np.float32)
            gradient_right = np.linspace(1, 0, right_half).astype(np.float32)
            gradient = np.concatenate((gradient_left, gradient_right), axis=0)
            return gradient
        
        def palabolic(reflect_width_size:float=0.05):
            # Calculate mid-point for gradient split
            mid = width // 2
            
            # Define the width of the narrow reflection area
            reflection_width = int(width * reflect_width_size)  # Adjust this value to change the width of the reflection

            # Create a narrower parabolic gradient for the left half
            x_left = np.linspace(1, 0, reflection_width)
            parabolic_left = 1 - np.square(x_left)  # Normalize x to [0, 1] before squaring
            
            # Create a narrower parabolic gradient for the right half
            x_right = np.linspace(0, 1, reflection_width)
            parabolic_right = 1 - np.square(x_right)  # Normalize x to [0, 1] before squaring
            
            # Combine the two parabolic gradients
            gradient = np.concatenate((
                np.ones(mid - reflection_width) * 0,  # Padding on the left
                parabolic_left,
                parabolic_right,
                np.ones(width - mid - reflection_width) * 0  # Padding on the right
            ), axis=0)
            
            return gradient

        def gradient_zigzag(gradient, zigzag_amplitude=10, zigzag_frequency=2):
            for row in range(height):
                shift_sin = int(zigzag_amplitude * np.sin(2 * np.pi * zigzag_frequency * row / height))
                shift_cos = int(zigzag_amplitude * np.cos(2 * np.pi * zigzag_frequency * row / height))
                shift_tan = int(zigzag_amplitude * np.tan(2 * np.pi * zigzag_frequency * row / height))
                
                # Randomly choose between shift_sin, shift_cos, and shift_tan
                shift = random.choices(
                    [shift_sin, shift_cos, shift_tan], 
                    weights=(0.4, 0.4, 0.2),
                    k=1,
                )

                gradient[row] = np.roll(gradient[row], shift)
            return gradient
        
        def gradient_simple_shift(gradient, max_shift=100):
            # Apply a random zigzag effect by shifting rows in the gradient randomly
            for row in range(height):
                shift = random.randint(-max_shift, max_shift)
                gradient[row] = np.roll(gradient[row], shift)
            return gradient

        def gradient_random_position(gradient):
            # Generate a random percentage shift in float between -0.5 and 0.5
            shift_percentage = random.uniform(-0.5, 0.5)
            
            # Calculate the actual shift value based on the width of the gradient
            shift = int(width * shift_percentage)
            
            # Apply the shift to the gradient
            gradient = np.roll(gradient, shift, axis=1)
            return gradient
        
        # initialize gradient
        # gradient = random.choice([palabolic(0.08), linear()])
        # gradient = linear()
        gradient = palabolic(0.08)
        gradient = np.tile(gradient, (height, 1))
        gradient = random.choice([gradient_zigzag(gradient, 20, 3), gradient_simple_shift(gradient, 40)])
        gradient = gradient_random_position(gradient)
        
        # Apply Gaussian blur to the gradient to smooth the edges
        gradient = cv2.GaussianBlur(gradient, (11, 11), 0)

        reflection_rgba = np.ones((height, width, 4), dtype=np.float32) * 255  # White reflection
        
        reflection_rgba[:, :, 3] *= gradient # Overwrite alpha
        reflection_rgba[:, :, 2] *= gradient # Overwrite B
        reflection_rgba[:, :, 1] *= gradient # Overwrite G
        reflection_rgba[:, :, 0] *= gradient # Overwrite R

        # Ensure both images are in the same data type for blending
        base_image = image.astype(np.float32)
        
        # Apply the gradient reflection
        overlay_result = cv2.addWeighted(base_image, 1, reflection_rgba, alpha, 0)
        
        # Convert result back to uint8
        overlay_result = np.clip(overlay_result, 0, 255).astype(np.uint8)
        
        return overlay_result