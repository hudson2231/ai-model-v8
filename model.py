import os
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from cog import BasePredictor, Input, Path
import warnings
warnings.filterwarnings('ignore')

print("🚨 FINAL VERSION 8.0 - COMPLETELY NEW APPROACH!")


class SimpleLineArtConverter:
    """Simple, effective line art converter focused on clean results"""
    
    def preprocess_image(self, image: Image.Image, target_size: int = 1024) -> np.ndarray:
        """Simple preprocessing for line art"""
        # Resize
        w, h = image.size
        if max(w, h) > target_size:
            ratio = target_size / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return np.array(image)
    
    def create_line_art(self, img: np.ndarray) -> np.ndarray:
        """Create clean line art using a simple but effective approach"""
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply strong bilateral filter to create smooth regions
        # This is key - it groups similar areas together
        smooth = cv2.bilateralFilter(gray, 15, 200, 200)
        smooth = cv2.bilateralFilter(smooth, 15, 200, 200)  # Apply twice for stronger effect
        
        # Create line art using adaptive threshold
        # This creates clean boundaries between regions
        line_art = cv2.adaptiveThreshold(
            smooth, 
            255, 
            cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, 
            7,  # Smaller block size for finer details
            10  # Constant subtracted from mean
        )
        
        # Clean up the result
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        line_art = cv2.morphologyEx(line_art, cv2.MORPH_OPEN, kernel)
        
        # Connect nearby lines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        line_art = cv2.morphologyEx(line_art, cv2.MORPH_CLOSE, kernel)
        
        # Ensure we have BLACK lines on WHITE background
        if np.mean(line_art) < 127:
            line_art = 255 - line_art
            
        return line_art
    
    def enhance_lines(self, line_art: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Enhance the line art with additional details"""
        
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        
        # Add some edge details using gentle Canny
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        edges = cv2.Canny(blurred, 80, 160)
        
        # Clean up edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Combine with main line art (edges should be black on white)
        if np.mean(edges) > 127:
            edges = 255 - edges
            
        # Merge - take the darkest pixels (black lines)
        enhanced = np.minimum(line_art, 255 - edges)
        
        return enhanced
    
    def final_cleanup(self, line_art: np.ndarray) -> np.ndarray:
        """Final cleanup for perfect coloring book style"""
        
        # Ensure pure black and white
        _, cleaned = cv2.threshold(line_art, 127, 255, cv2.THRESH_BINARY)
        
        # Remove tiny isolated pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Ensure lines are connected
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Final check - ensure black lines on white background
        if np.mean(cleaned) < 127:
            cleaned = 255 - cleaned
            
        return cleaned
    
    def process(self, image: Image.Image) -> Image.Image:
        """Main processing function"""
        
        print("🎨 Starting simple line art conversion...")
        
        # Preprocess
        print("📸 Preprocessing...")
        img_array = self.preprocess_image(image)
        
        # Create base line art
        print("🔍 Creating line art...")
        line_art = self.create_line_art(img_array)
        
        # Enhance with additional details
        print("✨ Enhancing details...")
        enhanced = self.enhance_lines(line_art, img_array)
        
        # Final cleanup
        print("🛠️ Final cleanup...")
        final = self.final_cleanup(enhanced)
        
        print("✅ Line art complete!")
        
        return Image.fromarray(final)


class Predictor(BasePredictor):
    def setup(self):
        """Initialize the predictor"""
        print("🚀 Setting up Simple Line Art Converter v6...")
        self.converter = SimpleLineArtConverter()
        print("✅ Setup complete!")
    
    def predict(
        self,
        input_image: Path = Input(description="Photo to convert to line art"),
        target_size: int = Input(
            description="Maximum image size", 
            default=1024, 
            ge=512, 
            le=2048
        ),
        line_intensity: str = Input(
            description="Line intensity",
            default="medium",
            choices=["light", "medium", "strong"]
        ),
    ) -> Path:
        """Convert image to line art"""
        
        print(f"📥 Loading: {input_image}")
        
        # Load image
        try:
            image = Image.open(input_image)
            if image.mode not in ['RGB', 'RGBA']:
                image = image.convert('RGB')
        except Exception as e:
            raise ValueError(f"Could not load image: {e}")
        
        print(f"📐 Size: {image.size}")
        
        # Process
        result = self.converter.process(image)
        
        # Adjust line intensity
        result_array = np.array(result)
        
        if line_intensity == "light":
            # Make lines thinner
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            result_array = cv2.erode(result_array, kernel, iterations=1)
            # Fix colors if needed
            if np.mean(result_array) < 127:
                result_array = 255 - result_array
        elif line_intensity == "strong":
            # Make lines thicker  
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            # Invert for dilation (work with white lines on black)
            if np.mean(result_array) > 127:
                result_array = 255 - result_array
            result_array = cv2.dilate(result_array, kernel, iterations=1)
            # Convert back to black lines on white
            result_array = 255 - result_array
        
        result = Image.fromarray(result_array)
        
        # Resize if needed
        if max(result.size) != target_size:
            w, h = result.size
            if max(w, h) > target_size:
                ratio = target_size / max(w, h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                result = result.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        print(f"📤 Final size: {result.size}")
        
        # Save
        output_path = "/tmp/simple_line_art.png"
        result.save(output_path, "PNG")
        
        print(f"💾 Saved: {output_path}")
        return Path(output_path)
