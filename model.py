import os
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from cog import BasePredictor, Input, Path
import warnings
warnings.filterwarnings('ignore')

print("🚨 PERFECTED VERSION 8.1 - PROFESSIONAL COLORING BOOK QUALITY!")


class ProfessionalLineArtConverter:
    """Professional line art converter matching high-quality coloring book standards"""
    
    def preprocess_image(self, image: Image.Image, target_size: int = 1024) -> np.ndarray:
        """Enhanced preprocessing for professional results"""
        # Resize
        w, h = image.size
        if max(w, h) > target_size:
            ratio = target_size / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Enhance contrast and sharpness for better edge detection
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
            
        return np.array(image)
    
    def create_major_outlines(self, img: np.ndarray) -> np.ndarray:
        """Create major structural outlines first"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Very strong bilateral filtering to create distinct regions
        smooth = cv2.bilateralFilter(gray, 25, 300, 300)
        smooth = cv2.bilateralFilter(smooth, 25, 300, 300)
        smooth = cv2.bilateralFilter(smooth, 15, 200, 200)  # Third pass for ultra-smooth regions
        
        # Use adaptive threshold to find major boundaries
        major_outlines = cv2.adaptiveThreshold(
            smooth, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            15,  # Larger block for major features
            8   # Lower constant for more boundaries
        )
        
        # Clean up major outlines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        major_outlines = cv2.morphologyEx(major_outlines, cv2.MORPH_CLOSE, kernel, iterations=2)
        major_outlines = cv2.morphologyEx(major_outlines, cv2.MORPH_OPEN, kernel)
        
        return major_outlines
    
    def create_detail_lines(self, img: np.ndarray) -> np.ndarray:
        """Create fine detail lines for features"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Medium bilateral filtering for details
        smooth = cv2.bilateralFilter(gray, 12, 150, 150)
        
        # Adaptive threshold for fine details
        detail_lines = cv2.adaptiveThreshold(
            smooth, 
            255, 
            cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, 
            7,   # Smaller block for fine details
            12   # Higher constant to reduce noise
        )
        
        # Clean up detail lines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        detail_lines = cv2.morphologyEx(detail_lines, cv2.MORPH_OPEN, kernel)
        detail_lines = cv2.morphologyEx(detail_lines, cv2.MORPH_CLOSE, kernel)
        
        return detail_lines
    
    def enhance_with_edges(self, img: np.ndarray) -> np.ndarray:
        """Add clean edge details using optimized Canny"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Strong noise reduction before edge detection
        blurred = cv2.GaussianBlur(gray, (7, 7), 2.0)
        
        # Use higher thresholds for cleaner edges
        edges = cv2.Canny(blurred, 120, 240)
        
        # Clean up edges significantly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        return edges
    
    def combine_and_refine(self, major_outlines: np.ndarray, detail_lines: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Intelligently combine all line sources"""
        
        # Start with major outlines as the foundation
        combined = major_outlines.copy()
        
        # Add detail lines where they don't conflict with major outlines
        # Use bitwise operations to combine intelligently
        combined = cv2.bitwise_or(combined, detail_lines)
        
        # Add clean edges for fine details
        combined = cv2.bitwise_or(combined, edges)
        
        # Final refinement using contour analysis
        contours, hierarchy = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create refined output
        refined = np.zeros_like(combined)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Filter by area and draw with appropriate thickness
            if area > 100:  # Major features
                # Smooth the contour
                epsilon = 0.015 * cv2.arcLength(contour, True)
                smoothed = cv2.approxPolyDP(contour, epsilon, True)
                cv2.drawContours(refined, [smoothed], -1, 255, 2)
            elif area > 20:  # Medium features
                epsilon = 0.02 * cv2.arcLength(contour, True)
                smoothed = cv2.approxPolyDP(contour, epsilon, True)
                cv2.drawContours(refined, [smoothed], -1, 255, 1)
            elif area > 5:  # Fine details
                cv2.drawContours(refined, [contour], -1, 255, 1)
        
        return refined
    
    def final_professional_cleanup(self, line_art: np.ndarray) -> np.ndarray:
        """Final cleanup for professional coloring book appearance"""
        
        # Ensure we start with the right orientation
        if np.mean(line_art) < 127:
            line_art = 255 - line_art
        
        # Apply strong threshold for pure black/white
        _, cleaned = cv2.threshold(line_art, 127, 255, cv2.THRESH_BINARY)
        
        # Remove tiny isolated pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Connect nearby line segments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Final smoothing pass using contour refinement
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final = np.zeros_like(cleaned)
        
        for contour in contours:
            if cv2.contourArea(contour) > 3:  # Keep most details
                # Very light smoothing to maintain detail while removing jaggedness
                epsilon = 0.005 * cv2.arcLength(contour, True)
                smoothed = cv2.approxPolyDP(contour, epsilon, True)
                cv2.drawContours(final, [smoothed], -1, 255, 1)
        
        # Ensure black lines on white background
        if np.mean(final) < 127:
            final = 255 - final
            
        return final
    
    def process(self, image: Image.Image) -> Image.Image:
        """Main processing pipeline for professional line art"""
        
        print("🎨 Starting professional line art conversion...")
        
        # Step 1: Preprocess
        print("📸 Enhanced preprocessing...")
        img_array = self.preprocess_image(image)
        
        # Step 2: Create major structural outlines
        print("🏗️ Creating major outlines...")
        major_outlines = self.create_major_outlines(img_array)
        
        # Step 3: Create fine detail lines
        print("🔍 Adding fine details...")
        detail_lines = self.create_detail_lines(img_array)
        
        # Step 4: Add edge enhancement
        print("✨ Enhancing with clean edges...")
        edges = self.enhance_with_edges(img_array)
        
        # Step 5: Intelligently combine all sources
        print("🎯 Combining and refining...")
        combined = self.combine_and_refine(major_outlines, detail_lines, edges)
        
        # Step 6: Final professional cleanup
        print("🛠️ Professional cleanup...")
        final = self.final_professional_cleanup(combined)
        
        print("✅ Professional line art complete!")
        
        return Image.fromarray(final)


class Predictor(BasePredictor):
    def setup(self):
        """Initialize the predictor"""
        print("🚀 Setting up Professional Line Art Converter v8.1...")
        self.converter = ProfessionalLineArtConverter()
        print("✅ Professional setup complete!")
    
    def predict(
        self,
        input_image: Path = Input(description="Photo to convert to professional line art"),
        target_size: int = Input(
            description="Maximum image size", 
            default=1024, 
            ge=512, 
            le=2048
        ),
        line_style: str = Input(
            description="Line art style",
            default="balanced",
            choices=["fine", "balanced", "bold"]
        ),
        detail_level: str = Input(
            description="Amount of detail to preserve",
            default="medium",
            choices=["low", "medium", "high"]
        ),
    ) -> Path:
        """Convert image to professional line art"""
        
        print(f"📥 Loading: {input_image}")
        
        # Load image
        try:
            image = Image.open(input_image)
            if image.mode not in ['RGB', 'RGBA']:
                image = image.convert('RGB')
        except Exception as e:
            raise ValueError(f"Could not load image: {e}")
        
        print(f"📐 Size: {image.size}")
        
        # Process with professional pipeline
        result = self.converter.process(image)
        
        # Apply style adjustments
        result_array = np.array(result)
        
        if line_style == "fine":
            # Make lines more delicate
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            result_array = cv2.erode(result_array, kernel, iterations=1)
            if np.mean(result_array) < 127:
                result_array = 255 - result_array
        elif line_style == "bold":
            # Make lines more prominent
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            if np.mean(result_array) > 127:
                result_array = 255 - result_array
            result_array = cv2.dilate(result_array, kernel, iterations=1)
            result_array = 255 - result_array
        
        # Apply detail level adjustments
        if detail_level == "low":
            # Reduce fine details, keep major outlines
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            result_array = cv2.morphologyEx(result_array, cv2.MORPH_OPEN, kernel, iterations=1)
        elif detail_level == "high":
            # Enhance fine details
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            result_array = cv2.morphologyEx(result_array, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        result = Image.fromarray(result_array)
        
        # Resize if needed
        if max(result.size) != target_size:
            w, h = result.size
            if max(w, h) > target_size:
                ratio = target_size / max(w, h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                result = result.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        print(f"📤 Final size: {result.size}")
        
        # Save with maximum quality
        output_path = "/tmp/professional_line_art.png"
        result.save(output_path, "PNG", optimize=False)
        
        print(f"💾 Professional result saved: {output_path}")
        return Path(output_path)
