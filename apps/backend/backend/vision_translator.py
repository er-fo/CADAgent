"""
Vision Translation Module for CAD Sketches

Translates user-uploaded sketches with markup and annotations into
structured textual CAD instructions using GPT-5.2 with medium reasoning.
"""

import logging
import base64
import io
import os
from typing import Tuple, Optional, Dict, Any
from PIL import Image
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class VisionTranslator:
    """Translates CAD sketches with markup into structured prompts"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the vision translator.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=self.api_key)

        # Model configuration
        self._model_name = os.getenv("VISION_MODEL", "gpt-5.2")
        self._reasoning_effort = os.getenv("VISION_REASONING_EFFORT", "none")

        logger.info(f"VisionTranslator initialized with model: {self._model_name}, reasoning effort: {self._reasoning_effort}")
    
    async def translate_sketch_to_prompt(
        self,
        image_data: str,
        image_format: str = "png",
        user_context: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Translate a sketch image into a structured CAD prompt.
        
        Args:
            image_data: Base64-encoded image string
            image_format: Image format (png, jpg, jpeg)
            user_context: Optional user-provided text context
            
        Returns:
            Tuple of (translated_prompt_text, metadata_dict)
            
        Raises:
            ValueError: If image cannot be decoded or API call fails
        """
        logger.info(f"Translating sketch (format={image_format}, has_context={bool(user_context)})")
        
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            logger.debug(f"Image decoded: size={image.size}, mode={image.mode}")
            
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            raise ValueError(f"Invalid image data: {e}")
        
        # Build specialized CAD vision prompt
        vision_prompt = self._build_vision_prompt(user_context)

        # Logging removed to reduce terminal output

        try:
            # Prepare image data URL for OpenAI
            # OpenAI expects data URLs in format: data:image/png;base64,{base64_data}
            image_data_url = f"data:image/{image_format};base64,{image_data}"

            # Call OpenAI Responses API with vision capabilities

            # Build input with text and image
            input_items = [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": vision_prompt
                        },
                        {
                            "type": "input_image",
                            "image_url": image_data_url,
                            "detail": "high"
                        }
                    ]
                }
            ]


            # Note: GPT-5.2 with reasoning doesn't support temperature parameter
            response = await self.client.responses.create(
                model=self._model_name,
                input=input_items,
                reasoning={
                    "effort": self._reasoning_effort
                },
                max_output_tokens=4096
            )

            # Extract text response from the response
            translated_prompt = ""
            for item in response.output:
                if item.type == "message" and hasattr(item, "content"):
                    for content in item.content:
                        if content.type == "output_text":
                            translated_prompt += content.text

            translated_prompt = translated_prompt.strip()

            # Log response metadata if available
            if hasattr(response, 'usage'):
                logger.info(f"Token usage: {response.usage}")
            if hasattr(response, 'model'):
                logger.info(f"Model used: {response.model}")
            
            # Build metadata
            metadata = {
                "image_size": image.size,
                "image_format": image_format,
                "model": self._model_name,
                "has_user_context": bool(user_context),
                "prompt_length": len(translated_prompt)
            }
            
            return translated_prompt, metadata

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}", exc_info=True)
            raise ValueError(f"Vision translation failed: {e}")
    
    def _build_vision_prompt(self, user_context: Optional[str]) -> str:
        """
        Build specialized prompt for CAD sketch analysis.
        
        This prompt instructs Gemini to extract:
        - Dimensions with units
        - Geometric shapes
        - Features (holes, cutouts, etc.)
        - Spatial relationships
        - Operations (extrude, revolve, etc.)
        """
        
        base_prompt = """You are a visual markup interpreter for CAD. Analyze this image and translate the user's markups, drawings, and annotations into a clear textual description of what changes they want.

**YOUR TASK:**
Convert visual markup → textual prompt describing intended changes

**WHAT TO ANALYZE:**

1. **Visual Markup Elements**:
   - Circles/shapes drawn on the model → Indicate feature locations (holes, cutouts, etc.)
   - Arrows → Point to specific locations or show direction
   - Highlighting/underlining → Emphasis on areas to modify
   - Dimension lines → Show measurements between points
   - Text annotations → Specifications, sizes, notes

2. **Extract All Text**:
   - Dimension values (e.g., "50mm", "Ø10", "2.5cm")
   - Feature labels (e.g., "mounting hole", "cutout", "fillet")
   - Notes or instructions written by user
   - If units missing, assume millimeters (mm)

3. **Interpret Markup Intent**:
   - Circle on a face → User wants a hole at that location
   - Rectangle outline → User wants a cutout or extrusion
   - Arrow with dimension → User specifies size or offset
   - Highlighted edge → User wants to modify that edge (fillet, chamfer)
   - Cross mark → User wants to remove that feature

4. **Determine Placement from Visual Context**:
   - Where on the model is the markup? (top face, side, corner, edge)
   - How is it positioned? (centered, offset, aligned, corner)
   - Are there patterns? (4 corners, evenly spaced, symmetric)

5. **Provide Complete Specifications**:
   - If markup shows a hole without size, infer logical diameter from visual proportion
   - If position looks centered, state "centered" definitively
   - If depth not specified, assume "through-all" for holes
   - Use engineering judgment to fill in any missing specifications

**OUTPUT FORMAT:**

Translate markup into a decisive textual prompt in FIRST PERSON:

```
I want you to create the following, its based on a drawing: [intent summary]. [Complete specifications for each marked feature including size, position, and type]. [Any additional operations or modifications indicated by markup].
```

**EXAMPLE INPUT:**
- Image shows a cube with two circles drawn on it
- One circle on top face (centered), labeled "Ø8mm"
- One circle on side face, no label but similar size
- Arrow pointing downward on top circle

**EXAMPLE OUTPUT:**
"I want you to create the following, its based on a drawing: add holes to the cube. Add a hole with diameter 8mm on the top face, centered, through-all as indicated by the downward arrow. Add a second hole with diameter 8mm on the side face, centered."

**CRITICAL RULES:**
- BE DECISIVE: State all specifications with confidence
- NO HEDGING: Avoid "appears", "seems", "possibly", "approximately"
- COMPLETE SPECS: Every feature needs size, exact position, and operation type
- IGNORE SKETCH QUALITY: Interpret intent from rough drawings
- ASSUME LOGICAL DEFAULTS: Fill in missing info using engineering judgment
- CONVERT ALL UNITS TO MM: All dimensions must be converted to millimeters (mm). Convert inches (1 inch = 25.4mm), centimeters (1cm = 10mm), meters (1m = 1000mm), feet (1ft = 304.8mm), etc. Always output the final value in mm.
- NO FOLLOW-UP NEEDED: Output should be 100% actionable as-is
"""
        
        if user_context:
            base_prompt += f"\n\n**USER CONTEXT:** {user_context}\n"
            base_prompt += "(Use this context to disambiguate or prioritize certain interpretations)\n"
        
        return base_prompt


# Singleton instance (initialized on first use)
_vision_translator: Optional[VisionTranslator] = None


def get_vision_translator() -> VisionTranslator:
    """Get or create the singleton VisionTranslator instance."""
    global _vision_translator
    
    if _vision_translator is None:
        _vision_translator = VisionTranslator()
    
    return _vision_translator
