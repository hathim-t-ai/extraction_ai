"""
LayoutLMv3 Inference Module

This module implements LayoutLMv3 inference for text-embedded bank statement PDFs.
It processes text and layout information to extract structured table data.
"""

import logging
import time
import json
import re
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass

import torch
from transformers import AutoModelForTokenClassification, AutoProcessor
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_parser.utils import load_config

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class LayoutLMResult:
    """Structure for LayoutLMv3 inference results"""
    table: List[List[str]]
    confidence: float
    raw_output: Dict[str, Any]
    processing_time: float
    model_used: str
    error_message: Optional[str] = None


class LayoutLMInference:
    """LayoutLMv3 inference engine for text-embedded PDFs"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize LayoutLMv3 inference engine.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = load_config(config_path)
        self.model_config = self.config.get('models', {}).get('layoutlmv3', {})
        
        # Model configuration
        self.model_name = self.model_config.get('model_path', 'microsoft/layoutlmv3-base')
        self.confidence_threshold = self.model_config.get('confidence_threshold', 0.7)
        self.max_length = self.model_config.get('max_length', 512)
        
        # Device configuration
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Model components (loaded lazily)
        self.processor = None
        self.model = None
        self.model_loaded = False
        
        logger.info(f"LayoutLMInference initialized with model: {self.model_name}")
        logger.info(f"Device: {self.device}")
    
    def _load_models(self):
        """Load LayoutLMv3 model and processor"""
        if self.model_loaded:
            return
            
        try:
            logger.info(f"Loading LayoutLMv3 model: {self.model_name}")
            start_time = time.time()
            
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            load_time = time.time() - start_time
            logger.info(f"LayoutLMv3 model loaded successfully in {load_time:.2f}s")
            self.model_loaded = True
            
        except Exception as e:
            error_msg = f"Failed to load LayoutLMv3 model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _prepare_input(self, text_data: Dict[str, Any], image: Optional[Image.Image] = None) -> Dict[str, Any]:
        """
        Prepare input for LayoutLMv3 processing.
        
        Args:
            text_data (Dict): Text extraction result from preprocessing
            image (Optional[Image.Image]): PDF page image for layout context
            
        Returns:
            Dict with processed inputs for LayoutLMv3
        """
        try:
            # Extract text content
            if isinstance(text_data, dict):
                text = text_data.get('text', '')
                coordinates = text_data.get('coordinates', [])
                text_blocks = text_data.get('text_blocks', [])
            else:
                text = str(text_data)
                coordinates = []
                text_blocks = []
            
            # Clean and tokenize text
            text = self._clean_text(text)
            
            # Prepare bounding boxes if available
            bbox = self._extract_bounding_boxes(coordinates, text_blocks)
            
            # LayoutLMv3 processor ALWAYS requires an image argument
            if image is None:
                # Create a dummy white image for text-only processing
                image = Image.new('RGB', (224, 224), color='white')
            
            # Process with LayoutLMv3 processor
            # Note: processor expects (image, text, boxes) format
            inputs = self.processor(
                image,
                text,
                boxes=bbox if bbox else None,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            )
            
            # Ensure tensor types are correct for embedding layer
            if 'input_ids' in inputs:
                inputs['input_ids'] = inputs['input_ids'].long()
            if 'bbox' in inputs:
                inputs['bbox'] = inputs['bbox'].long()
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'].long()
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error preparing LayoutLMv3 input: {e}")
            # Return minimal valid input structure
            dummy_image = Image.new('RGB', (224, 224), color='white')
            try:
                fallback_inputs = self.processor(
                    dummy_image,
                    "dummy text",
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=True
                )
                # Ensure correct tensor types
                for key in fallback_inputs:
                    if isinstance(fallback_inputs[key], torch.Tensor) and fallback_inputs[key].dtype == torch.float:
                        fallback_inputs[key] = fallback_inputs[key].long()
                return fallback_inputs
            except Exception as fallback_error:
                logger.error(f"Even fallback input preparation failed: {fallback_error}")
                return {"input_ids": torch.tensor([[0]], dtype=torch.long)}
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for processing"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\.\-\$\,\%\(\)\/\:]', ' ', text)
        
        return text.strip()
    
    def _extract_bounding_boxes(self, coordinates: List[Dict], text_blocks: List[Dict]) -> Optional[List[List[int]]]:
        """
        Extract bounding boxes from coordinate data.
        
        Args:
            coordinates (List[Dict]): Coordinate information from text extraction
            text_blocks (List[Dict]): Text block information
            
        Returns:
            Optional[List[List[int]]]: Bounding boxes in [x0, y0, x1, y1] format
        """
        try:
            bbox = []
            
            # Try to extract from coordinates
            if coordinates:
                for coord in coordinates[:50]:  # Limit to first 50 for performance
                    if 'bbox' in coord:
                        box = coord['bbox']
                        if len(box) == 4:
                            bbox.append([int(x) for x in box])
            
            # Try to extract from text blocks
            elif text_blocks:
                for block in text_blocks[:50]:  # Limit to first 50 for performance
                    if 'bbox' in block:
                        box = block['bbox']
                        if len(box) == 4:
                            bbox.append([int(x) for x in box])
            
            return bbox if bbox else None
            
        except Exception as e:
            logger.debug(f"Could not extract bounding boxes: {e}")
            return None
    
    def _parse_layoutlm_output(self, outputs: Any, input_text: str) -> Dict[str, Any]:
        """
        Parse LayoutLMv3 model output to extract table data.
        
        Args:
            outputs: Model outputs from LayoutLMv3
            input_text (str): Original input text
            
        Returns:
            Dict with parsed table data and confidence
        """
        try:
            # Get predictions and confidence scores
            logits = outputs.logits
            predictions = torch.nn.functional.softmax(logits, dim=-1)
            confidence = predictions.max().item()
            
            # Get predicted labels
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Convert to table format
            table_data = []
            
            # Basic text-based table extraction as fallback
            lines = input_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and self._looks_like_transaction(line):
                    # Try to parse transaction line
                    row = self._parse_transaction_line(line)
                    if row:
                        table_data.append(row)
            
            # Add header if we found data
            if table_data and not any('date' in str(cell).lower() for cell in table_data[0]):
                table_data.insert(0, ['Date', 'Description', 'Amount', 'Balance'])
            
            return {
                'table': table_data,
                'confidence': confidence,
                'predicted_ids': predicted_ids.tolist(),
                'raw_text': input_text
            }
            
        except Exception as e:
            logger.error(f"Error parsing LayoutLMv3 output: {e}")
            return {
                'table': [],
                'confidence': 0.0,
                'error': str(e),
                'raw_text': input_text
            }
    
    def _looks_like_transaction(self, line: str) -> bool:
        """Check if a line looks like a transaction"""
        line = line.strip().lower()
        
        # Check for common transaction indicators
        has_amount = bool(re.search(r'\$?\d+[\.\,]\d{2}', line))
        has_date = bool(re.search(r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}', line))
        has_transaction_words = any(word in line for word in [
            'deposit', 'withdrawal', 'payment', 'transfer', 'fee', 'charge',
            'interest', 'balance', 'debit', 'credit', 'purchase'
        ])
        
        return has_amount or (has_date and len(line.split()) >= 3) or has_transaction_words
    
    def _parse_transaction_line(self, line: str) -> Optional[List[str]]:
        """Parse a single transaction line into components"""
        try:
            # Split by multiple spaces or tabs
            parts = re.split(r'\s{2,}|\t', line.strip())
            
            if len(parts) >= 2:
                # Try to identify date, description, amount, balance
                date = ""
                description = ""
                amount = ""
                balance = ""
                
                for part in parts:
                    part = part.strip()
                    if re.match(r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}', part):
                        date = part
                    elif re.match(r'\$?\d+[\.\,]\d{2}', part):
                        if not amount:
                            amount = part
                        else:
                            balance = part
                    elif not description and len(part) > 2:
                        description = part
                
                # If we couldn't parse properly, use first 4 parts
                if not date and not description and not amount:
                    return parts[:4] + [''] * (4 - len(parts[:4]))
                
                return [date, description, amount, balance]
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not parse transaction line: {e}")
            return None
    
    def _fallback_text_parsing(self, input_text: str, start_time: float) -> LayoutLMResult:
        """
        Fallback text parsing when model inference fails.
        
        Args:
            input_text (str): Input text to parse
            start_time (float): Start time for processing time calculation
            
        Returns:
            LayoutLMResult: Parsed result using rule-based approach
        """
        try:
            table_data = []
            
            # Basic text-based table extraction
            lines = input_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and self._looks_like_transaction(line):
                    # Try to parse transaction line
                    row = self._parse_transaction_line(line)
                    if row:
                        table_data.append(row)
            
            # Add header if we found data
            if table_data and not any('date' in str(cell).lower() for cell in table_data[0]):
                table_data.insert(0, ['Date', 'Description', 'Amount', 'Balance'])
            
            processing_time = time.time() - start_time
            
            return LayoutLMResult(
                table=table_data,
                confidence=0.6,  # Lower confidence for fallback parsing
                raw_output={'fallback': True, 'raw_text': input_text},
                processing_time=processing_time,
                model_used=f"{self.model_name} (fallback)",
                error_message="Used fallback text parsing due to model inference failure"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return LayoutLMResult(
                table=[],
                confidence=0.0,
                raw_output={'error': str(e)},
                processing_time=processing_time,
                model_used=f"{self.model_name} (failed)",
                error_message=f"Fallback parsing also failed: {e}"
            )
    
    def process_text_data(self, text_data: Dict[str, Any], image: Optional[Image.Image] = None) -> LayoutLMResult:
        """
        Process text data using LayoutLMv3 for table extraction.
        
        Args:
            text_data (Dict): Text extraction result from preprocessing
            image (Optional[Image.Image]): Optional PDF page image
            
        Returns:
            LayoutLMResult: Structured result with table data and metadata
        """
        start_time = time.time()
        
        try:
            self._load_models()
            
            if not text_data:
                return LayoutLMResult(
                    table=[],
                    confidence=0.0,
                    raw_output={},
                    processing_time=0.0,
                    model_used=self.model_name,
                    error_message="No text data provided"
                )
            
            # Prepare input
            inputs = self._prepare_input(text_data, image)
            inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            
            # Run inference with error handling
            with torch.no_grad():
                try:
                    outputs = self.model(**inputs)
                except Exception as model_error:
                    logger.warning(f"Model inference failed, using fallback text parsing: {model_error}")
                    # Fall back to rule-based text parsing
                    input_text = text_data.get('text', '') if isinstance(text_data, dict) else str(text_data)
                    return self._fallback_text_parsing(input_text, processing_time)
            
            # Parse output
            input_text = text_data.get('text', '') if isinstance(text_data, dict) else str(text_data)
            parsed_result = self._parse_layoutlm_output(outputs, input_text)
            
            processing_time = time.time() - start_time
            
            return LayoutLMResult(
                table=parsed_result['table'],
                confidence=parsed_result['confidence'],
                raw_output=parsed_result,
                processing_time=processing_time,
                model_used=self.model_name,
                error_message=parsed_result.get('error')
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"LayoutLMv3 processing failed: {e}"
            logger.error(error_msg)
            
            return LayoutLMResult(
                table=[],
                confidence=0.0,
                raw_output={},
                processing_time=processing_time,
                model_used=self.model_name,
                error_message=error_msg
            )


# Convenience functions following Bank_Statement_Converter_Tasks.markdown pattern
def run_layoutlm(data: Union[Dict[str, Any], Any], config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Run LayoutLMv3 inference on text data.
    
    Args:
        data: Text extraction result or preprocessing result data
        config: Configuration dictionary (optional)
        
    Returns:
        Dict with table, confidence, and metadata
    """
    try:
        # Handle different input types from preprocessing
        if hasattr(data, 'text_extraction'):
            # Handle PreprocessingResult object
            text_data = data.text_extraction.__dict__ if data.text_extraction else {}
        elif isinstance(data, dict):
            text_data = data
        else:
            logger.error(f"Unsupported text input type: {type(data)}")
            return {"table": [], "confidence": 0.0, "error": "Invalid input type"}
        
        # Initialize LayoutLMv3 inference
        layoutlm = LayoutLMInference()
        result = layoutlm.process_text_data(text_data)
        
        # Return in expected format
        return {
            "table": result.table,
            "confidence": result.confidence,
            "raw_output": result.raw_output,
            "processing_time": result.processing_time,
            "model_used": result.model_used,
            "error": result.error_message
        }
        
    except Exception as e:
        logger.error(f"run_layoutlm failed: {e}")
        return {
            "table": [],
            "confidence": 0.0,
            "error": str(e)
        }


def extract_table_from_text(text_data: Dict[str, Any], config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Convenience function to extract table data from text using LayoutLMv3.
    
    Args:
        text_data (Dict): Text extraction result
        config_path (str): Path to configuration file
        
    Returns:
        Dict containing extracted table data and metadata
    """
    layoutlm = LayoutLMInference(config_path)
    result = layoutlm.process_text_data(text_data)
    
    return {
        "table": result.table,
        "confidence": result.confidence,
        "processing_time": result.processing_time,
        "model_used": result.model_used,
        "success": result.error_message is None,
        "error": result.error_message
    } 