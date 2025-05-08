from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Dict, Any
import asyncio
import os
import fitz  # PyMuPDF
import cv2
from PIL import Image, ImageEnhance
from deskew import determine_skew
from io import BytesIO
import pytesseract
import numpy as np
import logging
import uuid
from datetime import datetime
import json

from models.resume_schema import Resume
from services.cv_parser import extract_resume_from_text

router = APIRouter()
logger = logging.getLogger(__name__)

# Ensure necessary directories exist
os.makedirs("extracted_markdown", exist_ok=True)
os.makedirs("temp_files", exist_ok=True)


@router.post("/cv", response_model=Resume)
async def ocr_resume(file: UploadFile = File(...)):
    """
    Extract text and structured data from a resume file.
    
    Args:
        file: The CV/resume file (PDF preferred)
        
    Returns:
        Resume: Structured resume data
    """
    try:
        content = await file.read()
        file_name = file.filename
        base, ext = os.path.splitext(file_name)
        
        if ext.lower() != '.pdf':
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        generated_uuid = str(uuid.uuid4())
        file_uuid = f"{base}_{generated_uuid}{ext}"
        
        # Process the PDF file
        try:
            with fitz.open(stream=content, filetype="pdf") as pdf:
                pages = list(range(len(pdf)))
                logger.info(f"File {file_name} contains {len(pages)} pages")
                extracted_text = await ocr_scanned_doc_process(content, pages)
                
                # Combine all extracted text
                full_text = ""
                for idx in sorted(extracted_text.keys()):
                    if isinstance(idx, int):  # Skip 'error' key if present
                        full_text += extracted_text[idx] + "\n\n"
                
                # Parse the text into structured resume data
                resume = extract_resume_from_text(full_text)
                
                # Save the extracted text and resume for debugging
                temp_file_path = f"temp_files/extracted_text_{generated_uuid}.txt"
                with open(temp_file_path, "w", encoding="utf-8") as f:
                    f.write(full_text)
                    
                resume_json_path = f"temp_files/resume_{generated_uuid}.json"
                with open(resume_json_path, "w", encoding="utf-8") as f:
                    json.dump(resume.model_dump(), f, ensure_ascii=False, indent=4)
                
                return resume
                
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
            raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")
    
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


async def ocr_scanned_doc_process(bytesdoc, page_indices):
    """
    Process a document using OCR and extract text from each page.
    
    Args:
        bytesdoc: The document as bytes
        page_indices: List of page indices to process
        
    Returns:
        dict: Dictionary with page index as key and extracted text as value
    """
    async def process_page(page, page_index):
        try:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            image_array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n
            )
            if pixmap.n == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

            processed_img = preprocess_image(Image.fromarray(image_array))
            text = await ocr_page(Image.fromarray(np.array(processed_img)), page_index)

            output_dir = "extracted_markdown"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_path = os.path.join(output_dir, f"ocr_page_{page_index}_{timestamp}.md")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

            return page_index, text
        except Exception as e:
            logger.error(f"Error processing page {page_index}: {e}")
            return page_index, f"Error: {str(e)}"

    output = {}
    try:
        with fitz.open(stream=bytesdoc, filetype="pdf") as pdf:
            tasks = [process_page(pdf.load_page(i), i) for i in page_indices]
            results = await asyncio.gather(*tasks)
            output = {index: text for index, text in results}

    except Exception as e:
        logger.error(f"Failed to process document: {e}")
        output["error"] = str(e)

    return output


def preprocess_image(image):
    """
    Preprocess an image for better OCR results.
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        PIL Image: Processed image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    gray = np.array(image.convert("L"))
    mean_intensity = np.mean(gray)
    
    if mean_intensity < 10:
        logger.info("The image appears to be blackened. Skipping further processing.")
        return image
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        blurred_correction = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.addWeighted(gray, 1.5, blurred_correction, -0.5, 0)
    
    denoised = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)
    processed_image = Image.fromarray(denoised)
    processed_image = ImageEnhance.Contrast(processed_image).enhance(1.3)
    processed_image = ImageEnhance.Sharpness(processed_image).enhance(1.8)
    processed_image, angle = correct_orientation(np.array(processed_image))
    logger.info(f"Image corrected with an angle of {angle} degrees")
    
    return processed_image


async def ocr_page(image: Image.Image, page_index: int) -> str:
    """
    Perform OCR on an image and extract text.
    
    Args:
        image: PIL Image
        page_index: Page index for output file naming
        
    Returns:
        str: Extracted text
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, ImageFormatOption
        from docling.datamodel.document import DocumentStream
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        
        pipeline_options = PdfPipelineOptions(do_table_structure=True)
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        doc_converter = DocumentConverter(
            format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options)},
            allowed_formats=[InputFormat.IMAGE]
        )
        
        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)   
        doc_stream = DocumentStream(stream=img_bytes, name="image.png")
        conversion_result = doc_converter.convert(source=doc_stream)
        text_output = conversion_result.document.export_to_text()
    except ImportError:
        # Fallback to pytesseract if docling is not available
        logger.warning("Docling not available, falling back to pytesseract")
        text_output = pytesseract.image_to_string(image)

    output_file_path = f"temp_files/output_file_{page_index}.txt"
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(text_output)

    logger.info(f"✅ Text extracted and saved to: {output_file_path}")
    return text_output


def correct_orientation(image, ocr_confirmation=True):
    """
    Correct the orientation of the image.
    
    Args:
        image: Image array
        ocr_confirmation: Whether to use OCR to confirm orientation
        
    Returns:
        tuple: (corrected_image, angle)
    """
    logger.info("Initial angle detection with Deskew...")
    angle_deskew = determine_skew(image)
    logger.info(f"Detected angle by Deskew: {angle_deskew}°")

    if abs(angle_deskew) > 0:
        logger.info(f"Deskew correction rotation: {-angle_deskew}°")
        image = rotate_image(image, angle_deskew)

    if ocr_confirmation:
        try:
            logger.info("Verifying orientation via OCR (OSD)...")
            ocr_osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
            angle_osd = ocr_osd['rotate']
            conf_osd = ocr_osd['orientation_conf']
            logger.info(f"Angle detected by OSD: {angle_osd}° (Confidence: {conf_osd})")

            if abs(angle_osd) > 0 and conf_osd > 0:
                logger.info(f"Final correction with OSD: {-angle_osd}°")
                image = rotate_image(image, -angle_osd)
        except pytesseract.TesseractError as e:
            logger.warning(f"OCR error: {e}. Deskew correction retained.")

    return image, angle_deskew


def rotate_image(image, angle):
    """
    Rotate an image by a given angle.
    
    Args:
        image: Image array
        angle: Angle to rotate
        
    Returns:
        numpy.ndarray: Rotated image
    """
    if angle == 0:
        return image
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(
        image, 
        M, 
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    
    return rotated