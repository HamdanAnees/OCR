from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import easyocr
import cv2
import numpy as np
import logging
from uvicorn import run
import os
from PIL import Image, UnidentifiedImageError
import io
import shutil
import re
from datetime import datetime
from typing import Dict, Optional, List
import glob
from pathlib import Path

# Initialize logger and app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

# Initialize EasyOCR reader once at startup
reader = None

@app.on_event("startup")
def load_model():
    global reader
    logger.info("Loading OCR model...")
    try:
        reader = easyocr.Reader(['en'])
        logger.info("OCR model loaded")
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise RuntimeError("OCR model initialization failed") from e

def rescale(image: np.ndarray) -> np.ndarray:
    """Upscale image by 1.5x using Lanczos interpolation"""
    return cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LANCZOS4)

def extract_invoice_data(text: str) -> Dict:
    """Extract structured invoice data from OCR text with enhanced patterns"""
    
    # Preprocess text to fix common OCR errors
    text = re.sub(r'(\d)\s*\.\s*(\d{2})', r'\1.\2', text)  # Fix spaced decimals
    text = re.sub(r'\s{2,}', ' ', text)  # Reduce multiple spaces
    text = re.sub(r'(\w)-\s+(\w)', r'\1-\2', text)  # Fix hyphenated words
    
    # Initialize result structure
    invoice_data = {
        "vendor_name": None,
        "invoice_number": None,
        "date": None,
        "subtotal": None,
        "taxes": None,
        "total": None,
        "payment_method": None,
        "extraction_details": {
            "text_length": len(text),
            "line_count": len(text.split('\n')),
            "found_amounts": [],
            "extraction_notes": []
        }
    }
    
    lines = text.split('\n')
    text_lower = text.lower()
    
    # Extract vendor name - more robust patterns
    vendor_patterns = [
        r'^\*\*([^*]+)\*\*',  # Match **Vendor Name** pattern
        r'^([A-Z][A-Za-z\s&\.,]+(?:LLC|Inc|Corp|Co|Ltd|Company|Corporation)?)',
        r'\b(?:receipt from|vendor:?)\s*(.+)',  # "Receipt from Starbucks"
    ]
    
    for i, line in enumerate(lines[:10]):  # Check first 10 lines
        line = line.strip()
        if not line:
            continue
            
        for pattern in vendor_patterns:
            match = re.search(pattern, line)
            if match and not invoice_data["vendor_name"]:
                vendor = match.group(1).strip()
                
                # Clean common OCR artifacts
                if "  " in vendor:
                    vendor = vendor.split("  ")[0]
                if vendor.endswith('.') or vendor.endswith(','):
                    vendor = vendor[:-1]
                    
                invoice_data["vendor_name"] = vendor
                invoice_data["extraction_details"]["extraction_notes"].append(f"Vendor found in line {i+1}: {vendor}")
                break
    
    # Extract invoice number
    invoice_patterns = [
        r'invoice\s*#?\s*:?\s*([A-Za-z0-9\-]{3,})',
        r'inv\s*#?\s*:?\s*([A-Za-z0-9\-]+)',
        r'#\s*([A-Za-z0-9\-]{5,})',
        r'(?:invoice|receipt)\s+number\s*:?\s*([A-Za-z0-9\-]+)'
    ]
    
    for pattern in invoice_patterns:
        match = re.search(pattern, text_lower)
        if match:
            invoice_num = match.group(1).upper()
            invoice_data["invoice_number"] = invoice_num
            invoice_data["extraction_details"]["extraction_notes"].append(f"Invoice number found: {invoice_num}")
            break
    
    # Extract date - prioritize structured formats
    date_patterns = [
        r'date\s*\n([\d/]+)',  # Date: 05/17/2015
        r'(\d{1,2}/\d{1,2}/\d{4})',
        r'(\d{4}-\d{2}-\d{2})',
        r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            date_str = match.group(1)
            invoice_data["date"] = date_str
            invoice_data["extraction_details"]["extraction_notes"].append(f"Date found: {date_str}")
            break
    
    # Enhanced money extraction with better context handling
    money_patterns = [
        r'\$\s*(\d+\.\d{2})\b',  # Match $x.xx specifically
        r'\b(\d+\.\d{2})\s*\$',   # Match x.xx $ format
        r'total\s*\$\s*(\d+\.\d{2})',  # Total: $x.xx
        r'subtotal\s*\$\s*(\d+\.\d{2})'  # Subtotal: $x.xx
    ]
    
    amounts = []
    # First pass: Look for amounts in key-value pairs
    for i, line in enumerate(lines):
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                key, value = parts[0].strip().lower(), parts[1].strip()
                
                # Check for amounts in values
                for pattern in money_patterns:
                    match = re.search(pattern, value)
                    if match:
                        try:
                            amount_val = float(match.group(1).replace(',', ''))
                            if amount_val > 0:
                                amounts.append((amount_val, line))
                                
                                # Directly map based on key
                                if 'subtotal' in key:
                                    invoice_data["subtotal"] = amount_val
                                    invoice_data["extraction_details"]["extraction_notes"].append(f"Subtotal found in key-value: ${amount_val}")
                                elif 'tax' in key or 'vat' in key or 'gst' in key:
                                    invoice_data["taxes"] = amount_val
                                    invoice_data["extraction_details"]["extraction_notes"].append(f"Tax found in key-value: ${amount_val}")
                                elif 'total' in key or 'amount due' in key:
                                    invoice_data["total"] = amount_val
                                    invoice_data["extraction_details"]["extraction_notes"].append(f"Total found in key-value: ${amount_val}")
                                    
                        except ValueError:
                            continue
    
    # Second pass: Scan all lines for amounts
    for line in lines:
        for pattern in money_patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                try:
                    amount_val = float(match.replace(',', ''))
                    if amount_val > 0:
                        # Check if we already added this amount from key-value
                        if not any(abs(a[0] - amount_val) < 0.01 and a[1] == line for a in amounts):
                            amounts.append((amount_val, line))
                except ValueError:
                    continue
    
    # Populate found amounts
    for amount_val, line in amounts:
        invoice_data["extraction_details"]["found_amounts"].append({
            "amount": amount_val,
            "context": line.strip(),
            "formatted": f"${amount_val:,.2f}"
        })
    
    # Set amounts if not found in key-value
    for amount_val, line in amounts:
        if any(keyword in line.lower() for keyword in ['subtotal', 'sub total', 'sub-total']) and not invoice_data["subtotal"]:
            invoice_data["subtotal"] = amount_val
            invoice_data["extraction_details"]["extraction_notes"].append(f"Subtotal found: ${amount_val}")
        elif any(keyword in line.lower() for keyword in ['tax', 'vat', 'gst']) and not invoice_data["taxes"]:
            invoice_data["taxes"] = amount_val
            invoice_data["extraction_details"]["extraction_notes"].append(f"Tax found: ${amount_val}")
        elif any(keyword in line.lower() for keyword in ['total', 'amount due', 'balance']) and not invoice_data["total"]:
            if not invoice_data["total"] or amount_val > invoice_data["total"]:
                invoice_data["total"] = amount_val
                invoice_data["extraction_details"]["extraction_notes"].append(f"Total found: ${amount_val}")
    
    # Calculate subtotal if missing
    if not invoice_data["subtotal"] and invoice_data["total"] and invoice_data["taxes"]:
        invoice_data["subtotal"] = round(invoice_data["total"] - invoice_data["taxes"], 2)
        invoice_data["extraction_details"]["extraction_notes"].append(f"Subtotal calculated: ${invoice_data['subtotal']}")
    
    # Extract payment method with better context
    payment_patterns = [
        r'(credit card|debit card|visa|mastercard|amex|american express)',
        r'(cash|check|cheque)',
        r'(paypal|venmo|zelle)',
        r'payment\s+method\s*:?\s*([A-Za-z\s]+)',
        r'paid\s+by\s*:?\s*([A-Za-z\s]+)'
    ]
    
    for pattern in payment_patterns:
        match = re.search(pattern, text_lower)
        if match:
            payment_method = match.group(1).strip().title()
            if payment_method not in ['The', 'And', 'Or', 'To', 'By']:
                invoice_data["payment_method"] = payment_method
                invoice_data["extraction_details"]["extraction_notes"].append(f"Payment method found: {payment_method}")
                break
    
    # Special case handling for donation receipts
    if "gap fund donation" in text_lower or "cup fund donation" in text_lower:
        invoice_data["payment_method"] = "Donation"
        invoice_data["extraction_details"]["extraction_notes"].append("Payment method set to Donation based on context")
    
    # Add summary statistics
    fields_extracted = sum(1 for k in ["vendor_name", "invoice_number", "date", "subtotal", "taxes", "total", "payment_method"] 
                          if invoice_data[k] is not None)
    
    invoice_data["extraction_details"]["summary"] = {
        "fields_extracted": fields_extracted,
        "total_amounts_found": len(amounts),
        "extraction_confidence": "high" if fields_extracted >= 4 else "medium" if fields_extracted >= 2 else "low"
    }
    
    return invoice_data

async def save_raw_text(image_filename: str, raw_text: str) -> str:
    """Save raw OCR text to a text file in the uploaded_images folder"""
    folder = "uploaded_images"
    
    # Create filename based on original image filename (same base name, .txt extension)
    base_name = os.path.splitext(image_filename)[0]
    txt_filename = f"{base_name}.txt"
    txt_path = os.path.join(folder, txt_filename)
    
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Original Image: {image_filename}\n")
            f.write(f"Processing Date: {datetime.now().isoformat()}\n")
            f.write("-" * 50 + "\n")
            f.write(raw_text)
        
        logger.info(f"Raw text saved to: {txt_path}")
        return txt_filename
    except Exception as e:
        logger.error(f"Error saving raw text: {str(e)}")
        return None

def search_files(query: str) -> List[Dict]:
    """Search for files in uploaded_images folder based on query"""
    folder = "uploaded_images"
    
    if not os.path.exists(folder):
        return []
    
    matching_files = []
    query_lower = query.lower()
    
    # Get all files in uploaded_images folder
    all_files = os.listdir(folder)
    
    for filename in all_files:
        file_path = os.path.join(folder, filename)
        
        # Skip directories
        if not os.path.isfile(file_path):
            continue
            
        # Search in filename
        if query_lower in filename.lower():
            matching_files.append({
                "filename": filename,
                "path": file_path,
                "match_type": "filename",
                "file_type": "image" if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')) else "text",
                "modified_date": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            })
        elif filename.endswith('.txt'):
            # Search within text file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if query_lower in content.lower():
                        matching_files.append({
                            "filename": filename,
                            "path": file_path,
                            "match_type": "content",
                            "file_type": "text",
                            "modified_date": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                        })
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {str(e)}")
                continue
    
    # Sort by modification date (newest first)
    matching_files.sort(key=lambda x: x["modified_date"], reverse=True)
    return matching_files

def get_file_content(filename: str) -> Optional[str]:
    """Get content of a specific file from uploaded_images folder"""
    folder = "uploaded_images"
    file_path = os.path.join(folder, filename)
    
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return None

async def process_image(image: UploadFile = File(...)):
    """Process uploaded image and perform OCR with invoice data extraction"""
    file_path = await save_image(image)
    
    try:
        # Method 1: Use PIL to handle various formats
        pil_image = Image.open(file_path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        img = np.array(pil_image)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Enhance image quality
        img_enhanced = rescale(img)
        
        logger.info(f"Processing started {file_path}")
        
        # Perform OCR
        results = reader.readtext(img_enhanced)
        raw_text = '\n'.join([result[1] for result in results])
        
        # Extract structured invoice data
        invoice_data = extract_invoice_data(raw_text)
        
        # Save raw text to file
        txt_filename = await save_raw_text(image.filename, raw_text)
        
        # Create JSON response with detailed invoice data
        response_data = {
            "success": True,
            "filename": image.filename,
            "invoice_data": invoice_data,
            "raw_text_file": txt_filename,
            "processing_info": {
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "filename": image.filename
        }
    # Note: No cleanup - files are permanently stored

async def save_image(image: UploadFile = File(...)):
    """Save uploaded image permanently"""
    folder = "uploaded_images"
    
    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    file_path = os.path.join(folder, image.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        return file_path
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Invoice OCR API",
        "description": "Upload an invoice image to extract structured data",
        "endpoints": {
            "POST /ocr": "Process invoice image and extract structured data",
            "GET /search?query=<search_term>": "Search for files by filename or content",
            "GET /file/<filename>": "Get specific file content (text files only)",
            "GET /list": "List all stored files (images and text)",
            "GET /health": "Health check"
        }
    }

@app.post("/ocr")
async def process_ocr(file: UploadFile = File(...)):
    """Endpoint for invoice OCR processing with structured data extraction"""
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    logger.info(f"Processing invoice file: {file.filename}")
    
    try:
        result = await process_image(file)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/search")
async def search_stored_files(query: str):
    """Search for files in uploaded_images folder based on query parameter"""
    if not query or len(query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters long")
    
    try:
        matching_files = search_files(query.strip())
        
        return JSONResponse(content={
            "success": True,
            "query": query,
            "total_matches": len(matching_files),
            "files": matching_files
        })
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/file/{filename}")
async def get_stored_file_content(filename: str):
    """Get content of a specific text file from uploaded_images folder"""
    
    # Basic filename validation
    if not filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Invalid file format. Only .txt files are supported for content retrieval")
    
    # Prevent directory traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    try:
        content = get_file_content(filename)
        
        if content is None:
            raise HTTPException(status_code=404, detail="File not found")
        
        return JSONResponse(content={
            "success": True,
            "filename": filename,
            "content": content,
            "retrieved_at": datetime.now().isoformat()
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving file: {str(e)}")

@app.get("/list")
async def list_all_stored_files():
    """List all stored files in uploaded_images folder with their relationships"""
    folder = "uploaded_images"
    
    if not os.path.exists(folder):
        return JSONResponse(content={
            "success": True,
            "total_files": 0,
            "files": []
        })
    
    try:
        all_files = os.listdir(folder)
        files_info = []
        
        # Group files by base name to show image-text relationships
        file_groups = {}
        
        for filename in all_files:
            file_path = os.path.join(folder, filename)
            
            # Skip directories
            if not os.path.isfile(file_path):
                continue
                
            base_name = os.path.splitext(filename)[0]
            extension = os.path.splitext(filename)[1].lower()
            
            if base_name not in file_groups:
                file_groups[base_name] = {
                    "base_name": base_name,
                    "image_file": None,
                    "text_file": None,
                    "has_both": False
                }
            
            stat_info = os.stat(file_path)
            file_info = {
                "filename": filename,
                "size_bytes": stat_info.st_size,
                "created_date": datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                "modified_date": datetime.fromtimestamp(stat_info.st_mtime).isoformat()
            }
            
            if extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                file_groups[base_name]["image_file"] = file_info
            elif extension == '.txt':
                file_groups[base_name]["text_file"] = file_info
        
        # Check which files have both image and text
        for group in file_groups.values():
            group["has_both"] = group["image_file"] is not None and group["text_file"] is not None
            files_info.append(group)
        
        # Sort by most recent modification date
        files_info.sort(key=lambda x: max(
            x["image_file"]["modified_date"] if x["image_file"] else "1970-01-01T00:00:00",
            x["text_file"]["modified_date"] if x["text_file"] else "1970-01-01T00:00:00"
        ), reverse=True)
        
        return JSONResponse(content={
            "success": True,
            "total_groups": len(files_info),
            "files": files_info
        })
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    folder = "uploaded_images"
    image_count = 0
    text_count = 0
    
    if os.path.exists(folder):
        all_files = os.listdir(folder)
        for filename in all_files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_count += 1
            elif filename.endswith('.txt'):
                text_count += 1
    
    return {
        "status": "healthy",
        "ocr_model_loaded": reader is not None,
        "stored_images": image_count,
        "stored_text_files": text_count,
        "storage_folder": folder,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    run(app, host="0.0.0.0", port=port)