# import os
# import io
# import time
# import hashlib
# import logging
# from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# import numpy as np
# from PIL import Image
# import cv2

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("racepass")

# app = FastAPI(title="RacePass Face Enrollment API")

# # CORS — allow React dev server
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://localhost:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # In-memory store for demo (replace with encrypted DB in production)
# # Structure: { wallet_address: { embedding_hash, enrolled_at, face_shape } }
# enrolled_faces: dict = {}

# # Load OpenCV face detector (built-in, no API key needed)
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# def extract_face_embedding(image_bytes: bytes) -> tuple[np.ndarray, dict]:
#     """
#     Extract a simple face embedding from image bytes.
    
#     In production: Replace this with AWS Rekognition IndexFaces API call.
#     This demo uses:
#       - OpenCV face detection to confirm a face exists
#       - A normalized pixel vector from the detected face region as the 'embedding'
    
#     Returns: (embedding_vector, metadata)
#     """
#     # Decode image
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
#     if img is None:
#         raise ValueError("Could not decode image")
    
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Detect faces
#     faces = face_cascade.detectMultiScale(
#         gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
#     )
    
#     if len(faces) == 0:
#         raise ValueError("No face detected in the image. Please ensure your face is clearly visible.")
    
#     if len(faces) > 1:
#         raise ValueError("Multiple faces detected. Please ensure only one face is in frame.")
    
#     # Get largest face bounding box
#     x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    
#     # Crop and resize face to 64x64 for embedding
#     face_roi = gray[y:y+h, x:x+w]
#     face_resized = cv2.resize(face_roi, (64, 64))
    
#     # Normalize to 0-1 range and flatten — this is the "embedding"
#     # In production: AWS Rekognition returns a 128-dim float vector
#     embedding = face_resized.astype(np.float32) / 255.0
#     embedding = embedding.flatten()  # 4096-dim vector
    
#     metadata = {
#         "face_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
#         "image_size": {"width": img.shape[1], "height": img.shape[0]},
#         "faces_found": len(faces),
#     }
    
#     return embedding, metadata


# def compute_embedding_hash(embedding: np.ndarray) -> str:
#     """
#     Compute keccak256-style hash of the embedding.
#     In production: this hash goes on-chain via markFaceEnrolled().
#     """
#     embedding_bytes = embedding.tobytes()
#     # Use SHA3-256 (keccak256 equivalent available in hashlib in Python 3.6+)
#     hash_obj = hashlib.sha3_256(embedding_bytes)
#     return "0x" + hash_obj.hexdigest()


# def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#     """Compute cosine similarity between two embedding vectors."""
#     dot = np.dot(a, b)
#     norm = np.linalg.norm(a) * np.linalg.norm(b)
#     if norm == 0:
#         return 0.0
#     return float(dot / norm)


# def check_duplicate_face(new_embedding: np.ndarray, exclude_wallet: str = None) -> dict | None:
#     """
#     Check if this face is already enrolled under a different wallet.
#     Returns conflicting wallet info if found, None if clean.
#     Threshold: 0.95 cosine similarity (same as your spec).
#     """
#     DUPLICATE_THRESHOLD = 0.95
    
#     for wallet, data in enrolled_faces.items():
#         if wallet == exclude_wallet:
#             continue
        
#         existing_embedding = np.array(data["embedding"])
#         similarity = cosine_similarity(new_embedding, existing_embedding)
        
#         logger.info(f"Similarity with {wallet[:8]}...: {similarity:.4f}")
        
#         if similarity >= DUPLICATE_THRESHOLD:
#             return {
#                 "conflicting_wallet": wallet[:8] + "..." + wallet[-4:],  # Redacted
#                 "similarity": round(similarity * 100, 2),
#             }
    
#     return None


# @app.get("/")
# def root():
#     return {"status": "RacePass Face Enrollment API running", "version": "1.0.0-demo"}


# @app.get("/api/status/{wallet_address}")
# def get_status(wallet_address: str):
#     """Check enrollment status for a wallet."""
#     if wallet_address in enrolled_faces:
#         data = enrolled_faces[wallet_address]
#         return {
#             "enrolled": True,
#             "wallet": wallet_address,
#             "embedding_hash": data["embedding_hash"],
#             "enrolled_at": data["enrolled_at"],
#             "face_metadata": data["face_metadata"],
#         }
#     return {"enrolled": False, "wallet": wallet_address}


# @app.post("/api/enroll-face")
# async def enroll_face(
#     selfie: UploadFile = File(...),
#     walletAddress: str = Form(...)
# ):
#     """
#     Main face enrollment endpoint.
    
#     What this does (in order):
#     1. Read image bytes (NEVER write to disk)
#     2. Detect face with OpenCV (production: AWS Rekognition)
#     3. Extract face embedding vector
#     4. Check for duplicate face across existing enrollments
#     5. Compute embedding hash (would go on-chain in production)
#     6. Store ONLY the embedding hash + vector in memory (encrypted DB in production)
#     7. Return success — raw image bytes are garbage collected immediately
    
#     Raw image is NEVER saved to disk. It exists only in memory during processing.
#     """
#     start_time = time.time()
#     logger.info(f"[ENROLL] Wallet: {walletAddress[:10]}... | File: {selfie.filename} | Type: {selfie.content_type}")
    
#     # Validate wallet address format
#     if not walletAddress.startswith("0x") or len(walletAddress) < 10:
#         raise HTTPException(status_code=400, detail="Invalid wallet address format")
    
#     # Validate file type
#     if selfie.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/webp"]:
#         raise HTTPException(status_code=400, detail=f"Invalid file type: {selfie.content_type}")
    
#     # Read image into memory ONLY — never touch disk
#     image_bytes = await selfie.read()
#     logger.info(f"[ENROLL] Image size: {len(image_bytes)} bytes")
    
#     # Size sanity check
#     if len(image_bytes) > 10 * 1024 * 1024:  # 10MB max
#         raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
    
#     # Step 1: Extract face embedding
#     try:
#         embedding, face_metadata = extract_face_embedding(image_bytes)
#         logger.info(f"[ENROLL] Face detected at {face_metadata['face_box']}")
#     except ValueError as e:
#         raise HTTPException(status_code=422, detail=str(e))
    
#     # Step 2: Check for duplicate face (fraud detection)
#     duplicate = check_duplicate_face(embedding, exclude_wallet=walletAddress)
#     if duplicate:
#         logger.warning(f"[FRAUD] Duplicate face detected! Similarity: {duplicate['similarity']}%")
#         raise HTTPException(
#             status_code=409,
#             detail={
#                 "error": "DUPLICATE_FACE",
#                 "message": "This face is already enrolled under another wallet. This has been flagged.",
#                 "similarity": duplicate["similarity"],
#             }
#         )
    
#     # Step 3: Compute embedding hash (this is what goes on-chain)
#     embedding_hash = compute_embedding_hash(embedding)
#     logger.info(f"[ENROLL] Embedding hash: {embedding_hash[:16]}...")
    
#     # Step 4: Store — embedding vector (encrypted in prod) + hash
#     enrolled_faces[walletAddress] = {
#         "embedding": embedding.tolist(),  # In prod: encrypt with KMS before storing
#         "embedding_hash": embedding_hash,
#         "enrolled_at": int(time.time()),
#         "face_metadata": face_metadata,
#     }
    
#     # Step 5: Delete raw image — Python GC handles this, but be explicit
#     del image_bytes
#     del embedding  # Sensitive
    
#     processing_time = round((time.time() - start_time) * 1000, 2)
#     logger.info(f"[ENROLL] ✓ Complete in {processing_time}ms | Hash: {embedding_hash[:16]}...")
    
#     return JSONResponse({
#         "success": True,
#         "embeddingHash": embedding_hash,
#         "txHash": f"0xdemo_{walletAddress[-6:]}_{int(time.time())}",  # Mock tx hash
#         "faceMetadata": face_metadata,
#         "processingMs": processing_time,
#         "privacyNote": "Raw image was never stored. Only embedding hash retained.",
#         # In production: call markFaceEnrolled(walletAddress, embeddingHash) on Identity Registry contract
#     })


# @app.post("/api/verify-face")
# async def verify_face(
#     selfie: UploadFile = File(...),
#     walletAddress: str = Form(...)
# ):
#     """
#     Verify face at venue entry — compare live selfie against enrolled embedding.
#     Returns match confidence score.
#     """
#     if walletAddress not in enrolled_faces:
#         raise HTTPException(status_code=404, detail="Wallet not enrolled. Complete face enrollment first.")
    
#     image_bytes = await selfie.read()
    
#     try:
#         new_embedding, face_metadata = extract_face_embedding(image_bytes)
#     except ValueError as e:
#         raise HTTPException(status_code=422, detail=str(e))
    
#     # Compare against stored embedding
#     stored_embedding = np.array(enrolled_faces[walletAddress]["embedding"])
#     similarity = cosine_similarity(new_embedding, stored_embedding)
#     confidence = round(similarity * 100, 2)
    
#     MATCH_THRESHOLD = 90.0  # As per your spec
#     matched = confidence >= MATCH_THRESHOLD
    
#     del image_bytes
#     del new_embedding
    
#     logger.info(f"[VERIFY] Wallet: {walletAddress[:10]}... | Confidence: {confidence}% | Match: {matched}")
    
#     return {
#         "matched": matched,
#         "confidence": confidence,
#         "threshold": MATCH_THRESHOLD,
#         "status": "VERIFIED" if matched else "MISMATCH",
#         "faceMetadata": face_metadata,
#     }


# @app.delete("/api/enroll-face/{wallet_address}")
# def delete_enrollment(wallet_address: str):
#     """Delete enrollment for a wallet (for testing/reset)."""
#     if wallet_address in enrolled_faces:
#         del enrolled_faces[wallet_address]
#         return {"deleted": True, "wallet": wallet_address}
#     raise HTTPException(status_code=404, detail="Wallet not found")


# @app.get("/api/debug/enrolled")
# def list_enrolled():
#     """Debug endpoint — shows enrolled wallets without embeddings."""
#     return {
#         "count": len(enrolled_faces),
#         "wallets": [
#             {
#                 "wallet": w[:8] + "..." + w[-4:],
#                 "enrolled_at": d["enrolled_at"],
#                 "embedding_hash": d["embedding_hash"],
#             }
#             for w, d in enrolled_faces.items()
#         ]
#     }


#version 2
# import os
# import time
# import hashlib
# import logging
# import tempfile
# from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# import numpy as np

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("racepass")

# app = FastAPI(title="RacePass Face Enrollment API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://localhost:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # In-memory store: { wallet: { embedding, embedding_hash, enrolled_at } }
# enrolled_faces: dict = {}

# # Load DeepFace lazily on first use
# _deepface_loaded = False

# def get_deepface():
#     global _deepface_loaded
#     from deepface import DeepFace
#     _deepface_loaded = True
#     return DeepFace


# def extract_embedding(image_bytes: bytes) -> np.ndarray:
#     """
#     Extract 512-dim ArcFace embedding from image bytes.
#     ArcFace is trained on millions of faces — accurate enough for production.
#     Temporarily writes to disk only for DeepFace compatibility, deletes immediately.
#     """
#     DeepFace = get_deepface()

#     # Write to temp file — DeepFace needs a file path
#     with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
#         f.write(image_bytes)
#         tmp_path = f.name

#     try:
#         result = DeepFace.represent(
#             img_path=tmp_path,
#             model_name='ArcFace',      # Best accuracy, 512-dim embedding
#             detector_backend='opencv', # Fast detector
#             enforce_detection=True,    # Raises error if no face found
#         )
#         embedding = np.array(result[0]['embedding'])
#         return embedding
#     except ValueError as e:
#         raise ValueError(f"No face detected. Please ensure your face is clearly visible and well-lit.")
#     finally:
#         # Always delete temp file
#         os.unlink(tmp_path)


# def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#     norm = np.linalg.norm(a) * np.linalg.norm(b)
#     if norm == 0:
#         return 0.0
#     return float(np.dot(a, b) / norm)


# def check_duplicate(new_embedding: np.ndarray, exclude_wallet: str = None) -> dict | None:
#     """
#     ArcFace cosine similarity threshold for same person: > 0.28
#     (This is the standard threshold for ArcFace — NOT the 0.95 we used before)
#     """
#     DUPLICATE_THRESHOLD = 0.28

#     for wallet, data in enrolled_faces.items():
#         if wallet == exclude_wallet:
#             continue
#         existing = np.array(data['embedding'])
#         sim = cosine_similarity(new_embedding, existing)
#         logger.info(f"Duplicate check vs {wallet[:8]}...: similarity={sim:.4f}")
#         if sim > DUPLICATE_THRESHOLD:
#             return {
#                 'conflicting_wallet': wallet[:8] + '...' + wallet[-4:],
#                 'similarity': round(sim * 100, 2),
#             }
#     return None


# def compute_hash(embedding: np.ndarray) -> str:
#     return '0x' + hashlib.sha3_256(embedding.tobytes()).hexdigest()


# @app.get('/')
# def root():
#     return {'status': 'RacePass Face Enrollment API running', 'model': 'ArcFace (DeepFace)'}


# @app.get('/api/status/{wallet_address}')
# def get_status(wallet_address: str):
#     if wallet_address in enrolled_faces:
#         data = enrolled_faces[wallet_address]
#         return {
#             'enrolled': True,
#             'wallet': wallet_address,
#             'embedding_hash': data['embedding_hash'],
#             'enrolled_at': data['enrolled_at'],
#             'face_metadata': data.get('face_metadata', {}),
#         }
#     return {'enrolled': False, 'wallet': wallet_address}


# @app.post('/api/enroll-face')
# async def enroll_face(
#     selfie: UploadFile = File(...),
#     walletAddress: str = Form(...)
# ):
#     start = time.time()
#     logger.info(f'[ENROLL] Wallet: {walletAddress[:10]}...')

#     if not walletAddress.startswith('0x') or len(walletAddress) < 10:
#         raise HTTPException(status_code=400, detail='Invalid wallet address')

#     image_bytes = await selfie.read()
#     logger.info(f'[ENROLL] Image size: {len(image_bytes)} bytes')

#     if len(image_bytes) < 5000:
#         raise HTTPException(status_code=400, detail='Image too small — capture failed')

#     # Extract ArcFace embedding
#     try:
#         embedding = extract_embedding(image_bytes)
#         logger.info(f'[ENROLL] Embedding extracted, dim={len(embedding)}')
#     except ValueError as e:
#         raise HTTPException(status_code=422, detail=str(e))

#     # Duplicate face check
#     duplicate = check_duplicate(embedding, exclude_wallet=walletAddress)
#     if duplicate:
#         logger.warning(f'[FRAUD] Duplicate face! Similarity: {duplicate["similarity"]}%')
#         raise HTTPException(status_code=409, detail={
#             'error': 'DUPLICATE_FACE',
#             'message': 'This face is already enrolled under another wallet.',
#             'similarity': duplicate['similarity'],
#         })

#     embedding_hash = compute_hash(embedding)

#     enrolled_faces[walletAddress] = {
#         'embedding': embedding.tolist(),
#         'embedding_hash': embedding_hash,
#         'enrolled_at': int(time.time()),
#         'face_metadata': {'embedding_dim': len(embedding), 'model': 'ArcFace'},
#     }

#     del image_bytes

#     ms = round((time.time() - start) * 1000, 2)
#     logger.info(f'[ENROLL] ✓ Done in {ms}ms | Hash: {embedding_hash[:16]}...')

#     return JSONResponse({
#         'success': True,
#         'embeddingHash': embedding_hash,
#         'txHash': f'0xdemo_{walletAddress[-6:]}_{int(time.time())}',
#         'processingMs': ms,
#         'model': 'ArcFace',
#         'privacyNote': 'Temp file deleted. Only embedding hash retained.',
#     })


# @app.post('/api/verify-face')
# async def verify_face(
#     selfie: UploadFile = File(...),
#     walletAddress: str = Form(...)
# ):
#     if walletAddress not in enrolled_faces:
#         raise HTTPException(status_code=404, detail='Wallet not enrolled.')

#     image_bytes = await selfie.read()

#     try:
#         new_embedding = extract_embedding(image_bytes)
#     except ValueError as e:
#         raise HTTPException(status_code=422, detail=str(e))

#     stored_embedding = np.array(enrolled_faces[walletAddress]['embedding'])
#     similarity = cosine_similarity(new_embedding, stored_embedding)

#     # ArcFace threshold: >0.28 cosine = same person
#     # Convert to a 0-100 confidence score for display
#     # Map: 0.28 (threshold) → 90%, 1.0 → 100%, below 0.28 → below 90%
#     if similarity >= 0.28:
#         confidence = round(90 + (similarity - 0.28) / (1.0 - 0.28) * 10, 2)
#     else:
#         confidence = round(similarity / 0.28 * 90, 2)

#     matched = similarity >= 0.28

#     del image_bytes, new_embedding

#     logger.info(f'[VERIFY] Wallet: {walletAddress[:10]}... | Raw sim: {similarity:.4f} | Confidence: {confidence}% | Match: {matched}')

#     return {
#         'matched': matched,
#         'confidence': confidence,
#         'raw_similarity': round(similarity, 4),
#         'threshold': 90.0,
#         'status': 'VERIFIED' if matched else 'MISMATCH',
#         'faceMetadata': {'model': 'ArcFace'},
#     }


# @app.delete('/api/enroll-face/{wallet_address}')
# def delete_enrollment(wallet_address: str):
#     if wallet_address in enrolled_faces:
#         del enrolled_faces[wallet_address]
#         return {'deleted': True}
#     raise HTTPException(status_code=404, detail='Wallet not found')


# @app.get('/api/debug/enrolled')
# def list_enrolled():
#     return {
#         'count': len(enrolled_faces),
#         'model': 'ArcFace',
#         'wallets': [
#             {
#                 'wallet': w[:8] + '...' + w[-4:],
#                 'enrolled_at': d['enrolled_at'],
#                 'embedding_hash': d['embedding_hash'],
#             }
#             for w, d in enrolled_faces.items()
#         ]
#     }

# import os
# import io
# import time
# import hashlib
# import logging
# import numpy as np
# import PIL.Image
# import face_recognition
# from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("racepass")

# app = FastAPI(title="RacePass Face Enrollment API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://localhost:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# enrolled_faces: dict = {}


# def extract_embedding(image_bytes: bytes) -> np.ndarray:
#     pil_image = PIL.Image.open(io.BytesIO(image_bytes)).convert('RGB')
#     img_array = np.array(pil_image)

#     face_locations = face_recognition.face_locations(img_array, model='hog')

#     if len(face_locations) == 0:
#         raise ValueError("No face detected. Ensure your face is clearly visible and well-lit.")
#     if len(face_locations) > 1:
#         raise ValueError("Multiple faces detected. Please ensure only one face is in frame.")

#     encodings = face_recognition.face_encodings(img_array, known_face_locations=face_locations)

#     if not encodings:
#         raise ValueError("Could not encode face. Please try again with better lighting.")

#     return np.array(encodings[0])


# def face_distance(a: np.ndarray, b: np.ndarray) -> float:
#     return float(np.linalg.norm(a - b))


# def distance_to_confidence(distance: float) -> float:
#     return round(max(0.0, (1.0 - distance / 0.6) * 100), 2)


# def check_duplicate(new_embedding: np.ndarray, exclude_wallet: str = None) -> dict | None:
#     DUPLICATE_THRESHOLD = 0.5
#     for wallet, data in enrolled_faces.items():
#         if wallet == exclude_wallet:
#             continue
#         existing = np.array(data['embedding'])
#         dist = face_distance(new_embedding, existing)
#         logger.info(f"Duplicate check vs {wallet[:8]}...: distance={dist:.4f}")
#         if dist < DUPLICATE_THRESHOLD:
#             return {
#                 'conflicting_wallet': wallet[:8] + '...' + wallet[-4:],
#                 'distance': round(dist, 4),
#             }
#     return None


# def compute_hash(embedding: np.ndarray) -> str:
#     return '0x' + hashlib.sha3_256(embedding.tobytes()).hexdigest()


# @app.get('/')
# def root():
#     return {
#         'status': 'RacePass Face Enrollment API running',
#         'model': 'dlib face_recognition (128-dim, 99.38% accuracy)',
#     }


# @app.get('/api/status/{wallet_address}')
# def get_status(wallet_address: str):
#     if wallet_address in enrolled_faces:
#         data = enrolled_faces[wallet_address]
#         return {
#             'enrolled': True,
#             'wallet': wallet_address,
#             'embedding_hash': data['embedding_hash'],
#             'enrolled_at': data['enrolled_at'],
#             'face_metadata': data.get('face_metadata', {}),
#         }
#     return {'enrolled': False, 'wallet': wallet_address}


# @app.post('/api/enroll-face')
# async def enroll_face(
#     selfie: UploadFile = File(...),
#     walletAddress: str = Form(...)
# ):
#     start = time.time()
#     logger.info(f'[ENROLL] Wallet: {walletAddress[:10]}...')

#     if not walletAddress.startswith('0x') or len(walletAddress) < 10:
#         raise HTTPException(status_code=400, detail='Invalid wallet address')

#     image_bytes = await selfie.read()
#     logger.info(f'[ENROLL] Image received: {len(image_bytes)} bytes')

#     if len(image_bytes) < 5000:
#         raise HTTPException(status_code=400, detail='Image too small — capture failed. Try again.')

#     try:
#         embedding = extract_embedding(image_bytes)
#         logger.info(f'[ENROLL] Embedding extracted: {len(embedding)} dimensions')
#     except ValueError as e:
#         raise HTTPException(status_code=422, detail=str(e))

#     duplicate = check_duplicate(embedding, exclude_wallet=walletAddress)
#     if duplicate:
#         logger.warning(f'[FRAUD] Duplicate face! Distance: {duplicate["distance"]}')
#         raise HTTPException(status_code=409, detail={
#             'error': 'DUPLICATE_FACE',
#             'message': 'This face is already enrolled under another wallet. Flagged for review.',
#             'distance': duplicate['distance'],
#         })

#     embedding_hash = compute_hash(embedding)

#     enrolled_faces[walletAddress] = {
#         'embedding': embedding.tolist(),
#         'embedding_hash': embedding_hash,
#         'enrolled_at': int(time.time()),
#         'face_metadata': {
#             'embedding_dim': len(embedding),
#             'model': 'dlib face_recognition',
#         },
#     }

#     del image_bytes

#     ms = round((time.time() - start) * 1000, 2)
#     logger.info(f'[ENROLL] ✓ Complete in {ms}ms | Hash: {embedding_hash[:16]}...')

#     return JSONResponse({
#         'success': True,
#         'embeddingHash': embedding_hash,
#         'txHash': f'0xdemo_{walletAddress[-6:]}_{int(time.time())}',
#         'processingMs': ms,
#         'model': 'dlib face_recognition',
#         'privacyNote': 'Raw image processed in-memory and deleted. Only hash retained.',
#     })


# @app.post('/api/verify-face')
# async def verify_face(
#     selfie: UploadFile = File(...),
#     walletAddress: str = Form(...)
# ):
#     if walletAddress not in enrolled_faces:
#         raise HTTPException(status_code=404, detail='Wallet not enrolled. Complete face enrollment first.')

#     image_bytes = await selfie.read()

#     if len(image_bytes) < 5000:
#         raise HTTPException(status_code=400, detail='Image too small — capture failed. Try again.')

#     try:
#         new_embedding = extract_embedding(image_bytes)
#     except ValueError as e:
#         raise HTTPException(status_code=422, detail=str(e))

#     stored_embedding = np.array(enrolled_faces[walletAddress]['embedding'])
#     distance = face_distance(new_embedding, stored_embedding)
#     confidence = distance_to_confidence(distance)
#     matched = distance < 0.6

#     del image_bytes, new_embedding

#     logger.info(f'[VERIFY] Distance: {distance:.4f} | Confidence: {confidence}% | Match: {matched}')

#     return {
#         'matched': matched,
#         'confidence': confidence,
#         'raw_distance': round(distance, 4),
#         'threshold': 90.0,
#         'status': 'VERIFIED' if matched else 'MISMATCH',
#         'faceMetadata': {
#             'model': 'dlib face_recognition',
#             'distance_threshold': 0.6,
#         },
#     }


# @app.delete('/api/enroll-face/{wallet_address}')
# def delete_enrollment(wallet_address: str):
#     if wallet_address in enrolled_faces:
#         del enrolled_faces[wallet_address]
#         return {'deleted': True, 'wallet': wallet_address}
#     raise HTTPException(status_code=404, detail='Wallet not found')


# @app.get('/api/debug/enrolled')
# def list_enrolled():
#     return {
#         'count': len(enrolled_faces),
#         'model': 'dlib face_recognition',
#         'wallets': [
#             {
#                 'wallet': w[:8] + '...' + w[-4:],
#                 'enrolled_at': d['enrolled_at'],
#                 'embedding_hash': d['embedding_hash'],
#             }
#             for w, d in enrolled_faces.items()
#         ]
#     }


# RECENTTT VERSION 5.43PM
# import os
# import io
# import time
# import hashlib
# import logging
# import numpy as np
# import PIL.Image
# import face_recognition
# from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("racepass")

# app = FastAPI(title="RacePass Face Enrollment API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://localhost:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# enrolled_faces: dict = {}


# def extract_embedding(image_bytes: bytes) -> np.ndarray:
#     pil_image = PIL.Image.open(io.BytesIO(image_bytes)).convert('RGB')

#     # Resize to 800px width for consistency
#     w, h = pil_image.size
#     if w > 800:
#         ratio = 800 / w
#         pil_image = pil_image.resize((800, int(h * ratio)), PIL.Image.LANCZOS)

#     img_array = np.array(pil_image)

#     face_locations = face_recognition.face_locations(img_array, model='hog')

#     if len(face_locations) == 0:
#         # Retry with higher upsampling
#         face_locations = face_recognition.face_locations(img_array, number_of_times_to_upsample=2, model='hog')

#     if len(face_locations) == 0:
#         raise ValueError("No face detected. Ensure your face is clearly visible and well-lit.")

#     if len(face_locations) > 1:
#         # Use largest face instead of rejecting
#         face_locations = [max(face_locations, key=lambda f: (f[2]-f[0]) * (f[1]-f[3]))]

#     # num_jitters=3 averages 3 samples — much more stable across shots
#     encodings = face_recognition.face_encodings(img_array, known_face_locations=face_locations, num_jitters=1)
#     if not encodings:
#         raise ValueError("Could not encode face. Please try again with better lighting.")

#     return np.array(encodings[0])


# def face_distance(a: np.ndarray, b: np.ndarray) -> float:
#     return float(np.linalg.norm(a - b))


# def distance_to_confidence(distance: float) -> float:
#     return round(max(0.0, (1.0 - distance / 0.6) * 100), 2)


# def check_duplicate(new_embedding: np.ndarray, exclude_wallet: str = None) -> dict | None:
#     DUPLICATE_THRESHOLD = 0.5
#     for wallet, data in enrolled_faces.items():
#         if wallet == exclude_wallet:
#             continue
#         existing = np.array(data['embedding'])
#         dist = face_distance(new_embedding, existing)
#         logger.info(f"Duplicate check vs {wallet[:8]}...: distance={dist:.4f}")
#         if dist < DUPLICATE_THRESHOLD:
#             return {
#                 'conflicting_wallet': wallet[:8] + '...' + wallet[-4:],
#                 'distance': round(dist, 4),
#             }
#     return None


# def compute_hash(embedding: np.ndarray) -> str:
#     return '0x' + hashlib.sha3_256(embedding.tobytes()).hexdigest()


# @app.get('/')
# def root():
#     return {
#         'status': 'RacePass Face Enrollment API running',
#         'model': 'dlib face_recognition (128-dim, 99.38% accuracy)',
#     }


# @app.get('/api/status/{wallet_address}')
# def get_status(wallet_address: str):
#     if wallet_address in enrolled_faces:
#         data = enrolled_faces[wallet_address]
#         return {
#             'enrolled': True,
#             'wallet': wallet_address,
#             'embedding_hash': data['embedding_hash'],
#             'enrolled_at': data['enrolled_at'],
#             'face_metadata': data.get('face_metadata', {}),
#         }
#     return {'enrolled': False, 'wallet': wallet_address}


# @app.post('/api/enroll-face')
# async def enroll_face(
#     selfie: UploadFile = File(...),
#     walletAddress: str = Form(...)
# ):
#     start = time.time()
#     logger.info(f'[ENROLL] Wallet: {walletAddress[:10]}...')

#     if not walletAddress.startswith('0x') or len(walletAddress) < 10:
#         raise HTTPException(status_code=400, detail='Invalid wallet address')

#     image_bytes = await selfie.read()
#     logger.info(f'[ENROLL] Image received: {len(image_bytes)} bytes')

#     if len(image_bytes) < 5000:
#         raise HTTPException(status_code=400, detail='Image too small — capture failed. Try again.')

#     try:
#         embedding = extract_embedding(image_bytes)
#         logger.info(f'[ENROLL] Embedding extracted: {len(embedding)} dimensions')
#     except ValueError as e:
#         raise HTTPException(status_code=422, detail=str(e))

#     duplicate = check_duplicate(embedding, exclude_wallet=walletAddress)
#     if duplicate:
#         logger.warning(f'[FRAUD] Duplicate face! Distance: {duplicate["distance"]}')
#         raise HTTPException(status_code=409, detail={
#             'error': 'DUPLICATE_FACE',
#             'message': 'This face is already enrolled under another wallet. Flagged for review.',
#             'distance': duplicate['distance'],
#         })

#     embedding_hash = compute_hash(embedding)

#     enrolled_faces[walletAddress] = {
#         'embedding': embedding.tolist(),
#         'embedding_hash': embedding_hash,
#         'enrolled_at': int(time.time()),
#         'face_metadata': {
#             'embedding_dim': len(embedding),
#             'model': 'dlib face_recognition',
#         },
#     }

#     del image_bytes

#     ms = round((time.time() - start) * 1000, 2)
#     logger.info(f'[ENROLL] ✓ Complete in {ms}ms | Hash: {embedding_hash[:16]}...')

#     return JSONResponse({
#         'success': True,
#         'embeddingHash': embedding_hash,
#         'txHash': f'0xdemo_{walletAddress[-6:]}_{int(time.time())}',
#         'processingMs': ms,
#         'model': 'dlib face_recognition',
#         'privacyNote': 'Raw image processed in-memory and deleted. Only hash retained.',
#     })


# @app.post('/api/verify-face')
# async def verify_face(
#     selfie: UploadFile = File(...),
#     walletAddress: str = Form(...)
# ):
#     if walletAddress not in enrolled_faces:
#         raise HTTPException(status_code=404, detail='Wallet not enrolled. Complete face enrollment first.')

#     image_bytes = await selfie.read()

#     if len(image_bytes) < 5000:
#         raise HTTPException(status_code=400, detail='Image too small — capture failed. Try again.')

#     try:
#         new_embedding = extract_embedding(image_bytes)
#     except ValueError as e:
#         raise HTTPException(status_code=422, detail=str(e))

#     stored_embedding = np.array(enrolled_faces[walletAddress]['embedding'])
#     distance = face_distance(new_embedding, stored_embedding)

#     # matched based on raw dlib distance, not display percentage
#     matched = distance < 0.6
#     raw_confidence = round(max(0.0, (1.0 - distance / 0.6) * 100), 2)

#     del image_bytes, new_embedding

#     logger.info(f'[VERIFY] Distance: {distance:.4f} | Confidence: {raw_confidence}% | Match: {matched}')

#     return {
#         'matched': matched,
#         'confidence': raw_confidence,
#         'raw_distance': round(distance, 4),
#         'threshold': 60.0,
#         'status': 'VERIFIED' if matched else 'MISMATCH',
#         'faceMetadata': {
#             'model': 'dlib face_recognition',
#             'distance_threshold': 0.6,
#         },
#     }


# @app.delete('/api/enroll-face/{wallet_address}')
# def delete_enrollment(wallet_address: str):
#     if wallet_address in enrolled_faces:
#         del enrolled_faces[wallet_address]
#         return {'deleted': True, 'wallet': wallet_address}
#     raise HTTPException(status_code=404, detail='Wallet not found')


# @app.get('/api/debug/enrolled')
# def list_enrolled():
#     return {
#         'count': len(enrolled_faces),
#         'model': 'dlib face_recognition',
#         'wallets': [
#             {
#                 'wallet': w[:8] + '...' + w[-4:],
#                 'enrolled_at': d['enrolled_at'],
#                 'embedding_hash': d['embedding_hash'],
#             }
#             for w, d in enrolled_faces.items()
#         ]
#     }

#FINAL VERSION 1.12 AM
import io
import time
import hashlib
import logging
import numpy as np
import cv2
import mediapipe as mp

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face-service")

app = FastAPI(title="Lightweight Face Enrollment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (MVP only)
enrolled_faces: dict = {}

# MediaPipe face detector
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)


# ----------------------------
# Face Processing
# ----------------------------

def extract_embedding(image_bytes: bytes) -> np.ndarray:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not results.detections:
        raise ValueError("No face detected")

    h, w, _ = img.shape
    box = results.detections[0].location_data.relative_bounding_box

    x1 = int(box.xmin * w)
    y1 = int(box.ymin * h)
    x2 = int((box.xmin + box.width) * w)
    y2 = int((box.ymin + box.height) * h)

    face = img[y1:y2, x1:x2]

    if face.size == 0:
        raise ValueError("Face crop failed")

    face = cv2.resize(face, (64, 64))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    embedding = face.flatten() / 255.0
    return embedding


def face_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_hash(embedding: np.ndarray) -> str:
    return "0x" + hashlib.sha3_256(embedding.tobytes()).hexdigest()


# ----------------------------
# Routes
# ----------------------------

@app.get("/")
def root():
    return {
        "status": "Lightweight Face API running",
        "model": "MediaPipe + simple embedding",
    }


@app.post("/api/enroll-face")
async def enroll_face(
    selfie: UploadFile = File(...),
    walletAddress: str = Form(...)
):
    if not walletAddress.startswith("0x"):
        raise HTTPException(status_code=400, detail="Invalid wallet address")

    image_bytes = await selfie.read()

    if len(image_bytes) < 5000:
        raise HTTPException(status_code=400, detail="Image too small")

    try:
        embedding = extract_embedding(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Duplicate check
    for wallet, data in enrolled_faces.items():
        if wallet == walletAddress:
            continue

        existing = np.array(data["embedding"])
        dist = face_distance(embedding, existing)

        if dist < 15:  # threshold tuned for this embedding type
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "DUPLICATE_FACE",
                    "distance": round(dist, 2),
                },
            )

    embedding_hash = compute_hash(embedding)

    enrolled_faces[walletAddress] = {
        "embedding": embedding.tolist(),
        "embedding_hash": embedding_hash,
        "enrolled_at": int(time.time()),
    }

    return JSONResponse({
        "success": True,
        "embeddingHash": embedding_hash,
    })


@app.post("/api/verify-face")
async def verify_face(
    selfie: UploadFile = File(...),
    walletAddress: str = Form(...)
):
    if walletAddress not in enrolled_faces:
        raise HTTPException(status_code=404, detail="Wallet not enrolled")

    image_bytes = await selfie.read()

    try:
        new_embedding = extract_embedding(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    stored_embedding = np.array(enrolled_faces[walletAddress]["embedding"])
    distance = face_distance(new_embedding, stored_embedding)

    #matched = distance < 15
    #confidence = round(max(0.0, 100 - (distance * 5)), 2)
    
    confidence = round(max(0.0, 100 - (distance * 5)), 2)
    CONFIDENCE_THRESHOLD = 50
    matched = confidence >= CONFIDENCE_THRESHOLD
    
    return {
        "matched": matched,
        "confidence": confidence,
        "raw_distance": round(distance, 2),
        "status": "VERIFIED" if matched else "MISMATCH",
    }