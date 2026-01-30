def extract_features_from_yolo(image_path, yolo_model, output_dir):
    """
    Use YOLO to detect features, then crop and save them
    """
    # Run YOLO
    results = yolo_model(image_path)
    
    # Load original image
    img = cv2.imread(image_path)
    
    # For each detection
    for detection in results.boxes:
        class_id = int(detection.cls)
        bbox = detection.xyxy[0]  # [x1, y1, x2, y2]
        
        # Crop feature
        x1, y1, x2, y2 = map(int, bbox)
        crop = img[y1:y2, x1:x2]
        
        # Save crop
        feature_name = CLASS_NAMES[class_id]
        output_path = f"{output_dir}/{feature_name}/{filename}"
        cv2.imwrite(output_path, crop)
        