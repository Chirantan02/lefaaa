import numpy as np
import cv2
from PIL import Image, ImageDraw
from skimage.measure import label, regionprops
from src.utils_mask_contour import get_img_agnostic_upper_contour, get_img_agnostic_lower_contour, get_img_agnostic_dresses_contour, remove_small


def get_img_agnostic_shorts_contour(im_parse, pose_data, offset_top, offset_bottom, offset_left, offset_right):
    """
    Special case handling for shorts - masks the full lower body regardless of shorts length
    while maintaining the contour-following approach for better fit.
    """
    foot1 = pose_data[18:21]  # right
    foot2 = pose_data[21:24]  # left
    faces = pose_data[24:92]
    hands1 = pose_data[92:113]
    hands2 = pose_data[113:]
    body = pose_data[:18]
    parse_array = np.array(im_parse)

    # Create masks for different body parts
    parse_upper = ((parse_array == 4).astype(np.float32) +
                   (parse_array == 14).astype(np.float32) +
                   (parse_array == 15).astype(np.float32)
                   )
    
    # For shorts, we want to include the full lower body area
    # This includes pants/shorts (5), skirt (6), dress (7), and belt (8)
    parse_lower = ((parse_array == 5).astype(np.float32) +
                   (parse_array == 6).astype(np.float32) +
                   (parse_array == 7).astype(np.float32) +
                   (parse_array == 8).astype(np.float32))
    
    parse_lower = np.uint8(parse_lower * 255)
    parse_lower = remove_small(parse_lower, min_area=1000 * (parse_lower.shape[0] * parse_lower.shape[1] / 1024 / 768))
    
    parse_head = ((parse_array == 3).astype(np.float32) +
                  (parse_array == 1).astype(np.float32) +
                  (parse_array == 11).astype(np.float32))
    
    parse_shoes = ((parse_array == 9).astype(np.float32) +
                   (parse_array == 10).astype(np.float32))
    
    parse_fixed = (parse_array == 16).astype(np.float32)

    # Create blank images
    agnostic = Image.new(mode='L', size=(parse_array.shape[1], parse_array.shape[0]), color=0)
    img_black = Image.new(mode='L', size=(im_parse.shape[1], im_parse.shape[0]), color=0)

    # Find contours
    contours, _ = cv2.findContours(parse_lower, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the lower body
    lower_mask = np.zeros_like(parse_array, dtype=np.uint8)
    
    # If contours exist, draw them on the mask
    if len(contours) > 0:
        # Apply dilation to create a slightly larger mask for better coverage
        kernel = np.ones((15, 15), np.uint8)  # Adjust kernel size as needed
        
        # Draw all contours on the mask
        cv2.drawContours(lower_mask, contours, -1, 255, -1)
        
        # Apply dilation to expand the mask slightly
        lower_mask = cv2.dilate(lower_mask, kernel, iterations=1)
        
        # For shorts, we need to extend the mask down to the ankles
        # Get the bounding box of the current maskHey, Cortana, today is the day of a lot of rain, so there's not much I can do, including. We chose a big front system, believing it to be the best available. Things are priorities and find a way to secure the roof. But before we could tackle that we needed to excavate the hill behind the barn to access the room for a thorough renovation. The big question was should we rent an excavator or buy one to get the job done? The dates were getting chillier and wetter, so nothing was better than enjoying comfort food and the warmth of a wood stove after a hard day's work. There's something undeniably magical about a wood stove. It radiates a gentle warmth that wrapped around me like a soft woolen blackboard. And it's more than drisky. It had become a central part of our daily routine, bending to the fire from morning until evening. As our own source of war, it truly became the heart of our home, making Odin's first birthday feel especially cozy. The property went from looking like this. To this, with much of the surrounding yard now cleared and manageable for maintenance. We had made quite some progress outside, but our attention shifted to the next fake challenge. 
        x, y, w, h = cv2.boundingRect(lower_mask)
        
        # Find ankle positions from pose data
        ankle_points = []
        if body[10, 0] != 0 and body[10, 1] != 0:  # Right ankle
            ankle_points.append((int(body[10, 0]), int(body[10, 1])))
        if body[13, 0] != 0 and body[13, 1] != 0:  # Left ankle
            ankle_points.append((int(body[13, 0]), int(body[13, 1])))
            
        # If ankle points are detected, extend the mask to include them
        if ankle_points:
            max_y_ankle = max([pt[1] for pt in ankle_points])
            # Create a rectangle that extends from the top of the shorts to the ankles
            extended_mask = np.zeros_like(lower_mask)
            cv2.rectangle(extended_mask, (x, y), (x + w, max_y_ankle), 255, -1)
            # Combine with the original mask
            lower_mask = cv2.bitwise_or(lower_mask, extended_mask)
        
        # Apply offset adjustments
        if offset_top != 0 or offset_bottom != 0 or offset_left != 0 or offset_right != 0:
            # Create a slightly expanded bounding box for the mask
            x, y, w, h = cv2.boundingRect(lower_mask)
            expanded_mask = np.zeros_like(lower_mask)
            # Apply offsets to the bounding box
            x1 = max(0, x - abs(offset_left) if offset_left < 0 else x)
            y1 = max(0, y - abs(offset_top) if offset_top < 0 else y)
            x2 = min(lower_mask.shape[1], x + w + abs(offset_right) if offset_right > 0 else x + w)
            y2 = min(lower_mask.shape[0], y + h + abs(offset_bottom) if offset_bottom > 0 else y + h)
            # Create a new mask with the expanded bounding box
            cv2.rectangle(expanded_mask, (x1, y1), (x2, y2), 255, -1)
            # Combine the original contour mask with the expanded bounding box
            lower_mask = cv2.bitwise_or(lower_mask, expanded_mask)
    
    # Convert the mask to PIL Image
    lower_mask_pil = Image.fromarray(lower_mask)
    
    # Create the final mask by combining the lower body mask with the head and fixed masks
    agnostic.paste(128, None, lower_mask_pil)
    agnostic.paste(img_black, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(img_black, None, Image.fromarray(np.uint8(parse_shoes * 255), 'L'))
    agnostic.paste(img_black, None, Image.fromarray(np.uint8(parse_fixed * 255), 'L'))
    
    # Create the gray mask and the binary mask
    mask_gray = agnostic.copy()
    mask = agnostic.point(lambda p: p == 128 and 255)
    
    return mask, mask_gray


def get_img_agnostic_tank_top_contour(im_parse, pose_data, offset_top, offset_bottom, offset_left, offset_right):
    """
    Special case handling for tank tops - ensures proper masking of the shoulder and arm areas
    while maintaining the contour-following approach.
    """
    foot = pose_data[18:24]
    faces = pose_data[24:92]
    hands1 = pose_data[92:113]
    hands2 = pose_data[113:]
    body = pose_data[:18]
    
    # Convert PIL Image to numpy array if needed
    if isinstance(im_parse, Image.Image):
        parse_array = np.array(im_parse)
    else:
        parse_array = im_parse
    
    # Create masks for different body parts
    # For tank tops, we need to include all upper body parts
    parse_upper_all = ((parse_array == 4).astype(np.float32) +
                    (parse_array == 7).astype(np.float32) +
                    (parse_array == 14).astype(np.float32) +
                    (parse_array == 15).astype(np.float32)
                    )
    
    parse_upper = ((parse_array == 4).astype(np.float32) +
                     (parse_array == 7).astype(np.float32)
                    )
                    
    parse_head = ((parse_array == 3).astype(np.float32) +
                    (parse_array == 1).astype(np.float32) + 
                    (parse_array == 11).astype(np.float32))
                    
    parse_fixed = (parse_array == 16).astype(np.float32)

    # Create blank images with the correct size
    h, w = parse_array.shape[:2]
    agnostic = Image.new(mode='L', size=(w, h), color=0)
    img_black = Image.new(mode='L', size=(w, h), color=0)

    # Convert to uint8 for OpenCV operations
    parse_upper = np.uint8(parse_upper * 255)
    parse_upper_all = np.uint8(parse_upper_all * 255)

    # Find contours
    contours_all, _ = cv2.findContours(parse_upper_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(parse_upper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the upper body
    upper_mask = np.zeros_like(parse_array, dtype=np.uint8)
    
    # If contours exist, draw them on the mask
    if len(contours_all) > 0:
        # Apply dilation to create a slightly larger mask for better coverage
        kernel = np.ones((15, 15), np.uint8)  # Adjust kernel size as needed
        
        # Draw all contours on the mask
        cv2.drawContours(upper_mask, contours_all, -1, 255, -1)
        
        # Apply dilation to expand the mask slightly
        upper_mask = cv2.dilate(upper_mask, kernel, iterations=1)
        
        # For tank tops, we need to ensure the shoulder areas are properly covered
        # Get shoulder positions from pose data
        shoulder_points = []
        if body[2, 0] != 0 and body[2, 1] != 0:  # Right shoulder
            shoulder_points.append((int(body[2, 0]), int(body[2, 1])))
        if body[5, 0] != 0 and body[5, 1] != 0:  # Left shoulder
            shoulder_points.append((int(body[5, 0]), int(body[5, 1])))
            
        # If shoulder points are detected, ensure they're included in the mask
        if shoulder_points:
            # Get the bounding box of the current mask
            x, y, w, h = cv2.boundingRect(upper_mask)
            
            # Create a mask that includes the shoulders and extends down
            shoulder_mask = np.zeros_like(upper_mask)
            min_x = min([pt[0] for pt in shoulder_points])
            max_x = max([pt[0] for pt in shoulder_points])
            min_y = min([pt[1] for pt in shoulder_points]) - 10  # Add padding
            
            # Get the actual contour bounds for better fitting
            max_y = y + h
            
            # Create a more natural shoulder shape using the actual garment contour
            pts = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(shoulder_mask, [pts], 255)
            
            # Apply Gaussian blur to smooth the edges
            shoulder_mask = cv2.GaussianBlur(shoulder_mask, (15, 15), 0)
            
            # Combine with the original mask using intersection to prevent overflow
            shoulder_mask = cv2.bitwise_and(shoulder_mask, upper_mask)
            upper_mask = cv2.bitwise_or(upper_mask, shoulder_mask)
        
        # Apply offset adjustments
        if offset_top != 0 or offset_bottom != 0 or offset_left != 0 or offset_right != 0:
            # Create a slightly expanded bounding box for the mask
            x, y, w, h = cv2.boundingRect(upper_mask)
            expanded_mask = np.zeros_like(upper_mask)
            # Apply offsets to the bounding box
            x1 = max(0, x - abs(offset_left) if offset_left < 0 else x)
            y1 = max(0, y - abs(offset_top) if offset_top < 0 else y)
            x2 = min(upper_mask.shape[1], x + w + abs(offset_right) if offset_right > 0 else x + w)
            y2 = min(upper_mask.shape[0], y + h + abs(offset_bottom) if offset_bottom > 0 else y + h)
            # Create a new mask with the expanded bounding box
            cv2.rectangle(expanded_mask, (x1, y1), (x2, y2), 255, -1)
            # Combine the original contour mask with the expanded bounding box
            upper_mask = cv2.bitwise_or(upper_mask, expanded_mask)
    
    # Convert the mask to PIL Image
    upper_mask_pil = Image.fromarray(upper_mask)
    
    # Create the final mask by combining the upper body mask with the head mask
    agnostic.paste(128, None, upper_mask_pil)
    agnostic.paste(img_black, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    
    # Create the gray mask and the binary mask
    mask_gray = agnostic.copy()
    mask = agnostic.point(lambda p: p == 128 and 255)
    
    return mask, mask_gray


def get_mask_location_special(category, garment_type, model_parse, pose_data, width, height, 
                      offset_top, offset_bottom, offset_left, offset_right):
    """
    Enhanced mask location function that handles special garment types.
    
    Args:
        category (str): The garment category ('Upper-body', 'Lower-body', 'Dresses')
        garment_type (str): The specific garment type (e.g., 'shorts', 'tank top')
        model_parse: The parsed model image
        pose_data: The pose keypoints data
        width, height: Dimensions for resizing
        offset_top, offset_bottom, offset_left, offset_right: Offset parameters
        
    Returns:
        tuple: (mask, mask_gray)
    """
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)
    
    # Convert garment_type to lowercase for case-insensitive comparison
    if garment_type:
        garment_type = garment_type.lower()
    
    # Handle special cases based on garment type
    if garment_type == 'shorts':
        mask, mask_gray = get_img_agnostic_shorts_contour(parse_array, pose_data, offset_top, offset_bottom, offset_left, offset_right)
        return mask, mask_gray
    elif garment_type == 'tank top':
        mask, mask_gray = get_img_agnostic_tank_top_contour(parse_array, pose_data, offset_top, offset_bottom, offset_left, offset_right)
        return mask, mask_gray
    
    # If no special case, use the standard contour-following approach based on category
    if category == "Upper-body":
        mask, mask_gray = get_img_agnostic_upper_contour(parse_array, pose_data, offset_top, offset_bottom, offset_left, offset_right)
        return mask, mask_gray
    elif category == "Dresses":
        mask, mask_gray = get_img_agnostic_dresses_contour(parse_array, pose_data, offset_top, offset_bottom, offset_left, offset_right)
        return mask, mask_gray
    elif category == "Lower-body":
        mask, mask_gray = get_img_agnostic_lower_contour(parse_array, pose_data, offset_top, offset_bottom, offset_left, offset_right)
        return mask, mask_gray