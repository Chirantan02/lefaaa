import numpy as np
import cv2
from PIL import Image, ImageDraw
from skimage.measure import label, regionprops

def remove_small(binary_mask, min_area=10000):
    binary_mask = binary_mask.astype(np.uint8)

    label_image = label(binary_mask)
    regions = regionprops(label_image)
    small_white_regions = [region for region in regions if region.area < min_area]

    for region in small_white_regions:
        min_row, min_col, max_row, max_col = region.bbox
        binary_mask[min_row:max_row, min_col:max_col] = 0
    return binary_mask


def get_img_agnostic_upper_contour(im_parse, pose_data, offset_top, offset_bottom, offset_left, offset_right):
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
    parse_upper_all = ((parse_array == 4).astype(np.float32) +
                    (parse_array == 7).astype(np.float32) +
                    (parse_array == 14).astype(np.float32) +
                    (parse_array == 15).astype(np.float32) +
                    (parse_array == 11).astype(np.float32)
                    )
    parse_upper = ((parse_array == 4).astype(np.float32) +
                     (parse_array == 7).astype(np.float32) +
                     (parse_array == 11).astype(np.float32)
                    )
    parse_head = ((parse_array == 3).astype(np.float32) +
                    (parse_array == 1).astype(np.float32) + 
                    (parse_array == 11).astype(np.float32))
    parse_fixed = (parse_array == 16).astype(np.float32)

    # Create blank images with the correct size
    h, w = parse_array.shape[:2]
    agnostic = Image.new(mode='L', size=(w, h), color=0)
    img_black = Image.new(mode='L', size=(w, h), color=0)
    agnostic_draw = ImageDraw.Draw(agnostic)

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


def get_img_agnostic_lower_contour(im_parse, pose_data, offset_top, offset_bottom, offset_left, offset_right):
    foot1 = pose_data[18:21] # right
    foot2 = pose_data[21:24] # left
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
    agnostic_draw = ImageDraw.Draw(agnostic)

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
    agnostic.paste(img_black, None, Image.fromarray(np.uint8(parse_fixed * 255), 'L'))
    
    # Create the gray mask and the binary mask
    mask_gray = agnostic.copy()
    mask = agnostic.point(lambda p: p == 128 and 255)
    
    return mask, mask_gray


def get_img_agnostic_dresses_contour(im_parse, pose_data, offset_top, offset_bottom, offset_left, offset_right):
    foot1 = pose_data[18:21] # right
    foot2 = pose_data[21:24] # left
    faces = pose_data[24:92]
    hands1 = pose_data[92:113]
    hands2 = pose_data[113:]
    body = pose_data[:18]
    parse_array = np.array(im_parse)
    
    # Create masks for different body parts
    parse_upper_all = ((parse_array == 4).astype(np.float32) +
                       (parse_array == 5).astype(np.float32) +
                       (parse_array == 6).astype(np.float32) +
                        (parse_array == 7).astype(np.float32) +
                        (parse_array == 8).astype(np.float32) +
                        (parse_array == 14).astype(np.float32) +
                        (parse_array == 15).astype(np.float32)
                    )
    parse_head = ((parse_array == 3).astype(np.float32) +
                    (parse_array == 1).astype(np.float32) + 
                    (parse_array == 11).astype(np.float32))
    parse_shoes = ((parse_array == 9).astype(np.float32) + 
                   (parse_array == 10).astype(np.float32))
    parse_fixed = (parse_array == 16).astype(np.float32)

    # Create blank images
    agnostic = Image.new(mode='L', size=(parse_array.shape[1], parse_array.shape[0]), color=0)
    img_black = Image.new(mode='L', size=(im_parse.shape[1], im_parse.shape[0]), color=0)
    agnostic_draw = ImageDraw.Draw(agnostic)

    # Convert to uint8 for OpenCV operations
    parse_upper_all = np.uint8(parse_upper_all * 255)
    
    # Find contours
    contours_all, _ = cv2.findContours(parse_upper_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the dress
    dress_mask = np.zeros_like(parse_array, dtype=np.uint8)
    
    # If contours exist, draw them on the mask
    if len(contours_all) > 0:
        # Apply dilation to create a slightly larger mask for better coverage
        kernel = np.ones((15, 15), np.uint8)  # Adjust kernel size as needed
        
        # Draw all contours on the mask
        cv2.drawContours(dress_mask, contours_all, -1, 255, -1)
        
        # Apply dilation to expand the mask slightly
        dress_mask = cv2.dilate(dress_mask, kernel, iterations=1)
        
        # Apply offset adjustments
        if offset_top != 0 or offset_bottom != 0 or offset_left != 0 or offset_right != 0:
            # Create a slightly expanded bounding box for the mask
            x, y, w, h = cv2.boundingRect(dress_mask)
            expanded_mask = np.zeros_like(dress_mask)
            # Apply offsets to the bounding box
            x1 = max(0, x - abs(offset_left) if offset_left < 0 else x)
            y1 = max(0, y - abs(offset_top) if offset_top < 0 else y)
            x2 = min(dress_mask.shape[1], x + w + abs(offset_right) if offset_right > 0 else x + w)
            y2 = min(dress_mask.shape[0], y + h + abs(offset_bottom) if offset_bottom > 0 else y + h)
            # Create a new mask with the expanded bounding box
            cv2.rectangle(expanded_mask, (x1, y1), (x2, y2), 255, -1)
            # Combine the original contour mask with the expanded bounding box
            dress_mask = cv2.bitwise_or(dress_mask, expanded_mask)
    
    # Convert the mask to PIL Image
    dress_mask_pil = Image.fromarray(dress_mask)
    
    # Create the final mask by combining the dress mask with the head, shoes, and fixed masks
    agnostic.paste(128, None, dress_mask_pil)
    agnostic.paste(img_black, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(img_black, None, Image.fromarray(np.uint8(parse_shoes * 255), 'L'))
    agnostic.paste(img_black, None, Image.fromarray(np.uint8(parse_fixed * 255), 'L'))
    
    # Create the gray mask and the binary mask
    mask_gray = agnostic.copy()
    mask = agnostic.point(lambda p: p == 128 and 255)
    
    return mask, mask_gray


def get_mask_location(category, model_parse, pose_data, width, height, 
                      offset_top, offset_bottom, offset_left, offset_right):
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)

    if category == "Upper-body":
        mask, mask_gray = get_img_agnostic_upper_contour(parse_array, pose_data, offset_top, offset_bottom, offset_left, offset_right)
        return mask, mask_gray
    elif category == "Dresses":
        mask, mask_gray = get_img_agnostic_dresses_contour(parse_array, pose_data, offset_top, offset_bottom, offset_left, offset_right)
        return mask, mask_gray
    elif category == "Lower-body":
        mask, mask_gray = get_img_agnostic_lower_contour(parse_array, pose_data, offset_top, offset_bottom, offset_left, offset_right)
        return mask, mask_gray