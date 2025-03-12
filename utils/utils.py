import cv2
import matplotlib.pyplot as plt
import numpy as np


def find_and_draw_good_matches(image_1, image_2, keypoints_1, keypoints_2, matches, ratio, is_plot=True):
    """
    Find and draw good matches between reference and query images.

    Args:
        image_1 (numpy.ndarray): Reference image.
        image_2 (numpy.ndarray): Query image.
        ref_keypoints (list): List of keypoints in the reference image.
        query_keypoints (list): List of keypoints in the query image.
        matches (list): List of matches between keypoints.
        ratio (float): Ratio threshold for the ratio test.
        is_plot (bool, optional): Whether to display the matches plot. Defaults to True.

    Returns:
        list: List of good matches.

    """
    # Apply ratio test to find good matches
    good_matches = []
    for higher_match, lower_match in matches:
        if higher_match.distance < ratio * lower_match.distance:
            good_matches.append(higher_match)
    print(f"Number of good matches: {len(good_matches)}")

    # Draw matches
    img_matches = cv2.drawMatches(image_1, keypoints_1, image_2, keypoints_2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    if not is_plot:
        return good_matches
    else: 
        # Display the matches
        plt.figure(figsize=(12, 12))
        plt.imshow(img_matches)
        plt.title("Draw matches")
        plt.axis('off')
        plt.show()
        return good_matches


def stitch_two_imgs(image_1, image_2):
    orb = cv2.ORB_create() #Using default values for number of features = 500

    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2, None)

    print(f"Number of keypoints in image 1: {len(keypoints_1)}")
    print(f"Number of keypoints in image 2: {len(keypoints_2)}")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2) 
    print(len(matches))

    # good_matches = find_and_draw_good_matches(image_1, image_2, keypoints_1, keypoints_2, matches, 0.5)

    src = np.float32([keypoints_1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst = np.float32([keypoints_2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    dst = cv2.warpPerspective(image_1,H,(image_2.shape[1] + image_1.shape[1], image_2.shape[0]))
    plt.figure(figsize=(80, 80))  # Increase the figure size here
    plt.subplot(122)
    plt.imshow(dst)
    plt.title('Warped Image')
    plt.show()

    plt.figure(figsize=(80, 80))  # Increase the figure size here
    dst[0:image_2.shape[0], 0:image_2.shape[1]] = image_2

    # Trim all zero pixels on the edge
    nonzero_rows, nonzero_cols = np.nonzero(dst)
    trimmed_dst = dst[np.min(nonzero_rows):np.max(nonzero_rows)+1, np.min(nonzero_cols):np.max(nonzero_cols)+1]

    cv2.imwrite('output.jpg', trimmed_dst)
    plt.imshow(trimmed_dst)
    plt.show()