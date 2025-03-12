import cv2
import numpy as np
import math

def evaluate_ellipse( a, b ):
    '''
    Evaluate error between two ellipses based on:
    - KL Divergence
    - Gaussian Angle
    - Intersection over Union
    - Absolute error in ellipse parameters
    
    Arguments:
    'a' and 'b' are lists such that:
    [ x centre, y centre, semimajor axis, semiminor axis, angle (radians) ]
    '''
    
    error = {}
    error['x_error'] = abs( a[0] - b[0] )
    error['y_error'] = abs( a[1] - b[1] )
    error['a_error'] = abs( a[2] - b[2] )
    error['b_error'] = abs( a[3] - b[3] )
    error['theta_error'] = abs( a[4] - b[4] )
    error['absolute_error'] = np.sum( np.abs( np.array( a ) - np.array( b ) ) )
    
    # Intersection over union!
    img_shape = ( 1024, 1024, 3 )
    img1 = np.zeros( img_shape )
    img2 = np.zeros( img_shape )
    
    # intersection_img = np.zeros( img_shape )
    
    # Draw predicted ellipse in Red channel (filled)
    cv2.ellipse(
        img1,
        ( int( a[0] ), int( a[1] ) ), # Center point
        ( int( a[2] ), int( a[3] ) ), # Semiminor and Semimajor axes
        a[4], 
        0, # Start Angle for drawing
        360, # End Angle for drawing
        ( 1, 0, 0 ),
        -1,
    )
    
    # cv2.ellipse(
    #     intersection_img,
    #     ( int( a[0] ), int( a[1] ) ), # Center point
    #     ( int( a[2] ), int( a[3] ) ), # Semiminor and Semimajor axes
    #     a[4], 
    #     0, # Start Angle for drawing
    #     360, # End Angle for drawing
    #     ( 255, 0, 0 ),
    #     2,
    # )
    
    cv2.ellipse(
        img2,
        ( int( b[0] ), int( b[1] ) ), # Center point
        ( int( b[2] ), int( b[3] ) ), # Semiminor and Semimajor axes
        b[4] , # Angle (convert from radians to degrees)
        0, # Start Angle for drawing
        360, # End Angle for drawing
        ( 1, 0, 0 ),
        -1,
    )
    
    # cv2.ellipse(
    #     intersection_img,
    #     ( int( b[0] ), int( b[1] ) ), # Center point
    #     ( int( b[2] ), int( b[3] ) ), # Semiminor and Semimajor axes
    #     b[4], # Angle (convert from radians to degrees)
    #     0, # Start Angle for drawing
    #     360, # End Angle for drawing
    #     ( 0, 255, 0 ),
    #     2,
    # )
    
    
    intersection = np.logical_and( img1[:,:,0], img2[:,:,0] )
    
    union = np.logical_or( img1[:,:,0], img2[:,:,0] )
    error['IoU'] = np.sum( intersection ) / np.sum( union )
    # cv2.imshow("Intersection Image", intersection_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return error['IoU']

# evaluate_ellipse([500, 400, 80, 60, 0], [500, 500, 80, 60, 30])