# This pipeline was implemented in a previous project titled Advanced Lane Lines, 
# for details please refer to project link below
# project link: https://github.com/wphw/CarND-Advanced-Lane-Lines/



import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# camera parameters
# camera matrix (c_mtx) and distortion coefficients (dist) were
# computed previously.
c_mtx = np.array([
    [1.15396091e+03, 0.00000000e+00, 6.69706056e+02],
    [0.00000000e+00, 1.14802495e+03, 3.85655654e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[
    -2.41016964e-01,
    -5.30788794e-02,
    -1.15812035e-03,
    -1.28281652e-04,
    2.67259026e-02]])



# we will need to convert from BGR to RGB
def bgr2rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)



# write a function to get calibration coeffs and camera matrix
def camera_calibration_params(images, nx, ny):
    """Get camera calibration parameters.
    
    This function uses several chessboard images and returns 
    the calibrations parameters which are camera matrix and
    calibration coefficients.
    
    Args:
        images: A list of strings defining the path of each image.
        
        nx: the number of corners along x-axis inside a chessboard.
        
        ny: the number of corners along y-axis inside a chessboard.
    """
    
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane

    # prepare object points like (0, 0, 0), (1, 0, 0), 
    # (2, 0, 0), ..., (7, 5, 0)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    
    ret = False
    c_mtx, dist = None, None
    for path in images:
        img = cv2.imread(path)
        # use `COLOR_RGB2GRAY` if read by `matplotlib.image.imread`
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # if corners are found, add object and image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp) # all object points will be the same for each imgpoints
            print('corners found for image: ' + path)
            # draw and display corners
            #img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            #plt.imshow(img)
            #plt.show()
            #plt.close()

    # now do the calibration, here c_mtx is the camera matrix
    # and dist are the distiortion coefficients
    # and rvecs is the rotation and the tvecs is the
    # translation vectors.
    if ret:
        ret, c_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
    
    return ret, c_mtx, dist



# define a function to undistort the raw images of cameras
def undistort(raw, c_mtx, dist):
    """Undistort raw camera images.
    
    This function uses the camera matrix (c_mtx), distortion
    coefficients (dist) to undistort raw camera images.
    
    Returns BGR images!
    
    Args:
        raw (ndarray): The image taken by the cameraof which no 
            distortion correction applied. Image should be in `BGR` 
            format, read with `cv2.imread`.
            
        c_mtx: Camera calibration matrix, can be obtained using the
            `cv2.calibrateCamera` module.
        
        dist: Distortion coefficients, can be obtained using the
            `cv2.calibrateCamera` module.
    """
    
    # get undistorted destination image
    undist = cv2.undistort(raw, c_mtx, dist, None, c_mtx)
    
    return undist



# define a color thresholding function
def color_thresholding(
    img, ch_type='rgb', 
    binary=True, plot=False, 
    thr=(220, 255), save_path=None):
    """Apply color thresholding.
    
    Arg:
        img (numpy array): numpy image array, should be in `RGB`
            color space, NOT in `BGR`.
            
        ch_type (str): can be 'rgb', 'hls', 'hsv', 'yuv', 'ycrcb',
            'lab', 'luv'.
            
        binary (bool): If `True` then show and returns binary
            images. If not, returns original images in defined 
            color spaces.
            
        plot: If `True`, shows images.
        
        thr: min, max value for threasholding.
        
        save_path: if defines, saves figures.
    """
    # get channels
    if ch_type is 'hls':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif ch_type is 'hsv':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif ch_type is 'yuv':    
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif ch_type is 'ycrcb':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif ch_type is 'lab':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    elif ch_type is 'luv':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
    
    img_ch1 = img[:,:,0]
    img_ch2 = img[:,:,1]
    img_ch3 = img[:,:,2]

    # apply thresholding
    bin_ch1 = np.zeros_like(img_ch1)
    bin_ch2 = np.zeros_like(img_ch2)
    bin_ch3 = np.zeros_like(img_ch3)

    bin_ch1[(img_ch1 > thr[0]) & (img_ch1 <= thr[1])] = 1
    bin_ch2[(img_ch2 > thr[0]) & (img_ch2 <= thr[1])] = 1
    bin_ch3[(img_ch3 > thr[0]) & (img_ch3 <= thr[1])] = 1
    
    if binary:
        imrep_ch1 = bin_ch1
        imrep_ch2 = bin_ch2
        imrep_ch3 = bin_ch3
    else:
        imrep_ch1 = img_ch1
        imrep_ch2 = img_ch2
        imrep_ch3 = img_ch3
    if plot:
        n_rows = 2
        n_cols = 3
        fig, axarr = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 5, n_rows * 3),
            subplot_kw={'xticks': [], 'yticks': []})
        axarr[0, 0].imshow(img_ch1, cmap='gray')
        axarr[0, 0].set_title('original ' + ch_type + ' space: ch1')
        axarr[0, 1].imshow(img_ch2, cmap='gray')
        axarr[0, 1].set_title('original ' + ch_type + ' space: ch2')
        axarr[0, 2].imshow(img_ch3, cmap='gray')
        axarr[0, 2].set_title('original ' + ch_type + ' space: ch3')
        
        axarr[1, 0].imshow(imrep_ch1, cmap='gray')
        axarr[1, 0].set_title('binary ' + ch_type + ' space: ch1')
        axarr[1, 1].imshow(imrep_ch2, cmap='gray')
        axarr[1, 1].set_title('binary ' + ch_type + ' space: ch2')
        axarr[1, 2].imshow(imrep_ch3, cmap='gray')
        axarr[1, 2].set_title('binary ' + ch_type + ' space: ch3')
        plt.show()
        plt.close()
    
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.imsave(save_path + '/' + ch_type + '_ch1.png', imrep_ch1, cmap='gray')
        plt.imsave(save_path + '/' + ch_type + '_ch2.png', imrep_ch2, cmap='gray')
        plt.imsave(save_path + '/' + ch_type + '_ch3.png', imrep_ch3, cmap='gray')
    
    return imrep_ch1, imrep_ch2, imrep_ch3



# define a function to compare two different images
def compare_2img_color_thr(img1, img2, binary=False, save_path=[None, None]):
    """Compares two images for various color thresholdings.
     
    Args:
        img1 (numpy array): numpy image array, should be in `RGB` color
            space, NOT in `BGR`.
        
        img2 (numpy array): numpy image array, should be in `RGB` color
            space, NOT in `BGR`.
        
        binary (bool): If `True`, returns binary representations.
            If `False`, returns original images in defined
            color spaces.
        
        save_path (str): if not None, figures are saved.
    """
    
    ch_types = ['rgb', 'hls', 'hsv', 'yuv', 'ycrcb', 'lab', 'luv']
    
    dst_images = {}
    dst_images['img1'] = {}
    dst_images['img2'] = {}
    for ch_type in ch_types:
        dst_images['img1'][ch_type] =\
            color_thresholding(img1, ch_type=ch_type, binary=binary, save_path=save_path[0])
        dst_images['img2'][ch_type] =\
            color_thresholding(img2, ch_type=ch_type, binary=binary, save_path=save_path[1])
        
    n_cols = 3 * 2
    n_rows = len(ch_types)
    
    fig, axarr = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2),
        subplot_kw={'xticks': [], 'yticks': []})
    
    k = 0
    for ch_type in ch_types:
        axarr[k, 0].imshow(dst_images['img1'][ch_type][0], cmap='gray')
        axarr[k, 0].set_title(ch_type + ' space: ch1')
        
        axarr[k, 1].imshow(dst_images['img1'][ch_type][1], cmap='gray')
        axarr[k, 1].set_title(ch_type + ' space: ch2')
        
        axarr[k, 2].imshow(dst_images['img1'][ch_type][2], cmap='gray')
        axarr[k, 2].set_title(ch_type + ' space: ch3')
        
        axarr[k, 3].imshow(dst_images['img2'][ch_type][0], cmap='gray')
        axarr[k, 3].set_title(ch_type + ' space: ch1')
        
        axarr[k, 4].imshow(dst_images['img2'][ch_type][1], cmap='gray')
        axarr[k, 4].set_title(ch_type + ' space: ch2')
        
        axarr[k, 5].imshow(dst_images['img2'][ch_type][2], cmap='gray')
        axarr[k, 5].set_title(ch_type + ' space: ch3')
        
        k += 1

    plt.show()
    plt.close()



# write a masking function for defining a region of interest
def region_of_interest(img, points):
    """Mask the region of interest
    
    Only keeps the region of the image defined by the polygon
    formed from `points`. The rest of the image is set to black.
    
    Args:
        img: numpy array representing the image.
        points: verticies, example:
            [[(x1, y1), (x2, y2), (x4, y4), (x3, y3)]]
    """
    # define empty binary mask
    mask = np.zeros_like(img)
    
    # define a mask color to ignore the masking area
    # if there are more than one color channels consider
    # the shape of ignoring mask color
    if len(img.shape) > 2:
        n_chs = img.shape[2]
        ignore_mask_color = (255,) * n_chs
    else:
        ignore_mask_color = 255
    
    # define unmasking region 
    cv2.fillPoly(mask, points, ignore_mask_color)
    
    # mask the image
    masked_img = cv2.bitwise_and(img, mask)
    
    return masked_img

# define a perspective transformation function
def birds_eye_transform(img, points, offsetx):
    """Transforms the viewpoint to a bird's-eye view.
    
    Applies a perspective transformation. Returns
    the inverse matrix and the warped destination
    image.
    
    Args:
        img: A numpy image array.
        points: A list of four points to be flattened.
            Example: points = [[x1,y1], [x2,y2], [x4,y4], [x3,y3]].
        offsetx: offset value for x-axis.
    """
    
    img_size = img[:,:,0].shape[::-1]
    
    
    # get the region of interest
    img = region_of_interest(img, np.array([points]))
    
    src = np.float32(
        [
            points[0],
            points[1],
            points[2],
            points[3],
        ])

    pt1 = [offsetx, 0]
    pt2 = [img_size[0] - offsetx, 0]
    pt3 = [img_size[0] - offsetx, img_size[1]]
    pt4 = [offsetx, img_size[1]]
    dst = np.float32([pt1, pt2, pt3, pt4])
    
    mtx = cv2.getPerspectiveTransform(src, dst)
    invmtx = cv2.getPerspectiveTransform(dst, src)
    
    warped = cv2.warpPerspective(img, mtx, img_size)
    
    return invmtx, warped



# Define a function to combine color thresholding
def combined_color_thresholding(
    img, thr_rgb=(230,255), thr_hsv=(230,255), thr_luv=(157,255)):
    """Combines color thresholding on different channels
    
    Returns a binary image.
    
    Args:
        img: Numpy image array, should be in `RGB` color
            space.
        
        thr_rgb: min and max thresholding values for RGB color
            space.
        
        thr_hsv: min and max thresholding values for HSV color
            space.
        
        thr_luv: min and max thresholding values for LUV color
            space.
    """
    
    bin_rgb_ch1, bin_rgb_ch2, bin_rgb_ch3 =\
        color_thresholding(img, ch_type='rgb', thr=thr_rgb)
        
    bin_hsv_ch1, bin_hsv_ch2, bin_hsv_ch3 =\
    color_thresholding(img, ch_type='hsv', thr=thr_hsv)
    
    bin_luv_ch1, bin_luv_ch2, bin_luv_ch3 =\
    color_thresholding(img, ch_type='luv', thr=thr_luv)
    
    binary = np.zeros_like(bin_rgb_ch1)
    binary[
        (bin_rgb_ch1 == 1)
        | (bin_hsv_ch3 == 1)
        | (bin_luv_ch3 == 1)
        ] = 1
    
    return binary



def search_lane(warped, n_stepsy=9, n_stepsx=2, std_max=5., min_samp=50):
    """Search binary images to detect lane lines.
    
    Takes the top-view (warped) image as input, applies color
    thresholding to get a binary representation. Searches binary 
    image with sliding windows and detects lane pixels. In the 
    sliding window, if the standart deviation value for the white 
    pixels is lower than a threshold, detects as line pixels and 
    appends to result.
    
    returns an output image and the fitting parameters.
    
    Args:
        warped (ndarray): numpy array representing the top-view road image.
            Image should be in RGB format.
        
        n_stepsy (int): number of slides on y-axis.
        
        n_stepsx (int): number of slides on x-axis (n_stepsx >= 2).
        
        std_max (float): Float to represent the maximum value of
            standart deviation.
        
        min_samp (int): minimum number of white pixels to considering
            to check if lane or not.
    """
    
    def stdx(binary_window, starty=0, startx=0):
        """Computes mean and standart deviation of a binary window.
    
        Computes and returns the standart deviation value of lane pixels
        in the binary search window, along x-axis. Also returns number of 
        samples (number of white pixels in this case) and the lists of the
        absolute position of white pixels (absolute x and y coordinates of 
        lane pixels) in the whole binary image.
        
        Args:
            binary_window: An numpy array to represent a small,
                rectangular portion of an image as a sliding 
                search window/kernel.
                
            startx: integer to represent window position start
                value on x-axis.
                
            starty: integer to represent window position start
                value on y-axis.
        
        """
        y, x = binary_window.nonzero()
        # get absolute coordinates in whole image
        y_absolute = y + starty
        x_absolute = x + startx
        n_samples = len(x) # number of samples
        x = x + 1
        if n_samples > 0:
            std = x.std()
        else:
            std = np.inf
        return n_samples, std, x_absolute.tolist(), y_absolute.tolist()
    
    if n_stepsx < 2:
        n_stepsx = 2
    offsety = warped.shape[0] // n_stepsy
    offsetx = warped.shape[1] // n_stepsx
    halfx = warped.shape[1] // 2
    
    margin = 50 # margin of detected cluster squares (just to draw a rectangle).
    
    # define output images for visualization
    out_img = np.zeros_like(warped)
    
    # list left and right lane indices
    left_lane_inds = [[],[]]
    right_lane_inds = [[],[]]
    
    # thresholds for combined color channels
    thr_hsv = (230,255)
    thr_luv = (157,255)
    thr_rgb_list = [
        (230, 255),
        (185, 230)
    ]
    # get binary images list
    binary_images = []
    for thr_rgb in thr_rgb_list:
        binary = combined_color_thresholding(
            warped, thr_rgb=thr_rgb, thr_hsv=thr_hsv, thr_luv=thr_luv)
        binary_images.append(binary)
    
    for i in range(n_stepsy):
        starty = i * offsety
        endy = (i+1) * offsety
        
        left_res, right_res = False, False
        for binary in binary_images:
            
            for j in range(n_stepsx):
                
                startx = j * offsetx
                endx = (j+1) * offsetx
                
                window = binary[starty:endy,startx:endx]
                
                n_samples, std, x, y = stdx(window, starty, startx)
                
                if startx < halfx:
                    found_flag = left_res
                else:
                    found_flag = right_res
                
                if (std < std_max) & (n_samples > min_samp) & (not found_flag):
                    histogram = np.sum(window, axis=0)
                    peak_base = np.argmax(histogram) + startx
                    winx_start = peak_base - margin
                    winx_end = peak_base + margin
                    # append x and y coors for detected lane pixels
                    # decide if left or right lane pixels
                    if peak_base < halfx:
                        left_res = True
                        left_lane_inds[0].extend(x)
                        left_lane_inds[1].extend(y)
                        color = (255,0,0)
                    else:
                        right_res = True
                        right_lane_inds[0].extend(x)
                        right_lane_inds[1].extend(y)
                        color = (0,0,255)
                    # draw output image
                    out_img[y,x,:] = 255
                    cv2.rectangle(
                        out_img,
                        (winx_start,starty),
                        (winx_end,endy),
                        color, 2)
            if left_res & right_res:
                break
    
    return out_img, left_lane_inds, right_lane_inds



# define a function for fitting and measurements
def fit_lane(lane_img, left_lane_inds, right_lane_inds):
    """Fit lane lines and do the measurements."""
    
    res = False
    marked_lane_img = None
    filled_lane_img = None
    fit_lane_img = None
    avg_curve_radi = None
    if (len(left_lane_inds[0]) > 0) & (len(left_lane_inds[1]) > 0)\
        & (len(right_lane_inds[0]) > 0) & (len(right_lane_inds[1]) > 0):
        
        res = True
        # fit
        # 2nd order polynomial
        left_lane_inds = np.array(left_lane_inds)
        right_lane_inds = np.array(right_lane_inds)
        # we will fit y, rather than x just because most x values may
        # be the same for different y values (f(y)=Ay^2+By+C).
        left_fit = np.polyfit(left_lane_inds[1], left_lane_inds[0], 2)
        right_fit = np.polyfit(right_lane_inds[1], right_lane_inds[0], 2)
    
        # generate x and y values for plotting
        ploty = np.linspace(0, lane_img.shape[0]-1, lane_img.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # mark and fill detected lanes
        # mark first
        zeros_img = np.zeros_like(lane_img[:,:,0]).astype(np.uint8)
        marked_lane_img = np.dstack((zeros_img, zeros_img, zeros_img))
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        mark_margin = 10
        left_line_window1 =\
            np.array([np.transpose(np.vstack([left_fitx-mark_margin, ploty]))])
        left_line_window2 =\
            np.array([np.flipud(np.transpose(np.vstack([left_fitx+mark_margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 =\
            np.array([np.transpose(np.vstack([right_fitx-mark_margin, ploty]))])
        right_line_window2 =\
            np.array([np.flipud(np.transpose(np.vstack([right_fitx+mark_margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(marked_lane_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(marked_lane_img, np.int_([right_line_pts]), (0,255, 0))
        marked_lane_img = cv2.addWeighted(lane_img, 1, marked_lane_img, 0.4, 0)
        
        # mark and fill
        filled_lane_img = np.dstack((zeros_img, zeros_img, zeros_img))
        # recast x and y to usable format
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(filled_lane_img, np.int_([pts]), (208, 255, 64))
    
        # measure the radious of the curvature
        # choose the max y value
        y_eval = np.max(ploty)
        # To apply pixel to meter conversion we will consider
        # lane is about 3.7 meters wide and 30 meters long.
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/300 # meters per pixel in x dimension
        left_fit_cr = np.polyfit(
            left_lane_inds[1]*ym_per_pix, left_lane_inds[0]*xm_per_pix, 2)
        right_fit_cr = np.polyfit(
            right_lane_inds[1]*ym_per_pix, right_lane_inds[0]*xm_per_pix, 2)
        left_curverad = (
            (1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5)\
            / np.absolute(2*left_fit_cr[0])
        right_curverad = (
            (1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5)\
            / np.absolute(2*right_fit_cr[0])
        avg_curve_radi = (left_curverad+right_curverad)*0.5 # in meters
        
        # measure the position of the vehicle
        # the left and right x positions at the bottom of the image
        h = lane_img.shape[0]
        w = lane_img.shape[1]
        left_point = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
        right_point = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
        # measure lane midpoint
        lane_midpoint = (right_point-left_point)*0.5 + left_point
        # get the position of the vehicle with respect to 
        # the midpoint of the lane in meters
        pos = (lane_midpoint - w*0.5) * xm_per_pix
        
    
    return res, marked_lane_img, filled_lane_img, avg_curve_radi, pos




# A helper function to combine resulting images and add text
def combine_output(output_images, measurements):
    """Combine output images and write measurements."""
    
    combined_img = None
    if not None in output_images:
        # crop top views
        cropped_birds = bgr2rgb(output_images[0])[:,400:880]
        cropped_birds2 = bgr2rgb(output_images[2])[:,400:880]
        # concat 2 top view images and resize
        combined_birds = np.concatenate((cropped_birds2, cropped_birds), axis=1)
        combined_birds = cv2.resize(combined_birds, (combined_birds.shape[1]//2, combined_birds.shape[0]//2))
        # combine all 3 images
        combined_img = np.copy(bgr2rgb(output_images[4]))
        combined_img[0:360,800:1280] = combined_birds
        # add text
        radius = measurements[0]
        pos = measurements[1]
        fontface = cv2.FONT_HERSHEY_SIMPLEX
        fontscale = 1
        thickness = 2
        text = 'Predicted Lines'
        cv2.putText(combined_img, text, (807, 30), fontface, fontscale, (64,255,208), thickness)
        text = 'Top View'
        cv2.putText(combined_img, text, (1100, 30), fontface, fontscale, (64,255,208), thickness)
        text = 'Curve Radius = %.2f' % radius + 'm'
        cv2.putText(combined_img, text, (10, 30), fontface, fontscale, (64,255,208), thickness)
        if pos < 0:
            side = 'right'
        else:
            side = 'left'
        text = 'Vehicle is %.2f' % np.abs(pos) + 'm ' + side + ' of center'
        cv2.putText(combined_img, text, (10, 60), fontface, fontscale, (64,255,208), thickness)
    
    return combined_img



def lane_detector(image, c_mtx, dist, std_max=15., min_samp=30):
    """Detect the lane and do the measurements.
    
    Steps:
    * Takes an image of road as the input.
    * Undistort the image with camera calibration.
    * Transforms the region of interest into bird's-eye view,
        gets a top-view of the road. Also returns this image.
    * Applies color thresholding to get a binary representation
        of lane lines.
    * Detects the lane pixels and fits a second order polynomial 
        to those pixel coordinates. Returns the output image.
    * Measures the curvature radius of the lane and the position of
        the car in the lane and returns the results.
    * Colors the lane, transforms back to it's previous viewpoint
        and returns the final output image.
    
    Args:
        image: original road image in BGR colors, read with
            `cv2.imread`.
        
        c_mtx: Camera matrix for camera calibration.
        
        dist: Distortion coefficients for camera calibration.
        
        std_max: Float to represent maximum standard deviation of
            white pixel clusters to consider if they are lane pixels
            or not.
        
        min_samp: Integer to represent minimum number of white pixels
            required to consider if they are lane line pixels or not.
    """
    
    output_images = []
    measurements = []
    
    # undistort the raw image with camera calibration
    undist = undistort(image, c_mtx, dist)
    
    # source points for the top-view transformation
    points = np.array([
        [545, 457],
        [735, 457],
        [1280, 720],
        [0, 720],
        ])
    
    # get top-view image (warped) and the inverse matrix (invmtx)
    invmtx, warped = birds_eye_transform(undist, points, offsetx=430)
    
    # search lane pixels
    lane_lines, left_lane_inds, right_lane_inds =\
        search_lane(bgr2rgb(warped), n_stepsy=9, n_stepsx=2, std_max=std_max, min_samp=min_samp)
    
    # fit
    res, marked_lane_img, filled_lane_img, curve_radi, pos =\
        fit_lane(lane_lines, left_lane_inds, right_lane_inds)
    
    # unwrap, get back to prev viewpoint and output the final image.
    img_size = image[:,:,0].shape[::-1]
    if res:
        unwarped = cv2.warpPerspective(filled_lane_img, invmtx, img_size)
        result_img = cv2.addWeighted(undist, 1, unwarped, 0.5, 0)
    else:
        result_img = undist
    
    output_images.append(warped)
    output_images.append(lane_lines)
    output_images.append(marked_lane_img)
    output_images.append(filled_lane_img)
    output_images.append(result_img)
    measurements.append(curve_radi)
    measurements.append(pos)
    
    return output_images, measurements