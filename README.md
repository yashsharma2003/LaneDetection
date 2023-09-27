# LaneDetection
Project on lane detection ussing opencv and huffman codding (Machine Learning)

The general method of lane detection is to first take an image of road with the help of a camera fixed in the vehicle. Then the image is converted to a grayscale image in order to minimize the processing time. Secondly, as presence of noise in the image will hinder the correct edge detection. Therefore, filters should be applied to remove noises like bilateral filter, gabor filter, trilateral filter Then the edge detector is used to produce an edge image by using canny filter with automatic thresholding to obtain the edges. Then edged image is sent to the line detector after detecting the edges which will produces a right and left lane boundary segment. The lane boundary scan uses the information in the edge image detected by the Hough transform to perform the scan. The scan returns a series of points on the right and left side. Finally pair of hyperbolas is fitted to these data points to represent the lane boundaries. For visualization purposes the hyperbolas are displayed on the original color image 
The algorithm undergoes various changes and detection of patterns in the images of roads for detecting the lanes. Some of the images are shown in Figure 3-6. Figure 3 a , shows the input image. Figure 3b represents the filtered image of fig 3a. In Figure 4a, the filtered image is converted to grayscale image for reducing the processing time. Then this image is segmented to binary image 4b. It is done to locate the lanes in captured image.

I have Uploaded the test video for the data set named as test 2 , test 3 , etc

uploaded the result to help you through it named res & res1
