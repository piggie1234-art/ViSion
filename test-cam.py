import sys
import argparse
import cv2
from jetson_utils import videoSource, videoOutput,cudaImage, cudaToNumpy
import time
from test import RectifyCam
rec_cam = RectifyCam("xxx")
# parse command line
rtsp_url1 = "rtsp://xxx/user=admin&password=&channel=1&stream=0.sdp?real_stream" 
rtsp_url2 = "rtsp://xxx/user=admin&password=&channel=1&stream=0.sdp?real_stream" 
# parser = argparse.ArgumentParser()
# parser.add_argument("input", type=str, default="rtsp://192.168.1.12:554/user=admin&password=&channel=1&stream=0.sdp?real_stream" ,help="URI of the input stream")
# parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")

# try:
#     args = parser.parse_known_args()[0]
# except SystemExit:
#     print("Error: Unrecognized command line argument(s). Please check your input and try again.")
# # create video sources & outputs
input1 = videoSource(rtsp_url1, argv=sys.argv)    # default:  options={'width': 1280, 'height': 720, 'framerate': 30}
input2 = videoSource(rtsp_url2, argv=sys.argv) 
#output = videoOutput(args.output, argv=sys.argv)  # default:  options={'codec': 'h264', 'bitrate': 4000000}

# capture frames until end-of-stream (or the user exits)
idx = 0 
while True:
    idx += 1
    # format can be:   rgb8, rgba8, rgb32f, rgba32f (rgb8 is the default)
    # timeout can be:  -1 for infinite timeout (blocking), 0 to return immediately, >0 in milliseconds (default is 1000ms)
    time.sleep(0.08)
    image1 = input1.Capture(format='rgb8', timeout=1000)
    image2 = input2.Capture(format='rgb8', timeout=1000)
   
	
    if image1 is None:  # if a timeout occurred
        continue
    array1 = cudaToNumpy(image1)
    array2 = cudaToNumpy(image2)
    array1 = rec_cam.remap(array1,0)
    array2 = rec_cam.remap(array2,0)
    cv2.imwrite(f"left_{idx}.jpg",array1)
    cv2.imwrite(f"right_{idx}.jpg",array2)
    # img = cv2.resize(array,(640,480))
    # cv2.imshow("11",img[:,:,::-1])
    # cv2.waitKey(1)
		
    #output.Render(image)

    # exit on input/output EOS
    # if not input.IsStreaming() or not output.IsStreaming():
    #     break