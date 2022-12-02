import gxipy as gx
import cv2
import os, shutil
import time

binning = 2





gain = 15.0


device_manager = gx.DeviceManager()
sn1 = 'FCS22070911'
sn2 = 'FCS22070914'
dev_num, dev_info_list = device_manager.update_device_list()
print(f"Number of enumerated devices is {dev_num}")

# Open camera by serialnumber
cam1 = device_manager.open_device_by_sn(sn1)

# binning
cam1.BinningHorizontal.set(binning)
cam1.BinningVertical.set(binning)
cam1.size = (cam1.Width.get(), cam1.Height.get())

# Get frame rate
frame_rate = cam1.CurrentAcquisitionFrameRate.get()

# Set continuous acquisition
cam1.TriggerMode.set(gx.GxSwitchEntry.OFF)

# Set exposure time
cam1.ExposureTime.set(10000.0)

# Set gain (Autogain of niet nodig???)
cam1.Gain.set(gain)

# Set acquisition buffer count
cam1.data_stream[0].set_acquisition_buffer_number(1)

# Start data acquisition
cam1.stream_on()

print(f"Camera {sn1} is initialized")

# Open camera by serialnumber
cam2 = device_manager.open_device_by_sn(sn2)

# binning
cam2.BinningHorizontal.set(binning)
cam2.BinningVertical.set(binning)
cam2.size = (cam2.Width.get(), cam2.Height.get())

# Get frame rate
frame_rate = cam2.CurrentAcquisitionFrameRate.get()

# Set continuous acquisition
cam2.TriggerMode.set(gx.GxSwitchEntry.OFF)

# Set exposure time
cam2.ExposureTime.set(10000.0)

# Set gain (Autogain of niet nodig???)
cam2.Gain.set(gain)

# Set acquisition buffer count
cam2.data_stream[0].set_acquisition_buffer_number(1)

# Start data acquisition
cam2.stream_on()

print(f"Camera {sn2} is initialized")

num = 0


while True:
    img1 = cam1.data_stream[0].get_image()
    img1 = img1.get_numpy_array()
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)


    k = cv2.waitKey(5)

    if k == ord('q'):
        break
    if k == ord('s'):
        if num == 0:
            folder = '/home/arthur/PycharmProjects/python_stereo_camera_calibrate/images/stereoLeft'
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

        cv2.imwrite('images/stereoLeft/imageL' + str(num) + '.png', img1)
        num += 1

    if binning == 1:      
        img1 = cv2.resize(img1, (1920,1080))

    cv2.imshow(sn1, img1)

cv2.destroyAllWindows()

num = 0
while True:
    img2 = cam2.data_stream[0].get_image()
    img2 = img2.get_numpy_array()
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    k = cv2.waitKey(5)

    if k == ord('q'):
        break
    if k == ord('s'):
        if num == 0:
            folder = '/home/arthur/PycharmProjects/python_stereo_camera_calibrate/images/stereoRight'
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

        cv2.imwrite('images/stereoRight/imageR' + str(num) + '.png', img2)
        num += 1

    if binning == 1:      
        img2 = cv2.resize(img2, (1920,1080))

    cv2.imshow(sn2, img2)

cv2.destroyAllWindows()

num = 0
while True:
    img1 = cam1.data_stream[0].get_image()
    import time
    img1 = img1.get_numpy_array()
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    img2 = cam2.data_stream[0].get_image()
    img2 = img2.get_numpy_array()
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    k = cv2.waitKey(5)

    if k == ord('q'):
        break
    if k == ord('s'):
        if num == 0:
            folder = '/home/arthur/PycharmProjects/python_stereo_camera_calibrate/images/screen/stereoLeft'
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

            folder = '/home/arthur/PycharmProjects/python_stereo_camera_calibrate/images/screen/stereoRight'
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

        cv2.imwrite('images/screen/stereoLeft/imageL' + str(num) + '.png', img1)
        cv2.imwrite('images/screen/stereoRight/imageR' + str(num) + '.png', img2)
        num += 1

    if binning == 1:      
        img1 = cv2.resize(img1, (1920,1080))
        img2 = cv2.resize(img2, (1920,1080))

    cv2.imshow(sn1, img1)
    cv2.imshow(sn2, img2)

cv2.destroyAllWindows()