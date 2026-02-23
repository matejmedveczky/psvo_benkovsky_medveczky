from ximea import xiapi
import cv2
### runn this command first echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb  ###

# create instance for first connected camera
cam = xiapi.Camera()

# start communication
# to open specific device, use:
# cam.open_device_by_SN('41305651')
# (open by serial number)
print('Opening first camera...')
cam.open_device()

# settings
cam.set_exposure(50000)
cam.set_param("imgdataformat","XI_RGB32")
cam.set_param("auto_wb",1)

print('Exposure was set to %i us' %cam.get_exposure())

# create instance of Image to store image data and metadata
img = xiapi.Image()

# start data acquisition
print('Starting data acquisition...')
cam.start_acquisition()

# Counter for saved images
saved_count = 0
max_images = 4

print(f'Press SPACE to capture images ({saved_count}/{max_images})')
print('Press Q to quit')

while saved_count < max_images:
    cam.get_image(img)
    image = img.get_image_data_numpy()
    image_display = cv2.resize(image, (240, 240))

    # Display counter on image
    cv2.putText(image_display, f'Captured: {saved_count}/{max_images}', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("test", image_display)
    
    key = cv2.waitKey(1)
    
    if key == ord(' '):  # Spacebar pressed
        filename = f'captured_image_{saved_count + 1}.png'
        cv2.imwrite(filename, image)
        saved_count += 1
        print(f'Image saved as {filename} ({saved_count}/{max_images})')
        
        if saved_count >= max_images:
            print('All 4 images captured!')
            break
    
    elif key == ord('q'):  # Q pressed to quit early
        print('Quit early')
        break

# stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()

# stop communication
cam.close_device()

cv2.destroyAllWindows()
print('Done.')