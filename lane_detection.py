import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def color_selection(image, red_threshold, green_threshold, blue_threshold):
    color_select = np.copy(image)
    
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    thresholds = (image[:,:,0] < rgb_threshold[0]) | (image[:,:,1] < rgb_threshold[1]) | (image[:,:,2] < rgb_threshold[2])
    color_select[thresholds] = [0, 0, 0]

    return color_select

def region_masking(image, left_bottom, right_bottom, apex, rgb_threshold):
    color_select = np.copy(image)
    line_image = np.copy(image)

    # Define the vertices of a triangular mask.
    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

    color_thresholds = (image[:,:,0] < rgb_threshold[0]) | (image[:,:,1] < rgb_threshold[1]) | (image[:,:,2] < rgb_threshold[2])

    XX, YY = np.meshgrid(np.arange(0, image.shape[1]), np.arange(0, image.shape[0]))
    region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & (YY > (XX*fit_right[0] + fit_right[1])) & (YY < (XX*fit_bottom[0] + fit_bottom[1]))

    color_select[color_thresholds | ~region_thresholds] = [0, 0, 0]
    line_image[~color_thresholds & region_thresholds] = [0, 255, 0]

    return color_select, line_image

def visualize_results(image, left_bottom, right_bottom, apex, color_select, line_image):
    plt.imshow(image)
    plt.title("Input Image")
    plt.show()

    plt.imshow(image)
    x = [left_bottom[0], right_bottom[0], apex[0], left_bottom[0]]
    y = [left_bottom[1], right_bottom[1], apex[1], left_bottom[1]]
    plt.plot(x, y, 'r--', lw=4)
    plt.title("Region Of Interest")
    plt.show()

    plt.imshow(color_select)
    plt.title("Color Selection")
    plt.show()

    plt.imshow(line_image)
    plt.title("Lane Lines Detected")
    plt.show()

def lane_detection_pipeline(image_path, red_threshold, green_threshold, blue_threshold, left_bottom, right_bottom, apex):
    image = mpimg.imread(image_path)

    color_select = color_selection(image, red_threshold, green_threshold, blue_threshold)

    rgb_threshold = [red_threshold, green_threshold, blue_threshold]
    color_select, line_image = region_masking(image, left_bottom, right_bottom, apex, rgb_threshold)

    visualize_results(image, left_bottom, right_bottom, apex, color_select, line_image)

if __name__ == "__main__":
    image_path = 'test_images/solidWhiteRight.jpg'
    red_threshold = 200
    green_threshold = 200
    blue_threshold = 200
    left_bottom = [100, 539]
    right_bottom = [950, 539]
    apex = [480, 290]

    lane_detection_pipeline(image_path, red_threshold, green_threshold, blue_threshold, left_bottom, right_bottom, apex)
