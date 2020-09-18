import config
import cv2

FIXED_HEIGHT = 640
# The Mean values for the ImageNet training set are R=103.93, G=116.77, B=123.68
MEAN_R = 103.93
MEAN_G = 116.77
MEAN_B = 123.68


def inference(model, image):
    model_name = f"{config.MODEL_PATH}{model}.t7"
    model = cv2.dnn.readNetFromTorch(model_name)

    height, width = int(image.shape[0]), int(image.shape[1])
    new_width = int((FIXED_HEIGHT / height) * width)
    resized_image = cv2.resize(image, (new_width, FIXED_HEIGHT), interpolation=cv2.INTER_AREA)

    # Create our blob from the image, then perform a forward pass run of the network

    inp_blob = cv2.dnn.blobFromImage(
        resized_image,
        1.0,
        (new_width, FIXED_HEIGHT),
        (MEAN_R, MEAN_G, MEAN_B),
        swapRB=False,
        crop=False,
    )

    model.setInput(inp_blob)
    output = model.forward()

    # Reshape the output Tensor, add back the mean subtraction, re-order the channels
    output = output.reshape(3, output.shape[2], output.shape[3])
    output[0] += MEAN_R
    output[1] += MEAN_G
    output[2] += MEAN_B

    output = output.transpose(1, 2, 0)
    return output, resized_image
