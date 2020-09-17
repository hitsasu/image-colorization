# image-colorization
Neural Network based automatic image colorization

Image Colorization is an exciting active area of research which enables us to see moments and people in history that have never been seen in color. The aim of this project is to train neural networks to automatically colorize gray-scale images without manual effort. We change color channels from RGB to L-ab. The network predict ”a” and ”b” color spectrum channels using ”L” light channel as input. TensorFlow framework is used for developing the neural network.

The implementation can be summarized as follows:
1) Input Image Resizing: Each image is resized to 2242243 so that it can be fed to the encoder layer.
2) Encoder: or feature extraction
3) Decoder: Color histogram and cost function calculation and colorization.
4) Output Image scaling: Upsampling the resulting RGB image.

*****The framework shown in Figure/framework.jpg illustrates the entire process of coloring a grayscale image to a LAB color image.*******
