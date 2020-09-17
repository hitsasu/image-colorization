# image-colorization
Neural Network based automatic image colorization

***INTRODUCTION***

Image Colorization is an exciting active area of research which enables us to see moments and people in history that have never been seen in color. The aim of this project is to train neural networks to automatically colorize gray-scale images without manual effort. We change color channels from RGB to L-ab. The network predict ”a” and ”b” color spectrum channels using ”L” light channel as input. TensorFlow framework is used for developing the neural network.


***IMPLEMENTATION***

The implementation can be summarized as follows:
1) Input Image Resizing: Each image is resized to 2242243 so that it can be fed to the encoder layer.
2) Encoder: or feature extraction
3) Decoder: Color histogram and cost function calculation and colorization.
4) Output Image scaling: Upsampling the resulting RGB image.

*****The framework shown in Figure/framework.png illustrates the entire process of coloring a grayscale image to a LAB color image.***


***RESULTS***

We trained the model using appropriately 2500 images (classified as either face, animal or landscape images) of combined sets and tested it with another separate small sample of unseen images. Our subjective results shows that the coloring approach performed reasonably well when compared to the ground truth. The results are shown in ***Figure/results.png***


***CONCLUSIONS AND RECOMMENDATIONS***

In this work, we successfully implemented automatic image colorization using CNNs. The performance of the model can be further increased by training on larger and diverse dataset. Furthermore, we trained the model using nearly 2500 images classified as either face, animal or landscape images. We trained, validated, and tested with each individually, and all combined together to compare and contrast the model performance and quality under different scenarios. Our main findings is that training on related images, i.e., landscape images with certain feature, help in determining the right amount of colors needed to color new grayscale images underthe landscape classification.
As future works, we can increase testing time and include more diverse images that can be used to learn more features. Furthermore, an interesting future direction could be to use custom tailored encoder that can extract features targeting specific classes of objects and color spectrum.

