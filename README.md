# GrowCuts
C++/CUDA implementation of GrowCuts to find the boundary of an object

an example of using a CPU/GPU version of grow cuts to refine an initial segmentation mask of metal in a projections image.   Grow cuts is a segmentation algorithm which uses seed points to iteratively generate a segmentation. Mask0_UINT8_1076x884x10 contains the initial blurry mask image that extend pass the metal boundaries. Image_UINT16_1076x884x10 contains the projection image. These files contain a stack of 10 images.  The mask0 image is used to generate seed points which are then used by growcuts along with the projection image to refine the of the metal. Mask_UINT8_1076x884x10 contains the refined boundary metal masks. There are two versions of the GPU growcuts.  The d_growcuts_Checker() version updates the mask in place.  Each iteration is split into two passes. The first pass updates the even pixels of the mask and the second pass updates the odd pixels of the mask.

# Reference
V. Vezhnevets, V. Konouchine. "Grow-Cut" - Interactive Multi-Label N-D Image Segmentation".
In Proceedings of the 2005 Conference, Graphicon. Pages 150 – 156.
