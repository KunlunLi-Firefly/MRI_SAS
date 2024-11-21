TaskA Registration
This code, named Registration_t1w_mni.py, is designed to perform linear registration of T1w MRI images to the MNI standard brain template. The purpose is to align individual T1w images with the standardized MNI template. After running the script, the registered images will be saved in the specified output directory.

This code, named Registration_t2w_t1w.py, performs MRI image alignment, which is capable of aligning a T2w image with a registered T1w image. The script will process each pair of T1w and T2w images, aligning the T2w images to the corresponding registered T1w images. The registered images will be saved in the specified output directory.


TaskB Segmentation of Ventricle and EvansRatio Calculation
I have tried to construct the enviroment for the Fastsurfer on the WSL locally so there is no code for the segmentation part,the setup method can be seen in the(https://github.com/Deep-MI/FastSurfer/blob/dev/doc/overview/INSTALL.md#windows). After use fastsurfer segment the ventricle part, we can use the command in the fastsurfer to extract the brain mask and ventricle mask for the further calculation. Then you can use the TaskB_EvansRatio.ipynb in the TaskB folder to implement basic transition for the mask (like flipping) to make the ventricle align with the brain mask. Then you can use the rest of code to calculate the ratio of the maximum width of the frontal horns of the lateral ventricles and the maximal internal diameter of the skull at the same level(which is EvansRatio). After getting the result, you can use the last part of code to save all the information into the .txt file.


TaskC
T1w To T2w Workflow
This section outlines the workflow for processing T1-weighted (t1w) MRI images using a series of image manipulation and deep learning techniques to transform these images into T2-weighted (t2w) MRI images. The workflow is divided into several key steps:

1. Preprocessing of NIfTI Files
NIfTI to PNG Conversion: Initially, NIfTI files (.nii) are segmented into individual slices and saved as PNG files. This process is handled using the mri_image_process code(https://colab.research.google.com/drive/1WlL-Kbca4c8HumHk0jqM4vcpu4t4OBm0?usp=sharing). Alongside the PNG images, the corresponding affine matrices are stored to facilitate the accurate reassembly of the slices back into the original 3D format after processing.
2. Image Manipulation
Rotation and Cropping: Depending on the specific requirements and characteristics of the MRI slices, the mri_image_process code is used to perform rotations and cropping. This step ensures that the slices are in the desired orientation and size for further processing. The manipulated images are then saved in a new folder.
3. Model Prediction
Generating t2w Slices: The slices processed in the previous step are input into a pre-trained model to generate t2w images. This model, loaded with pre-trained weights(https://drive.google.com/drive/folders/1TXyuC1aQzX1PTFhuK-K6Bjwzdvjp_Nq5?usp=sharing), predicts t2w slices using the mri_test script(https://colab.research.google.com/drive/1_PpsmnQ2ydLYrS4x407tTJtxZmE6V7ws?usp=sharing). The generated t2w slices are saved in a dedicated folder for each subject.
4. Post-Processing of Generated Slices
Reorientation and Resizing: Using the mri_image_process code again, the generated t2w slices are processed to match the original size and orientation of the initial PNG slices, ensuring that the transformation preserves spatial consistency.
5. Reassembly of 3D Volumes
Combining Slices into 3D NIfTI: The final step involves reassembling the t2w slices back into a 3D NIfTI format using the saved affine matrices. This reassembly ensures that the new t2w images maintain the correct anatomical structure and orientation.
Training Code
The model training code, named "t1 to t2"(https://colab.research.google.com/drive/1rb3bI05z2asR56vjoRo6C7HBYqPdfiJt?usp=sharing), was developed based on insights and methodologies adapted from two projects focusing on the application of CycleGANs in brain MRI analysis.
## References

This part was inspired by and adapted from the following projects:

- Rathi, Raj. "Image to Image Translation for Brain MRI." Accessed on [date]. GitHub repository. [image2imageBrainMRI](https://github.com/rajrathi/image2imageBrainMRI/blob/main/.ipynb_checkpoints/model-checkpoint.ipynb).

- Hitha. "MRI Style Transfer Using CycleGAN." Accessed on [date]. GitHub repository. [MRI-styletransfer-CycleGAN](https://github.com/Hitha83/MRI-styletransfer-CycleGAN/blob/main/mri_gan.ipynb).

SAS Volume Calculation
This part performs the calculation of the Subarachnoid Space (SAS) volume from MRI images. It uses a combination of image preprocessing, K-means clustering, region growing, and morphological operations to segment the SAS region and calculate its volume.

Algorithm overview:
The algorithm for SAS volume calculation consists of the following steps:
1. Image Preprocessing:
Gaussian denoising is applied to reduce noise in the image data.
Normalization is performed to standardize the intensity values.
2. K-means Clustering:
The preprocessed image data is reshaped into a 1D array of pixels.
K-means clustering is applied to group similar pixels together based on their intensity values.
The number of clusters (n_clusters) is specified as a parameter.
3. Cluster Selection:
The clustered data is visualized in 3D, with each cluster assigned a different color.
The user is prompted to select the appropriate cluster label corresponding to the SAS region.
4. Region Growing:
The initial mask is created by selecting the pixels belonging to the chosen cluster label.
Seed points are obtained from the initial mask.
Region growing is performed starting from the seed points.
Pixels are added to the region if their intensity difference from the seed point falls within the specified thresholds (threshold1 and threshold2).
5. Morphological Operations:
The grown region is smoothed and trimmed using erosion to remove small isolated regions.
Holes are filled, and components are connected using dilation and erosion operations.
6. Refined Region Growing:
The processed mask from the previous step is used as the new initial mask.
Region growing is performed again with adjusted thresholds to further refine the segmentation.
7. SAS Volume Calculation:
The segmented SAS region is obtained from the refined mask.
The volume of the SAS region is calculated by multiplying the number of non-zero voxels by the voxel size.

Usage:
1. Run the main function with the desired image paths:
image_paths = ['./RegisteredC/subject_52_t1w.nii', './RegisteredC/subject_53_t1w.nii', './RegisteredC/subject_59_t1w.nii', './RegisteredC/subject_60_t1w.nii', './RegisteredC/subject_61_t1w.nii']
for image_path in image_paths:
    print(f"Processing image: {image_path}")
    main(image_path)
2. The code will process each image and display the clustered data visualization.
3. Enter the desired cluster label for the SAS region when prompted.
4. The code will then perform region growing, morphological operations, and visualization of the segmented SAS region.
5. The SAS volume will be calculated and saved in a text file named <image_name>_sas_volume.txt.

Functions:
preprocess_image(image_data): Preprocesses the image data by applying Gaussian denoising and normalization.
cluster_pixels(image_data, n_clusters): Performs K-means clustering on the image data.
generate_cluster_colors(n_clusters): Generates colors for each cluster label.
visualize_3d(image_data, n_clusters): Visualizes the clustered data in 3D.
region_growing(image_data, seed_points, threshold1, threshold2): Performs region growing on the image data starting from the seed points.
smooth_and_trim_mask(mask, kernel_size=2): Smooths and trims the mask using morphological operations.
fill_holes_and_connect(mask, kernel_size=1): Fills holes and connects components in the mask using morphological operations.
calculate_volume(segmented_region, voxel_size): Calculates the volume of the segmented region.
visualize_sas(preprocessed_data, sas_mask, mask_color=(1.0, 0.0, 0.0), alpha=0.3): Visualizes the segmented SAS region overlaid on the preprocessed image data.



