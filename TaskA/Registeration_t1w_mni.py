import os
import SimpleITK as sitk


def register_images(t1_path, mni_template_path, output_dir):
    # Loading images with SimpleITK
    t1_img = sitk.ReadImage(t1_path)
    mni_img = sitk.ReadImage(mni_template_path)

    # T1 register to MNI
    t1_to_mni_transformed = simple_linear_registration(mni_img, t1_img)

    # Save registered T1 images
    transformed_t1_filename = os.path.join(output_dir, os.path.basename(t1_path))
    sitk.WriteImage(t1_to_mni_transformed, transformed_t1_filename)


def preprocess_image(input_image, reference_image):
    # Resample the image to match the pixel pitch and size of the reference image
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference_image)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(sitk.AffineTransform(input_image.GetDimension()))
    resample.SetOutputSpacing(reference_image.GetSpacing())
    resample.SetSize(reference_image.GetSize())
    resample.SetOutputDirection(reference_image.GetDirection())
    resample.SetOutputOrigin(reference_image.GetOrigin())
    output_image = resample.Execute(input_image)

    return output_image


def simple_linear_registration(fixed_image, moving_image):
    # Setting up the registration framework
    registration_method = sitk.ImageRegistrationMethod()

    # Use Mattes to mutualize information and set the number of histograms
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    # Use gradient descent as an optimizer and adjust the parameters appropriately
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=500,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=20)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # multiresolution strategy
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Setting the initial transformation
    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.AffineTransform(fixed_image.GetDimension()))
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Using linear interpolation
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Perform registration
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Apply transformation to resample moving image to fixed image
    return sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())


def process_directory(t1_dir, mni_template_path, output_dir):
    t1_files = sorted([os.path.join(t1_dir, f) for f in os.listdir(t1_dir) if f.endswith('.nii')])

    for t1_path in t1_files:
        register_images(t1_path, mni_template_path, output_dir)
        print(t1_path, "has finished ")


if __name__ == '__main__':
    # Set path
    t1_dir = '/Users/panboshen/Desktop/MIA_Project2_MRI/t1w'
    mni_template_path = '/Users/panboshen/Desktop/MIA_Project2_MRI/mni_atlas/mni_icbm152_t1_tal_nlin_sym_09c_pow2.nii.gz'
    output_dir = '/Users/panboshen/Desktop/MIA_Project2_MRI/registered'

    # Execute processing
    process_directory(t1_dir, mni_template_path, output_dir)