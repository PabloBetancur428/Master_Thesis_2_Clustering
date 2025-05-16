##### Pseudo-code brain GRE
#### List of input images
MULTI_ECHO_MAG_FA4=”….” # Path of a 4D NIFTI file storing magnitude images of the multi-echo T2*-weighted acquisition for flip angle 4deg
MULTI_ECHO_MAG_FA15=”….” # Path of a 4D NIFTI file storing magnitude images of the multi-echo T2*-weighted acquisition for flip angle 15deg
MULTI_ECHO_MAG_FA25=”….” # Path of a 4D NIFTI file storing magnitude images of the multi-echo T2*-weighted acquisition for flip angle 25deg
VFA=”…” # Path of a 4D NIFTI file storing magnitude images of the variable flip angle T1- weighted experiments
B1map=”…” # Path of the B1 map --- We need to contact Marta Vidorreta again to mak sure how to calculate it correctly
MTRon=”….” # Path of MTR image with MT pulse on
MTRoff=”….” # Path of MTR image with MT pulse off
QSM_MAG=”…” # Path of the magnitude images of the qSM scan

### Pipeline
fslmerge -t ALL_MULTIECHO $MULTI_ECHO_MAG_FA4 $MULTI_ECHO_MAG_FA15
$MULTI_ECHO_MAG_FA25 # Merge all the multi-echo images at varying flip angles together
dwidenoise $ALL_MULTIECHO $ALL_MULTIECHO_DEN --noise $MULTIECHO_NOISE --extent
“3,3,3” # MPPCA enoising with kernel 3x3x3
unring $ALL_MULTIECHO_DEN $ALL_MULTIECHO_DEN_UNRING # Gibbs unringing
# Calculate mean T2*w echo from the qSM scan – it will be our reference for motion
correction
fslmaths $QSM_MAG -Tmean $REF_IMAGE

# Extract the images required for T2* and T1 fitting
fslroi $ALL_MULTIECHO_DEN_UNRING $T2star_FIT_SCAN 7 7 # Extract volumes 7th to 14th:
T2* fitting
fslroi $ALL_MULTIECHO_DEN_UNRING $VFA_IMG1 0 1 # Extract the first echo of the 3 scans
fslroi $ALL_MULTIECHO_DEN_UNRING $VFA_IMG1 7 1 # Extract the second echo of the 3
scans
fslroi $ALL_MULTIECHO_DEN_UNRING $VFA_IMG1 14 1 # Extract the third echo of the 3
scans

# Co-register all the volumes contained in T2star_FIT_SCAN to the reference image.
Estimate the registration transformation only once. Then fit T2*
fslmaths $T2star_FIT_SCAN -Tmean $T2star_FIT_SCAN_TMEAN
reg_aladin -rigidonly -ref $REF_IMAGE -flo $T2star_FIT_SCAN_TMEAN -res
$T2star_FIT_SCAN_TMEAN_REG -transform $T2star_FIT_TO_REF
reg_resample -ref $REF_IMAGE -flo $T2star_FIT_SCAN -res $T2star_FIT_SCAN_MOCO -
transform $T2star_FIT_TO_REF
python getT2T2star.py $ T2star_FIT_SCAN_MOCO $TE_FILE $OUTPUT_T2star_MAP
T2star_map=$OUTPUT_T2star_MAP”_TxyME.nii” # One of the output NIFTIs is the actual
T2* map

# Co-register the three images at different flip angle to the reference
reg_aladin -rigidonly -ref $REF_IMAGE -flo $ VFA_IMG1 -res $VFA_IMG1_REG -transform
$VFA_IMG1_TO_REF
reg_aladin -rigidonly -ref $REF_IMAGE -flo $ VFA_IMG1 -res $VFA_IMG2_REG -transform
$VFA_IMG2_TO_REF
reg_aladin -rigidonly -ref $REF_IMAGE -flo $ VFA_IMG1 -res $VFA_IMG3_REG -transform
$VFA_IMG3_TO_REF
fslmerge -t $VFA_PREPROC_MOCO $VFA_IMG1_REG $VFA_IMG2_REG $VFA_IMG3_REG
# Resample the B1 map to the preprocessed variable flip angle data
reg_resample -ref $VFA_PREPROC_MOCO -flo $B1map -res $B1mapres
# Calculate T1 map using the B1 map to correct actual vs nominal flip angle
python getT1VFA.py $VFA_PREPROC_MOCO $OUPUT_T1_map $FLIP_ANGLE_FILE --b1
$B1mapres
S0_FROM_VFA_FITTING=$OUPUT_T1_map”_S0VFA.nii” # One of the outputs is S0
(apparent proton density)
T1_FROM_VFA_FITTING=$OUPUT_T1_map”_T1VFA.nii” # One of the outputs is the T1 map
# Unring the MTRon and MTRoff
unring $MTRon $MTRon_unring
unring $MTRoff $MTRoff_unring
# Co-register MTRon and MTRoff to the reference image
reg_aladin -rigidonly -ref $REF_IMAGE -flo $$MTRon_unring -res $MTRon_unring_MOCO
reg_aladin -rigidonly -ref $REF_IMAGE -flo $$MTRoff_unring -res $MTRoff_unring_MOCO
# Calculate MTR
python getMTR.py $MTRon_unring_MOCO $MTRoff_unring_MOCO $MTRmap
# Calculate MTV
TISSUEMASK=”…..” # NIFTI containing the tissue mask. It contains CSF, normal-appearing
grey and white matter. In patients, ALL lesions should be removed and should not be
included in this mask

WATERMASK=”…” # NIFTI containing a “pure water” region-of-interest, as for example the
brain ventricles
TEfileVFA=”…” # Path storing a text file with the TE of the variable flip angle acquisition
OutMTV=”…” # Path of the base string for all output files
python getMTV.py $S0_FROM_VFA_FITTING $T1_FROM_VFA_FITTING $T2star_map
$TISSUEMASK $WATERMASK $TEfileVFA $OutMTV