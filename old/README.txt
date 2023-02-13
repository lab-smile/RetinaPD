
Software
	Python (Jupyter Notebook)
	Matlab

Directory Setup

Initial Master Folder
	Green
		(Empty)				
	Green_Vessel
		(Empty)			
	Normal		
		Original Images 			
	Normal_Vessel
		(Empty)
	Occulsion
		Green
			Images for Occulsion Maps
		Normal
			Images for Occulsion Maps
		Occulsion Creation.ipynb
		Occulsion Finalization.ipynb
		Occulsion Prediction.ipynb
		Scale and Save Image.ipynb
		Vessel Location.m
	Pickles
		(Empty)
	Grey.m
	Sqaurer.m
	SVM_Final.ipynb
	UFPD_deidentified_label_maker.m
	Label.csv				This file is preproduced or produced with UFPD_deidentified_label_maker
	README.txt

Final Master Folder
	Green
		Grey Scaled Images				
	Green_Vessel
		Vessel Maps of Grey Scaled Images			
	Normal		
		Cropped Original Images 			
	Normal_Vessel
		Vessel Maps of Cropped Original Images
	Occulsion
		Green
			Images for Occulsion Maps
			XXX.png
			XXXsmall.png
			XXXindex.xlsx
			XXXVesselOcculuded.npy
			XXXGreenLinearPDProb.npy
			
		Normal
			Images for Occulsion Maps
			XXX.png
			XXXsmall.png
			XXXindex.xlsx
			XXXVesselOcculuded.npy
			XXXGreenLinearPDProb.npy
		Occulsion Creation.ipynb
		Occulsion Finalization.ipynb
		Occulsion Prediction.ipynb
		Scale and Save Image.ipynb
		Vessel Location.m
	Pickles
		GreenLinear.pkl
		GreenRBF.pkl
		GreenSigmoid.pkl
		NormalLinear.pkl
		NormalRBF.pkl
		NormalSigmoid.pkl
	Grey.m
	Sqaurer.m
	SVM_Final.ipynb
	UFPD_deidentified_label_maker.m
	Label.csv				This file is preproduced or produced with UFPD_deidentified_label_maker
	README.txt

Overall Process For Developing Networks
1. Place all original images into the '/Master Folder/Normal' directory if not already done so
2. Label Generation
	This is achieved using either UFPD_deidentified_label_maker.m or preproduced if using UKB
	
	Run:UFPD_deidentified_label_maker.m
	
	Outputs: Label.csv to the Master Folder
3.Cropping
	This is done to remove excess black from around the images and square the images for the SVM network.
	
	Run: Sqaurer.m

	Warning: This step will overwrite the images.

4. Copy images from '/Master Folder/Normal' to '/Master Folder/Green'
5. Grey Scale images
	Run: Grey.m
	
	Outputs: Greyscaled images found in '/Master Folder/Green'

	Warning: This step will overwrite the images.

6. Generate Vessel Maps using the UNET Network developed by 2019 Summer Exchange students see JQ for documentation
	Copy Vessel Maps into '/Master Folder/Normal_Vessel' for maps of images in '/Master Folder/Normal'
	Copy Vessel Maps into '/Master Folder/Green_Vessel' for maps of images in '/Master Folder/Green'

7. Develop SVM Networks
	Run: SVM_Final.ipynb (See Notes below)
	
	Outputs: Pickles of trained networks in /Master Folder/Pickles'
		 Statistics on network performance (Optional)
		 Weight maps for Linear SVM (Optional)

Overall Process for Occulsion Maps
1. Run all steps from 'Overall Process For Developing Networks'
2. Copy desired images from '/Master Folder/Normal_Vessel'to '/Master Folder/Occulsion/Normal' and from '/Master Folder/Green_Vessel'to '/Master Folder/Occulsion/Green'
3. Scale Images to enable Occulsion Code
	Run: Scale and Save Image.ipynb
	Output: Scaled image located in same directory named ###small.png 
4. Obtain pixel location of vessels
	Run: Vessel_Location.m
	Output: ###index.xlsx
5. Create Occulsion Images for SVM Predictions
	Run: Occulsion Creation.ipynb
	Output: ###VesselOcculuded.npy
6. Obtain Prediction Probabilities of Occulsion Images
	Run: Occulsion Prediction.ipynb
	Output: "Image number" + "Image Type" +"Kernel"+"PDProb".npy
7. Create Occulsion Based Weight Map
	Run: Occulsion Finalization.ipynb
	Output: Weightmap of network performance on the image



README Instructions for Codes

UFPD_deidentified_label_maker

	Inputs:
		UF Deidentified images	

	Lines requiring changing:
		None
	Post Processing:
		Copy and past variable labels into a csv file with row one as 'Label'. Save file as Label.csv
	Output:
		Label.csv: the truths associated with each image
		Numerically labeled images

Sqaurer.m
	Setup
		A directory with all the images of the original images to be cropped. Note: images must be labeled numerically starting at 1
	

	Inputs:
		images: the total number of images
		directory name	
		

	Lines requiring changing:
		Line 10: Change to name of directory
		Line 13: Change to total number of images
		Lines 20 to 25: Change based on if cropping UKB or Remidio images
		Line 17/29: Change if not using jpeg
	Output:
		Cropped jpeg or pngs of inputed images

Grey.m

	Setup
		A directory named Green with all the cropped images of the original rgb images. Note: images must be labeled numerically starting at 1
	

	Inputs:
		images: the total number of images	
		

	Lines requiring changing:
		Liens 17 and 25: Change file type if not using jpg
	
	Output:
		Greyscaled jpeg or pngs of inputed images

SVM_Final

	Inputs:
		Label.csv: CSV file of labels with 1 for PD and 0 for CN. First cell should be labeled 'Label'
		Preprocessed vessel maps in png or jpeg form

	Lines requiring changing:
		Cell 3 Line 2: Change root to be name of desired directory where vessel maps are located
		Cell 3 Line 7: Change '.png' to '.jpg' if using jpegs
		Cell 7: Change to match number ranges for your specific case
		Cell 8 Line 4: Change kernel to match desired SVM kernel
		Cell 9 Line 3: Change kernel to match desired SVM kernel
		Cell 10 Line 2: Change path to location of Pickle folder
		Cell 10 Line 7: Change name to match desired name of pickle
		Cell 12 Line 4: Change name to pickle being used
		Cell 15 Line 25: Change name to desired name
		Cells 17/18/19 Line 1/3/4: Change to desired color scheme, names, and saving
	Output:
		Trained network pickle (Cell 10)
		Model Accuracy (Cell 13)
		Confusion Matrix (Cell 14)
		Sensitivity (Cell 14)
		Specificity (Cell 14)
		AUC Curve   (Cell 15)
		
		Linear Kernel Weight Maps (Cell 17 to 19)
		

Scale and Save Image
	
	Inputs:
		Preprocessed vessel map in .png or .jpeg

	Lines requiring changing:
		Cell 2 Line 1: Change Root to match destination directory 'Green' or 'Normal'
		Cell 2 Line 2: Change file name to desired image
		Cell 3 Line 1: Change path to that of the 'Green' or 'Normal' folder
		Cell 3 Line 1: Change name to 'image number'+ 'small.png' ie: 407small.png or 390small.png
	Output:
		###small.png or ###small.jpeg
			Example: 390small.png

Vessel_Location.m
	Inputs:
		###small.png or ###small.jpeg: Output image from "Scale and Save Image.ipynb"
	
	Lines requiring changing:
		Line 8: Change to directory of images "Green" or "Normal"
		Line 10: Change number to match desired image number
		Line 30: Change number to match desired image number
	Output:
		###index.xlsx: Pixel location of each part of the vessel

Occulsion Creation

	Inputs:
		Preprocessed vessel map in .png or .jpeg
		###index.xlsx: Output from 'Vessel_Location.m'

	Lines requiring changing:
		Cell 2 Line 1: Change Root to match destination directory
		Cell 2 Line 2: Change file name to desired image
		Cell 3 Line 1: Change to file directory for XXXindex.xlsx
		Cell 8 Line 1: Change path name to path for either 'Green' or 'Normal'
		Cell 8 Line 2: Change save name to desired name
	Output:
		###VesselOcculuded.npy:

Occulsion Prediction
	Inputs:
		###VesselOcculuded.npy: Output from'Occulsion Creation.ipynb'
		

	Lines requiring changing:
		
		Cell 4 Line 7: Change to path name for either 'Green' or 'Normal' or the path location of ###VesselOcculuded.npy
		Cell 5 line 2: Change index of A to be from 0 to A.shape[0]/2
		Cell 6 Line 2: Change index of A be from A.shape[0]/2 to A.shape[0]+1
		Cell 7 Line 2: Change path to location of pickle folder
		Cell 8 Line 4: Change to name of desired pickle
		Cell 11 Line 2: Change name to "Image number" + "Image Type" +"Kernel"+"PDProb"
	Output:
		"Image number" + "Image Type" +"Kernel"+"PDProb".npy : Probabilties associated with each occulsion image input to "Occulsion Image Finalization.ipynb"


Occulsion Image Finalization
	Inputs:
		"Image number" + "Image Type" +"Kernel"+"PDProb".npy: Output from 'Occulsion Prediction.ipynb'
		###index.xlsx: Output from 'Vessel_Location.m'

	Lines requiring changing:
		Cell 2 Line 46: Change to path of Occulsion Folder
		Cell 3 Line 1: Change to desired image number
		Cell 3 Line 2: Change to desired kernel
		Cell 3 Line 3: Change to desired image type
		Cell 3 Line 4: Change to desired heat map
		Cell 3 Line 6: Change path to location of Occulsion Folder
	Output:
		Occulsion generated heat map


