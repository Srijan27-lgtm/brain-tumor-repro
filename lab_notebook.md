Coverage for (Day 1-3):
Dates: 23/10/2026 to 25/10/2026 (till morning):
Objectives: Process BRISC and Sartaj datasets for NextSwin-X reproduction


Fix-1: BRISC aspect ratio issue:
What I tried: I used img.resize((256,256)) on the BRISC dataset.
What I expected: I expected all BRISC images to be uniform 512x512 and resize cleanly.
What actually happened (Mistake): I checked the manifest.csv and found 659 non-square images with varying widths (216 px to 1365 px). My code would have squashed these images, altering tumor morphology.
The fix and what I decided: I re-read the paper, which explicitly states: "Initially, each image was resized to a uniform dimension of 224x224 pixels." It does not mention aspect ratio preservation, padding, or cropping. To ensure strict reproduction method of the paper methodology, I updated the pipeline to directly force-resize all images to (224,224) using BICUBIC interpolation method. I accept that this squashes the 659 non-square images, as this is the most faithful interpretation of the authors' exact words without adding my own unverified assumptions. (I assumed that following the paper would be most accurate although I experimented with resize to 256x256 and then CenterCrop to 224224 but in the end I followed the paper quotes they stated and did this. Maybe after we reproduce this table we can do a better job at this by handling the squashing by some better method yet to be known. I think we can use this in stress testing or something like that doing a better thing (I am just assuming).)


Fix-2: Sartaj data leakage and duplicates
What I tried: I assumed the official Kaggle Train/Test split for Sartaj was perfectly cleaned and used it directly.
What I expected: I expected no overlap between the Training and Test sets.
What actually happened (Mistake): I ran SHA256 hash checks on the raw folders and found 127 images that existed in both the Training and Test sets along with internal duplicates too.
Fix and decision: I re-read the NextSwin paper. It explicitly states the Sartaj dataset consists of exactly ("3264 MRI images") and does not mention any data cleaning. To ensure the faithful reproduction thing for paper's Table-3 I retained the official, unmodified Kaggle split and documented this transparently. (Maybe just my thinking after we reproduce this table can we do better job than them by handling these things like allowing duplicates and leakage in dataset when we want to do something better (Just my assumptions).)


Fix-3: Label mapping crash and integrity:
What I tried: I uploaded to Hugging Face with string labels and a dictionary looking for 'no_tumor'.
What I expected: I expected a smooth upload with correct labels.
What actually happened (Mistake): The upload crashed with an 'ArrowInvalid' error. My string cleaning turned the folder 'no_tumor' into just 'no' but my dictionary looked for 'no_tumor'. The script skipped images but kept counting them, breaking array lengths.
The fix: I updated the dictionary to {"no": 2}, added a guard clause to skip unknown labels before processing and added 'failed_count' tracking with an assert to guarantee zero silent failures and saved the mapping to 'label_mapping.json' (uploaded to both HF repos).



Fix 4: Paper Preprocessing & Augmentation 
What I Tried: i used standard 'timm' defaults (Resize(256) then Centercrop(224)).
what I Expected: I expected standard preprocessing to match the paper's baseline.
what actually Happened: (Mistake): I re read NextSwin-X paper. The author explicitly state: "Initially , each image was resized to a uniform dimension of 224x224 pixels." They do not mention an intermediate 256 resize or CenterCrop.
The Fix: I changed the pipeline to **directly resize all images to 224x224** using BICUBIC interpolation 
My Assumptions: Because I resized images  to 224x224,  the paper's "on the fly" training augmentation (like rotation/flip) will now apply to the 224x224 images rather than the original images.
(i want to here maybe we can do something about this after reproduction we can make it better if we want by applying better augmentation techniques rather than on the fly approach but that is the story after the reproduction is done as good as possible)

Note: i have added those last line ideas in every fix because i assumed may be we can fix that after reproduction when we want to improve things (Again just My assumptions)


coverage for (26-30 October 2026):
For scratch CNN Parameter Verification 
what i tried:  I performed hand calculations for every layer's parameter count to verify the ScratchCNN architecture and ensure no mismatch before training.
what i expected: The hand-calculated total to perfectly match the torchinfo verification script output.
fix: Fixes: I realized I made a small mistake in the convolution part of Block 4 by not adding the bias (output channel) in the formula. I corrected the hand calculation to include the bias, which perfectly matched the 98,436 total.
what happened: My hand calculation resulted in 98,308 parameters, which was a shortage of 128 parameters compared to the torchinfo result of 98,436.
Some issues and  notes 
The bias issue: Missing the bias addition leads to a lower parameter count (as I experienced in step 1).
Flatten vs. Global Average Pool (GAP): If I had used flatten() simply on the 128x14x14 tensor, it would result in 25,088 input features. This would inflate the total final linear layer parameter count to over 1 lakh parameters, which would cause overfitting. GAP reduced this to exactly 128 features, keeping the total parameter count to 98,436 only.
BatchNorm issue: If I had not accounted for the 2 learnable parameters (scale and shift) in BatchNorm, it would result in missing parameters in the final count.
Note:
i performed hand calculations for understanding back propogation  algorithm did some small example and then calculated for a small conv layer where i did some mistakes like not calculating the gradient derivative properly and not writing the chain rule properly through more tries i calculated the output then calculated the error for 4 inputs that is where another error  occured as i calculated the error 9 times instead of 4 times because i assumed 9 uotput pixels were there to calculate the loss but upon trying again i found that the actual output is of 2by2 matrix and the 9 output pixel were just dot product of the filter going over the input matrix so i was made a mistake here and then upon revising the example i calculated the gradients individually and thr product of them was 18 and it was calculated for only w11 weight from the filter matrix as the filter hover overs the input matrix w11 comes in use 4 times giving total gradient for that run to be 72 and to add some more info the input and filter matrix used 0s and 1s so each gradient we choose mush give total gradient as 72 for that run only  for simplification and  hence hand calculation part was done i also verified it using the same tensor(same input) on pytorch through a script and it matched correctly 


coverage for (30-31 October 2026):
what i tried: i tried to run test on 100 images for sanity check of the data pipeline and training loop  
what i expected : i expect it run smoothly and give 100% accuracy(overfitting condition as data amount was too small)
what happened: it gave multiple errors then i t ran smoothly and gave 99% accuracy with error 0.189
fixes: 
fix1:PIL Crash: Realized Hugging Face stores raw pixel lists instead of PIL images, so I switched to direct tensor conversion.
fix2: Reshaped the flat pixel lists into proper 3D image grids so the model could actually read them.
fix3: Fixed a crash by explicitly casting the text-string labels into actual integers before making tensors.
fix4: Mapped the actual text labels (like "glioma") to numbers and turned off    Kaggle's background workers to stop silent crashes.
fix5:  Flipped the image dimensions to put the color channels first and divided pixels by 255 to stabilize the math.
Sanity Check: Ran the 100-image test and hit 99% accuracy, proving the whole data pipeline finally works.
next phase:
what i tried : i selected E2E Network for computation handling after analysis of what is feasible for me and sleected a gpu called NVIDIA A30 with respect to what i require and started setting up the environment for full scale  training of my scratch CNN and other models 
what i expected : i expected that after some settings and verifications and a complete setup the training will start 
what happened: it took a lot of time figuring out how to setup the system and stuff which came with a lot of errors so fixed them and started the training 
fixes: 
fix1: Realized the bare Ubuntu server logs in directly as root, so I dropped the sudo commands and just ran apt directly.
fix2: Pip got confused trying to find HuggingFace packages on the PyTorch server, so I split the installations into two separate commands.
fix3: Ubuntu 24.04 blocked me from modifying system packages, so I used the --ignore-installed idna flag to force the install.
fix4: The terminal login kept timing out and hitting rate limits, so I just hardcoded the authentication token directly into the Python script.
fix5:  Pasted long code blocks into the nano editor and it automatically wrapped them, breaking Python's indentation, so I refactored the code into shorter, multi-line blocks.
fix6: The script crashed on minor typos (ln instead of len, and dtypetorch instead of dtype), so I used the Linux sed command to instantly patch the file in the terminal without reopening the editor.
fix7: The script finished BRISC but crashed looking for the Sartaj dataset because the repo name was slightly off, so I updated the script to match the exact Hugging Face repo ID.
update:
currently fixing the failure with the Sartaj dataset 
Ran into a weird bug with the initial Sartaj training runs. The models were hitting 100% training accuracy by epoch 2, but evaluation showed they were just predicting "glioma" for every single image. Turns out there was a silent bug in the data pipeline mapping all the labels to index 0. I fixed it by rewriting the data transform to explicitly map every label variation correctly, and added a pre-flight check to verify the label distribution before training starts. I also added an emergency stop clause so if it hits 99.9% accuracy too early, it just kills the run instead of wasting GPU credits. Retraining the 3 seeds on E2E right now with the fixed pipeline.
coverage (31-1 Novermber):
What I tried: I retrained the Sartaj dataset using the new armored V2 pipeline with geometric augmentations (flips/rotations).
What I expected: The model to learn robust features without hitting the 100% mode collapse.
What actually happened: The emergency stop never triggered. All 3 seeds (42, 43, 44) successfully trained to ~85-88% accuracy. The augmentations prevented memorization, and the explicit label mapping fixed the silent data drop.
The fix and what I decided: I securely uploaded the validated weights as sartaj_seed{seed}_v2.pth to Hugging Face and permanently deleted the broken V1 files.







