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
