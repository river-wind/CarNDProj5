**CarND Project 5 writeup**
---

**Vehicle Detection Project**

The goals / steps of this project were:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./result.mp4

# [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here, each of the rubric points are considered individually, along with a description of how each is addressed each point in the project  implementation.  

---
**Writeup / README**

*1. Provide a Writeup / README that includes all the rubric points and how each one was addressed.*

This is that writeup.

**Histogram of Oriented Gradients (HOG)**

*1. Explain how (and identify where in your code) you extracted HOG features from the training images.*

The code for this step is contained in the second code cell of the IPython notebook named Project5-CarID, under the title "Then, define lesson code for functions to be used in processing features."  The function get_hog_fetures() was developed during the lesson; the official final version is used here in this project.

To actually use this function, an image is first fed to either the extract_features() function or the search_windows() function.  Once there, the got_hog_features() function is called, either against each HOG channel, or just one, based on the provided parameters.  I've settled on using all of the HOG channels, as the results appeared to be the best.

Once the hog_features are returned by the get_hog_features() function,  they are added to the features array to be returned to the main pipeline.

Different color spaces and different `skimage.hog()` parameters (including `orientations`, `pixels_per_cell`, and `cells_per_block`) were tried, with varying results.  I grabbed random images from each of the two classes or training images and displayed them with features overlaid to get a feel for what the output looked like with different settings.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=8` and `cells_per_block=2`  I'm extracting this separately from the main pipeline in order to include this image in the writeup; the code for this extraction is at the very bottom of the notebook.

![alt text][image2]

*1b. Adding spacial binning and color histograms*

I also added spacial binning and color histogram features per the lesson functions bin_spacial() and color_hist().  This was done with spacial size (16,16) and hist_bins of 16.  

*2. Explain how you settled on your final choice of HOG, spacial binning, and histogram parameters.*

I tried various combinations of parameters and found that between the randomness of the input dataset for training, and the number of parameters involved, that it was hard to determine what an optimal combination was.  After a week and a half of working heavily with LUV after some initial positive results, I then found that much of that effort had been fruitless, and YCrCb worked better; all the tests of combinations with the other parameters had to be thrown out.

I finally settled on a color_space of 'YCrCb', orientation of 9, pix per cell of 8, cell per block of 2, All HOG channels, a spacial size of (16,16), and a histogram bin count of 16. 

Were I to do this project again, I would create a structure to change each of these parameters one at a time, programmatically identifying the best combination on the test data.  As it stands, the method used was slipshod, trying values at random and slowing rangefinding something which worked acceptably.

*3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).*

In the third code block of the iPython notebook (under the heading "Train the model"), I read in all the training data, and brake it into two sets, "cars" and "notcars".  After shuffling the sets, I truncated them to a random collection of a maximum of 1200 each; any more than that would cause the Windows Edge browser to crash during training.  Since I'm stuck on a windows machine for running Jupyter at the moment, this presented a limitation I was unable to work around during the time available.  Moving the training to AWS would be preferable, but I would need to rework the code into individual .py files and upload everything to AWS, which I have not had the opportunity to do.

After setting the training variables, I trained a linear SVC using the extracted features from the car and notcar image sets, scaling them using StandardScaler(), and then breaking the set into 80% training and 20% testing data.  The SVC is trained with svc.fit(X_train, y_train), and the accuracy is calculated with svc.score(X_test, y_test). 

Finally, I saved the output of the trained svc to a pickle file for project review purposes.

**Sliding Window Search**

*1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?*

I set up 6 sliding window processes with different windows sizes and logic ranges within the screen to build a list of candidate "car" location for further validation and error checking.  I used 6 different sizes due to my difficulties in getting a good model early on; by fine tuning the sizes and ranges checked, I was able to heat up the correct car areas more effectively.  You can see the 6 sliding window calls in a few places, for example under the heading "Test the model - test 1".

The search_windows() function is then used to actually scan through each window in the collected slide_window grid, and the snipped image windows are used as input for SVC prediction, returning a list of "hot_windows", or windows which have been predicted as containing a car by the SVC.

*2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?*

The 6 window sizes added significant processing time to each video frame.  Due to this, I finally limited my sliding window search to 4, and limited to range of the image they searched (no tiny cars should be visible near the very bottom of the image, so I didn't scan that area with the 32x32 window).  Here are some example images:

![alt text][image4]
---

** Video Implementation**

*1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./result.mp4), with the file name result.mp4.

*2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In order to smooth out detections, limit overlapping boxes, and reduce the impact of false positives, I implemented a two-step averaging process of the "hot_windows" found above.  For each frame of video, I created a heatmap of the collected hot_windows, increasing the heatmap value of a pixel by 1 for every window it fell into.  For areas with cars, this method provided a higher heatmap value than in areas with false positives.  Because of this, the areas with high heatmap values can be expected to be cars, while low heatmap values indicate likely false positives.  

I then took the heatmap created, and as can be seen in the code section near the bottom of the iPython notebook titled "Process the project video in the same manner as the above", I added it to a deque object tracking the prior 10 heatmaps.  The average of these 10 heatmaps is then used to actually build the car bounding boxes.  This helps to limit jitteriness of the frames, as well as average out any intermittent false positive hot_windows which made it through to the heat map.

Using `scipy.ndimage.measurements.label()`, I had the system label the different blobs in the image in order to be abe to draw boxes around them.  The bounding boxes were then drawn around the labels, using the provided draw_labeled_boxes()

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

* Here are the six test frames and their corresponding heatmaps and labels:*

![alt text][image5]

* Here the resulting bounding boxes are drawn onto the last frame in the series:*
![alt text][image7]
---

**Discussion**

*1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?*

The largest problems I faced were with the model itself.  From the sheer number of possible combinations of settings on the feature extraction, to the limitation on how much data I could actually use to train the model, much of my time spent on this project likely went towards fighting with output from what I now know are bad models.  Being able to run the entire dataset should help reduce false positives further, so I will continue down that path after this submission.  

The time each frame takes to process was another limitation.  With all 6 multi-layered sliding windows active, each frame took about 170 seconds to process, making the video render in over 30 hours.  The final video, even with the limited sliding windows, still took over 6 hours to process.  Between these two issue, I spent nearly 150 hours spread over three weeks actively working on this project, causing it to be late.  I'd like to have merged the code with the advanced lane finding, given how well they would mesh together, but I did not have a chance to get to that.

Within the actual project behavior, my main struggle was with the heatmap.  By averaging the heatmaps over time, false positives are reduced; but only if there isn't a false positive in the first frame.  In that case, the first frame is over-weighted in subsequent frames; in most of my trained models, the first frame contained a false positive in the center-top of the road where the yellow line meets the horizon.  This was a strong hit, and would generally get 3-5 hot windows flagging it.  Averaging did not resolve this, so I attempted to eliminate the issue by starting my heatmaps array with a set of 10 all-black images.  Strangely, this did not work.  As soon as the next heatmap was added, the average heatmap reflected the new item only.  I'm not sure why this is, and further investigation is needed.

Also, the bounding boxes are not tight around the cars.  I suspect I may need to add another step to the pipeline; first identifying the likely cars, then re-analyzing that area to better define the car edges.  As it stands, the bounding box only include the heatmap thresholded area, which leaves out any of the outside edges of the car where only 1 or 2 bounding boxes found the vehicle.  Possibly redesigning the bounding box logic to use the outer edges of the hot_windows for that heatmap cluster would do a better job - though again, false positives would need to be handled.

In the end, the final result video is better than my initial results by a fair margin, however it remains less impressive than I would prefer.  The final training of the network produced a very low hit rate on the cars themselves, forcing me to use a threshold level of 1, which increased the false positives.  The averaging limited the number which appear in the final video, but many are still present.  The prior training of the network produced too many false positives, with a solid false positive in the center of the lane which stuck around for most of the video at a threshold level of 5.  Given this wide result diffrence, it is clear that the training of the model itself is still not optimal.  I'd like to keep working on this, but need to turn the project in to not be held back.  I'll continue working on it after I push the submit button.
