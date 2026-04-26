If we turn up the blur, we can identify the black notes
We can then use the positions of the black notes to parse for white notes

the black keys disappear when notes hit the keyboard because the color change makes the keys no longer recognizable in the black mask
- We can switch the method to look for rectangular shapes instead of color

When we search for black keys, compare the number of keys found to the number of keys detected in the last n frames
If the count is less than the peak

There is a stabilization period before the notes start falling
If we detect that the number of keys has been the same for some time, we declare that the new norm

We could also use a "voting" system
When we detect keys, note how many keys we detect. At every frame, we place a vote according to the number of keys we detect
The most voted for number is the ground truth. If we detect less than that many keys, then a key has been played

yt-dlp -- print filename -o "%(title)s.%(ext)s" iElUjQXQkPc

The current adjustment algorithm uses standard deviation of the area, but it's probably better to use standard deviation of the height / width or both

The current method using a black pixel mask just doesn't generalize well

Given the valley information about a stratum, we want to match each valley to a component
The components are not known at the start of the matching and must be dynamically created based on similarity
I did this before using the sort_into_buckets function but that function takes 2 pixel thresholds
Can we do this without using any hard pixel thresholds?
We could calculate the average separation between valleys and use a fraction of that to determine how big the pixel threshold should be


A valley in one strata is similar to a valley in an adjacent strata if 
- They have similar widths (more or less equivalent to the following 2)
- They have similar start values
- They have similar end values
- The second valley is the valley in that strata with the most similar start value to the first valley

1. Start with the index of all strata at 0
2. Check if the start of the valley at those indexes are similar
3. For any that are not similar, check if their start is less than or greater than the average of the similar ones
    3a. If less than, advance the index of that strata and jump to step 2
    3b. If greater than, ignore that strata and 


Given a big block of data from all the strata (~50). We want to identify which valleys come from the same key gap
1. Start with the index of all strata at 0
2. Check if the start of the valley at those indexes are similar, meaning:
    a. Their start values are within a small distance of each other
3. For any values that are considered not similar, check if their start is less than or greater than the average of the similar ones
    3a. If less than, advance the index of that strata and check again for similarity
    3b. If greater than, ignore the valley from that strata 
4. Search for an existing bucket to put the similar values into. The bucket is chosen based on:
    a. Similarity in average start. Pick the bucket with the lowest distance
    b. The distance from the average to the bucket must also be less than the largest difference between 2 similar
4. Assign all the similar values to a common bucket


Given a group of strata and a big array containing valleys from the same key gap, 
we want to assign the valleys in the group of strata to a key gap


We want to go up (SB / 2 - 1) rows
Each row is 512 pixels
So, we want to subtract 512 * (SB / 2 - 1) from the Middle Left Point

The middle left point is stored in t5
We want to subtract 512 * (SB / 2 - 1) from t5

A key is activated if there is a spike in color distance across all strata, 

Apply voting approach to finding a base color for the keys

A color detection approach won't always work because of missing frames. sometimes when a key is pressed fast, there is no in-between.
To fix this, we have to use a shape based approach. 
In this method, we can focus only on the strata containing both black and white keys, since it seems like the length of the falling notes 
    is only as large as those gaps

Multi-blur + canny using otsu's thresh seems to work well for detecting keys, so long as the search area is restricted.
If the image contains the keyboard or other major features (like in birdbrain the top of the screen is the sheet music),
    then otsu fails to find a good threshold. We could also just use a static threshold but i dislike that and would prefer
    dynamically determined thresholds

Actually, the aftereffects produced by keys messes up otsu (in birdbrain at least). The text seems to be an important part of why
    it gets messed up. Textured backgrounds do the same

Maybe we could do something with channels? Like what if we try to detect only in the green channel?

MOG Bg sub is useful but seems inconsistent. My guess is that because the notes fall along the same lines, bg sub will start to consider 
    the notes part of the background the longer a line has notes in it

KNN seems to get easily garbled by background noise but the contour is still there, so maybe we use contour detection over canny?
    The garbling seems to be a consequence of notes merging into bg

Otsu's binarization along the strata seems like the most consistent method when the notes are actually there. It's super sensitive to noise though
We might be able to replace the quantization with otsu's binarization actually

Using the value channel of HSV instead of grayscale has some massive advantages on birdbrain. 