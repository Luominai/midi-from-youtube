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