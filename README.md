SightReader
===========
Jordan Ford, James Eccles, Signe Munch, Pablo Jovani

Optical Music Recognition in OpenCV

The goal of this project is a semester-long venture into the methods used to recognize and analyze sheet music
in order to retrieve the notes and reconstruct the music digitally.

The main logic that processes sheet music images is located in main.cpp. 

Current Progress:
  1. Read in Sheet music in a digital format
  2. Calculate a median of longest lines contained in the image, and rotate to level the staff.
  3. Threshold the image, and calculate the horizontal and vertical projections. 
     Use the horizontal projection to find the staff.
  4. Traverse the staff, removing the staff lines to isolate the notes.
  5. Use contour function in OpenCV to find the bounding boxes for the notes and other objects.
  6. TBD - Define a set of row coordinates for the five lines of each staff in the image
  7. TBD - Find the centerpoints of the individual and grouped notes - compare these to the five lines to predict
    where the notes lie on the staff.
