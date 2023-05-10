
This repository contains the source code and the images used in the my undergraduation work.

The images in "full_images/heridal" and in "general_patches/heridal_patches" are available online at
< ipsar.fesb.unist.hr/HERIDAL database.html >. Their rights are not mine, and the authors were properly
referentiated in my work.

The article, will soon be available in this repository.

The directories "full_images" and "general_patches" contain only images, and no code. 
These two folders have only one subfolder each, because the directories organization of this repository
was thought to work with different sets of images.

The source code is in "Classification/src".
To run it properly you should run "split.py" first.
The instructions to do it are in its file.
Then, you should run "train_validade_test.py".
After that, you can run "sliding_window.py", "fbeta_precision.py" and "fbeta_recall.py" 
in any order. 
Note that you might want to comment some lines in the end of "fbeta_precision.py" and "fbeta_recall.py".
